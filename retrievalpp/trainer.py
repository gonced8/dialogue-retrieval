import json

from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from sentence_transformers.util import dot_score
import torch
from transformers import Trainer

from losses import *
from retrieval import generate_index, retrieve


class RetrievalTrainer(Trainer):
    def __init__(
        self,
        heuristic="rouge",
        n_candidates=10,
        retrieval_exclude_indices=None,
        loss_fn="cross_entropy",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Heuristic function
        if heuristic == "bleu":
            self.heuristic = BLEU(effective_order=True)
            self.heuristic_fn = (
                lambda hypothesis, reference: self.heuristic.sentence_score(
                    hypothesis=hypothesis, references=[reference]
                ).score
                / 100
            )
        elif heuristic == "rouge":
            self.heuristic = RougeScorer(["rougeL"])
            self.heuristic_fn = lambda hypothesis, reference: self.heuristic.score(
                target=reference, prediction=hypothesis
            )["rougeL"].fmeasure

        self.n_candidates = n_candidates
        self.loss_fn = loss_fn

        if retrieval_exclude_indices:
            with open(retrieval_exclude_indices, "r") as f:
                self.exclude_indices = json.load(f)
        else:
            self.exclude_indices = None

        self.index_idx = None

    def compute_loss(self, model, inputs, return_outputs=False):
        context_embeddings = model.encoder_question(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        answer_embeddings = model.encoder_answer(
            input_ids=inputs["answer_input_ids"],
            attention_mask=inputs["answer_attention_mask"],
        )

        # Compute loss
        if self.loss_fn == "cross_entropy":
            scores = dot_score(context_embeddings, answer_embeddings)
            loss = torch.nn.functional.cross_entropy(
                scores, torch.arange(0, scores.size(0), device=scores.device)
            )

        elif self.loss_fn == "heuristic":
            scores_h = heuristic_score(
                self.heuristic_fn, inputs["answer"], context_embeddings.device
            )
            scores_m = dot_score(context_embeddings, answer_embeddings)

            loss = compare_scores_diff(scores_h, scores_m)
            loss = torch.mean(loss)

        return (
            (loss, (context_embeddings, answer_embeddings)) if return_outputs else loss
        )

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Generate index
        self.index = generate_index(
            self.get_train_dataloader(), self.model.encoder_answer, index_key="answer"
        )

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=["answer"], metric_key_prefix="test"):
        # Generate index
        self.index = generate_index(
            self.get_train_dataloader(), self.model.encoder_answer, index_key="answer"
        )

        return super().predict(test_dataset, ignore_keys, metric_key_prefix)

    def prediction_step(
        self, model, inputs, prediction_loss_only=None, ignore_keys=None
    ):
        queries = inputs.pop("idx")
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            # Hack to get loss only for validation, not for test
            if ignore_keys is None:
                loss, (context_embeddings, _) = self.compute_loss(
                    model, inputs, return_outputs=True
                )
            else:
                loss = None
                context_embeddings = None

            indices, _ = retrieve(
                model.encoder_question,
                inputs,
                self.index,
                self.n_candidates,
                outputs=context_embeddings,
                index_key="context",
                queries_ids=inputs["id"],
            )

            candidates = torch.tensor(indices, dtype=torch.long)

        return loss, candidates, queries

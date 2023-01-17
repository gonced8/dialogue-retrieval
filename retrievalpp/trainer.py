from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from sentence_transformers.util import dot_score
import torch
from transformers import Trainer

from retrieval import generate_index, retrieve


class RetrievalTrainer(Trainer):
    def __init__(
        self,
        heuristic="bleu",
        loss_student_scale=1.0,
        n_candidates=10,
        index_key="answer",
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

        self.loss_student_scale = loss_student_scale
        self.n_candidates = n_candidates
        self.index_key = index_key
        self.index_idx = None

    def compute_loss(self, model, inputs, return_outputs=False):
        context_embeddings = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        if self.index_key == "answer":
            answer_embeddings = model(
                input_ids=inputs["answer_input_ids"],
                attention_mask=inputs["answer_attention_mask"],
            )
        else:
            answer_embeddings = context_embeddings

        # Compute loss that uses heuristic as teacher
        # scores_h = heuristic_score(
        #   self.heuristic_fn, inputs["answer"], context_embeddings.device
        # )
        # scores_m = dot_score(context_embeddings, answer_embeddings)

        # loss = compare_rank_scores_neighbors(scores_h, scores_m)
        # loss = torch.mean(loss)

        # Compute loss
        scores = dot_score(context_embeddings, answer_embeddings)
        loss = torch.nn.functional.cross_entropy(
            scores, torch.arange(0, scores.size(0), device=scores.device)
        )

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
        self.index, self.index_idx = generate_index(
            self.get_train_dataloader(), self.model, self.index_key
        )

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        # Generate index
        self.index, self.index_idx = generate_index(
            self.get_train_dataloader(), self.model, self.index_key
        )

        return super().predict(test_dataset, ignore_keys, metric_key_prefix)

    def prediction_step(
        self, model, inputs, prediction_loss_only=None, ignore_keys=None
    ):
        queries = inputs.pop("idx")
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, (context_embeddings, _) = self.compute_loss(
                model, inputs, return_outputs=True
            )
            indices, _ = retrieve(
                model,
                None,
                self.index,
                self.n_candidates,
                outputs=context_embeddings,
            )

        # Convert candidates to correct ids
        indices = [
            [self.index_idx[i] for i in sample_candidates]
            for sample_candidates in indices
        ]
        candidates = torch.tensor(indices, dtype=torch.long)

        return loss, candidates, queries

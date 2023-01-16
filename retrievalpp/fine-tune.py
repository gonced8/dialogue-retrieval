from argparse import ArgumentParser, BooleanOptionalAction
from itertools import islice
import json

from datasets import load_dataset
import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from sentence_transformers.util import dot_score
import torch
from tqdm.contrib import tzip
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

from data_collator import RetrievalDataCollator
from losses import *
from model import Encoder
from retrieval import generate_index, retrieve


def build_samples(samples, indices, args, tokenizer):
    # Select data
    contexts = [" ".join(turns[:-1]) for turns in samples["text"]]
    answers = [turns[-1] for turns in samples["text"]]

    # Tokenize
    context_inputs = tokenizer(
        contexts,
        max_length=args.max_length,
        truncation=True,
        return_length=True,
    )
    answer_inputs = tokenizer(
        answers,
        max_length=args.max_length,
        truncation=True,
        return_length=True,
    )

    # Verify if possible truncation
    if any(sample_len == args.max_length for sample_len in context_inputs["length"]):
        print("WARNING: Possible truncation occurring in input_ids.")

    if any(sample_len == args.max_length for sample_len in answer_inputs["length"]):
        print("WARNING: Possible truncation occurring in answer_input_ids.")

    return {
        "idx": indices,
        "answer": answers,
        "input_ids": context_inputs["input_ids"],
        "attention_mask": context_inputs["attention_mask"],
        "answer_input_ids": answer_inputs["input_ids"],
        "answer_attention_mask": answer_inputs["attention_mask"],
    }


class RetrievalMetrics:
    def __init__(self, index_dataset, query_dataset, output):
        # Load datasets
        with open(index_dataset, "r") as f:
            index_data = json.load(f)["data"]
        self.index_dataset = [sample["text"][-1] for sample in index_data]

        with open(query_dataset, "r") as f:
            query_data = json.load(f)["data"]
        self.query_dataset = [sample["text"][-1] for sample in query_data]

        # Name of file where output metrics and retrieval results will be saved
        self.output = self.output

        # Load metrics
        self.bleu = BLEU(effective_order=True)
        self.rouge = RougeScorer(["rougeL"])

    def __call__(self, eval_pred):
        candidates_ids = eval_pred.predictions
        candidates = [
            [self.index_dataset[idx] for idx in sample_candidates]
            for sample_candidates in candidates_ids
        ]

        answers_ids = eval_pred.label_ids
        answers = [self.query_dataset[idx] for idx in answers_ids]

        # Compute BLEUo scores
        bleu_scores = np.array(
            [
                [
                    self.bleu.sentence_score(candidate, [answer]).score
                    for candidate in sample_candidates
                ]
                for sample_candidates, answer in tzip(
                    candidates, answers, desc="Computing BLEU"
                )
            ]
        )

        # Compute ROUGE scores
        rouge_scores = np.array(
            [
                [
                    self.rouge.score(answer, candidate)["rougeL"].fmeasure
                    for candidate in sample_candidates
                ]
                for sample_candidates, answer in tzip(
                    candidates, answers, desc="Computing ROUGE"
                )
            ]
        )

        # Get scores from best candidate and average
        best_bleu = np.argmax(bleu_scores, axis=1)
        best_rouge = np.argmax(bleu_scores, axis=1)

        bleu_score = bleu_scores[best_bleu].mean()
        rouge_score = rouge_scores[best_rouge].mean()

        # Compute MRR from BLEU and ROUGE
        mrr_bleu = np.reciprocal(best_bleu + 1.0).mean()
        mrr_rouge = np.reciprocal(best_rouge + 1.0).mean()

        metrics = {
            "bleu": bleu_score,
            "rouge": rouge_score,
            "mrr_bleu": mrr_bleu,
            "mrr_rouge": mrr_rouge,
        }

        # Retrieval results

        # Save to file

        return metrics


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
        #    self.heuristic_fn, inputs["answer"], context_embeddings.device
        # )
        # scores_m = dot_score(context_embeddings, answer_embeddings)

        # loss = compare_scores_diff(scores_h, scores_m)
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


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--metric_for_best_model", type=str, default="bleu")
    parser.add_argument(
        "--resume_from_checkpoint", default=False, action=BooleanOptionalAction
    )
    parser.add_argument(
        "--index",
        type=str,
        choices=["context", "answer"],
        default="answer",
    )
    parser.add_argument(
        "--heuristic", type=str, choices=["bleu", "rouge"], default="rouge"
    )
    parser.add_argument("--n_candidates", type=int, default=10)
    parser.add_argument("--logging", default=True, action=BooleanOptionalAction)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    args = parser.parse_args()

    # Initialize model and tokenizer
    model = Encoder(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    data_files = {
        "train": args.train_dataset,
        "validation": args.val_dataset,
    }
    dataset = load_dataset("json", data_files=data_files, field="data")

    # Prepare samples
    dataset = dataset.map(
        build_samples,
        with_indices=True,
        batched=True,
        batch_size=args.preprocess_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer},
    )
    dataset.set_format(
        type="torch",
        columns=[
            "idx",
            "input_ids",
            "attention_mask",
            "answer_input_ids",
            "answer_attention_mask",
        ],
        output_all_columns=True,
    )

    # Setup data collator
    data_collator = RetrievalDataCollator(tokenizer=tokenizer, padding=True)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=f"checkpoints/{args.experiment_name}",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        metric_for_best_model=f"eval_{args.metric_for_best_model}",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_strategy="steps" if args.logging else "no",
    )

    # Create Trainer
    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=RetrievalMetrics(args.train_dataset, args.val_dataset),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10, early_stopping_threshold=1e-4
            )
        ],
        heuristic=args.heuristic,
        n_candidates=args.n_candidates,
        index_key=args.index,
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

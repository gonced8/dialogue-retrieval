from argparse import ArgumentParser, BooleanOptionalAction
from itertools import islice
import json

from datasets import load_dataset
import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
from tqdm.contrib import tzip
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

from data_collator import DataCollatorWithPaddingAndText
from losses import *
from model import Encoder
from retrieval import generate_index, retrieve


def build_samples(samples, indices, args, tokenizer):
    texts = []
    answers = []

    for turns in samples["text"]:
        if args.index == "context":
            text = " ".join(turns[:-1])
        elif args.index == "answer":
            text = turns[-1]
        else:
            text = " ".join(turns)

        texts.append(text)
        answers.append(text[-1])

    model_inputs = tokenizer(
        texts,
        max_length=args.max_length,
        truncation=True,
        return_length=True,
    )
    # Verify if possible truncation
    if any(sample_len == args.max_length for sample_len in model_inputs["length"]):
        print("WARNING: Possible truncation occurring in input_ids.")

    return {
        "idx": indices,
        "answer": answers,
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
    }


class RetrievalMetrics:
    def __init__(self, train_dataset):
        with open(train_dataset, "r") as f:
            data = json.load(f)["data"]
        self.dataset = {sample["id"]: sample["text"][-1] for sample in data}
        self.index_ids_labels = None

        # Load metrics
        self.bleu = BLEU(effective_order=True)
        self.rouge = RougeScorer(["rougeL"])

    def __call__(self, eval_pred):
        candidates_ids = eval_pred.predictions
        candidates = [
            [self.dataset[self.index_ids_labels[i]] for i in sample_candidates]
            for sample_candidates in candidates_ids
        ]

        answers_ids = eval_pred.label_ids
        answers = [next(islice(self.dataset.values(), i, i + 1)) for i in answers_ids]

        # Compute BLEUo scores
        print("Computing BLEU")
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
        bleu_score = bleu_scores.max(1).mean()
        rouge_score = rouge_scores.max(1).mean()

        # Compute MRR from BLEU and ROUGE
        mrr_bleu = np.reciprocal(np.argmax(bleu_scores, axis=1) + 1).mean()
        mrr_rouge = np.reciprocal(np.argmax(rouge_scores, axis=1) + 1).mean()

        return {
            "bleu": bleu_score,
            "rouge": rouge_score,
            "mrr_bleu": mrr_bleu,
            "mrr_rouge": mrr_rouge,
        }


class RetrievalTrainer(Trainer):
    def __init__(
        self, heuristic="bleu", loss_student_scale=1.0, n_candidates=10, **kwargs
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

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)

        scores_h = heuristic_score(self.heuristic_fn, inputs["answer"], outputs.device)
        scores_m = model_score(outputs) * self.loss_student_scale

        loss = compare_rank_scores(scores_h, scores_m)
        loss = torch.mean(loss)

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Generate index
        train_dataloader = self.get_train_dataloader()
        self.index, self.compute_metrics.index_ids_labels = generate_index(
            train_dataloader, self.model
        )

        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def prediction_step(
        self, model, inputs, prediction_loss_only=None, ignore_keys=None
    ):
        queries = inputs.pop("idx")
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            indices, _ = retrieve(
                model,
                None,
                self.index,
                self.n_candidates,
                outputs=outputs,
            )
        candidates = torch.tensor(indices, dtype=torch.long)

        return loss, candidates, queries


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--metric_for_best_model", type=str, default="bleu")
    parser.add_argument(
        "--resume_from_checkpoint", default=False, action=BooleanOptionalAction
    )
    parser.add_argument(
        "--index",
        type=str,
        choices=["context", "answer", "conversation"],
        default="answer",
    )
    parser.add_argument(
        "--heuristic", type=str, choices=["bleu", "rouge"], default="rouge"
    )
    parser.add_argument("--loss_student_scale", type=float, default=1.0)
    parser.add_argument("--n_candidates", type=int, default=10)
    args = parser.parse_args()

    # Initialize model and tokenizer
    model = Encoder(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    data_files = {
        "train": args.train_dataset,
        "validation": args.val_dataset,
        "test": args.test_dataset,
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
        columns=["idx", "input_ids", "attention_mask"],
        output_all_columns=True,
    )

    # Setup data collator
    data_collator = DataCollatorWithPaddingAndText(
        tokenizer=tokenizer, padding="longest"
    )

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
    )

    # Create Trainer
    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=RetrievalMetrics(args.train_dataset),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=1e-4
            )
        ],
        heuristic=args.heuristic,
        loss_student_scale=args.loss_student_scale,
        n_candidates=args.n_candidates,
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

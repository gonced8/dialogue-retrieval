from argparse import ArgumentParser, BooleanOptionalAction

from datasets import load_dataset
import evaluate
from sacrebleu.metrics import BLEU
from rouge_score.rouge_scorer import RougeScorer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pairwise_dot_score
import torch
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

from data_collator import DataCollatorWithPaddingAndText


def build_samples(samples, args, tokenizer):
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
        answers.append(turns[-1])

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
        "id": samples["id"],
        "answer": answers,
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
    }


def compute_metrics(eval_pred):
    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    pred_ids, labels = eval_pred

    # Pad masked tokens
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # Decode tensors
    predictions = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    references = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Compute metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rougeL"],
    )

    return {"bleu": bleu_score["bleu"], **rouge_score}


def heuristic_score(heuristic_fn, answers, device):
    scores = []

    for i, reference in enumerate(answers[:-1]):
        for j, hypothesis in enumerate(answers[i + 1 :]):
            scores.append(heuristic_fn(hypothesis, reference))

    return torch.tensor(scores, device=device).view(-1)


def model_score(embeddings):
    n, d = embeddings.shape

    references_index = (
        torch.tensor(
            [i for i in range(n - 1) for j in range(n - 1 - i)],
            dtype=torch.long,
            device=embeddings.device,
        )
        .view(-1, 1)
        .expand(-1, d)
    )
    hypothesis_index = (
        torch.tensor(
            [j for i in range(1, n) for j in range(i, n)],
            dtype=torch.long,
            device=embeddings.device,
        )
        .view(-1, 1)
        .expand(-1, d)
    )

    references = torch.gather(embeddings, dim=0, index=references_index)
    hypothesis = torch.gather(embeddings, dim=0, index=hypothesis_index)

    return pairwise_dot_score(references, hypothesis)


def compare_rank_scores(scores_h, scores_m, b=None):
    n = scores_m.size(0)

    index_i = [i for i in range(n - 1) for j in range(n - 1 - i)]
    index_j = [j for i in range(1, n) for j in range(i, n)]

    index_i = torch.tensor(
        index_i,
        dtype=torch.long,
        device=scores_m.device,
    )
    index_j = torch.tensor(
        index_j,
        dtype=torch.long,
        device=scores_m.device,
    )

    scores_h_i = torch.gather(scores_h, dim=0, index=index_i)
    scores_h_j = torch.gather(scores_h, dim=0, index=index_j)
    scores_m_i = torch.gather(scores_m, dim=0, index=index_i)
    scores_m_j = torch.gather(scores_m, dim=0, index=index_j)

    return (scores_h_i - scores_h_j) - (scores_m_i - scores_m_j)


class RetrievalTrainer(Trainer):
    def __init__(self, heuristic="bleu", loss_student_scale=1.0, **kwargs):
        super().__init__(**kwargs)
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

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
        )["sentence_embedding"]

        scores_h = heuristic_score(self.heuristic_fn, inputs["answer"], outputs.device)
        scores_m = model_score(outputs) * self.loss_student_scale

        # print("Scores Heuristic", scores_h, "", sep="\n")
        # print("Scores Model", scores_m, "", sep="\n")
        # print("========================================")
        # input()

        diff = compare_rank_scores(scores_h, scores_m)
        loss = torch.mean(diff**2, dim=0)

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
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
        default="context",
    )
    parser.add_argument(
        "--heuristic", type=str, choices=["bleu", "rouge"], default="bleu"
    )
    parser.add_argument("--loss_student_scale", type=float, default=1.0)
    args = parser.parse_args()

    # Initialize model and tokenizer
    model = SentenceTransformer(args.model)
    tokenizer = model.tokenizer

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
        batched=True,
        batch_size=args.preprocess_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer},
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True
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
        # metric_for_best_model=f"eval_{args.metric_for_best_model}",
        # load_best_model_at_end=True,
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
        # compute_metrics=compute_metrics,
        # callbacks=[
        #    EarlyStoppingCallback(
        #        early_stopping_patience=5, early_stopping_threshold=1e-4
        #    )
        # ],
        heuristic=args.heuristic,
        loss_student_scale=args.loss_student_scale,
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

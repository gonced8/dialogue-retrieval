from argparse import ArgumentParser, BooleanOptionalAction

from datasets import load_dataset
from datasets.features import Features
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from data import build_samples, RetrievalDataCollator, vary_context_length
from losses import *
from metrics import RetrievalMetrics
from model import RetrievalConfig, RetrievalModel
from trainer import RetrievalTrainer


def none_or_str(value):
    if value.lower() == "none":
        return None
    return value


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["train", "validation", "test"], default="train"
    )
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument(
        "--model", type=str, default="sentence-transformers/multi-qa-mpnet-base-dot-v1"
    )
    parser.add_argument(
        "--activation_fn", type=str, default="cls", choices=["cls", "mean"]
    )
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--preprocess_batch_size", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--metric_for_best_model", type=str, default="loss")
    parser.add_argument(
        "--resume_from_checkpoint", default=False, action=BooleanOptionalAction
    )
    parser.add_argument(
        "--heuristic", type=str, choices=["bleu", "rouge"], default="rouge"
    )
    parser.add_argument("--n_candidates", type=int, default=10)
    parser.add_argument("--logging", default=True, action=BooleanOptionalAction)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dual", default=False, action=BooleanOptionalAction)
    parser.add_argument(
        "--loss_fn",
        type=str,
        choices=["cross_entropy", "heuristic", "correlation"],
        default="cross_entropy",
    )
    parser.add_argument("--expand_samples", default=False, action=BooleanOptionalAction)
    parser.add_argument(
        "--max_nturns", type=int, default=5, help="Maximum length of context."
    )
    parser.add_argument("--data_field", type=none_or_str, default="data")
    parser.add_argument("--delexicalized", default=False, action=BooleanOptionalAction)
    parser.add_argument("--index_key", type=str, default="answer")
    args = parser.parse_args()

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        RetrievalModel(RetrievalConfig(args.model, args.dual, args.activation_fn))
        if not args.checkpoint
        else RetrievalModel.from_pretrained(args.checkpoint)
    )

    # Load dataset
    data_files = {}
    if args.train_dataset:
        data_files["train"] = args.train_dataset
    if args.val_dataset:
        data_files["validation"] = args.val_dataset
    if args.test_dataset:
        data_files["test"] = args.test_dataset

    dataset = load_dataset(
        "json",
        data_files=data_files,
        field=args.data_field,
    )

    # Prepare samples
    if args.expand_samples:
        dataset = dataset.map(
            vary_context_length,
            batched=True,
            batch_size=args.preprocess_batch_size,
            fn_kwargs={
                "args": args,
            },
        )

    dataset = dataset.map(
        build_samples,
        with_indices=True,
        batched=True,
        batch_size=args.preprocess_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer},
    )

    # Filter overflowing samples
    dataset = dataset.filter(lambda sample: not sample["overflow"])
    dataset = dataset.remove_columns("overflow")

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

    print(dataset)

    # Shuffle validation dataset (it will be better to compute a loss)
    if args.val_dataset:
        dataset["validation"] = dataset["validation"].shuffle(seed=42)

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
        save_total_limit=2,
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_strategy="steps" if args.logging else "no",
    )

    # Initialize metrics
    metrics = RetrievalMetrics(
        index_dataset=args.train_dataset,
        query_dataset=args.val_dataset if args.val_dataset else args.test_dataset,
        field=args.data_field,
        output=args.output,
        delexicalized=args.delexicalized,
    )

    # Initialize callbacks
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-4)
    ]

    # Create Trainer
    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"] if args.train_dataset else None,
        eval_dataset=dataset["validation"] if args.val_dataset else None,
        compute_metrics=metrics,
        callbacks=callbacks,
        heuristic=args.heuristic,
        n_candidates=args.n_candidates,
        loss_fn=args.loss_fn,
        index_key=args.index_key,
    )

    # Run
    if args.mode == "train":
        trainer.evaluate()
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elif args.mode == "validation":
        trainer.evaluate()
    elif args.mode == "test":
        trainer.predict(test_dataset=dataset["test"])

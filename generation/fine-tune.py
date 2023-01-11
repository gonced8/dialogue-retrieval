from argparse import ArgumentParser, BooleanOptionalAction

from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


def build_samples(samples):
    input_texts = []

    for context, knowledge in zip(samples["context"], samples["knowledge"]):
        input_text = " EOS ".join(context)
        if knowledge and args.candidates > 0:
            knowledge = " | ".join(knowledge[: args.candidates])
            input_text += " <|Knowledge|> " + knowledge
        input_text += " => "

        input_texts.append(input_text)

    model_inputs = tokenizer(
        input_texts,
        max_length=args.max_input_length,
        truncation=True,
        return_length=True,
    )
    labels = tokenizer(
        samples["response"],
        max_length=args.max_output_length,
        truncation=True,
        return_attention_mask=False,
        return_length=True,
    )

    # Verify if possible truncation
    if any(
        sample_len == args.max_input_length for sample_len in model_inputs["length"]
    ):
        print("WARNING: Possible truncation occurring in input_ids.")

    if any(sample_len == args.max_output_length for sample_len in labels["length"]):
        print("WARNING: Possible truncation occurring in labels.")

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"],
    }


def compute_metrics(eval_pred):
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
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )

    return {"bleu": bleu_score["bleu"], **rouge_score}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="microsoft/GODEL-v1_1-base-seq2seq"
    )
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--predict_with_generate", default=True, action=BooleanOptionalAction
    )
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--metric_for_best_model", type=str, default="rougeL")
    parser.add_argument(
        "--resume_from_checkpoint", default=False, action=BooleanOptionalAction
    )
    args = parser.parse_args()

    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_files = {"train": args.train_dataset, "validation": args.val_dataset}
    dataset = load_dataset("json", data_files=data_files, field="data")

    # Prepare samples
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    dataset = dataset.map(
        build_samples, batched=True, batch_size=args.preprocess_batch_size
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest"
    )

    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{args.experiment_name}",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=True,
        generation_max_length=args.max_output_length,
        generation_num_beams=4,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        metric_for_best_model=f"eval_{args.metric_for_best_model}",
        load_best_model_at_end=True,
    )

    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=None
        if args.predict_with_generate
        else preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5, early_stopping_threshold=1e-4
            )
        ],
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

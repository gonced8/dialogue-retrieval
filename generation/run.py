from argparse import ArgumentParser, BooleanOptionalAction
import json

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


def build_samples(samples, args, tokenizer):
    input_texts = []

    for context, knowledge in zip(samples["context"], samples["knowledge"]):
        input_text = " EOS ".join(context)
        if knowledge and args.candidates > 0:
            unique_knowledge = [*set(knowledge)]  # Remove duplicate candidates
            knowledge = " | ".join(unique_knowledge[: args.candidates])
            input_text += " <|Knowledge|> " + knowledge
        input_text += " => "

        input_texts.append(input_text)

    responses = samples["delexicalized"] if args.delexicalized else samples["response"]

    model_inputs = tokenizer(
        input_texts,
        max_length=args.max_input_length,
        truncation=True,
        return_length=True,
    )

    labels = tokenizer(
        responses,
        max_length=args.max_output_length,
        truncation=True,
        return_attention_mask=False,
        return_length=True,
    )

    # decoder_input_ids = tokenizer(
    #     "<pad>SYSTEM: ",
    #     max_length=args.max_output_length,
    #     truncation=True,
    #     add_special_tokens=False,
    #     return_attention_mask=False,
    #     return_length=True,
    # )

    # Verify if possible truncation
    if any(
        sample_len == args.max_input_length for sample_len in model_inputs["length"]
    ):
        print("WARNING: Possible truncation occurring in input_ids.")

    if any(sample_len == args.max_output_length for sample_len in labels["length"]):
        print("WARNING: Possible truncation occurring in labels.")

    # if any(
    #     sample_len == args.max_output_length
    #     for sample_len in decoder_input_ids["length"]
    # ):
    #     print("WARNING: Possible truncation occurring in labels.")

    # Repeat decoder input ids for every sample
    # decoder_input_ids = [decoder_input_ids["input_ids"].copy() for _ in samples["id"]]

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"],
        # "decoder_input_ids": decoder_input_ids,
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

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)

    # ROUGE
    rouge = evaluate.load("rouge")
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
    parser.add_argument(
        "--mode", type=str, choices=["train", "validation", "test"], default="train"
    )
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument(
        "--model", type=str, default="microsoft/GODEL-v1_1-base-seq2seq"
    )
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--predict_with_generate", default=True, action=BooleanOptionalAction
    )
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--metric_for_best_model", type=str, default="bleu")
    parser.add_argument(
        "--resume_from_checkpoint", default=False, action=BooleanOptionalAction
    )
    parser.add_argument("--delexicalized", default=False, action=BooleanOptionalAction)
    parser.add_argument("--do_sample", default=False, action=BooleanOptionalAction)
    parser.add_argument("--results", type=str)
    args = parser.parse_args()

    # Load dataset
    data_files = {}
    if args.train_dataset:
        data_files["train"] = args.train_dataset
    if args.val_dataset:
        data_files["validation"] = args.val_dataset
    if args.test_dataset:
        data_files["test"] = args.test_dataset
    dataset = load_dataset("json", data_files=data_files, field="data")

    # Prepare samples
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = dataset.map(
        build_samples,
        batched=True,
        batch_size=args.preprocess_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer},
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.checkpoint if args.checkpoint else args.model
    )

    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest"
    )

    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{args.experiment_name}",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size
        if args.mode == "test"
        else args.val_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=args.predict_with_generate,
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
        train_dataset=dataset["train"] if args.train_dataset else None,
        eval_dataset=dataset["validation"] if args.val_dataset else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=None
        if args.predict_with_generate
        else preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10, early_stopping_threshold=1e-4
            )
        ],
    )

    # Run
    if args.mode == "train":
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    elif args.mode == "validation":
        trainer.evaluate()

    elif args.mode == "test":
        output = trainer.predict(
            test_dataset=dataset["test"],
            max_length=args.max_output_length,
            do_sample=args.do_sample,
        )

        predictions = tokenizer.batch_decode(
            output.predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Save results
        dataset.reset_format()
        if "delexicalized" in dataset["test"][0]:
            results = [
                {
                    "id": sample["id"],
                    "context": sample["context"],
                    "response": sample["response"],
                    "delexicalized": sample["delexicalized"],
                    "model_answer": prediction,
                    "knowledge": sample["knowledge"][: args.candidates],
                }
                for sample, prediction in zip(dataset["test"], predictions)
            ]
        else:
            results = [
                {
                    "id": sample["id"],
                    "context": sample["context"],
                    "response": sample["response"],
                    "delexicalized": sample["delexicalized"],
                    "model_answer": prediction,
                    "knowledge": sample["knowledge"][: args.candidates],
                }
                for sample, prediction in zip(dataset["test"], predictions)
            ]

        with open(args.results, "w") as f:
            json.dump(
                {
                    "version": args.experiment_name,
                    "metrics": output.metrics,
                    "data": results,
                },
                f,
                indent=4,
            )
        print(f"Saved to: {args.results}")

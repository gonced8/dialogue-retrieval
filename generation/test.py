from argparse import ArgumentParser, BooleanOptionalAction
import json

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
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

    # Verify if possible truncation
    if any(
        sample_len == args.max_input_length for sample_len in model_inputs["length"]
    ):
        print("WARNING: Possible truncation occurring in input_ids.")

    decoder_input_ids = tokenizer(
        tokenizer.pad_token + "SYSTEM: ",
        max_length=args.max_output_length,
        truncation=True,
        return_attention_mask=False,
        return_length=True,
    )

    # Verify if possible truncation
    if any(
        sample_len == args.max_output_length
        for sample_len in decoder_input_ids["length"]
    ):
        print("WARNING: Possible truncation occurring in labels.")

    # Repeat decoder input ids for every sample
    decoder_input_ids = [decoder_input_ids["input_ids"].copy() for _ in samples["id"]]

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "decoder_input_ids": decoder_input_ids,
    }


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="microsoft/GODEL-v1_1-base-seq2seq"
    )
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument(
        "--predict_with_generate", default=True, action=BooleanOptionalAction
    )
    args = parser.parse_args()

    # Load dataset
    data_files = {"test": args.test_dataset}
    dataset = load_dataset("json", data_files=data_files, field="data")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

    # Prepare samples
    dataset = dataset.map(
        build_samples, batched=True, batch_size=args.preprocess_batch_size
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids"]
    )

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest"
    )

    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{args.experiment_name}",
        per_device_eval_batch_size=args.test_batch_size,
        predict_with_generate=True,
    )

    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    # Test
    output = trainer.predict(
        test_dataset=dataset["test"], max_length=args.max_output_length
    )
    predictions = tokenizer.batch_decode(
        output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # Save results
    dataset.set_format(type=None, columns=["id", "context", "response"])
    results = []
    for sample, prediction in zip(dataset["test"], predictions):
        results.append(
            {
                "id": sample["id"],
                "context": sample["context"],
                "truth_answer": sample["response"],
                "model_answer": prediction,
            }
        )

    with open(args.results, "w") as f:
        json.dump({"version": args.experiment_name, "data": results}, f, indent=4)

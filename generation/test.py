from argparse import ArgumentParser, BooleanOptionalAction

from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Text2TextGenerationPipeline,
)


def build_samples(samples):
    input_texts = []

    for context, knowledge in zip(samples["Context"], samples["Knowledge"]):
        input_text = context
        if knowledge:
            input_text += " <|Knowledge|> " + knowledge
        input_text += " => "

        input_texts.append(input_text)

    model_inputs = tokenizer(
        input_texts,
        max_length=args.max_input_length,
        truncation=True,
    )

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
    }


class GodelGenerationPipeline(Text2TextGenerationPipeline):
    def preprocess(self, inputs):
        model_input = build_samples(inputs)
        return {"model_input": model_input}


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=16)
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("json", data_files={"test": args.test_dataset})

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

    # Initialize pipeline
    generator = GodelGenerationPipeline(
        task="godel-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,
    )

    # Test
    for out in generator(
        dataset,
        batch_size=args.test_batch_size,
        clean_up_tokenization_spaces=True,
        generate_kwargs={},
    ):
        print(out)
        input()

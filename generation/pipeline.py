from argparse import ArgumentParser, BooleanOptionalAction

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Text2TextGenerationPipeline,
)


def build_samples(samples):
    # If samples is only one instance, make it into a dict of lists
    if not isinstance(samples["id"], list):
        samples = {k: [v] for k, v in samples.items()}

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
        padding=True,
        return_tensors="pt",
    )

    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
    }


class GodelGenerationPipeline(Text2TextGenerationPipeline):
    def preprocess(self, inputs):
        model_input = build_samples(inputs)
        return model_input


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_output_length", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=4)
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("json", data_files={"test": args.test_dataset})

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Initialize pipeline
    generator = GodelGenerationPipeline(
        task="godel-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=args.test_batch_size,
        # device=0,
    )

    # Test
    """
    for out in generator(
        iter(dataset["test"]),
        batch_size=args.test_batch_size,
        clean_up_tokenization_spaces=True,
    ):
        #print(out)
        pass
    """

    while True:
        text_input = input("input:\n")
        data = [{"id": "test", "Context": text_input, "Knowledge": ""}]
        out = generator(data, batch_size=1, clean_up_tokenization_spaces=True)
        print(out)

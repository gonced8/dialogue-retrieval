from argparse import ArgumentParser
import json
import pathlib

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

instruction = "You are a customer service system. Your task is to answer in a empathic, informative and useful way."


def build_samples(samples, args, tokenizer):
    # Get input text (prompts)
    global instruction
    prompts = []
    samples_knowledge = []

    for context, knowledge in zip(samples["context"], samples["knowledge"]):
        # Add instruction
        prompt = instruction

        # Add retrieved information
        if knowledge and args.candidates > 0:
            unique_knowledge = list(set(knowledge))  # Remove duplicate candidates
            samples_knowledge.append(unique_knowledge[: args.candidates])
            possible_answers = "\n".join(unique_knowledge[: args.candidates])

            prompt += f"Based on the possible answers below, answer the last conversation.\nPossible answers:\n{possible_answers}\n"
        else:
            prompt += "Answer the last conversation.\n"

        # Add context
        context = "\n".join(context)
        prompt += f"Conversation:\n{context}"

        prompts.append(prompt)

    # Tokenize inputs
    inputs = tokenizer(
        prompts,
        max_length=args.max_input_length,
        truncation=True,
        return_length=True,
    )

    # Tokenize decoder inputs
    decoder_inputs = tokenizer(
        "<pad>System: ",
        max_length=args.max_output_length,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    decoder_inputs["input_ids"] = [decoder_inputs["input_ids"]] * len(samples["id"])
    decoder_inputs["attention_mask"] = [decoder_inputs["attention_mask"]] * len(
        samples["id"]
    )

    # Tokenize labels
    labels = tokenizer(
        samples["response"],
        max_length=args.max_output_length,
        truncation=True,
        return_attention_mask=False,
        return_length=True,
    )

    # Verify if possible truncation
    overflow = {
        "overflow": [
            input_length == args.max_input_length
            or output_length == args.max_output_length
            for input_length, output_length in zip(inputs["length"], labels["length"])
        ]
    }

    possible_overflow = sum(overflow["overflow"])
    if possible_overflow:
        print(
            f"WARNING: Possible overflow in {possible_overflow} out of {len(samples['id'])} samples."
        )

    return {
        "knowledge": samples_knowledge,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": decoder_inputs["input_ids"],
        "decoder_attention_mask": decoder_inputs["attention_mask"],
        "labels": labels["input_ids"],
    } | overflow


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/flan-t5-xl")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("json", data_files={"test": args.dataset}, field="data")
    print(dataset)

    # Build samples
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = dataset.map(
        build_samples,
        batched=True,
        batch_size=args.preprocess_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer},
    )

    # Filter overflowing samples
    dataset = dataset.filter(lambda sample: not sample["overflow"])

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        ],
    )
    print(dataset)

    # Read results file if it exists (to avoid repeating generations)
    file_exists = pathlib.Path(args.results).is_file()

    if file_exists:
        with open(args.results, "r") as f:
            results = [json.loads(line) for line in f]
        results = [line["id"] for line in results if "id" in line]
    else:
        results = []

    # Open file for results
    with open(args.results, "a") as f:
        # Save instruction
        if not file_exists:
            f.write(json.dumps({"instruction": instruction}) + "\n")

        # Loop samples
        for sample in tqdm(dataset["test"], desc="Generating"):
            # Check if sample already computed
            if sample["id"] in results:
                continue

            while True:
                try:
                    # Generate
                    output = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=sample["messages"],
                        user=email_hash,
                    )
                except Exception as e:
                    print(e)
                else:
                    break

            # Append result to file
            model_answer = output["choices"][0]["message"]["content"]
            f.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "context": sample["context"],
                        "response": sample["response"],
                        "delexicalized": sample["delexicalized"],
                        "model_answer": model_answer,
                        "knowledge": sample["knowledge"],
                    }
                )
                + "\n"
            )

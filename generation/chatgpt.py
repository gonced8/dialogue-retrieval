from argparse import ArgumentParser
import json
import os
import pathlib

from datasets import load_dataset
from dotenv import load_dotenv
import openai
from tqdm import tqdm

# OpenAI Setup
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.organization = os.environ["OpenAI_Organization"]
email_hash = str(hash("goncalo.raposo.ext@unbabel.com"))

instruction = "You are a customer service system. Your task is to answer in a empathic, informative and useful way.\n\n"


def build_samples_prompt(samples, args):
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

            prompt += f"Based on the possible answers below, answer the conversation.\nPossible answers:\n{possible_answers}\n\n"
        else:
            samples_knowledge = [None] * len(samples["knowledge"])
            prompt += "Answer the conversation.\n\n"

        # Add context
        context = "\n".join(context)
        prompt += f"Conversation:\n{context}"

        prompts.append(prompt)

    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]

    return {"knowledge": samples_knowledge, "messages": messages}


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("json", data_files={"test": args.dataset}, field="data")
    print(dataset)

    # Build samples
    dataset = dataset.map(
        build_samples_prompt,
        batched=True,
        batch_size=args.preprocess_batch_size,
        fn_kwargs={
            "args": args,
        },
    )

    # Read results file if it exists (to avoid repeating generations)
    file_exists = pathlib.Path(args.results).is_file()

    if file_exists:
        with open(args.results, "r") as f:
            results = [json.loads(line) for line in f]
        results = [line["id"] for line in results if "id" in line]
    else:
        results = []

    # Open file for results
    with open(args.results, "a", buffering=1) as f:
        # Save instruction
        # if not file_exists:
        #    f.write(json.dumps({"instruction": instruction}) + "\n")

        # Loop samples
        for sample in tqdm(dataset["test"], desc="Generating"):
            # Check if sample already computed
            if sample["id"] in results:
                continue

            while True:
                try:
                    # Generate
                    output = openai.ChatCompletion.create(
                        model=args.model,
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
                        "reference": sample["response"],
                        "prediction": model_answer,
                        "knowledge": sample["knowledge"],
                    }
                )
                + "\n",
            )

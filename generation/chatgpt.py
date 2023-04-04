from argparse import ArgumentParser
import json
import pathlib

from datasets import load_dataset
import openai
from tqdm import tqdm

openai.api_key_path = "/home/gecr/.openai.key"
email_hash = str(hash("goncalo.raposo.ext@unbabel.com"))

instruction = {
    "role": "system",
    "content": """You are a customer service system. The user will ask you something and you must answer accordingly to the past answers.
Your response must be delexicalized, that is, certain values are replaced by their slots. The possible slots are: [address], [area], [arriveby], [bookday], [bookpeople], [bookstay], [booktime], [choice], [day], [departure], [destination], [duration], [entrancefee], [food], [leaveat], [name], [openhours], [phone], [postcode], [price], [pricerange], [ref], [stars], [trainid], [type].
Here are some examples of delexicalization:
- from Cambridge => from [departure]
- at 191 Histon Road Chesterton => at [address]
- around north part of town => around [area] part of town
- arrive by 19:00 => arrive by [arriveby]""",
}


def build_samples(samples, n_candidates=5):
    global instruction
    samples_messages = []
    samples_knowledge = []

    for context, knowledge in zip(samples["context"], samples["knowledge"]):
        context = {"role": "user", "content": "\n".join(context)}

        message = [instruction, context]

        if knowledge and args.candidates > 0:
            unique_knowledge = [*set(knowledge)]  # Remove duplicate candidates
            samples_knowledge.append(unique_knowledge[:n_candidates])
            past_answers = {
                "role": "assistant",
                "content": (
                    "Past answers:\n- " + "\n- ".join(unique_knowledge[:n_candidates])
                ),
            }

            message.append(past_answers)

        samples_messages.append(message)

    return {"knowledge": samples_knowledge, "messages": samples_messages}


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("json", data_files={"test": args.dataset}, field="data")
    print(dataset)

    # Build samples
    dataset = dataset.map(
        build_samples,
        batched=True,
        batch_size=args.preprocess_batch_size,
        fn_kwargs={
            "n_candidates": args.candidates,
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

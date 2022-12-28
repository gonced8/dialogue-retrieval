from argparse import ArgumentParser
import json
from pathlib import Path

from tqdm import tqdm

from data.multiwoz import MultiWOZ


def extract_subdialogues(args):
    if args.output is None:
        return

    # Check if dataset already exists
    output = Path(args.output)
    if output.exists():
        print(f"Dataset at {output} already exists.")
        option = input("Do you want to regenerate the dataset? [y/N] ")
        if option == "" or option.lower() == "n":
            return

    # Load original dataset
    with open(args.input, "r") as f:
        dataset = json.load(f)

    new_dataset = []
    exclude_indices = {}

    for d_id, dialogue in tqdm(
        dataset.items(), desc="Generating samples from dialogues"
    ):
        size = len(new_dataset)
        first = 0 if dialogue[0]["speaker"].lower() == "user" else 1

        for i in range(first + 2, len(dialogue) + 1, 2):
            start = max(first, i - args.max_nturns)

            sample_id = f"{d_id}_{start}-{i}"
            text = MultiWOZ.get_conversation(dialogue[start:i])

            sample = {"id": sample_id, "text": text}
            new_dataset.append(sample)

        exclude_indices[d_id] = list(range(size, size + (len(dialogue) - first) // 2))

    # Create output directory if it doesn't exist
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save new dataset
    with open(output, "w") as f:
        json.dump(new_dataset, f, indent=4)

    with open(output.parent / (output.stem + "_exclude_indices.json"), "w") as f:
        json.dump(exclude_indices, f, indent=4)

    print(f"Saved new dataset to directory {args.output}")
    print(f"New dataset contains {len(new_dataset)} samples.")

    return new_dataset, exclude_indices


def get_context_answer(text):
    return text.rsplit("\n", 1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="../data/multiwoz/processed/test.json"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="../data/multiwoz/processed2/test.json"
    )
    parser.add_argument("--max_nturns", type=int, default=6)
    args = parser.parse_args()

    # Run
    extract_subdialogues(args)

    print("Done")

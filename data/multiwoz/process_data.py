from argparse import ArgumentParser
import json
from pathlib import Path

from tqdm import tqdm


def get_sequence(
    dialogue, annotations=["domains", "acts", "slots", "values"], flatten=False
):
    sequence = []

    # Loop through turns
    for turn in dialogue:
        subsequence = []

        # Loop through dialogue acts
        for dialogue_act, slots_dict in turn["dialogue_acts"].items():
            domain, dialogue_act = dialogue_act.split("-")

            # Special case where there is no slots/values or we don't want them
            if not slots_dict or not (
                "slots" in annotations or "values" in annotations
            ):
                slots_dict = {None: None}

            # Loop through slots and values
            for slot, value in slots_dict.items():
                element = {}

                if "domains" in annotations:
                    element["domain"] = domain
                if "acts" in annotations:
                    element["act"] = dialogue_act
                if "slots" in annotations and slot is not None:
                    element["slot"] = slot
                if "values" in annotations and value is not None:
                    element["value"] = value

                if element:
                    subsequence.append(element)

        if subsequence:
            sequence.append(subsequence)

    return sequence


def process_data(args):
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
        sequence = get_sequence(dialogue)

        for i in range(2, len(sequence) + 1, 2):
            start = max(first, i - args.max_nturns)

            sample_id = f"{d_id}_{start}-{i}"
            text = [
                f"{turn['speaker']}: " + turn["utterance"] for turn in dialogue[start:i]
            ]
            annotations = sequence[start:i]

            sample = {"id": sample_id, "text": text, "annotations": annotations}
            new_dataset.append(sample)

        exclude_indices[d_id] = list(range(size, size + len(sequence) // 2))

    # Create output directory if it doesn't exist
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save new dataset
    with open(output, "w") as f:
        json.dump({"version": args.version, "data": new_dataset}, f, indent=4)

    with open(output.parent / (output.stem + "_exclude_indices.json"), "w") as f:
        json.dump(exclude_indices, f, indent=4)

    print(f"Saved new dataset to directory {args.output}")
    print(f"New dataset contains {len(new_dataset)} samples.")

    return new_dataset, exclude_indices


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-n", "--max_nturns", type=int, default=6)
    parser.add_argument("-v", "--version", type=str, required=True)
    args = parser.parse_args()

    process_data(args)

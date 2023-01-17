"""From the original PersonaChat files, extracts a clean dataset with utterances."""
import argparse
import itertools
import json
import os

import datasets


def build_samples(sample):
    context = [
        speaker + utterance
        for speaker, utterance in zip(
            itertools.cycle(["USER: ", "SYSTEM: "]), sample["history"][-5:]
        )
    ]
    knowledge = sample["personality"]
    response = "SYSTEM: " + sample["candidates"][-1]

    return {"context": context, "knowledge": knowledge, "response": response}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="From the original PersonaCHat files, extracts a clean dataset with utterances."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Folder for processed dialogues output files",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()

    # Load dialogues
    train_dataset = datasets.load_dataset("bavard/personachat_truecased", split="train")
    val_dataset, test_dataset = (
        datasets.load_dataset("bavard/personachat_truecased", split="validation")
        .train_test_split(test_size=0.5, shuffle=True)
        .values()
    )
    dataset = datasets.DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    # Build samples
    print("Building samples...")
    dataset = dataset.map(build_samples, remove_columns=dataset["train"].column_names)

    # Save data
    print("Saving samples...")
    for split, data in dataset.items():
        filename = os.path.join(args.output, split + ".json")
        with open(filename, "w") as f:
            json.dump(
                {
                    "version": f"personachat_{split}",
                    "data": [sample for sample in data],
                },
                f,
                indent=4,
            )
            print(f"Saved dataset to {filename}. Number of samples: {len(data)}")

"""From the original PersonaChat files, extracts a clean dataset with utterances."""
import argparse
import itertools
import json
import os

import datasets


def build_samples1(sample, index, args):
    context = [
        speaker + utterance
        for speaker, utterance in zip(
            itertools.cycle(["User: ", "System: "]),
            sample["history"][-args.nturns + 1 :],
        )
    ]
    response = "System: " + sample["candidates"][-1]
    persona = sample["personality"]

    return {
        "id": str(index),
        "context": context,
        "response": response,
        "persona": persona,
    }


def build_samples2(sample, index, args):
    context = [
        speaker + utterance
        for speaker, utterance in zip(
            itertools.cycle(["User: ", "System: "]),
            sample["history"][-args.nturns + 1 :],
        )
    ]
    knowledge = sample["personality"]
    response = "System: " + sample["candidates"][-1]

    return {
        "id": str(index),
        "context": context,
        "knowledge": knowledge,
        "response": response,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="From the original PersonaChat files, extracts a clean dataset with utterances."
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="retrieval",
        choices=["retrieval", "generation"],
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Folder for processed dialogues output files",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("-n", "--nturns", type=int, default=6)
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
    dataset = dataset.map(
        build_samples1 if args.mode == "retrieval" else build_samples2,
        remove_columns=dataset["train"].column_names,
        with_indices=True,
        fn_kwargs={"args": args},
    )

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

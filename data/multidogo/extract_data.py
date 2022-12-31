"""From the original MultiDoGo files, extracts a clean dataset with utterances."""
import argparse
import csv
import json
import os
import random

from tqdm import tqdm


def extract_data(input_directory, debug=False):
    data = {}

    # Loop dialogue files
    for name in os.listdir(input_directory):
        filename = os.path.join(input_directory, name)
        domain = os.path.splitext(name)[0]
        data[domain] = {}

        if not filename.endswith(".tsv"):
            continue

        # Load dialogues
        with open(filename, "r") as f:
            print(f"Load from {filename}")
            all_turns = [row for row in csv.DictReader(f)]

        # Initialize dialogue and turn variables
        dialogue_id = all_turns[0]["conversationId"]
        dialogue_data = []
        turn_id = 0
        speaker = all_turns[0]["authorRole"]
        utterance = []

        # Loop through turns
        for turn in tqdm(all_turns, desc=f"Processing dialogues in {filename}..."):
            # If new speaker or new dialogue
            if turn["authorRole"] != speaker or turn["conversationId"] != dialogue_id:
                # Add past turns to dialogue
                dialogue_data.append(
                    {
                        "turn_id": turn_id,
                        "speaker": "USER" if speaker == "customer" else "SYSTEM",
                        "utterance": " ".join(utterance),
                    }
                )

                # Update to new speaker
                turn_id += 1
                speaker = turn["authorRole"]
                utterance = []

            # If new dialogue
            if turn["conversationId"] != dialogue_id:
                # Add dialogue to dataset
                data[domain][dialogue_id] = dialogue_data

                # Update to new dialogue
                dialogue_id = turn["conversationId"]
                dialogue_data = []
                turn_id = 0

            # Add current turn text to list
            utterance.append(turn["utterance"])

        else:
            # Add last turn and add dialogue
            dialogue_data.append(
                {
                    "turn_id": turn_id,
                    "speaker": "USER" if speaker == "customer" else "SYSTEM",
                    "utterance": " ".join(utterance),
                }
            )
            data[domain][dialogue_id] = dialogue_data

        if debug:
            print(json.dumps(dialogue_data, indent=4))
            input()

    return data


def split_data(data, number, seed=42):
    fraction = number / sum(len(value) for value in data.values())
    splitted = {"train": {}, "val": {}, "test": {}}

    # Shuffle dialogues
    random.seed(seed)
    ids = {domain: list(dialogues.keys()) for domain, dialogues in data.items()}
    for dialogue_ids in ids.values():
        random.shuffle(dialogue_ids)

    # Split dialogues
    for domain, dialogue_ids in ids.items():
        a = int((1 - 2 * fraction) * len(dialogue_ids))
        b = int((1 - fraction) * len(dialogue_ids))

        # Train
        for dialogue_id in dialogue_ids[:a]:
            splitted["train"][dialogue_id] = data[domain][dialogue_id]

        # Validation
        for dialogue_id in dialogue_ids[a:b]:
            splitted["test"][dialogue_id] = data[domain][dialogue_id]

        # Test
        for dialogue_id in dialogue_ids[b:]:
            splitted["val"][dialogue_id] = data[domain][dialogue_id]

    return splitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="From the original MultiDoGo files, extracts a clean dataset with utterances."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Folder with original dialogues input files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Folder for processed dialogues output files",
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=1000,
        help="Number of samples for val and test splits each.",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()

    data = extract_data(args.input, args.debug)
    data = split_data(data, args.number, args.seed)

    # Save data
    for split, data in data.items():
        filename = os.path.join(args.output, split + ".json")
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
            print(f"Saved dataset to {filename}. Number of samples: {len(data)}")

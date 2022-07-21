"""From the original MultiWOZ files, extracts a clean dataset with utterances and annotations."""
import argparse
import json
import os
from tqdm import tqdm


def extract_data(input_files, dialogue_acts_file, debug=False):
    data = {}

    # Load dialogue acts
    with open(dialogue_acts_file, "r") as f:
        all_dialogue_acts = json.load(f)

    # Loop dialogue files
    for filename in input_files:
        if not filename.endswith(".json"):
            continue

        # Load dialogues
        with open(filename, "r") as f:
            print(f"Load from {filename}")
            all_dialogues = json.load(f)

        # Loop through conversations
        for dialogue in tqdm(
            all_dialogues, desc=f"Processing dialogues in {filename}..."
        ):
            dialogue_id = dialogue["dialogue_id"]
            services = dialogue["services"]
            dialogue_data = []

            # Loop dialogue intents and acts
            for turn, turn_da in zip(
                dialogue["turns"], all_dialogue_acts[dialogue_id].values()
            ):
                turn_id = turn["turn_id"]
                speaker = turn["speaker"]
                utterance = turn["utterance"]
                active_intents = []
                # Dialogue acts and slots
                # dialogue_acts = tuple([*turn_da["dialog_act"]])
                dialogue_acts = {
                    dialogue_act: {
                        k: v for (k, v) in slots_list if k != "none" and v != "non"
                    }
                    for dialogue_act, slots_list in turn_da["dialog_act"].items()
                }

                # Loop frames
                for frame in turn["frames"]:
                    if "state" not in frame:
                        continue

                    active_intent = frame["state"]["active_intent"]

                    if active_intent != "NONE":
                        active_intents.append(active_intent)

                # active_intents = tuple(sorted(active_intents))

                dialogue_data.append(
                    {
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "utterance": utterance,
                        "active_intents": active_intents,
                        "dialogue_acts": dialogue_acts,
                    }
                )

            # Add data from that dialogue
            data[dialogue_id] = dialogue_data

            if debug:
                print(json.dumps(dialogue_data, indent=4))
                input()

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="From the original MultiWOZ files, extracts a clean dataset with utterances and annotations."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=[],
        required=True,
        help="Original dialogues input files",
    )
    parser.add_argument(
        "-a",
        "--dialogue_acts",
        type=str,
        required=True,
        help="Original dialogue acts file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Processed dialogues output file",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()

    data = extract_data(args.input, args.dialogue_acts, args.debug)

    # Save data
    with open(args.output, "w") as f:
        json.dump(data, f, indent=4)
        print(f"Saved dataset to {args.output}")

import json
import os
from tqdm import tqdm

DEBUG = False

dataset_dir = "../data/multiwoz/original/data/MultiWOZ_2.2/train/"
filename_dialogue_acts = "../data/multiwoz/original/data/MultiWOZ_2.2/dialog_acts.json"
filename_data = "../data/multiwoz/processed/train.json"

if __name__ == "__main__":
    data = {}

    # Load dialogue acts
    with open(filename_dialogue_acts, "r") as f:
        all_dialogue_acts = json.load(f)

    # Loop dialogue files
    for root, _, files in os.walk(dataset_dir, topdown=False):
        for name in files:
            if not name.endswith(".json"):
                continue

            filename = os.path.join(root, name)

            # Load dialogues
            with open(filename, "r") as f:
                print(f"Load from {filename}")
                all_dialogues = json.load(f)

            # Loop through conversations
            for dialogue in tqdm(
                all_dialogues, desc=f"Processing dialogues in {name}..."
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

                if DEBUG:
                    print(json.dumps(dialogue_data, indent=4))
                    input()

    # Save data
    with open(filename_data, "w") as f:
        json.dump(data, f, indent=4)

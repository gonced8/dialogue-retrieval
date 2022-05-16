import difflib
import itertools
import json
import math
import matplotlib.pyplot as plt
import os
from rouge import Rouge
import seaborn as sns
from tqdm import tqdm

DEBUG = False

filedir = "../multiwoz/data/MultiWOZ_2.2/dev/"
filename_dialogue_acts = "../multiwoz/data/MultiWOZ_2.2/dialog_acts.json"
filename_data = "data.json"
filename_pairs = "pairs.json"

data = {}

# Load dialogue acts
with open(filename_dialogue_acts, "r") as f:
    all_dialogue_acts = json.load(f)

# Loop dialogue files
for root, _, files in os.walk(filedir, topdown=False):
    for name in files:
        if not name.endswith(".json"):
            continue

        filename = os.path.join(root, name)

        # Load dialogues
        with open(filename, "r") as f:
            print(f"Load from {filename}")
            all_dialogues = json.load(f)

        for dialogue in tqdm(all_dialogues, desc="Processing dialogues..."):
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
                dialogue_acts = tuple(sorted([*turn_da["dialog_act"]]))

                for frame in turn["frames"]:
                    if "state" not in frame:
                        continue

                    active_intent = frame["state"]["active_intent"]

                    if active_intent != "NONE":
                        active_intents.append(active_intent)

                active_intents = tuple(sorted(active_intents))

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

pairs = {}
rouge = Rouge()
n = 0

for (x_id, x), (y_id, y) in tqdm(
    itertools.combinations(data.items(), 2),
    desc="Processing pairs...",
    total=math.comb(len(data), 2),
):
    # if DEBUG:
    x_dialogue_acts = [turn["dialogue_acts"] for turn in x]
    y_dialogue_acts = [turn["dialogue_acts"] for turn in y]

    # Ratcliff and Obershelp algorithm
    sm = difflib.SequenceMatcher(None, x_dialogue_acts, y_dialogue_acts)
    ratio = sm.ratio()

    x_dialogue = "\n".join(turn["utterance"] for turn in x)
    y_dialogue = "\n".join(turn["utterance"] for turn in y)

    scores = rouge.get_scores(x_dialogue, y_dialogue)[0]

    # Add pair data
    pairs[str((x_id, y_id))] = {
        x_id: {"dialogue": x_dialogue, "dialogue_acts": x_dialogue_acts},
        y_id: {"dialogue": y_dialogue, "dialogue_acts": y_dialogue_acts},
        "ratio": ratio,
        "rouge": scores,
    }

    if DEBUG:
        print(x_id, x, "", sep="\n")
        print(y_id, y, "", sep="\n")
        print("x_dialogue_acts", x_dialogue_acts, "", sep="\n")
        print("y_dialogue_acts", y_dialogue_acts, "", sep="\n")
        print("ratio", ratio, "", sep="\n")
        print(json.dumps(scores, ident=4))
        input()

# Save pairs data
with open(filename_pairs, "w") as f:
    json.dump(pairs, f, indent=4)

ratios = [pair["ratio"] for pair in pairs.values()]
rouges = [pair["rouge"]["rouge-l"]["f"] for pair in pairs.values()]

plot = sns.relplot(x=ratios, y=rouges, kind="line")
plot.set_axis_labels("Ratio", "ROUGEL-F1")
plot.fig.savefig("plot.pdf")

"""Adapted from https://github.com/Tomiinek/MultiWOZ_Evaluation/blob/master/mwzeval/utils.py"""

from argparse import ArgumentParser, BooleanOptionalAction
from functools import partial
import json
import re

from mwzeval.normalization import normalize_slot_name
from mwzeval.utils import normalize_data
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm import tqdm


def load_raw(input_files, dialog_acts_file):
    # Load dialogue acts
    with open(dialog_acts_file, "r") as f:
        dialog_acts = json.load(f)

    # Load dialogues from all files
    raw_data = []

    for filename in input_files:
        if not filename.endswith(".json"):
            continue

        # Load dialogues
        with open(filename, "r") as f:
            print(f"Load from {filename}")
            raw_data.extend(json.load(f))

    return raw_data, dialog_acts


def delexicalize_utterance(utterance, span_info):
    span_info.sort(key=(lambda x: x[-2]))  # sort spans by start index
    new_utterance = ""
    prev_start = 0
    for span in span_info:
        intent, slot_name, value, start, end = span
        if start < prev_start or value == "dontcare":
            continue
        new_utterance += utterance[prev_start:start]
        new_utterance += f"[{slot_name}]"
        prev_start = end
    new_utterance += utterance[prev_start:]
    return new_utterance


def normalize_data(input_data):
    """In-place normalization of raw dictionary with input data. Normalize slot names, slot values, remove plurals and detokenize utterances."""

    mt, md = MosesTokenizer(lang="en"), MosesDetokenizer(lang="en")
    slot_name_re = re.compile(r"\[([\w\s\d]+)\](es|s|-s|-es|)")
    slot_name_normalizer = partial(
        slot_name_re.sub, lambda x: normalize_slot_name(x.group(1))
    )

    for dialogue in tqdm(input_data.values(), desc="Normalizing data"):
        for turn in dialogue:
            # turn["delexicalized"] = turn["delexicalized"].lower()
            turn["delexicalized"] = slot_name_normalizer(turn["delexicalized"])
            turn["delexicalized"] = md.detokenize(
                mt.tokenize(turn["delexicalized"].replace("-s", "").replace("-ly", ""))
            )


def load_multiwoz22(raw_data, dialog_acts, normalize=False):
    mwz22_data = {}
    for dialog in tqdm(raw_data, desc="Delexicalizing dialogues"):
        parsed_turns = []
        for i in range(len(dialog["turns"])):
            t = dialog["turns"][i]
            parsed_turns.append(
                {
                    "speaker": t["speaker"],
                    "response": t["utterance"],
                    "delexicalized": delexicalize_utterance(
                        t["utterance"],
                        dialog_acts[dialog["dialogue_id"]][t["turn_id"]]["span_info"],
                    ),
                }
            )
        mwz22_data[dialog["dialogue_id"].split(".")[0].lower()] = parsed_turns

    if normalize:
        normalize_data(mwz22_data)

    return mwz22_data


def get_subdialogues(dataset):
    new_dataset = []

    for dialogue, turns in tqdm(
        dataset.items(), desc="Generating samples from dialogues"
    ):
        first = 0 if turns[0]["response"].lower().startswith("user") else 1

        for i in range(1 if first else 2, len(turns), 2):
            start = max(0, i - args.max_nturns + 1)

            sample_id = f"{dialogue}_{start}-{i+1}"
            text = [
                f"{turn['speaker'].capitalize()}: {turn['response']}"
                for turn in turns[start:i]
            ]
            text.append(
                f"{turns[i]['speaker'].capitalize()}: {turns[i]['delexicalized']}"
            )

            sample = {"id": sample_id, "text": text}
            new_dataset.append(sample)

    return new_dataset


if __name__ == "__main__":
    parser = ArgumentParser(
        description="From the original MultiWOZ files, extracts a dataset with delexicalized utterances."
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
        "--dialog_acts",
        type=str,
        required=True,
        help="Original dialogue acts file",
    )
    parser.add_argument("--normalize", default=False, action=BooleanOptionalAction)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Processed dialogues output file",
    )
    parser.add_argument("-n", "--max_nturns", type=int, default=6)
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="Version of output file",
    )
    args = parser.parse_args()

    raw_data, dialog_acts = load_raw(args.input, args.dialog_acts)
    mwz22_data = load_multiwoz22(raw_data, dialog_acts, args.normalize)
    data = get_subdialogues(mwz22_data)
    output = {}

    if args.version:
        output["version"] = args.version

    output["data"] = data

    with open(args.output, "w") as f:
        json.dump(output, f, indent=4)

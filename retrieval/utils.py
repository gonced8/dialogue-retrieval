import collections
import itertools
import json
import random

from tqdm.contrib.concurrent import process_map


def get_sequence(dialogue, annotations):
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
                element = []

                if "domains" in annotations:
                    element.append(domain)
                if "acts" in annotations:
                    element.append(dialogue_act)
                if "slots" in annotations and slot is not None:
                    element.append(slot)
                if "values" in annotations and value is not None:
                    element.append(value)

                if element:
                    subsequence.append(tuple(element))

        if subsequence:
            sequence.append(subsequence)

    return sequence


def get_sequences(filename_data, mode="dialogue_acts", speaker="both", debug=False):
    with open(filename_data, "r") as f:
        data = json.load(f)

    if debug:
        acts = set()
        slots = set()
        values = set()
        acts_slots = set()

    sequences = collections.defaultdict(list)

    for dialogue, turns in data.items():
        # If speaker is not both, select user or agent turns
        if speaker == "user":
            turns = turns[::2]
        elif speaker == "agent":
            turns = turns[1::2]

        for turn in turns:
            subsequence = []

            for dialogue_act, slots_dict in turn["dialogue_acts"].items():
                if slots_dict:
                    acts_slots_i = [f"{dialogue_act}_{slot}" for slot in slots_dict]
                else:
                    acts_slots_i = [dialogue_act]

                if debug:
                    acts.add(dialogue_act)
                    slots.update(slots_dict.keys())
                    values.update(slots_dict.values())
                    acts_slots.update(acts_slots_i)

                if mode == "dialogue_acts":
                    subsequence.append(dialogue_act)
                elif mode == "acts_slots":
                    subsequence.extend(acts_slots_i)
                else:
                    raise NotImplementedError

            sequences[dialogue].append(subsequence)

    if debug:
        acts = sorted(acts)
        slots = sorted(slots)
        values = sorted(values)
        acts_slots = sorted(acts_slots)

        print("Dialogue acts: ", len(acts), "\n", acts, "\n", sep="")
        print("Slots: ", len(slots), "\n", slots, "\n", sep="")
        print("Values: ", len(values), "\n", sep="")
        # print("Acts and Slots: ", len(acts_slots), "\n", acts_slots, "\n", sep="")

        # print(
        #     "Example: ",
        #     next(iter(sequences.keys())),
        #     "\n",
        #     next(iter(sequences.values())),
        #     "\n",
        #     sep="",
        # )

    return sequences


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def flatten(sequence, concatenate=False):
    if concatenate:
        return [
            "".join(x.title().replace(" ", "") for x in element)
            for subsequence in sequence
            for element in subsequence
        ]
    else:
        return [
            x for subsequence in sequence for element in subsequence for x in element
        ]


def sample(it, length, k, max_workers=8):
    indices = random.sample(range(length), k)
    result = [None] * k

    for idx, value in enumerate(tqdm(it, desc="Sampling...", total=length)):
        try:
            i = indices.index(idx)
            result[i] = value

            # If filled k elements, end
            k -= 1
            if not k:
                break

        except ValueError:
            continue

    return result


def random_combinations(iterable, r, k=1, repeat=False, seed=None):
    """Random selection from itertools.combinations(iterable, r)
    Returns k combinations
    """
    pool = tuple(iterable)
    n = len(pool)
    random.seed(seed)

    for _ in range(k):
        if repeat:
            indices = random.choices(range(n), k=r)
        else:
            indices = random.sample(range(n), k=r)
        yield tuple(pool[i] for i in indices)


def get_conversation(d, speaker=True):
    return "\n".join(
        speaker + turn["utterance"]
        for speaker, turn in zip(
            itertools.cycle(["USER:  ", "AGENT: "] if speaker else [""]), d
        )
    )

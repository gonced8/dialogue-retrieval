import collections
import itertools
import json


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

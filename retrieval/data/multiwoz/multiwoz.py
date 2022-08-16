import json


class MultiWOZ:
    def __init__(self, filename):
        with open(filename, "r") as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, k):
        return self.dataset[k]

    def from_id(self, d_id, last=False):
        k, se = d_id.split("_")
        start, end = [int(i) for i in se.split("-")]
        if last:
            end += 1
        return self.dataset[k][start:end]

    @staticmethod
    def get_conversation(dialogue, speaker=True):
        if isinstance(dialogue, dict):
            dialogue = [dialogue]
        return "\n".join(
            (f"{turn['speaker']+': ':>8}" if speaker else "") + turn["utterance"]
            for turn in dialogue
        )

    @staticmethod
    def get_sequence(dialogue, annotations, flatten=False):
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

        # Flatten sequence
        if flatten == "concatenate":
            sequence = [
                "".join(x.title().replace(" ", "") for x in element)
                for subsequence in sequence
                for element in subsequence
            ]
        elif flatten:
            sequence = [
                x
                for subsequence in sequence
                for element in subsequence
                for x in element
            ]

        return sequence

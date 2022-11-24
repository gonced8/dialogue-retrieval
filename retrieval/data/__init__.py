from data.multiwoz.single import MultiWOZSingleDataModule
from data.multiwoz.combination import MultiWOZCombinationDataModule
from data.multiwoz.dialogue import MultiWOZDialogueDataModule


def get_data(data_name):
    data_name = data_name.lower().replace("_", "")

    if "multiwozsingle" in data_name:
        return MultiWOZSingleDataModule
    elif "multiwozcombination" in data_name:
        return MultiWOZCombinationDataModule
    elif "multiwozdialogue" in data_name:
        return MultiWOZDialogueDataModule

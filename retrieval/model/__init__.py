from model.retriever import Retriever
from model.next_turn import NextTurn


def get_model(model_name):
    model_name = model_name.lower().replace("_", "")

    if "retriever" in model_name:
        return Retriever
    elif "nextturn" in model_name:
        return NextTurn

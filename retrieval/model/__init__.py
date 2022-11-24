from model.retriever import Retriever
from model.next_turn import NextTurn
from model.retriever_answerer import RetrieverAnswererer


def get_model(model_name):
    model_name = model_name.lower().replace("_", "")

    if "retriever" == model_name:
        return Retriever
    elif "nextturn" == model_name:
        return NextTurn
    elif "retrieveranswerer" == model_name:
        return RetrieverAnswererer

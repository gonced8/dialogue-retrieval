from argparse import ArgumentParser

from pytorch_lightning import seed_everything
import streamlit as st
import torch

from model import get_model


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def init():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(args.model_name)(args)
    model.to(device)
    model.eval()
    torch.no_grad()

    return model


@st.cache(allow_output_mutation=True)
def Text():
    return ["USER: "]


def collate_fn(model, batch):
    # Get ids
    ids = [sample["id"] for sample in batch]

    # Get data
    contexts = [
        sample["text"].rsplit("\n", 1)[0] if "\n" in sample["text"] else sample["text"]
        for sample in batch
    ]
    # answers = [sample["text"].rsplit("\n", 1)[1] for sample in batch]

    # Tokenize and convert to tensors
    context_tokenized = model.retriever_tokenizer(
        contexts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    return {
        "ids": ids,
        "contexts": contexts,
        # "answers": answers,
        "context_tokenized": context_tokenized,
    }


def prepare_sample(model, text):
    batch = collate_fn(model, [{"id": "sample", "text": text}])
    batch["context_tokenized"].to(model.device)
    return batch


def demo(args):
    st.title("MultiWOZ: Retriever Answerer")

    left, right = st.columns(2)
    compute = left.button("Compute")
    reset = right.button("Reset")

    text_cache = Text()
    textbox = st.empty()

    model = init()

    if reset:
        print("reset")
        text_cache[0] = "USER: "

    if compute:
        print("compute")
        text = text_cache[0]
        history = text.split("\n")

        # If there is a new question
        if history[-1].startswith("USER: "):
            batch = prepare_sample(model, text)
            candidates = model.forward(batch)[0]
            candidates = [candidate["dialogue"]["text"] for candidate in candidates]
            answer = candidates[0].rsplit("\n", 1)[1]
            text_cache[0] += f"\n{answer}\nUSER: "

            st.caption("Retrieved Dialogues")
            st.text("\n\n".join(candidates))

    text_cache[0] = textbox.text_area("Conversation", text_cache[0], height=480)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    args = parser.parse_known_args()[0]

    parser = get_model(args.model_name).add_argparse_args(parser)

    parser.add_argument("--seed", type=int, default=42, help="seed")

    args = parser.parse_args()
    demo(args)

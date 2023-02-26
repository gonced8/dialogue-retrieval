from argparse import ArgumentParser

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Text2TextGenerationPipeline,
)
import streamlit as st


class GodelGenerationPipeline(Text2TextGenerationPipeline):
    def preprocess(self, inputs):
        model_input = self.build_sample(inputs, self.tokenizer)
        return model_input

    @staticmethod
    def build_sample(sample, tokenizer):
        input_text = " EOS ".join(sample["context"])
        if sample["knowledge"]:
            knowledge = " | ".join(sample["knowledge"])
            input_text += " <|Knowledge|> " + knowledge
        input_text += " => "

        model_inputs = tokenizer(
            input_text,
            max_length=args.max_input_length,
            truncation=True,
            return_attention_mask=False,
            return_tensors="pt",
        )

        return {
            "input_ids": model_inputs["input_ids"],
        }


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def init():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=256)
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).eval()

    # Initialize pipeline
    generator = GodelGenerationPipeline(
        task="godel-generation",
        model=model,
        tokenizer=tokenizer,
        device=torch.cuda.current_device() if torch.cuda.is_available() else -1,
    )

    return args, tokenizer, generator


@st.cache(allow_output_mutation=True)
def Knowledge():
    return [
        "\n".join(
            [
                "I like to run.",
                "I have 3 siblings.",
                "I am a PhD student of Computer Science.",
                "I'm doing an internship at Unbabel",
            ]
        )
    ]


@st.cache(allow_output_mutation=True)
def Context():
    return [
        "\n".join(["USER: Hello, how are you?", "SYSTEM: I'm good and you?", "USER: "])
    ]


#######################################################################

st.title("Persona-Chat Demo")
st.header("Unbabel")

args, tokenizer, generator = init()

left, middle, right = st.columns(3)
compute = left.button("Compute")
reset = middle.button("Reset")
refresh = right.button("Refresh")

knowledge_cache = Knowledge()
knowledge_textbox = st.empty()
context_cache = Context()
context_textbox = st.empty()

min_length = st.sidebar.slider(
    "min_length (default=10)",
    min_value=1,
    max_value=args.max_output_length,
    value=10,
    step=1,
)
max_length = st.sidebar.slider(
    "max_length (default=64)",
    min_value=1,
    max_value=args.max_output_length,
    value=64,
    step=1,
)
num_return_sequences = st.sidebar.slider(
    "num_return_sequences (default=5)", min_value=1, max_value=10, value=5, step=1
)
do_sample = st.sidebar.checkbox("do_sample (default=False)", value=False)
num_beams = st.sidebar.slider(
    "num_beams (default=10)", min_value=1, max_value=20, value=10, step=1
)
top_k = st.sidebar.slider(
    "top_k (defualt=50)", min_value=1, max_value=200, value=50, step=1
)
top_p = st.sidebar.slider(
    "top_p (default=1.0)", min_value=0.0, max_value=1.0, value=1.0, step=0.01
)
length_penalty = st.sidebar.slider(
    "length_penalty (default=1.0)", min_value=0.0, max_value=1.0, value=1.0, step=0.01
)
no_repeat_ngram_size = st.sidebar.slider(
    "no_repeat_ngram_size (default=0)", min_value=0, max_value=10, value=0, step=1
)


if reset:
    print("reset")
    context_cache[0] = "USER: "

if refresh:
    print("refresh")
    context_cache[0] = context_cache[0]
    knowledge_cache[0] = knowledge_cache[0]

if compute:
    print("compute")
    knowledge = knowledge_cache[0].split("\n")

    context = context_cache[0].split("\n")
    if context[-1].startswith("SYSTEM: "):
        *context, answer_start = context
        context_cache[0] = "\n".join(context)
    else:
        answer_start = "SYSTEM: "

    print("knowledge:", knowledge)
    print("context:", context)
    print("answer_start:", answer_start)

    data = {
        "id": "test",
        "context": context[-5:],
        "knowledge": knowledge,
    }

    print("data", data)

    # If there is anything
    if knowledge and context:
        # Prepare decoder input
        decoder_input_ids = tokenizer(
            "<pad>" + answer_start,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(generator.device)

        # Generate answers
        outs = generator(
            data,
            clean_up_tokenization_spaces=True,
            decoder_input_ids=decoder_input_ids,
            min_length=min_length,
            max_length=args.max_output_length,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        answers = [out["generated_text"] for out in outs]
        print("answers:", answers)

        context_cache[0] += f"\n{answers[0]}\nUSER: "

        st.caption("Generated Answers")
        st.write("\n\n".join(answers))

knowledge_cache[0] = knowledge_textbox.text_area(
    "Persona", knowledge_cache[0], height=240
)
context_cache[0] = context_textbox.text_area(
    "Conversation", context_cache[0], height=480
)

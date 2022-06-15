from itertools import cycle
import json

# from bert_score import
from rouge_score.rouge_scorer import RougeScorer
import streamlit as st


@st.cache(allow_output_mutation=True)
def init():
    filename_data = "../data/multiwoz/processed/data.json"
    with open(filename_data, "r") as f:
        data = json.load(f)

    scorer = RougeScorer(["rouge1", "rouge2", "rouge3", "rougeL"])

    return data, scorer


def hide_table_index():
    hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)


def get_conversation(d, speaker=True):
    return "\n".join(
        speaker + turn["utterance"]
        for speaker, turn in zip(cycle(["USER:  ", "AGENT: "] if speaker else [""]), d)
    )


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
                if "dialogue acts" in annotations:
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


def flatten(sequence):
    # return [x for subsequence in sequence for element in subsequence for x in element]
    return ["_".join(element) for subsequence in sequence for element in subsequence]


# Initialize

st.set_page_config(layout="wide")
hide_table_index()

data, scorer = init()


# Choose dialogues and write them
st.header("Dialogues")
col1, col2 = st.columns(2)
with col1:
    d1_id, d1 = st.selectbox(
        "Choose dialogue X", options=data.items(), format_func=lambda x: x[0]
    )

    st.write(f"No. turns: {len(d1)}")
    st.text(get_conversation(d1))

with col2:
    d2_id, d2 = st.selectbox(
        "Choose dialogue Y", options=data.items(), format_func=lambda x: x[0]
    )

    st.write(f"No. turns: {len(d2)}")
    st.text(get_conversation(d2))

st.subheader("Scores")
rouge = scorer.score(
    get_conversation(d1, speaker=False), get_conversation(d2, speaker=False)
)
rouge = {k: [v.fmeasure] for k, v in rouge.items()}
scores = rouge
st.table(scores)

st.header("Sequences")

# Select which annotations to use for the sequences
annotations = st.multiselect(
    "Annotations to use for the sequences",
    options=["domains", "dialogue acts", "slots", "values"],
    default=["domains", "dialogue acts"],
)


# Get sequences
seq1 = get_sequence(d1, annotations)
seq2 = get_sequence(d2, annotations)

col1, col2 = st.columns(2)
with col1:
    st.text("\n".join(str(subseq) for subseq in seq1))

with col2:
    st.text("\n".join(str(subseq) for subseq in seq2))


# Compute similarity scores
st.subheader("Flattened sequences")
flat_seq1 = flatten(seq1)
flat_seq2 = flatten(seq2)
st.text(flat_seq1)
st.text(flat_seq2)

st.subheader("Scores")
scores = scorer.score(" ".join(flat_seq1), " ".join(flat_seq2))
scores = {k: [v.fmeasure] for k, v in scores.items()}
st.table(scores)

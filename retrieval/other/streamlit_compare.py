from collections import defaultdict
from difflib import SequenceMatcher
import json
import os
import random

from bert_score import BERTScorer
import matplotlib.pyplot as plt
from rouge_score.rouge_scorer import RougeScorer
import streamlit as st
import textdistance
from transformers import logging

from utils import *

st.set_page_config(layout="wide")

filename_data = "../../data/multiwoz/processed/dev.json"
filename_pairs = "streamlit_pairs.json"


@st.cache(allow_output_mutation=True)
def init():
    data = read_json(filename_data)
    sorted_data_keys = sorted(data.keys())
    sorted_data_items = sorted(data.items())

    # Disable warnings
    logging.set_verbosity_error()
    plt.set_loglevel("WARNING")

    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    rouge_scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])

    # Load pairs
    if os.path.isfile(filename_pairs):
        pairs = read_json(filename_pairs)
        pairs = defaultdict(list, pairs)
    else:
        pairs = defaultdict(list)

    st.session_state.pairs = pairs
    st.session_state.stored_pair = None
    st.session_state.selected_pair = None

    return data, sorted_data_keys, sorted_data_items, bert_scorer, rouge_scorer


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def write_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def hide_table_index():
    # st.table
    hide_table_row_index = """
                <style>
                tbody th {display:none}
                .blank {display:none}
                </style>
                """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    # st.dataframe
    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)


def get_n_pairs():
    n = {s: len(pairs) for s, pairs in st.session_state.pairs.items()}
    return n


def sidebar_pairs():
    # Sidebar to load examples
    step = 0.25
    similarity = str(
        st.sidebar.slider(
            "Similarity", min_value=0.0, max_value=1.0, value=1.0, step=step
        )
    )

    pair_selector = st.sidebar.empty()
    scol1, scol2, scol3 = st.sidebar.columns(3)

    # Save pair
    with scol2:
        if (
            st.button("Save")
            and st.session_state.selected_pair not in st.session_state.pairs[similarity]
        ):
            st.session_state.pairs[similarity].append(st.session_state.selected_pair)
            write_json(filename_pairs, st.session_state.pairs)

    # Delete pair
    with scol3:
        if st.button("Delete"):
            try:
                st.session_state.pairs[similarity].remove(st.session_state.stored_pair)
            except ValueError as e:
                print(e)
            else:
                write_json(filename_pairs, st.session_state.pairs)

    # Select pairs
    n_pairs = get_n_pairs()
    sum_n_pairs = sum(n_pairs.values())

    st.session_state.stored_pair = pair_selector.selectbox(
        f"Pairs: {n_pairs[similarity]} of {sum_n_pairs}",
        st.session_state.pairs[similarity],
        index=n_pairs[similarity] - 1,
    )

    # Load pair
    with scol1:
        if st.button("Load"):
            d1_index = sorted_data_keys.index(st.session_state.stored_pair[0])
            d2_index = sorted_data_keys.index(st.session_state.stored_pair[1])

            st.session_state.dialogue_1 = sorted_data_items[d1_index]
            st.session_state.dialogue_2 = sorted_data_items[d2_index]

    # Plot histogram of saved pairs
    st.sidebar.caption("Plot of saved pairs")
    fig, ax = plt.subplots()
    ax.bar(list(n_pairs.keys()), n_pairs.values())
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Number of pairs")
    st.sidebar.pyplot(fig)


@st.cache(hash_funcs={BERTScorer: lambda _: None})
def compute_bert_score(bert_scorer, conversation1, conversation2):
    bert_score = bert_scorer.score([conversation1], [conversation2])
    P, R, F1 = [x.item() for x in bert_score]
    return {"P": P, "R": R, "F1": F1}


@st.cache(hash_funcs={RougeScorer: lambda _: None})
def compute_rouge_scores(rouge_scorer, prediction, reference):
    rouge_scores = rouge_scorer.score(reference, prediction)
    new_rouge_scores = {}

    for rouge_type, scores in rouge_scores.items():
        rouge_type = rouge_type.upper()

        new_rouge_scores[rouge_type] = {
            f"P": scores.precision,
            f"R": scores.recall,
            f"F1": scores.fmeasure,
        }

    return new_rouge_scores


# Initialize
hide_table_index()
data, sorted_data_keys, sorted_data_items, bert_scorer, rouge_scorer = init()
sidebar_pairs()

# Choose dialogues and write them
st.header("Dialogues from MultiWOZ 2.2")

# Swap selected dialogues or pick random
col1, col2 = st.columns(2)
with col1:
    if st.button("Swap"):
        d1_index = sorted_data_keys.index(st.session_state.selected_pair[0])
        d2_index = sorted_data_keys.index(st.session_state.selected_pair[1])

        st.session_state.dialogue_1 = sorted_data_items[d2_index]
        st.session_state.dialogue_2 = sorted_data_items[d1_index]
with col2:
    if st.button("Random"):
        st.session_state.dialogue_1, st.session_state.dialogue_2 = random.sample(
            sorted_data_items, 2
        )


col1, col2 = st.columns(2)
with col1:
    d1_id, d1 = st.selectbox(
        "Choose dialogue",
        options=sorted(data.items()),
        format_func=lambda x: x[0],
        key="dialogue_1",
    )

    st.write(f"No. turns: {len(d1)}")
    st.text(get_conversation(d1))

with col2:
    d2_id, d2 = st.selectbox(
        "Choose reference",
        options=sorted(data.items()),
        format_func=lambda x: x[0],
        key="dialogue_2",
    )

    st.write(f"No. turns: {len(d2)}")
    st.text(get_conversation(d2))

# Update selected pair
st.session_state.selected_pair = [d1_id, d2_id]

# Score text similarity
st.subheader("Scores")
conversation1 = get_conversation(d1, speaker=False)
conversation2 = get_conversation(d2, speaker=False)

compute = st.checkbox("Compute BERTScore", value=False)
if compute:
    bert_score = compute_bert_score(bert_scorer, conversation1, conversation2)
    st.table({f"BERTScore-{k}": [v] for k, v in bert_score.items()})

rouge_scores = compute_rouge_scores(rouge_scorer, conversation1, conversation2)
rouge_table = {"x": [x[len("ROUGE") :] for x in rouge_scores.keys()]}
rouge_table.update(
    {
        f"ROUGEx-{k}": [scores[k] for _, scores in rouge_scores.items()]
        for k in ["P", "R", "F1"]
    }
)
st.table(rouge_table)

st.header("Sequences")

# Select which annotations to use for the sequences
annotations = st.multiselect(
    "Annotations to use for the sequences",
    options=["domains", "acts", "slots", "values"],
    default=["domains", "acts"],
)


# Get sequences
seq1 = get_sequence(d1, annotations)
seq2 = get_sequence(d2, annotations)

col1, col2 = st.columns(2)
with col1:
    st.text("\n".join(str(subseq) for subseq in seq1))

with col2:
    st.text("\n".join(str(subseq) for subseq in seq2))


# Flatten sequences
st.subheader("Flattened sequences")
concatenate = st.checkbox("Concatenate annotations", value=True)
flat_seq1 = flatten(seq1, concatenate)
flat_seq2 = flatten(seq2, concatenate)
st.text(flat_seq1)
st.text(flat_seq2)

# Compute similarity scores
st.subheader("Scores")

# ROUGE
rouge_scores = compute_rouge_scores(
    rouge_scorer, " ".join(flat_seq1), " ".join(flat_seq2)
)
rouge_table = {"x": [x[len("ROUGE") :] for x in rouge_scores.keys()]}
rouge_table.update(
    {
        f"ROUGEx-{k}": [scores[k] for _, scores in rouge_scores.items()]
        for k in ["P", "R", "F1"]
    }
)
st.table(rouge_table)

# Ratcliff and Obershelp algorithm
sm = SequenceMatcher(None, flat_seq1, flat_seq2)
ro_ratio = sm.ratio()

# Levenshtein distance
ld = textdistance.levenshtein.normalized_similarity(flat_seq1, flat_seq2)

st.table({"ro_ratio": [ro_ratio], "levenshtein": [ld]})

# Longest common subsequence
st.subheader("Longest common subsequence")
st.text(textdistance.lcsstr(flat_seq1, flat_seq2))

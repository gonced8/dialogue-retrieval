from argparse import ArgumentParser
import json
from pathlib import Path
import random

import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pytorch_lightning import seed_everything
from tqdm import tqdm

from data.multiwoz.single import MultiWOZSingleDataModule
from utils.lcs import lcs_similarity


def retrieve_bm25(dataset, dialogues, index_directory):
    candidates_bm25 = []
    searcher = LuceneSearcher(index_directory)

    for dialogue in tqdm(dialogues, desc="Retrieving using BM25"):
        context = dataset.get_conversation(dialogue["context"])
        answer = dataset.get_conversation(dialogue["answer"])

        hit = searcher.search(q=context, k=2)[1]  # select 2nd because 1st is the query
        candidates_bm25.append(json.loads(hit.raw))

    candidates_bm25 = [
        {
            "id": dialogue["id"],
            "context": dialogue["contents"],
            "answer": dialogue["answer"],
        }
        for dialogue in candidates_bm25
    ]

    return candidates_bm25


def retrieve_lcs(dataset, dialogues):
    annotations = ["domains", "acts", "slots", "values"]
    flatten = True

    # Get sequences of selected dialogues
    selected_ids = [dialogue["id"] for dialogue in dialogues]
    sequences = [
        dataset.get_sequence(dialogue["context"], annotations, flatten)
        for dialogue in dialogues
    ]

    # Get sequences of entire dataset
    collection = {
        dialogue["id"]: dataset.get_sequence(dialogue["context"], annotations, flatten)
        for dialogue in dataset
    }

    # Retrieve using LCS similarity
    candidates_lcs = []

    for d_id, sequence in tqdm(
        zip(selected_ids, sequences),
        total=len(selected_ids),
        desc="Retrieving using LCS",
    ):
        results = {
            other_id: lcs_similarity(sequence, other_sequence)
            for other_id, other_sequence in collection.items()
            if other_id != d_id
        }

        idx, (hit, score) = max(enumerate(results.items()), key=lambda x: x[1][1])
        candidates_lcs.append(idx)

    candidates_lcs = [dataset[idx] for idx in candidates_lcs]
    candidates_lcs = [
        {
            "id": dialogue["id"],
            "context": dataset.get_conversation(dialogue["context"]),
            "answer": dataset.get_conversation(dialogue["answer"]),
        }
        for dialogue in candidates_lcs
    ]

    return candidates_lcs


def save_excel(dialogues, candidates_bm25, candidates_lcs, output):
    df = pd.DataFrame(
        [
            {
                "id": dialogue["id"],
                "context": dataset.get_conversation(dialogue["context"]),
                "answer": dataset.get_conversation(dialogue["answer"]),
                "bm25_id": retrieved_bm25["id"],
                "bm25_context": retrieved_bm25["context"],
                "bm25_answer": retrieved_bm25["answer"],
                "lcs_id": retrieved_lcs["id"],
                "lcs_context": retrieved_lcs["context"],
                "lcs_answer": retrieved_lcs["answer"],
            }
            for dialogue, retrieved_bm25, retrieved_lcs in zip(
                dialogues, candidates_bm25, candidates_lcs
            )
        ]
    )
    df.to_excel(output, index=False, freeze_panes=(1, 3))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = MultiWOZSingleDataModule.add_argparse_args(parser)
    parser.add_argument("--index_directory", type=str, required=True)
    parser.add_argument("--n_dialogues", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(raw=True)
    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed, workers=True)

    # Load test dataset
    data = MultiWOZSingleDataModule(args)
    data.prepare_data()
    data.setup("test")
    dataset = data.test_dataset

    # Randomly select dialogues for evaluation
    indices = random.sample(range(len(dataset)), args.n_dialogues)
    dialogues = [dataset[idx] for idx in indices]

    # Retrieve with BM25
    candidates_bm25 = retrieve_bm25(dataset, dialogues, args.index_directory)

    # Retrieve with LCS similarity
    candidates_lcs = retrieve_lcs(dataset, dialogues)

    # Save in Excel file
    save_excel(dialogues, candidates_bm25, candidates_lcs, args.output)

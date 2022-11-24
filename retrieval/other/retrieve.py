from argparse import ArgumentParser
import json
import os
from pathlib import Path

from faiss import read_index
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pytorch_lightning import seed_everything
from rouge_score.rouge_scorer import RougeScorer
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from other.generate_dataset_index import CollateFn, get_context_answer
from utils.lcs import lcs_similarity


def retrieve_bm25(args):
    if args.results_bm25 is None:
        return
        
    # Check if results already exists
    results_bm25_path = Path(args.results_bm25)
    if results_bm25_path.exists():
        print(f"Index at {results_bm25_path} already exists.")
        option = input("Do you want to recompute BM25 results? [y/N] ")
        if option == "" or option.lower() == "n":
            return

    # Load dataset and exclude indices
    dataset_path = Path(args.dataset)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    with open(
        dataset_path.parent / (dataset_path.stem + "_exclude_indices.json"), "r"
    ) as f:
        exclude_indices = json.load(f)

    # Load index
    searcher = LuceneSearcher(args.bm25_index)

    # Retrieve using BM25
    results = {}

    for sample in tqdm(dataset, desc="Retrieving using BM25"):
        context, answer = sample["text"].rsplit("\n", 1)

        # Get maximum number of sub-dialogues per dialogue
        max_n_subdialogues = len(exclude_indices[sample["id"].split("_")[0]])

        # Retrieve
        hits = searcher.search(q=context, k=args.k + max_n_subdialogues)
        labels = [hit.docid for hit in hits]
        scores = [hit.score for hit in hits]

        # Filter sub-dialogues hits from the same original dialogue
        sample_base_id = sample["id"].split("_")[0]
        sample_results = {}

        for hit_id, hit_distance in zip(labels, scores):
            hit_base_id = hit_id.split("_")[0]
            if hit_base_id != sample_base_id:
                sample_results[hit_id] = f"{hit_distance:.04f}"

                if len(sample_results) > args.k:
                    break

        results[sample["id"]] = sample_results

    with open(results_bm25_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


def retrieve_st(args):
    if args.results_st is None:
        return

    # Check if results already exists
    results_st_path = Path(args.results_st)
    if results_st_path.exists():
        print(f"Index at {results_st_path} already exists.")
        option = input("Do you want to recompute results? [y/N] ")
        if option == "" or option.lower() == "n":
            return

    # Load dataset and exclude indices
    dataset_path = Path(args.dataset)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    with open(
        dataset_path.parent / (dataset_path.stem + "_exclude_indices.json"), "r"
    ) as f:
        exclude_indices = json.load(f)

    # Load index
    index_directory = Path(args.st_index)
    index = read_index(str(index_directory / "index.bin"))
    with open(index_directory / "ids_labels.json", "r") as f:
        ids_labels = list(json.load(f).values())

    # Load model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SentenceTransformer(args.model_name)
    model.to(device)
    model.eval()

    # Initialize Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=CollateFn(model.tokenizer),
        shuffle=False,
        pin_memory=bool(torch.cuda.device_count()),
    )

    # Tokenize, encode, and retrieve
    results = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Retrieving using Sentence Transformer")
        ):
            # Get maximum number of sub-dialogues per dialogue
            batch_exclude_indices = [
                exclude_indices[sample_id.split("_")[0]] for sample_id in batch["ids"]
            ]
            max_n_subdialogues = max(len(indices) for indices in batch_exclude_indices)

            # Get input_ids and attention_mask into device
            x = {k: v.to(device) for k, v in batch["context_tokenized"].items()}

            # Compute dialogue embeddings
            embeddings = model(x)["sentence_embedding"]

            # Retrieve
            distances, indices = index.search(
                embeddings.cpu().numpy(), args.k + max_n_subdialogues
            )

            # Filter sub-dialogues hits from the same original dialogue
            for sample_id, sample_ids, sample_distances in zip(
                batch["ids"], indices, distances
            ):
                sample_base_id = sample_id.split("_")[0]
                sample_results = {}

                for hit_id, hit_distance in zip(sample_ids, sample_distances):
                    hit_id = ids_labels[hit_id]
                    hit_base_id = hit_id.split("_")[0]
                    if hit_base_id != sample_base_id:
                        sample_results[hit_id] = f"{hit_distance:.04f}"

                        if len(sample_results) > args.k:
                            break

                results[sample_id] = sample_results

    with open(results_st_path, "w") as f:
        json.dump(results, f, indent=4)

    return results


# TODO: NOT WORKING
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


def save_results(args):
    if args.results is None:
        return

    # Check if excel file already exists
    excel_path = Path(args.results)
    if excel_path.exists():
        print(f"Index at {excel_path} already exists.")
        option = input("Do you want to save the file again? [y/N] ")
        if option == "" or option.lower() == "n":
            return

    # Load dataset and exclude indices
    dataset_path = Path(args.dataset)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    dataset = {sample["id"]: sample["text"] for sample in dataset}

    # Load results
    with open(args.results_bm25, "r") as f:
        results_bm25 = json.load(f)

    with open(args.results_st, "r") as f:
        results_st = json.load(f)

    # Get data into pretty format for excel
    data = {"ids": list(dataset.keys()), "text": list(dataset.values())}

    # Load ROUGE scorer for answers
    rouge_scorer = RougeScorer(["rouge1", "rouge2", "rougeL"])

    for i in range(args.n_candidates if args.n_candidates is not None else args.k):
        data[f"bm25 id {i+1}"] = [list(v.keys())[i] for v in results_bm25.values()]
        data[f"bm25 score {i+1}"] = [list(v.values())[i] for v in results_bm25.values()]
        data[f"bm25 rougeL {i+1}"] = [
            rouge_scorer.score(
                get_context_answer(text)[1], get_context_answer(dataset[d_id])[1]
            )["rougeL"].fmeasure
            for text, d_id in zip(data["text"], data[f"bm25 id {i+1}"])
        ]
        data[f"bm25 text {i+1}"] = [dataset[d_id] for d_id in data[f"bm25 id {i+1}"]]
        data[f"st id {i+1}"] = [list(v.keys())[i] for v in results_st.values()]
        data[f"st score {i+1}"] = [list(v.values())[i] for v in results_st.values()]
        data[f"st rougeL {i+1}"] = [
            rouge_scorer.score(
                get_context_answer(text)[1], get_context_answer(dataset[d_id])[1]
            )["rougeL"].fmeasure
            for text, d_id in zip(data["text"], data[f"st id {i+1}"])
        ]
        data[f"st text {i+1}"] = [dataset[d_id] for d_id in data[f"st id {i+1}"]]

    df = pd.DataFrame(data)

    print(f"BM25 average rougeL-F1 (top-1): {df['bm25 rougeL 1'].mean()}")
    print(f"Sentence Transformer average rougeL-F1 (top-1): {df['st rougeL 1'].mean()}")

    # Truncate rows
    if args.n_results is not None:
        df = df.sample(n=args.n_results)

    print(f"Saving excel file to {excel_path}")
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, index=False, freeze_panes=(1, 2))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="../data/multiwoz/processed2/test.json"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument(
        "--bm25_index", type=str, default="data/multiwoz/index/test_bm25"
    )
    parser.add_argument("--st_index", type=str, default="data/multiwoz/index/test_st")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument(
        "--n_results", type=int, default=None, help="Number of results to export"
    )
    parser.add_argument(
        "--n_candidates", type=int, default=None, help="Number of candidates to export"
    )
    parser.add_argument(
        "--results_bm25", type=str, default="results/test_retrieval_bm25.json"
    )
    parser.add_argument(
        "--results_st", type=str, default="results/test_retrieval_st.json"
    )
    parser.add_argument(
        "--results", type=str, default="results/test_retrieval_bm25_st.xlsx"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed, workers=True)

    # Retrieve with BM25 and Sentence Transformer
    retrieve_bm25(args)
    retrieve_st(args)

    # Save in JSON and Excel file
    save_results(args)

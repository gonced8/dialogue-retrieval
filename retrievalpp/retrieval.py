from collections import defaultdict
import psutil

from autofaiss import build_index
import numpy as np
import torch
from tqdm import tqdm


def generate_index(index_dataloader, encoder, index_key="answer"):
    index_key = "" if index_key == "context" else f"{index_key}_"

    # Compute embeddings
    all_embeddings = []
    idx = []
    ids = []

    for batch in tqdm(index_dataloader, desc="Calculating embeddings"):
        # Get input_ids and attention_mask into device
        input_ids = batch[index_key + "input_ids"].to(encoder.device)
        attention_mask = batch[index_key + "attention_mask"].to(encoder.device)

        # Compute dialogue embeddings
        with torch.no_grad():
            embeddings = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Add embeddings to cache
        all_embeddings.append(embeddings.cpu().numpy())

        # Update ids arrays
        idx.extend(batch["idx"].tolist())
        ids.extend(batch["id"])

    all_embeddings = np.concatenate(all_embeddings)

    # Generate index
    current_memory_available = f"{psutil.virtual_memory().available * 2**-30:.0f}G"

    index, index_infos = build_index(
        embeddings=all_embeddings,
        max_index_memory_usage="32G",
        current_memory_available=current_memory_available,
        metric_type="ip",
        save_on_disk=False,
    )

    # Get excluded indices to avoid leakage
    exclude_idx = defaultdict(list)
    for i, doc_id in zip(idx, ids):
        exclude_idx[doc_id.rsplit("_", 1)[0]].append(i)

    return (index, idx, exclude_idx)


def retrieve(
    encoder,
    inputs,
    index,
    n_candidates,
    outputs=None,
    index_key="context",
    queries_ids=None,
):
    """May contain leakage of data if using same dataset for query and documents"""
    index, idx, exclude_idx = index
    index_key = "" if index_key == "context" else f"{index_key}_"

    # Encode
    if outputs is None:
        embeddings = encoder(
            input_ids=inputs[index_key + "input_ids"],
            attention_mask=inputs[index_key + "attention_mask"],
        )
    else:
        embeddings = outputs

    # Get maximum of possible leaked candidates
    if queries_ids:
        queries_base_ids = [query_id.rsplit("_", 1)[0] for query_id in queries_ids]
        max_leak = max(
            len(exclude_idx.get(query_base_id, []))
            for query_base_id in queries_base_ids
        )
    else:
        max_leak = 0

    # Retrieve
    distances, indices = index.search(
        embeddings.cpu().detach().numpy(),
        n_candidates + max_leak,
    )

    # Convert candidates to correct indices (index dataset might be shuffled)
    indices = [[idx[i] for i in sample_hits] for sample_hits in indices]

    # Filter leaked candidates
    if queries_ids:
        hits_distances = [
            [
                (hit, distance)
                for hit, distance in zip(sample_candidates, sample_distances)
                if hit not in exclude_idx.get(query_base_id, [])
            ][:n_candidates]
            for sample_candidates, sample_distances, query_base_id in zip(
                indices, distances, queries_base_ids
            )
        ]

        indices = [
            [elem[0] for elem in sample_pairs] for sample_pairs in hits_distances
        ]
        distances = [
            [elem[1] for elem in sample_pairs] for sample_pairs in hits_distances
        ]

    return indices, distances

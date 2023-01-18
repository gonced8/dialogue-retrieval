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

    for batch in tqdm(index_dataloader, desc="Calculating embeddings"):
        # Get input_ids and attention_mask into device
        input_ids = batch[index_key + "input_ids"].to(encoder.device)
        attention_mask = batch[index_key + "attention_mask"].to(encoder.device)

        # Compute dialogue embeddings
        with torch.no_grad():
            embeddings = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Add embeddings to cache
        all_embeddings.append(embeddings.cpu().numpy())

        # Update ids_labels array
        idx.extend(batch["idx"])

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

    return (index, idx)


def retrieve(
    encoder,
    inputs,
    index,
    n_candidates,
    outputs=None,
    index_key="context",
    queries=None,
    exclude_indices=None,
):
    """May contain leakage of data if using same dataset for query and documents"""
    index, idx = index
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
    if exclude_indices:
        exclude_indices = [
            exclude_indices[query.rsplit("_", 1)[0]] for query in queries
        ]
        max_leak = max(len(query_exclude) for query_exclude in exclude_indices)
    else:
        max_leak = 0

    # Retrieve
    distances, indices = index.search(
        embeddings.cpu().detach().numpy(),
        n_candidates + max_leak,
    )

    # Convert candidates to correct indices
    indices = [[idx[i] for i in sample_candidates] for sample_candidates in indices]

    # Filter leaked candidates
    if exclude_indices:
        pair = [
            [
                (hit, distance)
                for hit, distance in zip(sample_candidates, sample_distances)
                if hit not in query_exclude
            ][:n_candidates]
            for sample_candidates, sample_distances, query_exclude in zip(
                indices, distances, exclude_indices
            )
        ]
        indices = [[elem[0] for elem in sample_pairs] for sample_pairs in pair]
        distances = [[elem[1] for elem in sample_pairs] for sample_pairs in pair]

    return indices, distances

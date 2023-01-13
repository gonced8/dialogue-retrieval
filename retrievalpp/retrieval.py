import psutil

from autofaiss import build_index
import numpy as np
import torch
from tqdm import tqdm


def generate_index(index_dataloader, encoder):
    # Compute embeddings
    all_embeddings = []
    ids_labels = []

    for batch in tqdm(index_dataloader, desc="Calculating embeddings"):
        # Get input_ids and attention_mask into device
        batch["input_ids"] = batch["input_ids"].to(encoder.device)
        batch["attention_mask"] = batch["attention_mask"].to(encoder.device)

        # Compute dialogue embeddings
        with torch.no_grad():
            embeddings = encoder(**batch)

        # Add embeddings to cache
        all_embeddings.append(embeddings.cpu().numpy())

        # Update ids_labels array
        ids_labels.extend(batch["id"])

        # TODO: REMOVE
        break

    all_embeddings = np.concatenate(all_embeddings)

    # Generate index
    current_memory_available = f"{psutil.virtual_memory().available * 2**-30:.0f}G"

    index, index_infos = build_index(
        embeddings=all_embeddings,
        max_index_memory_usage="32G",
        current_memory_available=current_memory_available,
        metric_type="ip",
        # metric_type="l2",
        save_on_disk=False,
    )

    return index, ids_labels


def retrieve(encoder, inputs, index, n_candidates, outputs=None):
    """May contain leakage of data if using same dataset for query and documents"""

    # Encode
    if outputs is None:
        embeddings = encoder(**inputs)
    else:
        embeddings = outputs

    # Retrieve
    distances, indices = index.search(
        embeddings.cpu().detach().numpy(),
        n_candidates,
    )

    return indices, distances

from functools import partial
from itertools import combinations
import json
import math
import os
from tqdm.contrib.concurrent import process_map

import textdistance
from utils import *

filename_data = "../data/multiwoz/processed/train.json"
filename_output = "../data/multiwoz/processed/pairs_train.json"
annotations = ["domains", "acts", "slots", "values"]
concatenate = False  # Whether to concatenate annotations
round_n = 6
max_workers = 16
chunksize = 100
k = 10000


def compute_similarity(data, d, round_n=None):
    # Get dialogues
    d1_id, d2_id = d
    d1 = data[d1_id]
    d2 = data[d2_id]

    # Get sequences
    seq1 = get_sequence(d1, annotations)
    seq2 = get_sequence(d2, annotations)

    # Flatten sequences
    flat_seq1 = flatten(seq1, concatenate)
    flat_seq2 = flatten(seq2, concatenate)

    # Levenshtein distance
    ld = textdistance.levenshtein.normalized_similarity(flat_seq1, flat_seq2)

    if round_n is not None:
        ld = round(ld, round_n)

    return {f"{d1_id} {d2_id}": ld}


if __name__ == "__main__":
    with open(filename_data, "r") as f:
        data = json.load(f)

    total = math.comb(len(data), 2)
    print(f"Generating {k} out of {total} pairs")

    output_list = process_map(
        partial(compute_similarity, data, round_n=round_n),
        random_combinations(data.keys(), 2, k),
        max_workers=min(max_workers, os.cpu_count()),
        chunksize=chunksize,
        desc="Processing pairs...",
        total=k,
    )
    output = {}
    for d in output_list:
        output.update(d)

    name, ext = os.path.splitext(filename_output)
    filename_output = f"{name}_{'+'.join(annotations)}{ext}"
    with open(filename_output, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Saved: {filename_output}")

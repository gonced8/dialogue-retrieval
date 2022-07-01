import json
import math
import os

from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

filename_data = "../data/multiwoz/processed/train.json"
filename_pairs = "../data/multiwoz/processed/pairs_train_domains+acts+slots+values.json"
speaker = True
chunk_size = 16  # must have an integer sqrt

if __name__ == "__main__":
    annotations = os.path.splitext(filename_pairs)[0].split("_")[-1].split("+")
    ks = int(math.sqrt(chunk_size))
    assert ks ** 2 == chunk_size, "Chunk size must have an integer sqrt."

    # Load data and pairs
    with open(filename_data, "r") as f:
        data = json.load(f)
    with open(filename_pairs, "r") as f:
        pairs = json.load(f)

    # Get dialogues
    d_ids, ds = next(iter(pairs.items()))
    d1_id, d2_id = d_ids.split(" ")
    d1 = data[d1_id]
    d2 = data[d2_id]

    # Get conversations
    conversation1 = get_conversation(d1, speaker=speaker)
    conversation2 = get_conversation(d2, speaker=speaker)

    # Get sequences
    seq1 = get_sequence(d1, annotations)
    seq2 = get_sequence(d2, annotations)

    # Get embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb1 = model.encode(conversation1.split("\n"), convert_to_tensor=True)
    emb2 = model.encode(conversation2.split("\n"), convert_to_tensor=True)

    # Similarity matrix
    dot_sim = util.dot_score(emb1, emb2)
    # cos_sim = util.cos_sim(emb1, emb2)

    # Similarity tensor
    emb1u = emb1.unfold(0, ks, ks)
    emb2u = emb2.unfold(0, ks, ks)
    # TODO

    # Extract patches with chunk_size elements
    # TODO: take care of when not perfect patches
    patches = dot_sim.unfold(0, ks, ks).unfold(1, ks, ks)
    patches = patches.reshape(-1, chunk_size)

    # Estimate similarity
    linear = nn.Linear(chunk_size, 1)
    output1 = F.sigmoid(linear(patches))

    # Print
    print(d1_id, conversation1, "", sep="\n")
    print(d2_id, conversation2, "", sep="\n")
    # print("Dot-Product Similarity Matrix", dot_sim, "", sep="\n")
    # print("Cosine Similarity Matrix", cos_sim, "", sep="\n")
    print("Embeddings 1 shape", emb1.shape, "", sep="\n")
    print("Embeddings 2 shape", emb2.shape, "", sep="\n")
    print("Similarity Matrix shape", dot_sim.shape, "", sep="\n")
    print("Patches shape", patches.shape, "", sep="\n")
    print("Output 1 shape", output1.shape, "", sep="\n")

import json
import os

from sentence_transformers import SentenceTransformer, util

from utils import *

filename_data = "../data/multiwoz/processed/train.json"
filename_pairs = "../data/multiwoz/processed/pairs_train_domains+acts+slots+values.json"
speaker = True

if __name__ == "__main__":
    annotations = os.path.splitext(filename_pairs)[0].split("_")[-1].split("+")

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
    emb1 = model.encode(conversation1.split("\n"))
    emb2 = model.encode(conversation2.split("\n"))

    # Similarity matrix
    dot_sim = util.dot_score(emb1, emb2)
    cos_sim = util.cos_sim(emb1, emb2)

    # Print
    print(d1_id, conversation1, emb1.shape, "", sep="\n")
    print(d2_id, conversation2, emb2.shape, "", sep="\n")
    print("Cosine Similarity Matrix", dot_sim, "", sep="\n")
    print("Dot-Product Similarity Matrix", cos_sim, "", sep="\n")

import json
from pathlib import Path

from datasets import load_metric
import faiss  # , faiss.contrib.torch_utils
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from data.multiwoz import MultiWOZ
from utils import parse_rouge_score
from utils.lcs import lcs_similarity


class NextTurn(pl.LightningModule):
    def __init__(self, args, data=None):
        super().__init__()
        self.save_hyperparameters(args)

        try:
            self.model_name = self.hparams.model_name
        except AttributeError:
            self.model_name = self.hparams.original_model_name

        # Load encoder
        self.model = SentenceTransformer(self.hparams.original_model_name)
        self.tokenizer = self.model.tokenizer

        # Load index
        index_directory = Path(self.hparams.index_directory)
        self.index = faiss.read_index(str(index_directory / "index.bin"))
        with open(index_directory / "ids_labels.json", "r") as f:
            self.ids_labels = list(json.load(f).values())
        self.data = data

        # Update data module
        if data is not None:
            data.tokenizer = self.tokenizer
            self.randomize = data.randomize
            self.seed = data.hparams.seed
        else:
            self.seed = args.seed

        # Metrics
        self.rouge = {
            "dialogues": load_metric("rouge"),
            "answers": load_metric("rouge"),
        }

    def forward(self, x):
        # Compute embeddings and normalize (because of cosine simlarity)
        out = self.model(x)["sentence_embedding"]
        # Normalize embeddings because of cosine similarity
        return torch.nn.functional.normalize(out, dim=1)

    def test_step(self, batch, batch_idx):
        # Compute dialogue embeddings
        embeddings = self.forward(batch["contexts_tokenized"])

        # Retrieve similar dialogues
        distances, indices = self.index.search(embeddings.cpu().numpy(), self.hparams.k)
        candidates = [
            {self.ids_labels[idx]: distance for idx, distance in zip(*sample)}
            for sample in zip(indices, distances)
        ]

        # Filter candidates from the same dialogue as query
        candidates = [
            {
                c_id: d
                for c_id, d in results.items()
                if c_id.split("_")[0] != q_id.split("_")[0]
            }
            for q_id, results in zip(batch["ids"], candidates)
        ]

        # Get truth and retrieved dialogues
        truth_dialogues = [
            f"{context}\n{'SYSTEM: ':>8}{answer}"
            for context, answer in zip(batch["contexts"], batch["answers"])
        ]
        retrieved = [
            self.data.test_dataset.from_id(next(iter(candidates_i)), last=True)
            for candidates_i in candidates
        ]
        retrieved_dialogues = [
            MultiWOZ.get_conversation(dialogue) for dialogue in retrieved
        ]

        # Get ground truth and retrieved answers
        truth_answers = batch["answers"]
        retrieved_answers = [
            MultiWOZ.get_conversation(dialogue[-1], speaker=False)
            for dialogue in retrieved
        ]

        # Get truth and retrieved sequences of annotations
        annotations = ["domains", "acts", "slots", "values"]
        truth_annotations = [
            MultiWOZ.get_sequence(
                self.data.test_dataset.from_id(d_id, last=True), annotations, True
            )
            for d_id in batch["ids"]
        ]
        retrieved_annotations = [
            MultiWOZ.get_sequence(dialogue, annotations, True) for dialogue in retrieved
        ]

        # print("\n\n".join("\n".join(pair) for pair in zip(ground_truth, answers)))
        # input()

        # Compute metrics
        self.rouge["dialogues"].add_batch(
            predictions=retrieved_dialogues, references=truth_dialogues
        )
        self.rouge["answers"].add_batch(
            predictions=retrieved_answers, references=truth_answers
        )
        lcs_score = [
            lcs_similarity(X, Y)
            for X, Y in zip(truth_annotations, retrieved_annotations)
        ]

        return {"lcs_score": lcs_score}

    def test_epoch_end(self, outputs):
        rouge_score = {}
        for k, v in self.rouge.items():
            score = v.compute()
            score = parse_rouge_score(score, "precision")
            rouge_score[k] = score
        self.log_dict(rouge_score, prog_bar=True)

        lcs_score = {
            "lcs_score": round(
                sum(sum(results["lcs_score"]) for results in outputs)
                / sum(len(results["lcs_score"]) for results in outputs),
                4,
            )
        }
        self.log_dict(lcs_score, prog_bar=True)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Module: NextTurn")
        parser.add_argument(
            "--original_model_name",
            type=str,
            default="sentence-transformers/all-mpnet-base-v2",
        )
        parser.add_argument(
            "--index_directory", type=str, default="data/multiwoz/index"
        )
        parser.add_argument("--k", type=int, default=10)
        return parent_parser

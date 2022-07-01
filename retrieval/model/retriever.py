import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer


class Retriever(pl.LigtningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.model_name = "Retriever"
        self.original_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Initialize original model
        self.model = SentenceTransformer(self.original_model_name)

        # Loss
        self.loss = nn.MSELoss(reduction="mean")

    def training_step(self, batch, batch_idx):
        print(batch)

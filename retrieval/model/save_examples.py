import json
import os

from pytorch_lightning.callbacks import Callback
import torch


class SaveExamples(Callback):
    def __init__(self, filename="results_val.json"):
        self.filename = filename
        self.val_outs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kargs):
        self.val_outs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {
            m: torch.stack([step["metrics"][m] for step in self.val_outs]).mean()
            for m in outputs[0]["metrics"]
        }
        data = [{k: v.item() for k, v in metrics.items()}]
        data.extend(
            [
                {"id": sample_id, "label": label.item(), "output": output.item()}
                for step in self.val_outs
                for sample_id, label, output in zip(
                    step["ids"],
                    step["labels"],
                    step["outputs"],
                )
            ]
        )

        # Round float numbers
        data = [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in sample.items()}
            for sample in data
        ]

        # Save results to file
        output_filename = os.path.join(trainer.logger.log_dir, self.filename)

        with open(output_filename, "w") as f:
            json.dump(data, f, indent=4)

        # Clear validation outputs from Callback memory
        self.val_outs = []

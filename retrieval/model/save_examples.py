import json
import os
from statistics import mean

import numpy as np
from pytorch_lightning.callbacks import Callback
import torch


class SaveExamples(Callback):
    def __init__(self):
        self.val_outs = []
        self.test_outs = []

    def save(self, outs, filename, digits=4):
        metrics = {
            m: np.stack([step["metrics"][m] for step in outs]).mean()
            for m in outs[0]["metrics"]
        }
        data = [{k: v.item() for k, v in metrics.items()}]
        data.extend(
            [
                {
                    k.rstrip("s"): v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in step.items()
                    if k != "metrics"
                }
                for step in outs
                # TODO
            ]
        )

        # Round float numbers
        data = [
            {
                k: round(v, digits) if isinstance(v, float) else v
                for k, v in sample.items()
            }
            for sample in data
        ]

        # Save results to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        # Clear validation outputs from Callback memory
        outs.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kargs):
        self.val_outs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        filename = os.path.join(trainer.logger.log_dir, "results_val.json")
        self.save(self.val_outs, filename)

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kargs):
        self.test_outs.append(outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        filename = os.path.join(trainer.logger.log_dir, "results_test.json")
        self.save(self.test_outs, filename)

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
        # Average metrics and save
        metrics = {
            m: np.concatenate([step["metrics"][m] for step in outs]).mean()
            for m in outs[0]["metrics"]
        }

        data = [metrics]

        # Aggregate metrics per sample
        metrics_keys = outs[0]["metrics"].keys()
        for step in outs:
            step["metrics"] = [
                dict(zip(metrics_keys, metrics_values))
                for metrics_values in zip(*step["metrics"].values())
            ]

        # Get results for each sample
        sample_keys = [k.rstrip("s") if k != "metrics" else k for k in outs[0].keys()]

        data.extend(
            [
                dict(zip(sample_keys, sample_values))
                for step in outs
                for sample_values in zip(*step.values())
            ]
        )

        # Convert Tensors to lists
        data = [
            {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in sample.items()
            }
            for sample in data
        ]

        # Round float numbers
        round_recursively = (
            lambda x: round(x, digits)
            if isinstance(x, float)
            else {k: round_recursively(v) for k, v in x.items()}
            if isinstance(x, dict)
            else x
        )
        data = [round_recursively(sample) for sample in data]

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

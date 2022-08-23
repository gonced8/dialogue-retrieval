import itertools
import json
import os
from statistics import mean

from pytorch_lightning.callbacks import Callback
import torch


class SaveExamples(Callback):
    def __init__(self):
        self.val_outs = []
        self.test_outs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, *args, **kargs):
        self.val_outs.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = {
            m: torch.stack([step["metrics"][m] for step in self.val_outs]).mean()
            for m in self.val_outs[0]["metrics"]
        }
        data = [{k: v.item() for k, v in metrics.items()}]
        data.extend(
            [
                {"id": sample_id, "label": label.tolist(), "output": output.tolist()}
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
        output_filename = os.path.join(trainer.logger.log_dir, "results_val.json")

        with open(output_filename, "w") as f:
            json.dump(data, f, indent=4)

        # Clear validation outputs from Callback memory
        self.val_outs = []

    def on_test_batch_end(self, trainer, pl_module, outputs, *args, **kargs):
        self.test_outs.append(outputs)

    def on_test_epoch_end(self, trainer, pl_module):
        # Stacks outputs
        metrics = {
            m: list(
                itertools.chain.from_iterable(
                    step["metrics"][m] for step in self.test_outs
                )
            )
            for m in self.test_outs[0]["metrics"]
        }

        other = {
            k: list(itertools.chain.from_iterable(step[k] for step in self.test_outs))
            for k in self.test_outs[0]
            if k != "metrics"
        }

        # Average metrics
        average = {metric: mean(values) for metric, values in metrics.items()}

        # Transpose other (dict of lists to list of dicts)
        other = [
            dict(zip([*other, *metrics], sample_values))
            for sample_values in zip(*other.values(), *metrics.values())
        ]

        # Format output
        data = [average]
        data.extend(other)

        # Round float numbers
        digits = 4
        data = [
            {
                k: round(v, digits) if isinstance(v, float) else v
                for k, v in sample.items()
            }
            for sample in data
        ]

        # Save results to file
        output_filename = os.path.join(trainer.logger.log_dir, "results_test.json")

        with open(output_filename, "w") as f:
            json.dump(data, f, indent=4)

        # Clear validation outputs from Callback memory
        self.test_outs = []

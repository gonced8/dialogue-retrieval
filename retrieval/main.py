from argparse import ArgumentParser

from pytorch_lightning import seed_everything

from data import MultiWOZ
from model import Retriever
from trainer import Trainer


def main(args):
    # Seed everything
    seed_everything(args.seed, workers=True)

    # Load dataset
    data = MultiWOZ(args)

    # Load model
    model = Retriever(args, data)

    # Get Trainer
    trainer = Trainer.from_argparse_args(args)

    # Load from checkpoint
    if args.from_checkpoint:
        raise NotImplementedError

    # Train or test
    if "train" in args.mode:
        trainer.fit(model, data)
    elif "test" in args.mode:
        trainer.test(model, data, verbose=True)
    else:
        print(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = MultiWOZ.add_argparse_args(parser)
    parser = Retriever.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--seed", type=int, default=42, help="seed")

    args = parser.parse_args()
    main(args)

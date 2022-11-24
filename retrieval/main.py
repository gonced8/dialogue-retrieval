from argparse import ArgumentParser

from pytorch_lightning import seed_everything

from data import get_data
from model import get_model
from trainer import Trainer


def main(args):
    # Seed everything
    seed_everything(args.seed, workers=True)

    # Load dataset
    if args.data_name is not None:
        data = get_data(args.data_name)(args)
    else:
        data = None

    # Load model
    if args.model_name is not None:
        model = get_model(args.model_name)(args, data)

    # Get Trainer
    trainer = Trainer.from_argparse_args(args)

    # Train or test
    if "train" in args.mode:
        trainer.fit(model, data, ckpt_path=args.ckpt_path)
    elif "validate" in args.mode:
        trainer.validate(model, data, ckpt_path=args.ckpt_path)
    elif "test" in args.mode:
        trainer.test(model, data, ckpt_path=args.ckpt_path, verbose=True)
    else:
        print(f"Unrecognized mode: {args.mode}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_known_args()[0]

    if args.data_name is not None:
        parser = get_data(args.data_name).add_argparse_args(parser)
    if args.model_name is not None:
        parser = get_model(args.model_name).add_argparse_args(parser)

    parser.add_argument("--seed", type=int, default=42, help="seed")

    args = parser.parse_args()
    main(args)

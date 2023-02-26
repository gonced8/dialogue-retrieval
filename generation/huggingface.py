from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="microsoft/GODEL-v1_1-base-seq2seq"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.checkpoint if args.checkpoint else args.model
    )

    # Push to Hub
    print("Pushing to Hugging Face...")
    tokenizer.push_to_hub("gonced8/godel-multiwoz")
    model.push_to_hub("gonced8/godel-multiwoz")

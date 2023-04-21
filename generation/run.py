from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
import json
from pathlib import Path

from datasets import load_dataset
import evaluate
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled


instruction = "You are a customer service system. Your task is to answer in a empathic, informative and useful way."


def freeze_params(model):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    freeze_params(model.shared)
    for d in [model.encoder, model.decoder]:
        freeze_params(d.embed_tokens)


def build_samples(samples, args, tokenizer):
    input_texts = []
    samples_knowledge = []

    for context, knowledge in zip(samples["context"], samples["knowledge"]):
        input_text = " EOS ".join(context)
        if knowledge and args.candidates > 0:
            unique_knowledge = [*set(knowledge)]  # Remove duplicate candidates
            samples_knowledge.append(unique_knowledge[: args.candidates])
            knowledge = " | ".join(unique_knowledge[: args.candidates])
            input_text += " <|Knowledge|> " + knowledge
        input_text += " => "

        input_texts.append(input_text)

    responses = samples["delexicalized"] if args.delexicalized else samples["response"]

    # Tokenize inputs
    inputs = tokenizer(
        input_texts,
        max_length=args.max_input_length,
        truncation=True,
        return_length=True,
    )

    # Tokenize decoder inputs
    decoder_inputs = tokenizer(
        "<pad>System: ",
        max_length=args.max_output_length,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    decoder_inputs["input_ids"] = [decoder_inputs["input_ids"]] * len(samples["id"])
    decoder_inputs["attention_mask"] = [decoder_inputs["attention_mask"]] * len(
        samples["id"]
    )

    # Tokenize labels
    labels = tokenizer(
        responses,
        max_length=args.max_output_length,
        truncation=True,
        return_attention_mask=False,
        return_length=True,
    )

    # Verify if possible truncation
    overflow = {
        "overflow": [
            input_length == args.max_input_length
            or output_length == args.max_output_length
            for input_length, output_length in zip(inputs["length"], labels["length"])
        ]
    }

    possible_overflow = sum(overflow["overflow"])
    if possible_overflow:
        print(
            f"WARNING: Possible overflow in {possible_overflow} out of {len(samples['id'])} samples."
        )

    return {
        "knowledge": samples_knowledge,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": decoder_inputs["input_ids"],
        "decoder_attention_mask": decoder_inputs["attention_mask"],
        "labels": labels["input_ids"],
    } | overflow


def build_samples_prompt(samples, args, tokenizer):
    # Get input text (prompts)
    global instruction
    prompts = []
    samples_knowledge = []

    for context, knowledge in zip(samples["context"], samples["knowledge"]):
        # Add instruction
        prompt = instruction

        # Add retrieved information
        if knowledge and args.candidates > 0:
            unique_knowledge = list(set(knowledge))  # Remove duplicate candidates
            samples_knowledge.append(unique_knowledge[: args.candidates])
            possible_answers = "\n".join(unique_knowledge[: args.candidates])

            prompt += f"Based on the possible answers below, answer the conversation.\nPossible answers:\n{possible_answers}\n"
        else:
            samples_knowledge = [None] * len(samples["knowledge"])
            prompt += "Answer the conversation.\n"

        # Add context
        context = "\n".join(context)
        prompt += f"Conversation:\n{context}"

        prompts.append(prompt)

    # Tokenize inputs
    inputs = tokenizer(
        prompts,
        max_length=args.max_input_length,
        truncation=True,
        return_length=True,
    )

    # Tokenize decoder inputs
    decoder_inputs = tokenizer(
        "<pad>System: ",
        max_length=args.max_output_length,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=True,
    )
    decoder_inputs = {
        "decoder_input_ids": [decoder_inputs["input_ids"]] * len(samples["id"]),
        "decoder_attention_mask": [decoder_inputs["attention_mask"]]
        * len(samples["id"]),
    }

    # Tokenize labels
    labels = tokenizer(
        samples["response"],
        max_length=args.max_output_length,
        truncation=True,
        return_attention_mask=False,
        return_length=True,
    )
    labels["labels"] = labels.pop("input_ids")

    # Verify if possible truncation
    overflow = {
        "overflow": [
            input_length == args.max_input_length
            or output_length == args.max_output_length
            for input_length, output_length in zip(inputs["length"], labels["length"])
        ]
    }

    possible_overflow = sum(overflow["overflow"])
    if possible_overflow:
        print(
            f"WARNING: Possible overflow in {possible_overflow} out of {len(samples['id'])} samples."
        )

    inputs.pop("length")
    labels.pop("length")

    return (
        {"knowledge": samples_knowledge} | inputs | labels | decoder_inputs | overflow
    )


def compute_metrics(eval_pred):
    pred_ids, labels = eval_pred

    # Pad masked tokens
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    # Decode tensors
    predictions = tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    references = tokenizer.batch_decode(
        labels, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # BLEU
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)

    # ROUGE
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )

    return {"bleu": bleu_score["bleu"], **rouge_score}


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]

    return logits.argmax(dim=-1)


class Seq2SeqTrainerWithSave(Seq2SeqTrainer):
    def __init__(
        self,
        output: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output = output

        # Create output file
        if self.output:
            open(self.output, "w").close()
            print(f"Created file: {self.output}")

    def compute_loss(self, model, inputs, return_outputs=False):
        model_inputs = {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }
        return super().compute_loss(model, model_inputs, return_outputs)

    def prediction_step(
        self, model, inputs, prediction_loss_only=None, ignore_keys=None
    ):
        ################ Start copied code ################

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs and torch.is_tensor(
            inputs["labels"]
        )  # QUICK FIX
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs.copy()
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
        ):
            gen_kwargs["max_length"] = self.model.config.max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"]
            if gen_kwargs.get("num_beams") is not None
            else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"]
            if gen_kwargs.get("synced_gpus") is not None
            else default_synced_gpus
        )

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)
        if "global_attention_mask" in inputs:
            gen_kwargs["global_attention_mask"] = inputs.get(
                "global_attention_mask", None
            )

        # ADDED DECODER_INPUT_IDS
        if "decoder_input_ids" in inputs:
            gen_kwargs["decoder_input_ids"] = inputs["decoder_input_ids"]
        if "decoder_attention_mask" in inputs:
            gen_kwargs["decoder_attention_mask"] = inputs["decoder_attention_mask"]

        # prepare generation inputs
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if (
            hasattr(self.model, "encoder")
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if (
            gen_kwargs.get("max_length") is not None
            and generated_tokens.shape[-1] < gen_kwargs["max_length"]
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"]
            )
        elif gen_kwargs.get("max_new_tokens") is not None and generated_tokens.shape[
            -1
        ] < (gen_kwargs["max_new_tokens"] + 1):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_new_tokens"] + 1
            )

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    ##################### ADDED MODEL_INPUTS ######################
                    model_inputs = {
                        k: v
                        for k, v in inputs.items()
                        if k in ["input_ids", "attention_mask", "labels"]
                    }
                    outputs = model(**model_inputs)
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if (
                gen_kwargs.get("max_length") is not None
                and labels.shape[-1] < gen_kwargs["max_length"]
            ):
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
            elif gen_kwargs.get("max_new_tokens") is not None and labels.shape[-1] < (
                gen_kwargs["max_new_tokens"] + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, (gen_kwargs["max_new_tokens"] + 1)
                )
        else:
            labels = None

        ################ End copied code ################

        if self.output is not None:
            self.save_results(self.output, inputs, generated_tokens)

        return loss, generated_tokens, labels

    @staticmethod
    def save_results(output, samples, pred_ids):
        # Pad masked tokens
        pred_ids[pred_ids == -100] = tokenizer.pad_token_id

        # Decode tensors
        predictions = tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        with open(output, "a") as f:
            for id, context, reference, prediction, knowledge in zip(
                samples["id"],
                samples["context"],
                samples["response"],
                predictions,
                samples["knowledge"],
            ):
                f.write(
                    json.dumps(
                        {
                            "id": id,
                            "context": context,
                            "reference": reference,
                            "prediction": prediction,
                            "knowledge": knowledge,
                        }
                    )
                    + "\n"
                )


@dataclass
class DataCollatorForSeq2SeqWithText(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        text_keys = [k for k, v in features[0].items() if not torch.is_tensor(v)]
        text_features = {k: [sample.pop(k) for sample in features] for k in text_keys}

        batch = super().__call__(features, return_tensors)

        # Add text features
        batch.update(text_features)

        return batch


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["train", "validation", "test"], default="train"
    )
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument(
        "--model", type=str, default="microsoft/GODEL-v1_1-base-seq2seq"
    )
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--predict_with_generate", default=True, action=BooleanOptionalAction
    )
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--metric_for_best_model", type=str, default="bleu")
    parser.add_argument(
        "--resume_from_checkpoint", default=False, action=BooleanOptionalAction
    )
    parser.add_argument("--delexicalized", default=False, action=BooleanOptionalAction)
    parser.add_argument("--do_sample", default=False, action=BooleanOptionalAction)
    parser.add_argument("--results", type=str, help="Folder to save results on.")
    parser.add_argument("--prompt", default=False, action=BooleanOptionalAction)
    parser.add_argument("--logging", default=True, action=BooleanOptionalAction)
    parser.add_argument(
        "--freeze", type=bool, default=False, action=BooleanOptionalAction
    )
    parser.add_argument("--optimizer", type=str, default="adamw_hf")
    args = parser.parse_args()

    # Load dataset
    data_files = {}
    if args.train_dataset:
        data_files["train"] = args.train_dataset
    if args.val_dataset:
        data_files["validation"] = args.val_dataset
    if args.test_dataset:
        data_files["test"] = args.test_dataset
    dataset = load_dataset("json", data_files=data_files, field="data")

    # Prepare samples
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = dataset.map(
        build_samples_prompt if args.prompt else build_samples,
        batched=True,
        batch_size=args.preprocess_batch_size,
        fn_kwargs={"args": args, "tokenizer": tokenizer},
    )

    # Filter overflowing samples
    dataset = dataset.filter(lambda sample: not sample["overflow"])
    dataset = dataset.remove_columns("overflow")

    # Convert to PyTorch tensors
    columns = {
        "train": ["input_ids", "attention_mask", "labels"],
        "validation": [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        ],
        "test": [
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
        ],
    }

    for k, v in dataset.items():
        v.set_format(
            type="torch",
            columns=columns[k],
            output_all_columns=True,
        )

    print(dataset)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.checkpoint if args.checkpoint else args.model
    )
    if args.freeze:
        freeze_embeds(model)
        print("Froze embeddings")

    # Setup data collator
    data_collator = DataCollatorForSeq2SeqWithText(
        tokenizer=tokenizer, model=model, padding="longest"
    )

    # Setup training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{args.experiment_name}",
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size
        if args.mode == "test"
        else args.val_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.max_output_length,
        generation_num_beams=4,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        metric_for_best_model=f"eval_{args.metric_for_best_model}",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        logging_strategy="steps" if args.logging else "no",
        optim=args.optimizer,
    )

    # Create Trainer
    trainer = Seq2SeqTrainerWithSave(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"] if args.train_dataset else None,
        eval_dataset=dataset["validation"] if args.val_dataset else None,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=None
        if args.predict_with_generate
        else preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10, early_stopping_threshold=1e-4
            ),
        ],
        output=Path(args.results) / f"{args.experiment_name}.jsonl",
    )

    # Run
    if args.mode == "train":
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    elif args.mode == "validation":
        trainer.evaluate()

    elif args.mode == "test":
        output = trainer.predict(
            test_dataset=dataset["test"],
            max_length=args.max_output_length,
            do_sample=args.do_sample,
        )

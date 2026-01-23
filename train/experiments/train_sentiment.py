import argparse
import logging
from collections.abc import Iterable

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABEL2ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a sentiment classifier on FinanceInc/auditor_sentiment.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="FinanceInc/auditor_sentiment",
        help="Hugging Face dataset name to load.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="models/finBERT-tone",
        help="Base model checkpoint (local path or Hub id).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finBERT-tone-auditor",
        help="Where to save the fine-tuned model.",
    )
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-column",
        type=str,
        default="sentence",
        help="Text column name in the dataset.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Label column name in the dataset.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training when a CUDA device is available.",
    )
    return parser.parse_args()


def to_label_id(label: str | int) -> int:
    if isinstance(label, str):
        key = label.lower()
        if key not in LABEL2ID:
            raise ValueError(f"Unexpected label: {label}")
        return LABEL2ID[key]
    return int(label)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading dataset %s", args.dataset_name)
    dataset = load_dataset(args.dataset_name)
    if "train" not in dataset:
        raise ValueError("Dataset must provide a 'train' split")
    eval_split = "test" if "test" in dataset else "validation"
    if eval_split not in dataset:
        raise ValueError("Dataset must provide a 'test' or 'validation' split")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def preprocess(batch: dict[str, Iterable[str | int]]) -> dict[str, Iterable]:
        texts = batch[args.text_column]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
        )
        encodings["labels"] = [to_label_id(lbl) for lbl in batch[args.label_column]]
        return encodings

    logging.info("Tokenizing splits")
    train_dataset = dataset["train"].map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    eval_dataset = dataset[eval_split].map(
        preprocess,
        batched=True,
        remove_columns=dataset[eval_split].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        return {"accuracy": accuracy}

    fp16 = bool(args.fp16 and torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=args.seed,
        fp16=fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training")
    trainer.train()
    logging.info("Evaluating on %s split", eval_split)
    trainer.evaluate()
    logging.info("Saving model to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

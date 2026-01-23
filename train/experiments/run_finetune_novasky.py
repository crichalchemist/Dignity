"""
Full finetune runner for NovaSky-style models on multiple finance datasets.

Datasets default:
  - FinanceInc/auditor_sentiment
  - yale-nlp/FinanceMath
  - PatronusAI/financebench
  - Josephgflowers/Finance-Instruct-500k

Example:
  python train/run_finetune_novasky.py \\
    --model-name-or-path /path/to/NovaSky-AI/Sky-T1-32B-Flash \\
    --output-dir models/novasky-finetuned \\
    --per-device-train-batch-size 1 \\
    --gradient-accumulation-steps 16 \\
    --learning-rate 5e-6 \\
    --num-epochs 1 \\
    --max-length 2048 \\
    --save-steps 200

Notes:
- This is a full finetune. For 32B models on limited VRAM, expect very small
  per-device batches and heavy gradient accumulation. Gradient checkpointing is
  enabled by default.
- Interruptions are handled by frequent checkpointing; resume with
  --resume-from-checkpoint.
"""

import argparse
from collections.abc import Callable, Iterable

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

DEFAULT_DATASETS = [
    "FinanceInc/auditor_sentiment",
    "yale-nlp/FinanceMath",
    "PatronusAI/financebench",
    "Josephgflowers/Finance-Instruct-500k",
]


def _safe_get(example: dict, keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in example and example[k] is not None:
            val = str(example[k]).strip()
            if val:
                return val
    return None


def fmt_auditor(example: dict) -> str | None:
    text = _safe_get(example, ["sentence", "text"])
    label = example.get("label")
    if text is None or label is None:
        return None
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    if isinstance(label, str):
        label_str = label
    else:
        label_str = label_map.get(int(label), str(label))
    return f"Instruction: Classify the sentiment of the following financial statement as negative, neutral, or positive.\nStatement: {text}\nResponse: {label_str}"


def fmt_financemath(example: dict) -> str | None:
    question = _safe_get(example, ["question", "prompt", "input"])
    answer = _safe_get(example, ["answer", "output"])
    if question and answer:
        return f"Instruction: Solve the following finance/math question.\nQuestion: {question}\nResponse: {answer}"
    return None


def fmt_financebench(example: dict) -> str | None:
    question = _safe_get(example, ["question", "prompt"])
    answer = _safe_get(example, ["answer", "response"])
    if question and answer:
        return f"Instruction: Answer the financial question.\nQuestion: {question}\nResponse: {answer}"
    return None


def fmt_instruct(example: dict) -> str | None:
    instr = _safe_get(example, ["instruction", "prompt"])
    resp = _safe_get(example, ["output", "response", "completion"])
    if instr and resp:
        return f"Instruction: {instr}\nResponse: {resp}"
    # Fallback for generic text-only samples.
    text = _safe_get(example, ["text"])
    if text:
        return f"Instruction: Provide a helpful financial response.\nResponse: {text}"
    return None


FORMATTERS: dict[str, Callable[[dict], str | None]] = {
    "FinanceInc/auditor_sentiment": fmt_auditor,
    "yale-nlp/FinanceMath": fmt_financemath,
    "PatronusAI/financebench": fmt_financebench,
    "Josephgflowers/Finance-Instruct-500k": fmt_instruct,
}


def load_and_format_dataset(
        name: str,
        formatter: Callable[[dict], str | None],
        split: str,
        max_samples: int | None = None,
        streaming: bool = False,
) -> Dataset:
    ds = load_dataset(name, split=split, streaming=streaming)
    if streaming and max_samples is not None:
        ds = ds.take(max_samples)

    def _map_fn(example):
        txt = formatter(example)
        return {"text": txt} if txt else {"text": None}

    ds = ds.map(_map_fn, remove_columns=[c for c in ds.column_names if c != "text"])
    ds = ds.filter(lambda ex: ex["text"] is not None)
    if not streaming:
        ds = ds.shuffle(seed=42)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def build_dataset_list(
        dataset_names: list[str],
        max_samples: int | None,
        streaming: bool,
) -> Dataset:
    prepared: list[Dataset] = []
    for name in dataset_names:
        formatter = FORMATTERS.get(name, fmt_instruct)
        for split in ["train", "validation", "test"]:
            try:
                ds = load_and_format_dataset(
                    name,
                    formatter,
                    split=split,
                    max_samples=max_samples,
                    streaming=streaming,
                )
                if streaming:
                    prepared.append(ds)
                else:
                    if len(ds) > 0:
                        prepared.append(ds)
            except Exception:
                # Ignore missing splits; proceed with available ones.
                continue
    if not prepared:
        raise RuntimeError("No data loaded; check dataset names or connectivity.")
    return concatenate_datasets(prepared)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full finetune on finance datasets.")
    parser.add_argument("--model-name-or-path", required=True,
                        help="Base model path or HF id (e.g., local NovaSky checkpoint).")
    parser.add_argument("--output-dir", required=True, help="Where to save finetuned model.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated list of HF datasets.")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap total samples per split (after formatting).")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (reduces disk usage).")
    parser.add_argument("--max-length", type=int, default=1024, help="Max token length.")
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override steps; -1 uses epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to checkpoint to resume.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 weights if available.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 weights if available.")
    parser.add_argument("--no-gradient-checkpointing", action="store_true", help="Disable gradient checkpointing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    if args.bf16 and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif args.fp16 and torch.cuda.is_available():
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[info] Loading datasets: {dataset_names}")
    dataset = build_dataset_list(
        dataset_names=dataset_names,
        max_samples=args.max_samples,
        streaming=args.streaming,
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    remove_cols = dataset.column_names if isinstance(dataset, Dataset) else None
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

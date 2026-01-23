"""
Lightweight runner to query SkyThought models (e.g., NovaSky-AI/Sky-T1-32B-Flash)
for reasoning/decision-style prompts using Hugging Face transformers.

Example:
  python utils/run_skythought_inference.py \\
      --model NovaSky-AI/Sky-T1-32B-Flash \\
      --prompt "Should we increase GBPUSD exposure this hour given rising volatility?"
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "NovaSky-AI/Sky-T1-32B-Flash"

# Default SkyThought system prompt taken from the published config.
SKYTHOUGHT_SYSTEM_PROMPT = (
    "Act as an elite full-stack developer and mentor with deep expertise in "
    "shipping secure, reliable trading systems for FX/finance. Explore the "
    "question with a disciplined long-form reasoning process before giving the "
    "final answer. Assume inputs may include highly formatted results from two "
    "separate forecasting agents plus sentiment analysis routed into Sky-T1; "
    "tie these streams together, resolve conflicts, and keep the expected "
    "format. Highlight and correct security, robustness, deployment, and "
    "financial risk controls (e.g., data integrity, leakage prevention, model "
    "bias) as you reason so the reader learns from each fix. "
    "Structure your response into two sections: Thought and Solution. Use the "
    "exact markers below.\n"
    "<|begin_of_thought|>\n"
    "Step 1 - Clarify: restate the question, trading/analytics goal, upstream "
    "signals (forecasting vs. sentiment), and any assumptions.\n\n"
    "Step 2 - Plan: list candidate approaches (including data, modeling, "
    "execution, and how to normalize structured agent outputs) and choose the "
    "best one.\n\n"
    "Step 3 - Deep Dive: work through the approach step by step, explicitly "
    "calling out performance, security, correctness, and financial risk "
    "checks; track provenance of each agent input and handle formatting "
    "strictly.\n\n"
    "Step 4 - Validate: test reasoning against edge cases or counterexamples; "
    "fix any gaps, including failure modes like data drift, latency spikes, or "
    "misaligned agent outputs.\n\n"
    "Step 5 - Summarize: recap the logic, reconciled signals, and confirmed "
    "safeguards.\n"
    "<|end_of_thought|>\n"
    "Then provide the final answer in the Solution section using:\n"
    "<|begin_of_solution|>\n"
    "A concise, actionable solution that follows from the validated reasoning, "
    "including key steps, recommended commands, and explicit security and risk "
    "notes.\n"
    "<|end_of_solution|>\n"
    "Now, solve the following question using this structure:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkyThought model inference for a custom prompt.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local path.")
    parser.add_argument(
        "--prompt",
        help="Inline prompt text to send to the model. Mutually exclusive with --prompt-file.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a text file containing the prompt. Mutually exclusive with --prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Override the default SkyThought system prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling parameter.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Torch dtype for model weights.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Device map passed to transformers (e.g., "auto", "cuda", "cpu").',
    )
    return parser.parse_args()


def load_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    if prompt and prompt_file:
        raise ValueError("Provide exactly one of --prompt or --prompt-file.")
    if prompt_file:
        return prompt_file.read_text()
    if prompt:
        return prompt
    raise ValueError("A prompt is required. Use --prompt or --prompt-file.")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    user_prompt = load_prompt(args.prompt, args.prompt_file)
    system_prompt = args.system_prompt or SKYTHOUGHT_SYSTEM_PROMPT
    full_prompt = f"{system_prompt}\n\n{user_prompt}\n\nAssistant:"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None and tokenizer.pad_token_id is not None:
        eos_token_id = tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must provide an EOS or PAD token id for generation.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(completion.strip())


if __name__ == "__main__":
    main()

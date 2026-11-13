"""Rehearse the evaluation on tuning seeds, away from every committed result.

For each seed of `SELECTION_SEEDS` this writes an isolated config under
`runs/rehearsal/<stage>/<variant>/seed_<N>/` (the hard preset plus the variant's
overrides, with results, charts and checkpoints redirected into that directory) and
refuses any other seed, a repeated seed or an unknown variant before any data is
generated.

    python scripts/rehearse.py --stage candidate --variant base --seeds 1000,1001
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # where runs/rehearsal/ is written
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import SELECTION_SEEDS, DignityConfig  # noqa: E402

PRESET = Path(__file__).resolve().parent.parent / "config" / "synthetic_hard.yaml"
REHEARSAL_MIN_REVIEW_RATE = 0.02  # a rehearsal bar; verify_release.py never applies it
STAGES = ("timing", "variants", "candidate")
VARIANTS: dict[str, dict[str, object]] = {
    "base": {},
    "cold": {"train.forecast_warm_start": False, "policy.multiplier_lr": 0.05},
    "cap10": {"train.forecast_pos_weight_cap": 10.0, "policy.multiplier_lr": 0.2},
    "cap3": {"train.forecast_pos_weight_cap": 3.0},
    "narrow": {"model.hidden_size": 128},
    "lr1e-3": {"train.lr": 0.001},
}


def _seed_list(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",")]
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"seeds must be integers: {text!r}") from err


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rehearse the evaluation.")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seeds", type=_seed_list, required=True)
    parser.add_argument("--tasks", default="risk,forecast,policy")
    parser.add_argument("--privacy-sweep", action="store_true")
    return parser.parse_args(argv)


def seed_dir(stage: str, variant: str, seed: int) -> Path:
    return ROOT / "runs" / "rehearsal" / stage / variant / f"seed_{seed}"


def write_seed_config(stage: str, variant: str, seed: int) -> Path:
    """Write the seed's config.yaml; loading it fails on an unknown key."""
    root = seed_dir(stage, variant, seed)
    config = yaml.safe_load(PRESET.read_text())
    overrides = dict(VARIANTS[variant])
    overrides.update(
        {
            "eval.out_dir": str(root / "results"),
            "eval.img_dir": str(root / "img"),
            "train.checkpoint_dir": str(root / "ckpt"),
        }
    )
    for dotted, value in overrides.items():
        *parents, key = dotted.split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        node[key] = value
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    DignityConfig.from_yaml(str(path))
    return path


def _refusal(args: argparse.Namespace) -> str | None:
    outside = [s for s in args.seeds if s not in SELECTION_SEEDS]
    if outside:
        return f"seeds {outside} are not selection seeds {list(SELECTION_SEEDS)}"
    if len(set(args.seeds)) != len(args.seeds):
        return f"seeds {args.seeds} repeat a seed"
    if args.variant not in VARIANTS:
        return f"unknown variant {args.variant!r}; choose from {sorted(VARIANTS)}"
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    refusal = _refusal(args)
    if refusal is not None:
        print(f"rehearse: {refusal}", file=sys.stderr)
        return 2
    for seed in args.seeds:
        write_seed_config(args.stage, args.variant, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

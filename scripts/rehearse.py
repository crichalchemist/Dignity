"""Rehearse the evaluation on tuning seeds, away from every committed result.

For each seed of `SELECTION_SEEDS` this writes an isolated config under
`runs/rehearsal/<stage>/<variant>/seed_<N>/` (the hard preset plus the variant's
overrides, with results, charts and checkpoints redirected into that directory) and
refuses any other seed, a repeated seed or an unknown variant before any data is
generated.

    python scripts/rehearse.py --stage candidate --variant base --seeds 1000,1001
"""

import argparse
import functools
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO  # where runs/rehearsal/ is written
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval.run  # noqa: E402
from core.config import (  # noqa: E402
    SELECTION_SEEDS,
    DignityConfig,
    EvalConfig,
    PolicyConfig,
)
from train.checkpoint import load_checkpoint  # noqa: E402

PRESET = REPO / "config" / "synthetic_hard.yaml"
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


@functools.cache
def _release():
    """scripts/verify_release.py, loaded by path: its criteria are never copied."""
    path = Path(__file__).resolve().parent / "verify_release.py"
    spec = importlib.util.spec_from_file_location("verify_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dig(data: dict | None, dotted: str):
    """Value at a dotted path, None when any step is missing (reporting only)."""
    node = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _read_result(results: Path, task: str) -> dict | None:
    path = results / f"{task}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _risk_val_pr_auc(ckpt: Path, data_seed: int, base_seed: int, n: int):
    """Each training seed's best validation PR-AUC, None if any checkpoint is absent."""
    run = ckpt / f"eval_tuning_{data_seed}" / "risk"
    paths = [run / f"train_seed_{base_seed + k}" / "best.pt" for k in range(n)]
    if not all(p.exists() for p in paths):
        return None
    return [load_checkpoint(p)["best_metric"] for p in paths]


def _judge(results: dict[str, dict | None], n_train_seeds: int) -> dict:
    """Criteria 2 to 4 of verify_release.py on one seed's results (None: not run)."""
    release = _release()
    policy_defaults = PolicyConfig()
    verdict: dict[str, bool | None] = {"C2": None, "C3": None, "C4": None}
    if results["risk"] is not None:
        verdict["C2"] = release._criterion_2(results["risk"]).passed
    if results["forecast"] is not None:
        verdict["C3"] = release._criterion_3(results["forecast"], n_train_seeds).passed
    policy = results["policy"]
    if policy is not None:
        c4 = release._criterion_4(
            policy,
            policy_defaults.review_budget,
            n_train_seeds,
            policy_defaults.collapse_review_factor,
        )
        rates = _dig(policy, "metrics.seed_spread.review_rate.values")
        floor_met = (
            isinstance(rates, list)
            and bool(rates)
            and all(r is not None and r >= REHEARSAL_MIN_REVIEW_RATE for r in rates)
        )
        verdict["C4"] = c4.passed and floor_met
    for name, task in (("C3", "forecast"), ("C4", "policy")):
        if _dig(results[task], "metrics.raised") is not None:
            verdict[name] = False
    return verdict


def _summarise(results: dict[str, dict | None], risk_val, wall: float, n: int):
    """One seed's summary entry: verdicts, the raised payloads and the series."""
    raised = {
        task: _dig(results[task], "metrics.raised")
        for task in ("forecast", "policy")
        if _dig(results[task], "metrics.raised") is not None
    }

    def series(task, path):
        if task in raised:
            return None
        return _dig(results[task], f"metrics.seed_spread.{path}.values")

    return {
        **_judge(results, n),
        "raised": raised or None,
        "forecast_val_lift_upper": series("forecast", "val_lift_upper"),
        "forecast_floor": series("forecast", "risk_val_lift_upper"),
        "forecast_best_epoch": series("forecast", "best_epoch"),
        "policy_val_neg_cost": series("policy", "val_neg_cost"),
        "review_rate": series("policy", "review_rate"),
        "review_multiplier": series("policy", "review_multiplier"),
        "uncapped_review_share": series("policy", "uncapped_review_share"),
        "risk_val_pr_auc": risk_val,
        "wall_seconds": wall,
    }


def _git_sha() -> str:
    out = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    return out.decode().strip() if isinstance(out, bytes) else out.strip()


def _run_seed(args: argparse.Namespace, seed: int) -> tuple[dict, int]:
    """Run eval.run on one seed; return its summary entry and 0, or 1 on an error."""
    cfg_path = write_seed_config(args.stage, args.variant, seed)
    config = DignityConfig.from_yaml(str(cfg_path))
    # a result file left by an earlier run of this seed must never be judged
    for stale in (Path(config.eval.out_dir), Path(config.eval.img_dir)):
        shutil.rmtree(stale, ignore_errors=True)
    argv = ["--config", str(cfg_path), "--seed", "tuning", "--data-seed", str(seed)]
    argv += ["--tasks", args.tasks]
    if args.privacy_sweep:
        argv.append("--privacy-sweep")
    start = time.perf_counter()
    try:
        status = eval.run.main(argv)
        error = None if status in (0, 1) else f"eval.run exited {status}"
    except SystemExit as err:
        error = f"eval.run exited {err.code}"
    except Exception as err:  # noqa: BLE001 - recorded in the summary, then exit 1
        error = f"{type(err).__name__}: {err}"
    wall = time.perf_counter() - start
    if error is not None:
        print(f"rehearse: seed {seed}: {error}", file=sys.stderr)
        return {"error": error, "wall_seconds": wall}, 1
    # a recorded raise (status 1) is data: eval.run already printed TRAINING_RAISED
    try:
        results_dir = Path(config.eval.out_dir)
        tasks = {t.strip() for t in args.tasks.split(",")}
        results = {
            t: _read_result(results_dir, t) if t in tasks else None for t in tasks
        }
        results = {t: results.get(t) for t in ("risk", "forecast", "policy")}
        n = EvalConfig().n_train_seeds
        ckpt = Path(config.train.checkpoint_dir)
        risk_val = _risk_val_pr_auc(ckpt, seed, config.seed, n)
        return _summarise(results, risk_val, wall, n), 0
    except Exception as err:  # noqa: BLE001 - recorded in the summary, then exit 1
        message = f"unusable results: {type(err).__name__}: {err}"
        print(f"rehearse: seed {seed}: {message}", file=sys.stderr)
        return {"error": message, "wall_seconds": wall}, 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    refusal = _refusal(args)
    if refusal is not None:
        print(f"rehearse: {refusal}", file=sys.stderr)
        return 2
    summary = {
        "stage": args.stage,
        "variant": args.variant,
        "overrides": VARIANTS[args.variant],
        "git_sha": _git_sha(),
        "seeds": {},
    }
    out = seed_dir(args.stage, args.variant, args.seeds[0]).parent / "summary.json"
    if out.exists():
        earlier = json.loads(out.read_text())
        # seeds run by an earlier invocation at this commit stay in the summary
        if earlier.get("git_sha") == summary["git_sha"]:
            summary["seeds"] = dict(earlier.get("seeds", {}))
    exit_code = 0
    for seed in args.seeds:
        entry, failed = _run_seed(args, seed)
        summary["seeds"][str(seed)] = entry
        exit_code |= failed
        out.write_text(json.dumps(summary, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

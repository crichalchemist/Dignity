"""Evaluation run: train through ``fit``, score the test split once, write results.

python -m eval.run --config config/synthetic_hard.yaml --seed tuning
python -m eval.run --smoke --seed tuning
"""

import argparse
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from core.config import DignityConfig
from data.loader import create_dataloader
from data.pipeline import TransactionPipeline
from data.source.synthetic import RAW_COLUMNS
from eval.baselines import block_tree, majority, window_logistic
from eval.metrics import model_gate, risk_metrics
from eval.plots import plot_per_type_recall, plot_pr_curves
from train.checkpoint import build_model_from_meta, load_checkpoint
from train.engine import predict
from train.loop import FitResult, fit, prepare_data

_ROOT = Path(__file__).resolve().parent.parent
_PRINTED_METRICS = ("auroc", "pr_auc", "recall_at_1pct_fpr", "brier")


def config_hash(config: DignityConfig) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _train_seeds(
    config: DignityConfig, data, seed_name: str
) -> list[tuple[int, FitResult]]:
    """Train one model per training seed; model init is independent of the data."""
    runs = []
    for k in range(config.eval.n_train_seeds):
        s = config.seed + k
        run_dir = Path(config.train.checkpoint_dir) / f"eval_{seed_name}"
        run_dir = run_dir / f"train_seed_{s}"
        runs.append((s, fit(config, data, s, str(run_dir))))
    return runs


def _score(config: DignityConfig, result: FitResult, X: np.ndarray) -> np.ndarray:
    """Score windows with the best (validation-selected) checkpoint."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(result.best_path, map_location=device)
    model = build_model_from_meta(result.meta).to(device)
    model.load_state_dict(ckpt["model_state"])
    loader = create_dataloader(
        X,
        None,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    return predict(model, loader, device)["score"]


def _write_result(
    config: DignityConfig, name: str, metrics: dict, baselines: dict, run_ctx: dict
) -> Path:
    """Write ``<out_dir>/<name>.json`` with the contract's five top-level keys."""
    out_dir = Path(config.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_hash": run_ctx["config_hash"],
        "seed": int(run_ctx["seed"]),
        "git_sha": run_ctx["git_sha"],
        "metrics": metrics,
        "baselines": baselines,
    }
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _run_ctx(config: DignityConfig) -> dict:
    return {
        "config_hash": config_hash(config),
        "seed": config.data.synthetic.seed,
        "git_sha": _git_sha(),
    }


def _fmt(entry: dict) -> str:
    return f"{entry['value']:.4f}"


def _fmt_ci(entry: dict) -> str:
    low, high = entry["ci_low"], entry["ci_high"]
    if low is None or high is None:
        return f"{_fmt(entry)} [n/a]"
    return f"{_fmt(entry)} [{low:.4f}, {high:.4f}]"


def _spread(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) >= 2 else None,
    }


def run_risk(
    config: DignityConfig, seed_name: str, run_ctx: dict | None = None
) -> dict:
    """Train, score the test split once, write risk.json and charts; return payload."""
    run_ctx = run_ctx or _run_ctx(config)
    seed = config.data.synthetic.seed
    n_boot = config.eval.n_bootstrap

    df, data = prepare_data(config)
    train, test = data.train, data.test
    runs = _train_seeds(config, data, seed_name)
    scores = [_score(config, result, test.X) for _, result in runs]
    score_0 = scores[0]  # headline model: first training seed, declared in advance

    metrics = risk_metrics(
        test.y_risk,
        score_0,
        test.block_index,
        n_boot,
        seed,
        anomaly_type=test.anomaly_type,
    )

    df_train = df[df.block_id.isin(np.unique(train.block_index))]
    df_test = df[df.block_id.isin(np.unique(test.block_index))]
    y_block, s_tree = block_tree(df_train, df_test)
    model_block = np.array(
        [score_0[test.block_index == b].max() for b in np.unique(test.block_index)]
    )
    block_ids = np.arange(len(y_block))
    metrics["block_level"] = risk_metrics(y_block, model_block, block_ids, n_boot, seed)

    point = [risk_metrics(test.y_risk, s, test.block_index, 0, seed) for s in scores]
    metrics["seed_spread"] = {
        "train_seeds": [s for s, _ in runs],
        "pr_auc": _spread([float(m["pr_auc"]["value"]) for m in point]),
        "auroc": _spread([float(m["auroc"]["value"]) for m in point]),
    }

    # the window baseline is defined on the generator's raw columns (spec 5.2)
    raw = TransactionPipeline(
        seq_len=config.data.seq_len, features=list(RAW_COLUMNS)
    ).process_blocks(
        df,
        config.data.split,
        config.model.horizon,
        config.data.label_overlap,
        np.random.default_rng(config.data.synthetic.seed),
    )
    assert np.array_equal(raw.test.block_index, test.block_index)
    assert np.array_equal(raw.test.y_risk, test.y_risk)

    s_majority = majority(train.y_risk, len(test.y_risk))
    s_logistic = window_logistic(raw.train, raw.test)
    baselines = {
        "majority": risk_metrics(
            test.y_risk, s_majority, test.block_index, n_boot, seed
        ),
        "window_logistic": risk_metrics(
            test.y_risk, s_logistic, test.block_index, n_boot, seed
        ),
        "block_tree": risk_metrics(y_block, s_tree, block_ids, n_boot, seed),
    }

    window_gate = model_gate(
        metrics["pr_auc"],
        {
            "majority": baselines["majority"]["pr_auc"],
            "window_logistic": baselines["window_logistic"]["pr_auc"],
        },
    )
    block_gate = model_gate(
        metrics["block_level"]["pr_auc"],
        {"block_tree": baselines["block_tree"]["pr_auc"]},
    )
    metrics["model_gate"] = {
        "window": window_gate,
        "block": block_gate,
        "passed": window_gate["passed"] and block_gate["passed"],
    }

    img_dir = Path(config.eval.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    plot_pr_curves(
        {
            "model": (test.y_risk, score_0),
            "window_logistic": (test.y_risk, s_logistic),
            "majority": (test.y_risk, s_majority),
        },
        str(img_dir / "risk_pr_curve.png"),
    )
    plot_per_type_recall(metrics["per_type"], str(img_dir / "risk_per_type_recall.png"))

    path = _write_result(config, "risk", metrics, baselines, run_ctx)

    for name in _PRINTED_METRICS:
        print(
            f"{name}: model {_fmt_ci(metrics[name])} "
            f"| majority {_fmt(baselines['majority'][name])} "
            f"| window_logistic {_fmt(baselines['window_logistic'][name])}"
        )
    print(
        f"block-level pr_auc: model {_fmt(metrics['block_level']['pr_auc'])} "
        f"| block_tree {_fmt(baselines['block_tree']['pr_auc'])}"
    )
    verdict = "PASS" if metrics["model_gate"]["passed"] else "FAIL"
    print(
        f"model gate: {verdict} "
        f"(window {'pass' if window_gate['passed'] else 'fail'}, "
        f"block {'pass' if block_gate['passed'] else 'fail'})"
    )
    print(f"wrote {path}")

    return {
        "config_hash": run_ctx["config_hash"],
        "seed": int(run_ctx["seed"]),
        "git_sha": run_ctx["git_sha"],
        "metrics": metrics,
        "baselines": baselines,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Dignity evaluation.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=str, help="Path to config YAML file")
    source.add_argument(
        "--smoke", action="store_true", help="CI-sized run of config/smoke.yaml"
    )
    parser.add_argument("--seed", choices=("tuning", "eval"), required=True)
    parser.add_argument("--tasks", default="risk")
    args = parser.parse_args(argv)

    for task in args.tasks.split(","):
        if task.strip() != "risk":
            parser.error(f"task {task.strip()} is not implemented until M4")
    if args.smoke and args.seed == "eval":
        parser.error("--smoke never runs on the evaluation seed")

    config = DignityConfig.from_yaml(
        str(_ROOT / "config" / "smoke.yaml") if args.smoke else args.config
    )
    config.data.synthetic.seed = getattr(config.eval, f"{args.seed}_seed")
    run_ctx = _run_ctx(config)  # once, after the seed override

    run_risk(config, args.seed, run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

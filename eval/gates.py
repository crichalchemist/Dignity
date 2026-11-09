"""Tuning tool. Runs on the tuning seed only; seeds come from `core.config.EvalConfig`."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from core.config import DignityConfig, EvalConfig
from data.pipeline import TransactionPipeline, WindowSet
from data.source.synthetic import RAW_COLUMNS, SyntheticGenerator
from eval.baselines import block_tree, window_logistic

TUNING_SEED = EvalConfig().tuning_seed
EVAL_SEED = EvalConfig().eval_seed

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRESETS = {
    "easy": str(_ROOT / "config" / "synthetic_easy.yaml"),
    "medium": str(_ROOT / "config" / "synthetic_medium.yaml"),
    "hard": str(_ROOT / "config" / "synthetic_hard.yaml"),
}

AUROC_BAND = (0.55, 0.80)
FLATNESS_MAX_SPREAD = 0.1


def onset_bucket_recall(
    scores: np.ndarray, ws: WindowSet, n_buckets: int = 5
) -> list[float]:
    """Recall at 10% FPR per onset-position bucket of the positive windows.

    Only windows whose anomaly starts inside the window (onset_offset >= 1) count.
    """
    thr = np.quantile(scores[ws.y_risk == 0], 0.9)
    seq_len = ws.X.shape[1]
    pos = (ws.y_risk == 1) & (ws.onset_offset >= 1)
    bucket = ws.onset_offset * n_buckets // seq_len
    recalls = []
    for b in range(n_buckets):
        sel = pos & (bucket == b)
        if not sel.any():
            raise ValueError(f"onset bucket {b} has no positive windows")
        # strict >: with ties at the 90th percentile of negatives, > keeps FPR <= 10%;
        # intentionally differs from M3.2's threshold_at_fpr (>=), which uses a
        # roc_curve threshold.
        recalls.append(float(np.mean(scores[sel] > thr)))
    return recalls


def _score_preset(
    path: str | Path, seed: int
) -> tuple[dict[str, float], list[float] | None]:
    """Baseline AUROCs for one preset; onset recall too for the hard preset."""
    cfg = DignityConfig.from_yaml(path)
    syn = replace(cfg.data.synthetic, seed=seed)
    df = SyntheticGenerator(syn).generate()
    d = TransactionPipeline(
        seq_len=cfg.data.seq_len, features=list(RAW_COLUMNS)
    ).process_blocks(
        df,
        cfg.data.split,
        cfg.model.horizon,
        cfg.data.label_overlap,
        np.random.default_rng(seed),
    )
    win_scores = window_logistic(d.train, d.val)
    y_block, s_block = block_tree(
        df[df["block_id"].isin(np.unique(d.train.block_index))],
        df[df["block_id"].isin(np.unique(d.val.block_index))],
    )
    auroc = {
        "window_logistic": float(roc_auc_score(d.val.y_risk, win_scores)),
        "block_tree": float(roc_auc_score(y_block, s_block)),
    }
    recall = None
    if Path(path).stem == Path(DEFAULT_PRESETS["hard"]).stem:
        recall = onset_bucket_recall(win_scores, d.val)
    return auroc, recall


def data_gate(config_paths: dict[str, str], seed: int) -> dict:
    auroc: dict[str, dict[str, float]] = {}
    onset_recall: list[float] = []
    for preset in ("easy", "medium", "hard"):
        auroc[preset], recall = _score_preset(config_paths[preset], seed)
        if recall is not None:
            onset_recall = recall

    lo, hi = AUROC_BAND

    def in_band(name: str) -> bool:
        return lo <= auroc["hard"][name] <= hi

    def monotone(name: str) -> bool:
        return auroc["easy"][name] > auroc["medium"][name] > auroc["hard"][name]

    checks = {
        "window_logistic_in_band": in_band("window_logistic"),
        "block_tree_in_band": in_band("block_tree"),
        "monotone": monotone("window_logistic") and monotone("block_tree"),
        "positional_flatness": max(onset_recall) - min(onset_recall)
        <= FLATNESS_MAX_SPREAD,
    }
    return {
        "seed": seed,
        "auroc": auroc,
        "onset_recall": onset_recall,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic-data gate.")
    parser.add_argument("--seed", type=int, default=TUNING_SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.seed == EVAL_SEED:
        print(f"refusing seed {EVAL_SEED}: it is the evaluation seed")
        return 2
    result = data_gate(DEFAULT_PRESETS, args.seed)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

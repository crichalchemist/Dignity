"""Ten-seed data gate. Runs on the tuning seeds only (`core.config.TUNING_SEEDS`)."""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core.config import TUNING_SEEDS, DignityConfig
from data.pipeline import TransactionPipeline, WindowSet
from data.source.synthetic import RAW_COLUMNS, SyntheticGenerator
from eval.baselines import block_tree, window_logistic

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRESETS = {
    "easy": str(_ROOT / "config" / "synthetic_easy.yaml"),
    "medium": str(_ROOT / "config" / "synthetic_medium.yaml"),
    "hard": str(_ROOT / "config" / "synthetic_hard.yaml"),
}

AUROC_BAND = (0.55, 0.80)
FLATNESS_MAX_SPREAD = 0.1
STRENGTH_GRID = (0.6, 0.7, 0.8)
HEADROOM_HORIZONS = (1, 5, 10)
HEADROOM_LIFT_RANGE = (3.0, 15.0)
HEADROOM_MIN_SEED_LIFT = 1.5
HEADROOM_PR_AUC_RATIO = 1.5
HEADROOM_MIN_WIN_SHARE = 7 / 8
HEADROOM_PPV_RANGE = (0.25, 0.65)
MIN_TEST_ONSETS = 200
GATE_N_BOOT = 1000
BASELINES = ("window_logistic", "block_tree")
PRESET_ORDER = ("easy", "medium", "hard")


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


def preset_config_hash(config_paths: dict[str, str]) -> str:
    """First 16 hex chars of the sha256 of every preset's `data` block."""
    blocks = {
        name: asdict(DignityConfig.from_yaml(path).data)
        for name, path in config_paths.items()
    }
    return hashlib.sha256(json.dumps(blocks, sort_keys=True).encode()).hexdigest()[:16]


def _score_preset(
    path: str | Path,
    seed: int,
    n_blocks: int | None = None,
    strength: float | None = None,
) -> dict:
    """One preset on one seed: val and test AUROCs of both baselines.

    The hard preset also returns the onset recalls per split and the number of
    scorable test onsets. `n_blocks` and `strength` replace the preset's values in
    memory. `headroom` is filled by the forecast-headroom check.
    """
    cfg = DignityConfig.from_yaml(path)
    changes: dict = {"seed": seed}
    if n_blocks is not None:
        changes["n_blocks"] = n_blocks
    if strength is not None:
        changes["precursor_strength"] = strength
    df = SyntheticGenerator(replace(cfg.data.synthetic, **changes)).generate()
    d = TransactionPipeline(
        seq_len=cfg.data.seq_len, features=list(RAW_COLUMNS)
    ).process_blocks(
        df,
        cfg.data.split,
        cfg.model.horizon,
        cfg.data.label_overlap,
        np.random.default_rng(seed),
    )

    def blocks_of(ws: WindowSet) -> pd.DataFrame:
        return df[df["block_id"].isin(np.unique(ws.block_index))]

    train_blocks = blocks_of(d.train)
    auroc: dict[str, dict[str, float]] = {name: {} for name in BASELINES}
    scores = {}
    for split, ws in (("val", d.val), ("test", d.test)):
        scores[split] = window_logistic(d.train, ws)
        auroc["window_logistic"][split] = float(roc_auc_score(ws.y_risk, scores[split]))
        y_block, s_block = block_tree(train_blocks, blocks_of(ws))
        auroc["block_tree"][split] = float(roc_auc_score(y_block, s_block))
    is_hard = Path(path).stem == Path(DEFAULT_PRESETS["hard"]).stem
    return {
        "auroc": auroc,
        "onset_recall": {
            "val": onset_bucket_recall(scores["val"], d.val),
            "test": onset_bucket_recall(scores["test"], d.test),
        }
        if is_hard
        else None,
        "test_onsets": int((d.test.y_forecast[:, 0] == 1).sum()) if is_hard else None,
        "headroom": None,
    }


def _check_request(seeds: Sequence[int], strength: float | None) -> None:
    """Refuse before any data is generated."""
    if len(seeds) == 0:
        raise ValueError("no seeds given")
    for seed in seeds:
        if seeds.count(seed) > 1:
            raise ValueError(f"duplicate seed {seed}")
    for seed in seeds:
        if seed not in TUNING_SEEDS:
            raise ValueError(
                f"refusing seed {seed}: the gate runs on the tuning seeds "
                f"{TUNING_SEEDS[0]}-{TUNING_SEEDS[-1]} only"
            )
    if strength is not None and strength not in STRENGTH_GRID:
        raise ValueError(f"strength {strength} is not on the grid {STRENGTH_GRID}")


def _preset_strengths(
    config_paths: dict[str, str], strength: float | None
) -> dict[str, float]:
    """Build-up strength per preset: hard S, medium S + 0.1, easy S + 0.2."""
    if strength is not None:
        return {
            "hard": strength,
            "medium": round(strength + 0.1, 2),
            "easy": round(strength + 0.2, 2),
        }
    return {
        name: DignityConfig.from_yaml(
            config_paths[name]
        ).data.synthetic.precursor_strength
        for name in PRESET_ORDER
    }


def data_gate(
    config_paths: dict[str, str],
    seeds: Sequence[int],
    n_blocks: int | None = None,
    strength: float | None = None,
) -> dict:
    seeds = list(seeds)
    _check_request(seeds, strength)
    strengths = _preset_strengths(config_paths, strength)
    scored = {
        preset: [
            _score_preset(
                config_paths[preset],
                seed,
                n_blocks=n_blocks,
                strength=None if strength is None else strengths[preset],
            )
            for seed in seeds
        ]
        for preset in PRESET_ORDER
    }

    auroc: dict[str, dict[str, dict]] = {}
    for preset, results in scored.items():
        auroc[preset] = {}
        for name in BASELINES:
            val = [r["auroc"][name]["val"] for r in results]
            test = [r["auroc"][name]["test"] for r in results]
            auroc[preset][name] = {
                "val": val,
                "test": test,
                "pooled_mean": float(np.mean(val + test)),
            }

    hard = scored["hard"]
    onset_recall: dict = {}
    for split in ("val", "test"):
        by_bucket = np.mean([r["onset_recall"][split] for r in hard], axis=0)
        onset_recall[split] = [float(x) for x in by_bucket]
        onset_recall[f"{split}_spread"] = float(by_bucket.max() - by_bucket.min())
    test_onsets = [r["test_onsets"] for r in hard]

    lo, hi = AUROC_BAND

    def pooled(preset: str, name: str) -> float:
        return auroc[preset][name]["pooled_mean"]

    def monotone(name: str) -> bool:
        return pooled("easy", name) > pooled("medium", name) > pooled("hard", name)

    checks = {
        "window_logistic_in_band": lo <= pooled("hard", "window_logistic") <= hi,
        "block_tree_in_band": lo <= pooled("hard", "block_tree") <= hi,
        "monotone": all(monotone(name) for name in BASELINES)
        and strengths["easy"] > strengths["medium"] > strengths["hard"],
        "positional_flatness": onset_recall["val_spread"] <= FLATNESS_MAX_SPREAD,
        "test_onsets": all(n >= MIN_TEST_ONSETS for n in test_onsets),
    }
    checks = {name: bool(ok) for name, ok in checks.items()}
    hard_config = DignityConfig.from_yaml(config_paths["hard"])
    return {
        "seeds": seeds,
        "n_blocks": n_blocks or hard_config.data.synthetic.n_blocks,
        "strength_override": strength,
        "preset_config_hash": preset_config_hash(config_paths),
        "auroc": auroc,
        "onset_recall": onset_recall,
        "test_onsets": test_onsets,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _seed_list(text: str) -> list[int]:
    try:
        return [int(x) for x in text.split(",")] if text.strip() else []
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--seeds takes integers separated by commas, got {text!r}"
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic-data gate.")
    parser.add_argument(
        "--seeds",
        type=_seed_list,
        default=list(TUNING_SEEDS),
        help="comma-separated tuning seeds (default: all ten, 1000-1009)",
    )
    parser.add_argument(
        "--n-blocks", type=int, default=None, help="blocks per preset (default: preset)"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help=f"build-up strength of the hard preset, one of {STRENGTH_GRID}",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        _check_request(args.seeds, args.strength)
    except ValueError as err:
        print(err)
        return 2
    result = data_gate(
        DEFAULT_PRESETS, args.seeds, n_blocks=args.n_blocks, strength=args.strength
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

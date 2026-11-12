"""Ten-seed data gate. Runs on the tuning seeds only (`core.config.TUNING_SEEDS`)."""

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from core.config import TUNING_SEEDS, DignityConfig
from data.pipeline import SplitData, TransactionPipeline, WindowSet
from data.source.synthetic import RAW_COLUMNS, SyntheticGenerator
from eval.baselines import (
    block_tree,
    fit_forecast_probe,
    risk_score_forecast,
    window_logistic,
    window_logistic_forecast,
)
from eval.metrics import detection_gain, threshold_at_fpr, warned_onsets

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRESETS = {
    "easy": str(_ROOT / "config" / "synthetic_easy.yaml"),
    "medium": str(_ROOT / "config" / "synthetic_medium.yaml"),
    "hard": str(_ROOT / "config" / "synthetic_hard.yaml"),
}

AUROC_BAND = (0.55, 0.80)
FLATNESS_MAX_SPREAD = 0.15
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


def _stratum(ws: WindowSet) -> WindowSet:
    """Windows the forecast mask keeps and no anomaly has touched (`y_risk == 0`)."""
    keep = ws.forecast_mask & (ws.y_risk == 0)
    return WindowSet(**{f.name: getattr(ws, f.name)[keep] for f in fields(ws)})


def _build_up_position(build_up_block: np.ndarray) -> np.ndarray:
    """Position fraction `(i + 1) / n` inside each maximal run of equal nonzero values."""
    out = np.zeros(len(build_up_block))
    start = 0
    while start < len(build_up_block):
        end = start
        while (
            end < len(build_up_block) and build_up_block[end] == build_up_block[start]
        ):
            end += 1
        if build_up_block[start] != 0:
            out[start:end] = np.arange(1, end - start + 1) / (end - start)
        start = end
    return out


def _last_row_build_up(
    df: pd.DataFrame, ws: WindowSet
) -> tuple[np.ndarray, np.ndarray]:
    """`build_up` and the shape-only score at each window's last row."""
    build_up = df["build_up"].to_numpy()
    shape = np.zeros(len(df))
    for rows in df.groupby("block_id", sort=False).indices.values():
        shape[rows] = _build_up_position(build_up[rows])
    row_in_block = df.groupby("block_id", sort=False).cumcount().to_numpy()
    where = pd.Series(
        np.arange(len(df)),
        index=pd.MultiIndex.from_arrays([df["block_id"].to_numpy(), row_in_block]),
    )
    rows = where.reindex(
        pd.MultiIndex.from_arrays([ws.block_index, ws.end_row])
    ).to_numpy()
    return build_up[rows], shape[rows]


def _masked_ap(y: np.ndarray, score: np.ndarray, mask: np.ndarray) -> float | None:
    """Average precision on the masked rows; `None` when `y` holds one class there."""
    if len(np.unique(y[mask])) < 2:
        return None
    return float(average_precision_score(y[mask], score[mask]))


def _stratified_lift(d: SplitData, column: int) -> float | None:
    """AP over prevalence at one horizon, probe fitted and scored inside the stratum."""
    train, test = _stratum(d.train), _stratum(d.test)
    y = test.y_forecast[:, column]
    one_class = len(np.unique(y)) < 2 or any(
        len(np.unique(train.y_forecast[:, h])) < 2
        for h in range(train.y_forecast.shape[1])
    )
    if one_class:
        return None
    score = window_logistic_forecast(train, test)[:, column]
    return float(average_precision_score(y, score) / y.mean())


def forecast_headroom(df: pd.DataFrame, d: SplitData, horizon: int, seed: int) -> dict:
    """Does a window probe forecast anomalies that today's risk score cannot?

    All on the test split. The PR-AUC ratio and the wins read `probe_risk_aware`, the
    probe that is also given `y_risk`; the lift, the warnings and the build-up
    share read the plain probe. `arrays` holds the onset-level data the pooled
    bootstrap needs; `data_gate` pops it before anything is written.
    """
    val, test = d.val, d.test
    cols = {f"h{h}": h - 1 for h in HEADROOM_HORIZONS}
    plain = fit_forecast_probe(d.train)
    probe = plain(test)
    scores = {
        "probe": probe,
        "probe_risk_aware": fit_forecast_probe(d.train, risk_aware=True)(test),
        "stand_in": risk_score_forecast(window_logistic(d.train, test), horizon),
        "oracle": risk_score_forecast(test.y_risk.astype(float), horizon),
    }
    pr_auc = {
        name: {
            key: _masked_ap(test.y_forecast[:, c], score[:, c], test.forecast_mask)
            for key, c in cols.items()
        }
        for name, score in scores.items()
    }

    last_build_up, shape_test = _last_row_build_up(df, test)
    _, shape_val = _last_row_build_up(df, val)
    in_build_up = test.forecast_mask & (last_build_up != 0)
    ppv = float(test.y_forecast[in_build_up, 9].mean()) if in_build_up.any() else None

    def threshold(val_score: np.ndarray) -> float:
        return threshold_at_fpr(
            val.y_forecast[val.forecast_mask, 9], val_score[val.forecast_mask]
        )

    def warned(score: np.ndarray, val_score: np.ndarray):
        return warned_onsets(
            score,
            threshold(val_score),
            test.y_forecast,
            test.forecast_mask,
            test.block_index,
            test.end_row,
        )

    rng = np.random.default_rng(seed)
    random_val = rng.random(len(val.y_risk))
    random_test = rng.random(len(test.y_risk))
    probe_val = plain(val)[:, 9]
    onset_block, warned_probe = warned(probe[:, 9], probe_val)
    _, warned_random = warned(random_test, random_val)
    _, warned_shape = warned(shape_test, shape_val)
    onsets = test.y_forecast[:, 0] == 1
    return {
        "lift_h10": _stratified_lift(d, 9),
        "pr_auc": pr_auc,
        "ppv_build_up": ppv,
        "n_onsets": int(len(onset_block)),
        "n_ramped": int((last_build_up[onsets] == 1).sum()),
        "n_shape_warned": int(warned_shape.sum()),
        "arrays": {
            "onset_block": onset_block,
            "warned_probe": warned_probe,
            "warned_random": warned_random,
            "block_index": test.block_index,
        },
    }


def _mean_or_none(values: list) -> float | None:
    return None if any(v is None for v in values) else float(np.mean(values))


def headroom_summary(per_seed: list[dict], seeds: Sequence[int]) -> dict:
    """Aggregate `forecast_headroom` results over seeds into the JSON block.

    `check` is the conjunction of the headroom conditions; any missing component
    (`None`) fails it. Onsets are pooled across seeds with blocks keyed by seed.
    """
    n = len(per_seed)
    lifts = [r["lift_h10"] for r in per_seed]
    lift_mean = _mean_or_none(lifts)
    lift_ok = (
        lift_mean is not None
        and HEADROOM_LIFT_RANGE[0] <= lift_mean <= HEADROOM_LIFT_RANGE[1]
        and min(lifts) >= HEADROOM_MIN_SEED_LIFT
    )

    aware: dict[str, list] = {}
    plain: dict[str, list] = {}
    rival: dict[str, list] = {}
    for key in (f"h{h}" for h in HEADROOM_HORIZONS):
        aware[key] = [r["pr_auc"]["probe_risk_aware"][key] for r in per_seed]
        plain[key] = [r["pr_auc"]["probe"][key] for r in per_seed]
        rival[key] = [
            None
            if r["pr_auc"]["stand_in"][key] is None
            or r["pr_auc"]["oracle"][key] is None
            else max(r["pr_auc"]["stand_in"][key], r["pr_auc"]["oracle"][key])
            for r in per_seed
        ]
    ratio: dict[str, float | None] = {}
    ratio_plain: dict[str, float | None] = {}
    for key in ("h5", "h10"):
        top, bar = _mean_or_none(aware[key]), _mean_or_none(rival[key])
        ratio[key] = None if top is None or bar is None else top / bar
        low = _mean_or_none(plain[key])
        ratio_plain[key] = None if low is None or bar is None else low / bar
    ratio_ok = all(v is not None and v >= HEADROOM_PR_AUC_RATIO for v in ratio.values())

    needed = math.ceil(HEADROOM_MIN_WIN_SHARE * n)
    wins = {
        key: sum(
            p is not None and r is not None and p > r
            for p, r in zip(aware[key], rival[key], strict=True)
        )
        for key in aware
    }
    wins_ok = all(count >= needed for count in wins.values())

    arrays = [r["arrays"] for r in per_seed]
    keyed = [seed * 1_000_000 for seed in seeds]
    gain = detection_gain(
        [np.concatenate([a["warned_probe"] for a in arrays])],
        np.concatenate([a["warned_random"] for a in arrays]),
        np.concatenate(
            [a["onset_block"] + k for a, k in zip(arrays, keyed, strict=True)]
        ),
        np.concatenate(
            [a["block_index"] + k for a, k in zip(arrays, keyed, strict=True)]
        ),
        n_boot=GATE_N_BOOT,
        seed=0,
    )
    gain_ok = gain["gain"]["ci_low"] is not None and gain["gain"]["ci_low"] > 0

    ppv_mean = _mean_or_none([r["ppv_build_up"] for r in per_seed])
    ppv_ok = (
        ppv_mean is not None
        and HEADROOM_PPV_RANGE[0] <= ppv_mean <= HEADROOM_PPV_RANGE[1]
    )

    onsets = sum(r["n_onsets"] for r in per_seed)
    return {
        "per_seed": [{k: v for k, v in r.items() if k != "arrays"} for r in per_seed],
        "stratified_lift_h10_mean": lift_mean,
        "pr_auc_ratio": ratio,
        "pr_auc_ratio_plain": ratio_plain,
        "wins": wins,
        "detection_gain": gain,
        "ppv_build_up_mean": ppv_mean,
        "ramped_onset_share": sum(r["n_ramped"] for r in per_seed) / onsets
        if onsets
        else None,
        "shape_only_detection_rate": sum(r["n_shape_warned"] for r in per_seed) / onsets
        if onsets
        else None,
        "check": bool(lift_ok and ratio_ok and wins_ok and gain_ok and ppv_ok),
    }


def _feature_split(df: pd.DataFrame, cfg: DignityConfig, seed: int) -> SplitData:
    """The frame's windows over the preset's own `data.features`, as the model sees.

    Same blocks per split as the `RAW_COLUMNS` windows of the band baselines.
    """
    return TransactionPipeline(
        seq_len=cfg.data.seq_len, features=list(cfg.data.features)
    ).process_blocks(
        df,
        cfg.data.split,
        cfg.model.horizon,
        cfg.data.label_overlap,
        np.random.default_rng(seed),
    )


def _score_preset(
    path: str | Path,
    seed: int,
    n_blocks: int | None = None,
    strength: float | None = None,
) -> dict:
    """One preset on one seed: val and test AUROCs of both baselines.

    The hard preset also returns the onset recalls per split and the number of
    scorable test onsets and the forecast headroom (`forecast_headroom`, on windows
    of the preset's own features; `None` for the other presets). `n_blocks` and
    `strength` replace the preset's values in memory.
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
        "headroom": forecast_headroom(
            df, _feature_split(df, cfg, seed), cfg.model.horizon, seed
        )
        if is_hard
        else None,
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
    headroom = headroom_summary([r["headroom"] for r in hard], seeds)

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
        "forecast_headroom": headroom.pop("check"),
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
        "headroom": headroom,
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

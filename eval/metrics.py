"""Risk metrics with block-bootstrap confidence intervals (spec §5.2-§5.3)."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FPR_BUDGET = 0.01


def bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    block_index: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """95% percentile interval of ``stat_fn`` over whole-block resamples.

    ``stat_fn`` receives the window indices of one resample. Windows of one block
    overlap, so they are never drawn singly (DGNTY-21). ``nan`` statistics (an
    undefined resample) are ignored; all-``nan`` gives ``(nan, nan)``.
    """
    blocks = np.unique(block_index)
    members = [np.flatnonzero(block_index == b) for b in blocks]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        draw = rng.integers(0, len(blocks), size=len(blocks))
        stats[i] = stat_fn(np.concatenate([members[j] for j in draw]))
    if np.isnan(stats).all():
        return float("nan"), float("nan")
    return (
        float(np.nanpercentile(stats, 2.5)),
        float(np.nanpercentile(stats, 97.5)),
    )


def threshold_at_fpr(
    y: np.ndarray, score: np.ndarray, max_fpr: float = FPR_BUDGET
) -> float:
    """Lowest score whose false-positive rate is <= ``max_fpr``.

    An alarm is ``score >= threshold``. The result is ``+inf`` when no finite
    threshold stays inside the budget.
    """
    fpr, _, thr = roc_curve(y, score, drop_intermediate=False)
    return float(thr[np.flatnonzero(fpr <= max_fpr)[-1]])


def _recall_at_fpr(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    return float(tpr[fpr <= FPR_BUDGET].max())


def _guard_one_class(fn: Callable[[np.ndarray, np.ndarray], float]):
    def wrapped(y: np.ndarray, score: np.ndarray) -> float:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(fn(y, score))

    return wrapped


def _interval(
    stat_fn: Callable[[np.ndarray], float],
    block_index: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """CI as JSON-safe values: ``None`` when not requested or undefined."""
    if n_boot == 0:
        return None, None
    low, high = bootstrap_ci(stat_fn, block_index, n_boot, seed)
    if np.isnan(low) or np.isnan(high):
        return None, None
    return float(low), float(high)


def _entry(value, stat_fn, block_index, n_boot, seed) -> dict:
    low, high = _interval(stat_fn, block_index, n_boot, seed)
    return {"value": float(value), "ci_low": low, "ci_high": high}


def risk_metrics(
    y: np.ndarray,
    score: np.ndarray,
    block_index: np.ndarray,
    n_boot: int,
    seed: int,
    anomaly_type: np.ndarray | None = None,
) -> dict:
    """Point estimates and block-bootstrap CIs for a risk score.

    Every scalar is a Python float/int so the result passes
    ``json.dumps(..., allow_nan=False)``; undefined CIs are ``None``.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    block_index = np.asarray(block_index)
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"risk_metrics needs both classes; got prevalence {float(np.mean(y))}"
        )

    def over(fn):
        return lambda idx: fn(y[idx], score[idx])

    auroc = _guard_one_class(roc_auc_score)
    pr_auc = _guard_one_class(average_precision_score)

    def brier(yy, ss):
        return np.mean((ss - yy) ** 2)

    def prevalence(yy, ss):
        return np.mean(yy)

    out = {}
    for name, fn in [
        ("auroc", auroc),
        ("pr_auc", pr_auc),
        ("brier", brier),
        ("prevalence", prevalence),
        ("recall_at_1pct_fpr", _recall_at_fpr),
    ]:
        out[name] = _entry(fn(y, score), over(fn), block_index, n_boot, seed)

    out["per_type"] = {}
    if anomaly_type is not None:
        anomaly_type = np.asarray(anomaly_type)
        threshold = threshold_at_fpr(y, score)
        for t in sorted(set(anomaly_type[y == 1]) - {"none"}):

            def type_recall(idx, t=t):
                yy, ss = y[idx], score[idx]
                hit = (yy == 1) & (anomaly_type[idx] == t)
                if len(np.unique(yy)) < 2 or not hit.any():
                    return float("nan")
                return float(np.mean(ss[hit] >= threshold_at_fpr(yy, ss)))

            mask = (y == 1) & (anomaly_type == t)
            entry = _entry(
                np.mean(score[mask] >= threshold),
                type_recall,
                block_index,
                n_boot,
                seed,
            )
            entry["n_windows"] = int(mask.sum())
            out["per_type"][str(t)] = entry
    return out


def forecast_metrics(
    y: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    block_index: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Per-horizon PR-AUC with block-bootstrap CIs on ``mask`` rows only.

    ``y`` and ``score`` are ``[N, H]``; ``mask`` drops windows whose last row is
    inside an active anomaly (their score reacts to it, it is not a warning).
    A horizon whose masked column lacks a positive or a negative (or has no rows)
    reports ``None`` rather than a fake score; its prevalence is ``None`` when
    the mask selects no row.
    """
    y = np.asarray(y)
    score = np.asarray(score)
    mask = np.asarray(mask, dtype=bool)
    blocks = np.asarray(block_index)[mask]
    pr_auc = _guard_one_class(average_precision_score)
    by_horizon = []
    for h in range(y.shape[1]):
        yh, sh = y[mask, h], score[mask, h]
        if len(np.unique(yh)) < 2:
            by_horizon.append({"value": None, "ci_low": None, "ci_high": None})
            continue

        def stat(idx, yh=yh, sh=sh):
            return pr_auc(yh[idx], sh[idx])

        by_horizon.append(_entry(pr_auc(yh, sh), stat, blocks, n_boot, seed))
    return {
        "pr_auc_by_horizon": by_horizon,
        "prevalence_by_horizon": [
            float(y[mask, h].mean()) if mask.any() else None for h in range(y.shape[1])
        ],
    }


def lead_times(
    alarm: np.ndarray,
    threshold: float,
    y: np.ndarray,
    mask: np.ndarray,
    block_index: np.ndarray,
    end_row: np.ndarray,
) -> dict:
    """Rows between the first alarm and each anomaly onset (an alarm is ``>=``).

    An onset sits at ``o = end_row + 1`` of every window with ``y[:, 0] == 1``;
    it is scored only when a window ends at ``o - 1``. Only ``mask`` windows
    ending in ``[o - H, o - 1]`` of the same block can warn. Values are plain
    Python numbers so the result passes ``json.dumps(..., allow_nan=False)``.
    """
    horizon = y.shape[1]
    values: list[int] = []
    n_onsets = 0
    n_missed = 0
    for i in np.flatnonzero(y[:, 0] == 1):
        b, onset = block_index[i], int(end_row[i]) + 1
        same = block_index == b
        if not (same & (end_row == onset - 1)).any():
            continue
        n_onsets += 1
        eligible = same & mask & (end_row >= onset - horizon) & (end_row <= onset - 1)
        fired = eligible & (alarm >= threshold)
        if fired.any():
            values.append(onset - int(end_row[fired].min()))
        else:
            n_missed += 1
    median = float(np.median(values)) if values else None
    return {
        "median": median,
        "values": values,
        "n_onsets": n_onsets,
        "n_missed": n_missed,
    }


def model_gate(model_pr_auc: dict, baseline_pr_aucs: dict[str, dict]) -> dict:
    """Spec §5.2: the model must beat the best baseline with its CI lower bound."""
    if model_pr_auc.get("ci_low") is None:
        raise ValueError("model_gate needs a bootstrap CI (n_boot >= 1)")
    best = max(baseline_pr_aucs, key=lambda k: baseline_pr_aucs[k]["value"])
    best_value = float(baseline_pr_aucs[best]["value"])
    value = float(model_pr_auc["value"])
    ci_low = float(model_pr_auc["ci_low"])
    return {
        "best_baseline": best,
        "best_baseline_pr_auc": best_value,
        "model_pr_auc": value,
        "model_ci_low": ci_low,
        "passed": bool(value > best_value and ci_low > best_value),
    }

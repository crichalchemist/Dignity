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

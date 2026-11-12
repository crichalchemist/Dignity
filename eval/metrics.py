"""Risk metrics with block-bootstrap confidence intervals (spec §5.2-§5.3)."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from models.head.policy import ACTIONS

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


def warned_onsets(
    alarm: np.ndarray,
    threshold: float,
    y: np.ndarray,
    mask: np.ndarray,
    block_index: np.ndarray,
    end_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per scored onset, its block and whether any eligible window alarmed.

    The same onsets, eligibility and ``>=`` rule as ``lead_times``, in the order
    it visits them, so ``(~warned).sum()`` is its ``n_missed``. The per-onset
    form lets a bootstrap redraw blocks and recount the warned share.
    """
    horizon = y.shape[1]
    blocks: list[int] = []
    warned: list[bool] = []
    for i in np.flatnonzero(y[:, 0] == 1):
        b, onset = block_index[i], int(end_row[i]) + 1
        same = block_index == b
        if not (same & (end_row == onset - 1)).any():
            continue
        eligible = same & mask & (end_row >= onset - horizon) & (end_row <= onset - 1)
        blocks.append(int(block_index[i]))
        warned.append(bool((eligible & (alarm >= threshold)).any()))
    return np.array(blocks, dtype=np.int64), np.array(warned, dtype=bool)


_NO_GAIN = {"value": None, "ci_low": None, "ci_high": None}


def _block_counts(block_index: np.ndarray, n_boot: int, seed: int):
    """Per replicate, how many times each unique block is drawn (bootstrap_ci's draws)."""
    n = len(np.unique(block_index))
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, size=n), minlength=n)


def _gain_entry(value: float, stats: list[float]) -> dict:
    """``{"value", "ci_low", "ci_high"}`` with ``None`` bounds when undefined."""
    arr = np.array(stats, dtype=np.float64)
    if len(arr) == 0 or np.isnan(arr).all():
        return {"value": value, "ci_low": None, "ci_high": None}
    return {
        "value": value,
        "ci_low": float(np.nanpercentile(arr, 2.5)),
        "ci_high": float(np.nanpercentile(arr, 97.5)),
    }


def paired_forecast_gain(
    y: np.ndarray,
    mask: np.ndarray,
    block_index: np.ndarray,
    model_scores: list[np.ndarray],
    base_scores: list[np.ndarray],
    n_boot: int,
    seed: int,
) -> list[dict]:
    """Per horizon, the seed-matched average-precision gain over the baseline.

    ``model_scores[k]`` and ``base_scores[k]`` are ``[N, H]`` scores of training
    seed ``k``. The value is ``mean_k(AP(model_k) - AP(base_k))`` on ``mask`` rows.
    Model and baseline share every block draw, so the interval is a paired one;
    blocks are drawn over all rows, before the mask (``_block_counts``), exactly
    as ``bootstrap_ci`` would. A horizon with one class among masked rows reports
    ``None``; the bounds are ``None`` when ``n_boot == 0`` or undefined.
    """
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    block_index = np.asarray(block_index)
    ub = np.unique(block_index)
    pos = np.searchsorted(ub, block_index[mask])
    ym = y[mask]
    model = [np.asarray(s)[mask] for s in model_scores]
    base = [np.asarray(s)[mask] for s in base_scores]

    def gain(yh, ms, bs, w=None):
        if len(np.unique(yh)) < 2:
            return float("nan")
        return float(
            np.mean(
                [
                    average_precision_score(yh, m, sample_weight=w)
                    - average_precision_score(yh, b, sample_weight=w)
                    for m, b in zip(ms, bs, strict=True)
                ]
            )
        )

    out = []
    valid = [len(np.unique(ym[:, h])) == 2 for h in range(y.shape[1])]
    values = [
        gain(ym[:, h], [m[:, h] for m in model], [b[:, h] for b in base])
        if valid[h]
        else None
        for h in range(y.shape[1])
    ]
    replicates: list[list[float]] = [[] for _ in values]
    if n_boot > 0:
        for counts in _block_counts(block_index, n_boot, seed):
            w = counts[pos]
            keep = w > 0
            for h, ok in enumerate(valid):
                if ok:
                    replicates[h].append(
                        gain(
                            ym[keep, h],
                            [m[keep, h] for m in model],
                            [b[keep, h] for b in base],
                            w[keep],
                        )
                    )
    for h, value in enumerate(values):
        out.append(
            dict(_NO_GAIN) if value is None else _gain_entry(value, replicates[h])
        )
    return out


def detection_gain(
    warned: list[np.ndarray],
    random_warned: np.ndarray,
    onset_block: np.ndarray,
    block_index: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Share of onsets the model warns about, against a random score's share.

    ``warned[k]`` is ``warned_onsets``' flag per onset for training seed ``k``;
    ``random_warned`` is the same for a random score at the same false-alarm
    budget. The gain is the mean over seeds of the model's warned share minus the
    random share, with a block bootstrap that weights each onset by how often its
    block is drawn (the draws are the same as ``paired_forecast_gain``'s).
    """
    onset_block = np.asarray(onset_block)
    n_onsets = len(onset_block)
    if n_onsets == 0:
        return {
            "n_onsets": 0,
            "detection_rate": None,
            "detection_rate_by_seed": [None] * len(warned),
            "random_rate": None,
            "gain": dict(_NO_GAIN),
        }
    block_index = np.asarray(block_index)
    ub = np.unique(block_index)
    pos = np.searchsorted(ub, onset_block)
    flags = [np.asarray(w, dtype=float) for w in warned]
    rand = np.asarray(random_warned, dtype=float)
    by_seed = [float(f.mean()) for f in flags]
    rate, random_rate = float(np.mean(by_seed)), float(rand.mean())
    stats: list[float] = []
    if n_boot > 0:
        for counts in _block_counts(block_index, n_boot, seed):
            w = counts[pos]
            total = w.sum()
            if total == 0:
                continue
            stats.append(
                float(np.mean([(w * f).sum() / total for f in flags]))
                - float((w * rand).sum() / total)
            )
    return {
        "n_onsets": n_onsets,
        "detection_rate": rate,
        "detection_rate_by_seed": by_seed,
        "random_rate": random_rate,
        "gain": _gain_entry(rate - random_rate, stats),
    }


def policy_metrics(
    actions: np.ndarray,
    y: np.ndarray,
    cost_matrix: np.ndarray,
    block_index: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    """Triage scored on fixed actions, with block-bootstrap intervals.

    ``expected_cost`` is the mean of ``cost_matrix[action, y]`` per window;
    ``review_rate`` and ``block_rate`` are action shares; ``anomaly_recall`` is
    P(review or block | anomaly) and ``false_block_rate`` is P(block | normal).
    Resamples redraw blocks only; the decision rule is never re-run.
    """
    review, block = ACTIONS.index("review"), ACTIONS.index("block")
    actions = np.asarray(actions)
    y = np.asarray(y).astype(int)
    cost = np.asarray(cost_matrix, np.float64)[actions, y]
    is_review, is_block = actions == review, actions == block

    def expected_cost(idx):
        return float(cost[idx].mean())

    def review_rate(idx):
        return float(is_review[idx].mean())

    def block_rate(idx):
        return float(is_block[idx].mean())

    def anomaly_recall(idx):
        hit = (is_review | is_block)[idx][y[idx] == 1]
        return float(hit.mean()) if len(hit) else float("nan")

    def false_block_rate(idx):
        hit = is_block[idx][y[idx] == 0]
        return float(hit.mean()) if len(hit) else float("nan")

    stat_fns = {
        "expected_cost": expected_cost,
        "review_rate": review_rate,
        "block_rate": block_rate,
        "anomaly_recall": anomaly_recall,
        "false_block_rate": false_block_rate,
    }
    everything = np.arange(len(actions))
    out = {}
    for name, fn in stat_fns.items():
        value = fn(everything)
        low, high = _interval(fn, block_index, n_boot, seed)
        out[name] = {
            "value": None if np.isnan(value) else value,
            "ci_low": low,
            "ci_high": high,
        }
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

"""Risk charts for eval.run. matplotlib is imported lazily (it is an extra)."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str) -> None:
    """One precision-recall line per name; each entry is (y, score)."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (y, score) in curves.items():
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        ax.plot(recall, precision, label=f"{name} (AP {ap:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_type_recall(per_type: dict, path: str) -> None:
    """One bar per anomaly type; error bars from ci_low/ci_high when present."""
    plt = _pyplot()
    names = list(per_type)
    values = [float(per_type[n]["value"]) for n in names]
    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(names) + 2), 4.5))
    ax.bar(range(len(names)), values)
    for i, n in enumerate(names):
        low, high = per_type[n].get("ci_low"), per_type[n].get("ci_high")
        if low is None or high is None:
            continue
        err = [[max(values[i] - low, 0.0)], [max(high - values[i], 0.0)]]
        ax.errorbar([i], [values[i]], yerr=err, color="black", capsize=4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_pr_auc(result: dict, path: str) -> None:
    """PR-AUC per horizon for the model and the risk-score baseline, with CIs.

    ``result`` is a forecast.json payload; the dashed line is the prevalence.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sides = {
        "model": result["metrics"],
        "risk_score_forecast": result["baselines"]["risk_score_forecast"],
    }
    for name, side in sides.items():
        entries = side["pr_auc_by_horizon"]
        known = [h for h, e in enumerate(entries) if e["value"] is not None]
        values = [float(entries[h]["value"]) for h in known]
        line = ax.plot([h + 1 for h in known], values, marker="o", label=name)[0]
        for h in known:
            low, high = entries[h]["ci_low"], entries[h]["ci_high"]
            if low is None or high is None:
                continue
            v = float(entries[h]["value"])
            err = [[max(v - low, 0.0)], [max(high - v, 0.0)]]
            ax.errorbar([h + 1], [v], yerr=err, color=line.get_color(), capsize=3)
    prevalence = result["metrics"]["prevalence_by_horizon"]
    known = [h for h, p in enumerate(prevalence) if p is not None]
    ax.plot(
        [h + 1 for h in known],
        [prevalence[h] for h in known],
        linestyle="--",
        color="gray",
        label="prevalence",
    )
    ax.set_xlabel("horizon (rows ahead)")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lead_times(result: dict, path: str) -> None:
    """Histogram of rows of warning before each caught onset; misses in the title."""
    plt = _pyplot()
    lead = result["metrics"]["lead_time"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    values = lead["values"]
    if values:
        ax.hist(values, bins=range(0, int(max(values)) + 2), align="left")
    ax.set_xlabel("lead time (rows before onset)")
    ax.set_ylabel("onsets")
    ax.set_title(f"{lead['n_missed']} of {lead['n_onsets']} onsets missed")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_policy_cost(result: dict, path: str) -> None:
    """Expected cost per window for the model and the threshold baseline.

    ``result`` is a policy.json payload; whiskers are the block-bootstrap CIs and
    the subtitle gives each side's review and block rates.
    """
    plt = _pyplot()
    sides = {
        "model": result["metrics"],
        "threshold_policy": result["baselines"]["threshold_policy"],
    }
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for i, side in enumerate(sides.values()):
        entry = side["expected_cost"]
        v = float(entry["value"])
        ax.bar([i], [v], color=f"C{i}")
        low, high = entry.get("ci_low"), entry.get("ci_high")
        if low is not None and high is not None:
            err = [[max(v - low, 0.0)], [max(high - v, 0.0)]]
            ax.errorbar([i], [v], yerr=err, color="black", capsize=4)
    ax.set_xticks(range(len(sides)))
    ax.set_xticklabels(list(sides))
    ax.set_ylabel("expected cost per window")
    ax.set_title(
        "  |  ".join(
            f"{name}: review {side['review_rate']['value']:.1%}, "
            f"block {side['block_rate']['value']:.1%}"
            for name, side in sides.items()
        ),
        fontsize=8,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

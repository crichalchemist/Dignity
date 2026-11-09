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

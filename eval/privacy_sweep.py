"""Privacy sweep helpers: one allocation template, rescaled to any epsilon."""

from dataclasses import replace
from pathlib import Path

from core.config import DignityConfig, PrivacyConfig
from data.pipeline import describe_privacy
from data.source.synthetic import public_log1p_volume_bounds
from eval.baselines import window_logistic
from eval.metrics import risk_metrics
from eval.plots import _pyplot
from eval.run import _predict, _spread, _task_config
from train.loop import fit, prepare_data


def privacy_at(
    template: PrivacyConfig,
    epsilon: float,
    bounds: dict[str, tuple[float, float]],
) -> PrivacyConfig:
    """Rescale the template's Laplace epsilons so they sum to `epsilon`.

    The allocation (each feature's share) is kept. Laplace bounds come from the
    public `bounds` argument, never from data; they are in the feature's transform
    units (log1p units for ``transform: log1p``). Seeded mode is switched on.
    Generalize features, k, key_env and transforms are left unchanged.
    """
    laplace = {n: f for n, f in template.features.items() if f.mechanism == "laplace"}
    total = sum(f.epsilon for f in laplace.values())
    if not laplace or total <= 0:
        raise ValueError("template has no laplace feature to rescale")
    features = {}
    for name, feature in template.features.items():
        if name in laplace:
            if name not in bounds:
                raise ValueError(f"no public bounds for laplace feature '{name}'")
            feature = replace(
                feature,
                epsilon=feature.epsilon * epsilon / total,
                bounds=tuple(bounds[name]),
            )
        features[name] = feature
    return PrivacyConfig(
        epsilon_total=epsilon,
        k=template.k,
        key_env=template.key_env,
        deterministic_rng=True,
        features=features,
    )


def _scores(config: DignityConfig, data, seed: int, run_dir: Path) -> dict:
    """Train one risk model and score the test split (reporting only)."""
    result = fit(config, data, seed, str(run_dir))
    return _predict(config, result, data.test.X)["score"]


def _metrics(test, score, n_boot: int, seed: int) -> dict:
    return risk_metrics(test.y_risk, score, test.block_index, n_boot, seed)


def _spreads(metrics: list[dict]) -> dict:
    return {
        "pr_auc": _spread([float(m["pr_auc"]["value"]) for m in metrics]),
        "auroc": _spread([float(m["auroc"]["value"]) for m in metrics]),
    }


def sweep(config: DignityConfig, epsilons: list[float], seed: int) -> dict:
    """Train the risk model at every epsilon and without privacy; return privacy.json.

    ``seed`` is the generator seed: the data and the block split are identical for
    every replicate and every epsilon. Replicate r trains with ``config.seed + r``
    and noises with privacy seed ``seed + r``; within a replicate every epsilon
    shares that privacy seed. The reference (``privacy=None``, same features)
    trains the same seeds. Replicate 0 carries the bootstrap CI. Test-split scores
    are for reporting; nothing is selected on them.

    A replicate's checkpoint meta records ``synthetic.seed`` and
    ``deterministic_rng`` but not the privacy seed; the noise of replicate r is
    reproducible from privacy.json's ``privacy_seeds[r]``.
    """
    if config.privacy is None:
        raise ValueError("sweep needs config.privacy as the allocation template")
    if not epsilons:
        raise ValueError("sweep needs at least one epsilon")
    bounds = {"volume": public_log1p_volume_bounds(config.data.synthetic)}
    synthetic = replace(config.data.synthetic, seed=seed)
    base = replace(config, data=replace(config.data, synthetic=synthetic))
    risk_config = _task_config(base, "risk")
    n_boot = config.eval.n_bootstrap
    replicates = range(config.eval.n_train_seeds)
    root = Path(config.train.checkpoint_dir) / "privacy_sweep"

    points, logistic_points = [], []
    features = None
    for epsilon in epsilons:
        privacy = privacy_at(config.privacy, epsilon, bounds)
        risk, logistic = [], []
        for r in replicates:
            _, data = prepare_data(
                replace(risk_config, privacy=privacy), privacy_seed=seed + r
            )
            print(f"eps={epsilon} replicate={r} {describe_privacy(data, privacy)}")
            features = data.feature_names
            run_dir = root / f"eps_{epsilon}" / f"train_seed_{config.seed + r}"
            score = _scores(
                replace(risk_config, privacy=privacy), data, config.seed + r, run_dir
            )
            boot = n_boot if r == 0 else 0
            risk.append(_metrics(data.test, score, boot, seed))
            logistic.append(
                _metrics(data.test, window_logistic(data.train, data.test), boot, seed)
            )
            if r == 0:
                spent, mode = dict(data.epsilon_spent), data.privacy_mode
        points.append(
            {
                "epsilon": epsilon,
                "risk": risk[0],
                "epsilon_spent": spent,
                "privacy_mode": mode,
                "replicate_spread": _spreads(risk),
            }
        )
        logistic_points.append(
            {
                "epsilon": epsilon,
                "auroc": logistic[0]["auroc"],
                "pr_auc": logistic[0]["pr_auc"],
                "replicate_spread": _spreads(logistic),
            }
        )

    _, plain = prepare_data(replace(risk_config, privacy=None))
    print(f"reference {describe_privacy(plain, None)}")
    reference = []
    for r in replicates:
        run_dir = root / "reference" / f"train_seed_{config.seed + r}"
        score = _scores(
            replace(risk_config, privacy=None), plain, config.seed + r, run_dir
        )
        reference.append(_metrics(plain.test, score, n_boot if r == 0 else 0, seed))
    logistic_ref = _metrics(
        plain.test, window_logistic(plain.train, plain.test), n_boot, seed
    )

    return {
        "metrics": {
            "epsilons": list(epsilons),
            "features": features,
            "template": config.privacy.to_dict(),
            "train_seeds": [config.seed + r for r in replicates],
            "privacy_seeds": [seed + r for r in replicates],
            "points": points,
            "reference": {
                "epsilon": None,
                "risk": reference[0],
                "privacy_mode": plain.privacy_mode,
                "seed_spread": _spreads(reference),
            },
        },
        "baselines": {
            "window_logistic": {
                "points": logistic_points,
                "reference": {
                    "auroc": logistic_ref["auroc"],
                    "pr_auc": logistic_ref["pr_auc"],
                },
            }
        },
    }


def _bar(ax, x: float, entry: dict, **style) -> None:
    """Draw a point's CI bar; no bar when either bound is None."""
    low, high = entry.get("ci_low"), entry.get("ci_high")
    if low is None or high is None:
        return
    v = float(entry["value"])
    err = [[max(v - low, 0.0)], [max(high - v, 0.0)]]
    ax.errorbar([x], [v], yerr=err, capsize=3, **style)


def plot_curve(result: dict, path: str) -> None:
    """Risk PR-AUC against epsilon, with the non-private reference.

    ``result`` is a privacy.json payload. Bars are replicate 0's block-bootstrap
    CI; small unfilled markers are every replicate's PR-AUC (training seed and
    noise draw). The dashed line and band are the reference, the dotted line is
    the test prevalence (chance).
    """
    plt = _pyplot()
    metrics = result["metrics"]
    points = metrics["points"]
    reference = metrics["reference"]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    def draw(label: str, epsilons, entries, spreads, color: str, marker: str):
        known = [
            (e, p)
            for e, p in zip(epsilons, entries, strict=True)
            if p["value"] is not None
        ]
        ax.plot(
            [e for e, _ in known],
            [float(p["value"]) for _, p in known],
            marker=marker,
            color=color,
            label=label,
        )
        for e, p in known:
            _bar(ax, e, p, color=color)
        for e, spread in zip(epsilons, spreads, strict=True):
            values = [v for v in (spread or {}).get("values", []) if v is not None]
            ax.scatter(
                [e] * len(values),
                values,
                s=14,
                facecolors="none",
                edgecolors=color,
                alpha=0.6,
            )

    epsilons = [p["epsilon"] for p in points]
    draw(
        "model",
        epsilons,
        [p["risk"]["pr_auc"] for p in points],
        [p.get("replicate_spread", {}).get("pr_auc") for p in points],
        "C0",
        "o",
    )
    logistic = result["baselines"]["window_logistic"]["points"]
    draw(
        "window_logistic",
        [p["epsilon"] for p in logistic],
        [p["pr_auc"] for p in logistic],
        [p.get("replicate_spread", {}).get("pr_auc") for p in logistic],
        "C1",
        "s",
    )

    ref = reference["risk"]["pr_auc"]
    if ref["value"] is not None:
        ax.axhline(float(ref["value"]), linestyle="--", color="C2", label="non-private")
        low, high = ref.get("ci_low"), ref.get("ci_high")
        if low is not None and high is not None:
            ax.axhspan(low, high, color="C2", alpha=0.15)
    spread = reference.get("seed_spread", {}).get("pr_auc") or {}
    ref_values = [v for v in spread.get("values", []) if v is not None]
    if ref_values and epsilons:
        ax.scatter(
            [max(epsilons)] * len(ref_values),
            ref_values,
            s=14,
            marker="_",
            color="C2",
            alpha=0.8,
        )
    prevalence = reference["risk"]["prevalence"]["value"]
    if prevalence is not None:
        ax.axhline(
            float(prevalence), linestyle=":", color="gray", label="chance (prevalence)"
        )

    ax.plot([], [], "o", mfc="none", color="gray", label="replicates")
    if epsilons:
        ax.set_xscale("log")
    ax.set_xlabel("epsilon (log scale)")
    ax.set_ylabel("risk PR-AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Risk PR-AUC vs \u03b5 (test split, 95% block-bootstrap CI)")
    # outside the axes: the curve can sit anywhere between chance and the reference
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.text(
        0.5,
        -0.02,
        "markers: replicates (training seed and noise draw)",
        ha="center",
        fontsize=8,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

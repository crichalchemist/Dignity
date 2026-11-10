"""Privacy sweep helpers: one allocation template, rescaled to any epsilon."""

from dataclasses import replace
from pathlib import Path

from core.config import DignityConfig, PrivacyConfig
from data.pipeline import describe_privacy
from data.source.synthetic import public_log1p_volume_bounds
from eval.baselines import window_logistic
from eval.metrics import risk_metrics
from eval.run import _predict, _spread, _task_config
from train.loop import fit, prepare_data


def privacy_at(
    template: PrivacyConfig,
    epsilon: float,
    bounds: dict[str, tuple[float, float]],
) -> PrivacyConfig:
    """Rescale the template's Laplace epsilons so they sum to `epsilon`.

    The allocation (each feature's share) is kept. Laplace bounds come from the
    public `bounds` argument, never from data. Seeded mode is switched on.
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
    """
    if config.privacy is None:
        raise ValueError("sweep needs config.privacy as the allocation template")
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
    print(describe_privacy(plain, None))
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
                "privacy_mode": "off",
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

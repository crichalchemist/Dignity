"""Privacy sweep helpers: one allocation template, rescaled to any epsilon."""

from dataclasses import replace

from core.config import PrivacyConfig


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

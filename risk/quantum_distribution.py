"""Risk utilities driven by quantum-inspired distribution models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.quantum_emulation.walks.adg import SplitStepQuantumWalkADG, QWADGConfig


@dataclass
class DistributionRiskConfig:
    alpha: float = 0.05
    n_paths: int = 5000
    horizon: int = 1


def compute_var_cvar(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    flattened = samples.reshape(-1)
    var = np.quantile(flattened, alpha)
    cvar = flattened[flattened <= var].mean() if np.any(flattened <= var) else var
    return float(var), float(cvar)


def estimate_adg_risk(
    returns: np.ndarray,
    cfg: DistributionRiskConfig | None = None,
    adg_cfg: QWADGConfig | None = None,
) -> dict[str, float]:
    """Fit ADG and compute VaR/CVaR from generated scenarios."""
    cfg = cfg or DistributionRiskConfig()
    model = SplitStepQuantumWalkADG(adg_cfg)
    model.fit(returns)
    samples = model.sample(context=None, n_paths=cfg.n_paths, horizon=cfg.horizon)
    var, cvar = compute_var_cvar(samples, alpha=cfg.alpha)
    return {"var": var, "cvar": cvar}


__all__ = ["DistributionRiskConfig", "compute_var_cvar", "estimate_adg_risk"]

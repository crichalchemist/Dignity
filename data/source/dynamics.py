"""Normal-process dynamics for the synthetic generator.

Every function takes an explicit ``np.random.Generator`` and never touches the
global numpy RNG. Each one takes its driver series (or length) explicitly.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalProcessParams:
    """Starting values for the normal-process dynamics.

    The calibration builder tunes these inside the bands set in M2.10.
    """

    garch_omega: float = 7e-7
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    t_dof: float = 5.0
    regime_switch_prob: float = 0.0  # the generator overrides this from config
    regime_vol: tuple[float, float, float] = (0.6, 1.0, 2.0)
    regime_log_volume: tuple[float, float, float] = (-0.3, 0.0, 0.4)
    ar1_phi: float = 0.6
    log_volume_innov_sd: float = 0.5
    log_volume_mean: float = 4.6
    block_level_sd: float = 0.5
    price_level_mean: float = 6.9
    price_level_sd: float = 1.0
    tx_base: float = 50.0
    tx_volume_elasticity: float = 0.8
    tx_dispersion: float = 20.0
    capacity_ratio: float = 1.1
    fee_base: float = 1e-3
    fee_congestion_power: float = 1.5
    fee_noise_sd: float = 0.3


DEFAULT_PARAMS = NormalProcessParams()


def garch_returns(
    n: int, params: NormalProcessParams, rng: np.random.Generator
) -> np.ndarray:
    """GARCH(1,1) returns with standardised Student-t innovations."""
    omega, alpha, beta = params.garch_omega, params.garch_alpha, params.garch_beta
    nu = params.t_dof
    z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)
    r = np.empty(n)
    sigma2 = omega / (1.0 - alpha - beta)
    for t in range(n):
        if t > 0:
            sigma2 = omega + alpha * r[t - 1] ** 2 + beta * sigma2
        r[t] = np.sqrt(sigma2) * z[t]
    return r


def ar1_log_volume(
    n: int, params: NormalProcessParams, rng: np.random.Generator
) -> np.ndarray:
    """AR(1) log-volume deviations with centred lognormal innovations."""
    s = params.log_volume_innov_sd
    eps = np.exp(rng.normal(0.0, s, size=n)) - np.exp(s * s / 2.0)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = params.ar1_phi * x[t - 1] + eps[t]
    return x


def regime_path(
    n: int, params: NormalProcessParams, rng: np.random.Generator
) -> np.ndarray:
    """Calm/normal/turbulent regime path, starting in state 1 (normal)."""
    move = rng.random(n) < params.regime_switch_prob
    pick = rng.integers(0, 2, size=n)
    path = np.empty(n, dtype=np.int8)
    state = 1
    for t in range(n):
        path[t] = state
        if move[t]:
            others = [s for s in (0, 1, 2) if s != state]
            state = others[pick[t]]
    return path


def tx_counts(
    volume: np.ndarray, params: NormalProcessParams, rng: np.random.Generator
) -> np.ndarray:
    """Overdispersed transaction counts whose mean scales with volume."""
    r = params.tx_dispersion
    lam = (
        params.tx_base
        * (volume / np.exp(params.log_volume_mean)) ** params.tx_volume_elasticity
    )
    return rng.negative_binomial(r, r / (r + lam)).astype(np.int64)


def congestion_fee(
    tx_count: np.ndarray, params: NormalProcessParams, rng: np.random.Generator
) -> np.ndarray:
    """Fee that rises with the backlog of transactions above capacity."""
    n = len(tx_count)
    cap = params.capacity_ratio * np.median(tx_count)
    backlog = np.empty(n)
    b = 0.0
    for t in range(n):
        b = max(0.0, b + tx_count[t] - cap)
        backlog[t] = b
    noise = rng.normal(0.0, params.fee_noise_sd, size=n)
    return (
        params.fee_base
        * (1.0 + backlog / cap) ** params.fee_congestion_power
        * np.exp(noise)
    )

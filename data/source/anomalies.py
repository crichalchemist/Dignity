"""Anomaly typologies and benign spikes injected into synthetic blocks.

Each injector takes ``(frame, start, duration, severity, rng)`` and returns a
modified copy that touches only ``volume, price, fee_rate, tx_count``. Price
effects are applied in log-return space and the price is rebuilt from the first
row, so returns outside the span are unchanged and later prices keep any level
shift. Label columns are the generator's job, not the injectors'.
"""

import math
from collections.abc import Callable

import numpy as np
import pandas as pd

ANOMALY_TYPES = (
    "wash_trading",
    "pump_and_dump",
    "fee_spike",
    "flash_crash",
    "layering",
    "fee_evasion",
)


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "volume": frame["volume"].to_numpy(dtype=float, copy=True),
        "price": frame["price"].to_numpy(dtype=float, copy=True),
        "fee_rate": frame["fee_rate"].to_numpy(dtype=float, copy=True),
        "tx_count": frame["tx_count"].to_numpy(dtype=float, copy=True),
    }


def _returns(a: dict[str, np.ndarray]) -> np.ndarray:
    return np.diff(np.log(a["price"]))


def _build(frame: pd.DataFrame, a: dict[str, np.ndarray], r: np.ndarray):
    """Rebuild price from row 0 and assign whole columns into a frame copy."""
    out = frame.copy()
    out["volume"] = a["volume"]
    out["price"] = a["price"][0] * np.exp(np.r_[0.0, np.cumsum(r)])
    out["fee_rate"] = a["fee_rate"]
    out["tx_count"] = np.round(a["tx_count"]).astype(np.int64)
    return out


def _add_returns(r: np.ndarray, start: int, delta: np.ndarray) -> None:
    """Add ``delta[i]`` to the return into span row ``start + i``."""
    idx = start - 1 + np.arange(len(delta))
    keep = idx >= 0
    r[idx[keep]] += delta[keep]


def _wash_trading(frame, start, duration, severity, rng):
    a = _arrays(frame)
    r = _returns(a)
    span = slice(start, start + duration)
    a["volume"][span] *= 1 + 4 * severity
    a["tx_count"][span] *= 1 + 3 * severity
    idx = start - 1 + np.arange(duration)
    idx = idx[idx >= 0]
    r[idx] *= 1 - severity
    return _build(frame, a, r)


def _pump_and_dump(frame, start, duration, severity, rng):
    a = _arrays(frame)
    r = _returns(a)
    d = duration
    k = math.ceil(0.7 * d)
    i = np.arange(d)
    delta = np.where(i < k, 0.15 * severity / k, -0.20 * severity / max(d - k, 1))
    _add_returns(r, start, delta)
    a["volume"][start : start + d] *= 1 + 3 * severity * np.minimum(i + 1, k) / k
    return _build(frame, a, r)


def _fee_spike(frame, start, duration, severity, rng):
    a = _arrays(frame)
    span = slice(start, start + duration)
    a["fee_rate"][span] *= 1 + 9 * severity
    a["tx_count"][span] *= 1 + severity
    return _build(frame, a, _returns(a))


def _flash_crash(frame, start, duration, severity, rng):
    a = _arrays(frame)
    r = _returns(a)
    d = duration
    delta = np.full(d, 0.07 * severity / max(d - 1, 1))
    delta[0] = -0.10 * severity
    _add_returns(r, start, delta)
    mult = np.full(d, 1 + 2 * severity)
    mult[0] = 1 + 5 * severity
    a["volume"][start : start + d] *= mult
    return _build(frame, a, r)


def _layering(frame, start, duration, severity, rng):
    a = _arrays(frame)
    median = np.median(a["volume"])
    rows = start + np.arange(0, duration, 4)
    a["volume"][rows] += 3 * severity * median
    return _build(frame, a, _returns(a))


def _fee_evasion(frame, start, duration, severity, rng):
    a = _arrays(frame)
    span = slice(start, start + duration)
    a["fee_rate"][span] *= 1 - 0.9 * severity
    a["volume"][span] *= 1 + severity
    return _build(frame, a, _returns(a))


ANOMALY_INJECTORS: dict[
    str,
    Callable[[pd.DataFrame, int, int, float, np.random.Generator], pd.DataFrame],
] = {
    "wash_trading": _wash_trading,
    "pump_and_dump": _pump_and_dump,
    "fee_spike": _fee_spike,
    "flash_crash": _flash_crash,
    "layering": _layering,
    "fee_evasion": _fee_evasion,
}


def inject_benign_spike(
    block_frame: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Benign look-alike: volume, tx_count and |return| jump with no anomaly."""
    n = len(block_frame)
    d = int(rng.integers(3, 21))
    start = int(rng.integers(0, n - d + 1))
    m = float(rng.uniform(1.0, 4.0))
    a = _arrays(block_frame)
    r = _returns(a)
    span = slice(start, start + d)
    a["volume"][span] *= 1 + m
    a["tx_count"][span] *= 1 + m
    idx = start - 1 + np.arange(d)
    r[idx[idx >= 0]] *= 2
    return _build(block_frame, a, r)

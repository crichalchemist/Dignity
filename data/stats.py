"""Chunked calibration statistics shared by the calibration script and tests.

Pure numpy/scipy. No network imports: tests/test_operator.py scans data/.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats

STAT_NAMES: tuple[str, ...] = (
    "ret_excess_kurtosis",
    "abs_ret_lag1_acf",
    "log_volume_lag1_acf",
    "volume_cv",
    "log_volume_skew",
)
CALIBRATION_CHUNK_LEN: int = 160
DEFAULT_REFERENCE_STATS = (
    Path(__file__).resolve().parent / "calibration" / "reference_stats.json"
)


def _lag1_acf(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _chunk_stats(price: np.ndarray, volume: np.ndarray) -> dict[str, float]:
    r = np.diff(np.log(price))
    abs_r = np.abs(r)
    lv = np.log(volume)
    return {
        "ret_excess_kurtosis": float(stats.kurtosis(r)),
        "abs_ret_lag1_acf": _lag1_acf(abs_r),
        "log_volume_lag1_acf": _lag1_acf(lv),
        "volume_cv": float(np.std(volume) / np.mean(volume)),
        "log_volume_skew": float(stats.skew(lv)),
    }


def calibration_stats(
    price: np.ndarray,
    volume: np.ndarray,
    chunk_len: int = CALIBRATION_CHUNK_LEN,
) -> dict[str, float]:
    """Median of each stat over the whole chunks of a price/volume series.

    Args:
        price: Prices, one per bar.
        volume: Volumes aligned with ``price``.
        chunk_len: Bars per chunk. A trailing partial chunk is dropped.

    Raises:
        ValueError: If the series lengths differ or there are fewer than
            two whole chunks.
    """
    if len(price) != len(volume):
        raise ValueError(f"price and volume must align: {len(price)} != {len(volume)}")
    n_chunks = len(price) // chunk_len
    if n_chunks < 2:
        raise ValueError(
            f"need at least 2 whole chunks of {chunk_len} bars, got {n_chunks}"
        )
    per_chunk = [
        _chunk_stats(
            price[i * chunk_len : (i + 1) * chunk_len],
            volume[i * chunk_len : (i + 1) * chunk_len],
        )
        for i in range(n_chunks)
    ]
    return {name: float(np.median([c[name] for c in per_chunk])) for name in STAT_NAMES}


def widen_band(values: list[float], frac: float = 0.2) -> tuple[float, float]:
    """Widen the [min, max] of ``values`` by ``frac`` of its range, split evenly."""
    lo, hi = min(values), max(values)
    pad = frac / 2 * (hi - lo)
    return lo - pad, hi + pad


def load_reference_stats(path: str | Path = DEFAULT_REFERENCE_STATS) -> dict:
    """Load the committed reference statistics JSON."""
    with open(path) as f:
        return json.load(f)

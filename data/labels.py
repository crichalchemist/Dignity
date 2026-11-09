"""Window labels computed from the per-row event record of one block."""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_starts(block_len: int, seq_len: int, horizon: int) -> np.ndarray:
    """Valid window start offsets: 0 .. block_len - seq_len - horizon inclusive."""
    return np.arange(block_len - seq_len - horizon + 1)


def _windows(values: np.ndarray, starts: np.ndarray, seq_len: int) -> np.ndarray:
    return sliding_window_view(values, seq_len)[starts]


def risk_labels(
    is_anomalous_block: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    overlap: float,
) -> np.ndarray:
    """int8 [n]: 1 iff the anomalous fraction of the window exceeds ``overlap``."""
    flags = np.asarray(is_anomalous_block, dtype=bool)
    frac = _windows(flags, starts, seq_len).mean(axis=1)
    return (frac > overlap).astype(np.int8)


def forecast_labels(
    anomaly_id_block: np.ndarray,
    is_anomalous_block: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Early-warning labels: y float32 [n, H], mask bool [n].

    y[:, h-1] = 1 if an onset row lies in (t, t+h], t = start + seq_len - 1.
    mask is False where an anomaly is already active at row t.
    """
    aid = np.asarray(anomaly_id_block)
    onset = (aid != -1) & np.concatenate(([True], aid[1:] != aid[:-1]))
    cum = np.concatenate(([0], np.cumsum(onset)))  # cum[k] = onsets in rows < k
    t = np.asarray(starts) + seq_len - 1
    h = np.arange(1, horizon + 1)
    y = (cum[t[:, None] + h + 1] - cum[t[:, None] + 1] > 0).astype(np.float32)
    mask = ~np.asarray(is_anomalous_block, dtype=bool)[t]
    return y, mask


def window_meta(
    is_anomalous_block: np.ndarray,
    anomaly_type_block: np.ndarray,
    starts: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(onset_offset int64 [n], anomaly_type object [n]) per window.

    onset_offset is the row-in-window of the first anomalous row, or -1. The type is
    the one with the most anomalous rows (ties go to the type appearing first), or
    "none" if the window has no anomalous row.
    """
    flags = _windows(np.asarray(is_anomalous_block, dtype=bool), starts, seq_len)
    types = _windows(np.asarray(anomaly_type_block, dtype=object), starts, seq_len)
    has_any = flags.any(axis=1)
    offset = np.where(has_any, flags.argmax(axis=1), -1).astype(np.int64)
    out = np.full(len(flags), "none", dtype=object)
    for i in np.flatnonzero(has_any):
        counts: dict = {}
        for name in types[i][flags[i]]:
            counts[name] = counts.get(name, 0) + 1
        out[i] = max(counts, key=counts.get)  # max keeps the first of equal counts
    return offset, out

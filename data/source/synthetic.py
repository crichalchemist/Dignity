"""Seeded event-record generator for synthetic transaction data.

Every block is an independent normal process (``data.source.dynamics``). With
probability ``contamination`` a block gets typed anomalies at random onsets
(``data.source.anomalies``); the generator writes the label columns. One
``np.random.Generator`` drives everything and the global numpy RNG is never
touched (DGNTY-19).
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from core.config import SyntheticConfig
from data.source.anomalies import (
    ANOMALY_INJECTORS,
    ANOMALY_TYPES,
    inject_benign_spike,
)
from data.source.dynamics import (
    DEFAULT_PARAMS,
    ar1_log_volume,
    congestion_fee,
    garch_returns,
    regime_path,
    tx_counts,
)

RAW_COLUMNS = ("volume", "price", "fee_rate", "tx_count")

_T0_MS = 1_700_000_000_000
_STEP_MS = 3_600_000  # hourly rows, ms like every other source


class SyntheticGenerator:
    """Generate an event-record frame: one row per step, blocks contiguous."""

    def __init__(self, config: SyntheticConfig, rng: np.random.Generator | None = None):
        unknown = sorted(set(config.type_weights) - set(ANOMALY_TYPES))
        if unknown:
            raise ValueError(
                f"unknown anomaly types in type_weights: {unknown}; "
                f"expected a subset of {ANOMALY_TYPES}"
            )
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        self.params = replace(
            DEFAULT_PARAMS, regime_switch_prob=config.regime_switch_prob
        )
        self._types = list(config.type_weights)
        weights = np.array([config.type_weights[t] for t in self._types], dtype=float)
        self._type_p = weights / weights.sum()

    def generate(self) -> pd.DataFrame:
        cfg = self.config
        n = cfg.block_len
        raw = {column: [] for column in RAW_COLUMNS}
        is_anomalous, anomaly_type, anomaly_id, block_label = [], [], [], []
        next_id = 0

        for block_id in range(cfg.n_blocks):
            block = self._normal_block(block_id)
            mask = np.zeros(n, dtype=bool)
            types = np.full(n, "none", dtype=object)
            ids = np.full(n, -1, dtype=np.int64)

            if self.rng.random() < cfg.contamination:
                for start, duration in self._spans():
                    kind = str(self.rng.choice(self._types, p=self._type_p))
                    severity = float(self.rng.uniform(*cfg.severity_range))
                    block = ANOMALY_INJECTORS[kind](
                        block, start, duration, severity, self.rng
                    )
                    span = slice(start, start + duration)
                    mask[span] = True
                    types[span] = kind
                    ids[span] = next_id
                    next_id += 1
            elif self.rng.random() < cfg.benign_spike_rate:
                block = inject_benign_spike(block, self.rng)

            label = int(mask.any())
            if self.rng.random() < cfg.label_noise:
                label = 1 - label

            for column in RAW_COLUMNS:
                raw[column].append(block[column].to_numpy(copy=True))
            is_anomalous.append(mask)
            anomaly_type.append(types)
            anomaly_id.append(ids)
            block_label.append(np.full(n, label, dtype=np.int8))

        rows = np.arange(cfg.n_blocks * n, dtype=np.int64)
        return pd.DataFrame(
            {
                "timestamp": _T0_MS + rows * _STEP_MS,
                "volume": np.concatenate(raw["volume"]).astype(np.float64),
                "price": np.concatenate(raw["price"]).astype(np.float64),
                "fee_rate": np.concatenate(raw["fee_rate"]).astype(np.float64),
                "tx_count": np.concatenate(raw["tx_count"]).astype(np.int64),
                "block_id": rows // n,
                "is_anomalous": np.concatenate(is_anomalous),
                "anomaly_type": np.concatenate(anomaly_type),
                "anomaly_id": np.concatenate(anomaly_id),
                "block_label": np.concatenate(block_label),
            }
        )

    def _normal_block(self, block_id: int) -> pd.DataFrame:
        """Raw columns of one normal block, before any injection."""
        n, p, rng = self.config.block_len, self.params, self.rng
        regime = regime_path(n, p, rng)
        r = garch_returns(n, p, rng) * np.asarray(p.regime_vol)[regime]
        x = ar1_log_volume(n, p, rng) + np.asarray(p.regime_log_volume)[regime]
        log_vol_base = rng.normal(p.log_volume_mean, p.block_level_sd)
        volume = np.exp(log_vol_base + x)
        price = np.exp(rng.normal(p.price_level_mean, p.price_level_sd) + np.cumsum(r))
        tx = tx_counts(volume, p, rng)
        if np.median(tx) <= 0:
            # congestion_fee divides by capacity_ratio * median(tx_count)
            raise ValueError(
                f"block {block_id}: median tx_count is {np.median(tx)}, so the "
                "congestion fee would be infinite; tx_count must have a positive median"
            )
        fee = congestion_fee(tx, p, rng)
        return pd.DataFrame(
            {"volume": volume, "price": price, "fee_rate": fee, "tx_count": tx}
        )

    def _spans(self) -> list[tuple[int, int]]:
        """Non-overlapping (start, duration) pairs for one anomalous block."""
        cfg = self.config
        n = cfg.block_len
        lo, hi = cfg.anomalies_per_block
        k = int(self.rng.integers(lo, hi + 1))
        d_lo, d_hi = cfg.duration_range
        durations = [int(d) for d in self.rng.integers(d_lo, d_hi + 1, size=k)]
        # Drop the last anomaly until the spans fit. If even one span is longer
        # than block_len - 1 this ends at k = 0: the block was drawn anomalous but
        # carries no anomaly, so its truth is 0 (label_noise may still flip its
        # block_label). That is allowed (G14).
        while durations and sum(durations) > n - 1:
            durations.pop()
        # Gaps start at 1 so no span starts at row 0: injectors act through the
        # log-return into each span row, and row 0 has none (a flash crash there
        # would be labelled but never happen).
        gaps = np.sort(
            self.rng.integers(1, n - sum(durations) + 1, size=len(durations))
        )
        starts = gaps + np.cumsum([0, *durations])[:-1]  # start_i = g_i + sum d_j<i
        return [(int(s), d) for s, d in zip(starts, durations, strict=True)]

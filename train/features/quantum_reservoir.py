"""Quantum reservoir computing feature block (emulated)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.quantum_emulation.reservoir.ising_qrc import IsingQRCConfig, IsingQuantumReservoir


@dataclass
class QRCFeatureConfig:
    input_columns: list[str]


DEFAULT_INPUT_COLUMNS = [
    "return_1",
    "log_return_1",
    "high_low_spread",
    "close_open_spread",
]


def build_qrc_features(feature_df: pd.DataFrame, cfg: IsingQRCConfig) -> pd.DataFrame:
    input_cols = [c for c in DEFAULT_INPUT_COLUMNS if c in feature_df.columns]
    if not input_cols:
        # Fall back to numeric columns if base features missing.
        input_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if not input_cols:
        raise ValueError("No numeric columns available for QRC input")

    input_df = feature_df[input_cols].copy()
    input_df = input_df.bfill().ffill()
    X = input_df.to_numpy(dtype=np.float64)
    reservoir = IsingQuantumReservoir(cfg)
    qrc_features = reservoir.fit_transform(X)
    names = reservoir.feature_names()
    if qrc_features.shape[1] != len(names):
        names = [f"qrc_{i}" for i in range(qrc_features.shape[1])]
    qrc_df = pd.DataFrame(qrc_features, columns=names, index=feature_df.index)
    return qrc_df


def add_quantum_reservoir_features(feature_df: pd.DataFrame, cfg: IsingQRCConfig) -> pd.DataFrame:
    qrc_df = build_qrc_features(feature_df, cfg)
    return pd.concat([feature_df, qrc_df], axis=1)


__all__ = ["QRCFeatureConfig", "build_qrc_features", "add_quantum_reservoir_features"]

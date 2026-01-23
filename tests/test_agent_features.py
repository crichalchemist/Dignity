import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import FeatureConfig
from train.features.agent_features import build_feature_frame


def _sample_df(length: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=length, freq="min")
    base = np.linspace(1.0, 1.1, length)
    close = base + 0.0005 * np.sin(np.linspace(0, np.pi, length))
    open_ = base
    high = close + 0.0002
    low = open_ - 0.0002
    volume = np.linspace(1_000, 1_200, length)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_feature_group_inclusion_exclusion():
    df = _sample_df()
    config = FeatureConfig(
        sma_windows=[3],
        ema_windows=[3],
        rsi_window=3,
        bollinger_window=5,
        bollinger_num_std=1.5,
        atr_window=3,
        short_vol_window=3,
        long_vol_window=5,
        spread_windows=[3],
        include_groups=["trend", "atr"],
        exclude_groups=["momentum"],
    )
    feature_df = build_feature_frame(df, config=config)

    assert "sma_3" in feature_df.columns
    assert "ema_3" in feature_df.columns
    assert "rsi_3" not in feature_df.columns
    assert "bb_bandwidth" not in feature_df.columns
    assert "atr_3_norm" in feature_df.columns


def test_feature_frame_no_lookahead_and_normalized_ranges():
    df = _sample_df()
    config = FeatureConfig(
        sma_windows=[3],
        ema_windows=[3],
        rsi_window=3,
        bollinger_window=5,
        bollinger_num_std=2.0,
        atr_window=3,
        short_vol_window=3,
        long_vol_window=5,
        spread_windows=[3],
        imbalance_smoothing=2,
        exclude_groups=["regime"],
    )

    full_features = build_feature_frame(df, config=config)
    truncated_features = build_feature_frame(df.iloc[:-1].copy(), config=config)

    common_len = min(len(full_features), len(truncated_features))
    pd.testing.assert_frame_equal(
        full_features.iloc[:common_len].reset_index(drop=True),
        truncated_features.reset_index(drop=True),
    )

    engineered_cols = [
        c
        for c in full_features.columns
        if c not in {"datetime", "open", "high", "low", "close", "volume"}
    ]
    float_values = full_features[engineered_cols].select_dtypes(include=[float]).to_numpy()
    assert np.isfinite(float_values).all()
    assert np.nanmax(np.abs(float_values)) < 150
    assert full_features["wick_imbalance"].between(-1, 1).all()
    assert full_features["body_range_ratio"].between(-1, 1).all()


def test_feature_frame_with_quantum_reservoir_group():
    df = _sample_df(length=60)
    config = FeatureConfig(
        include_groups=["quantum_reservoir"],
        qrc_num_qubits=4,
        qrc_input_qubits=2,
        qrc_memory_qubits=2,
        qrc_steps=1,
        qrc_measure_pairs=False,
    )

    feature_df = build_feature_frame(df, config=config)
    qrc_cols = [c for c in feature_df.columns if c.startswith("qrc_")]
    assert qrc_cols, "Expected quantum reservoir features to be present"
    assert len(feature_df) > 0

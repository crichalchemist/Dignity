"""Parallel feature computation for large datasets.

Implements the API described in the docs by parallelising feature *groups*
using ``concurrent.futures``. The behaviour matches
``features.agent_features.build_feature_frame`` (same column names and NaN
handling) while enabling concurrent execution for CPU-bound groups.

Usage:
    from train.features.agent_features_parallel import build_feature_frame_parallel

    # Always parallelise
    df = build_feature_frame_parallel(raw_df, config, parallel=True)

    # Auto-switch to parallel only for larger datasets
    df = build_feature_frame_parallel(raw_df, config, parallel="auto")
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import pandas as pd

from config.config import FeatureConfig
from train.features.agent_features import (
    _should_add,
    add_base_features,
    average_true_range,
    bollinger_bandwidth,
    candle_imbalance_features,
    ema,
    rsi,
    sma,
    volatility_clustering,
)

# Threshold where "auto" mode enables parallel execution.
_DEFAULT_PARALLEL_ROW_THRESHOLD = 200_000


def _trend_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if not _should_add("trend", cfg):
        return pd.DataFrame(index=df.index)
    out = {}
    for window in cfg.sma_windows:
        out[f"sma_{window}"] = sma(df["close"], window)
    for window in cfg.ema_windows:
        out[f"ema_{window}"] = ema(df["close"], window)
    return pd.DataFrame(out, index=df.index)


def _momentum_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if not _should_add("momentum", cfg):
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(
        {f"rsi_{cfg.rsi_window}": rsi(df["close"], window=cfg.rsi_window)},
        index=df.index,
    )


def _bollinger_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if not _should_add("bollinger", cfg):
        return pd.DataFrame(index=df.index)
    width, percent_b, bb_low, bb_mid, bb_high = bollinger_bandwidth(
        df["close"],
        window=cfg.bollinger_window,
        num_std=cfg.bollinger_num_std,
    )
    return pd.DataFrame(
        {
            f"bb_low_{cfg.bollinger_window}_{cfg.bollinger_num_std}": bb_low,
            f"bb_mid_{cfg.bollinger_window}_{cfg.bollinger_num_std}": bb_mid,
            f"bb_high_{cfg.bollinger_window}_{cfg.bollinger_num_std}": bb_high,
            "bb_bandwidth": width,
            "bb_percent_b": percent_b,
        },
        index=df.index,
    )


def _atr_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if not _should_add("atr", cfg):
        return pd.DataFrame(index=df.index)
    atr, true_range = average_true_range(df, window=cfg.atr_window)
    return pd.DataFrame(
        {
            "true_range_norm": true_range / df["close"],
            f"atr_{cfg.atr_window}_norm": atr / df["close"],
        },
        index=df.index,
    )


def _vol_clustering_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if not _should_add("vol_clustering", cfg):
        return pd.DataFrame(index=df.index)
    return volatility_clustering(
        df["log_return_1"],
        short_window=cfg.short_vol_window,
        long_window=cfg.long_vol_window,
    )


def _imbalance_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if not _should_add("imbalance", cfg):
        return pd.DataFrame(index=df.index)
    return candle_imbalance_features(df, smoothing=cfg.imbalance_smoothing)


def _compute_group_features(df: pd.DataFrame, cfg: FeatureConfig, group: str) -> pd.DataFrame:
    """Dispatch to the appropriate feature-group builder."""
    if group == "trend":
        return _trend_features(df, cfg)
    if group == "momentum":
        return _momentum_features(df, cfg)
    if group == "bollinger":
        return _bollinger_features(df, cfg)
    if group == "atr":
        return _atr_features(df, cfg)
    if group == "vol_clustering":
        return _vol_clustering_features(df, cfg)
    if group == "imbalance":
        return _imbalance_features(df, cfg)
    return pd.DataFrame(index=df.index)


def build_feature_frame_parallel(
        df: pd.DataFrame,
        config: FeatureConfig | None = None,
        parallel: bool | Literal["auto"] = "auto",
        min_rows_for_parallel: int = _DEFAULT_PARALLEL_ROW_THRESHOLD,
) -> pd.DataFrame:
    """Parallelised drop-in replacement for ``build_feature_frame``.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLC dataframe.
    config : FeatureConfig, optional
        Feature configuration; defaults to ``FeatureConfig()``.
    parallel : bool | "auto"
        - True: always parallelise feature groups.
        - False: run sequentially (same as ``build_feature_frame``).
        - "auto": parallelise only if ``len(df) >= min_rows_for_parallel``.
    min_rows_for_parallel : int
        Row threshold for enabling parallel mode when ``parallel="auto"``.
    """
    cfg = config or FeatureConfig()

    # Base features are computed once (shared dependency for later groups).
    base_df = add_base_features(df, spread_windows=cfg.spread_windows)

    should_parallel = (
            parallel is True
            or (parallel == "auto" and len(base_df) >= min_rows_for_parallel)
    )

    feature_frames = [base_df]
    groups = ["trend", "momentum", "bollinger", "atr", "vol_clustering", "imbalance"]

    if should_parallel and len(groups) > 1:
        with ThreadPoolExecutor(max_workers=len(groups)) as executor:
            futures = {
                executor.submit(_compute_group_features, base_df, cfg, group): group
                for group in groups
            }
            for future in as_completed(futures):
                feature_frames.append(future.result())
    else:
        for group in groups:
            feature_frames.append(_compute_group_features(base_df, cfg, group))

    feature_df = pd.concat(feature_frames, axis=1)
    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df

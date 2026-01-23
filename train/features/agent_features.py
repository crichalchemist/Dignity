import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import FeatureConfig, ResearchConfig


def _load_generated_features():
    """Load generated feature modules with recursion protection."""
    try:
        cfg = ResearchConfig()
        gen_dir = Path(cfg.generated_code_dir)
        features = {}

        # Check if directory exists and is valid
        if not gen_dir.exists() or not gen_dir.is_dir():
            return features

        for py_file in gen_dir.glob("*.py"):
            module_name = py_file.stem

            # Skip if already loaded (recursion protection)
            if module_name in sys.modules:
                continue

            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = mod
                    spec.loader.exec_module(mod)
                    if hasattr(mod, module_name):
                        features[module_name] = getattr(mod, module_name)
            except Exception:
                # Silently skip modules that fail to load
                continue

        return features
    except Exception:
        # If anything fails, return empty dict (fail-safe)
        return {}


GENERATED_FEATURES = _load_generated_features()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(100)


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def average_true_range(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr_components = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    return true_range.rolling(window=window, min_periods=window).mean(), true_range


def bollinger_bandwidth(close: pd.Series, window: int, num_std: float) -> pd.Series:
    lower, mid, upper = bollinger_bands(close, window=window, num_std=num_std)
    width = (upper - lower) / mid.replace(0, pd.NA)
    percent_b = (close - lower) / (upper - lower)
    return width, percent_b, lower, mid, upper


def volatility_clustering(log_returns: pd.Series, short_window: int, long_window: int) -> pd.DataFrame:
    short_vol = log_returns.rolling(window=short_window, min_periods=short_window).std()
    long_vol = log_returns.rolling(window=long_window, min_periods=long_window).std()
    vol_ratio = short_vol / long_vol.replace(0, pd.NA)
    vol_diff = short_vol - long_vol
    return pd.DataFrame(
        {
            "volatility_ratio": vol_ratio,
            "volatility_diff": vol_diff,
            "volatility_short": short_vol,
            "volatility_long": long_vol,
        }
    )


def candle_imbalance_features(df: pd.DataFrame, smoothing: int) -> pd.DataFrame:
    true_range = (df["high"] - df["low"]).replace(0, pd.NA)
    body = df["close"] - df["open"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    body_ratio = (body / true_range).clip(-1, 1)
    upper_wick_ratio = (upper_wick / true_range).clip(lower=0)
    lower_wick_ratio = (lower_wick / true_range).clip(lower=0)
    wick_imbalance = (upper_wick_ratio - lower_wick_ratio).clip(-1, 1)

    if smoothing > 1:
        body_ratio = body_ratio.rolling(window=smoothing, min_periods=smoothing).mean()
        wick_imbalance = wick_imbalance.rolling(window=smoothing, min_periods=smoothing).mean()

    return pd.DataFrame(
        {
            "body_range_ratio": body_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "lower_wick_ratio": lower_wick_ratio,
            "wick_imbalance": wick_imbalance,
        }
    )


def add_base_features(df: pd.DataFrame, spread_windows: list[int] | None = None) -> pd.DataFrame:
    out = df.copy()
    pct_change = out["close"].pct_change()
    log_return = np.log1p(pct_change)
    log_return[~np.isfinite(log_return)] = np.nan

    out["return_1"] = pct_change
    out["log_return_1"] = log_return
    out["high_low_spread"] = (out["high"] - out["low"]) / out["close"]
    out["close_open_spread"] = (out["close"] - out["open"]) / out["open"]

    if spread_windows:
        for window in spread_windows:
            mean_spread = out["high_low_spread"].rolling(window=window, min_periods=window).mean()
            std_spread = out["high_low_spread"].rolling(window=window, min_periods=window).std()
            out[f"high_low_spread_z_{window}"] = (out["high_low_spread"] - mean_spread) / std_spread.replace(0, pd.NA)
    return out


def _should_add(group: str, config: FeatureConfig) -> bool:
    if config.include_groups and group not in config.include_groups:
        return False
    return not (config.exclude_groups and group in config.exclude_groups)


def build_feature_frame(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """
    Compute a configurable set of technical features.
    Keep this pure: the caller handles I/O and splitting.
    """

    config = config or FeatureConfig()
    raw_df = df.copy()  # Keep reference to raw data for regime detection
    feature_df = add_base_features(df, spread_windows=config.spread_windows)

    if _should_add("trend", config):
        for window in config.sma_windows:
            feature_df[f"sma_{window}"] = sma(feature_df["close"], window)
        for window in config.ema_windows:
            feature_df[f"ema_{window}"] = ema(feature_df["close"], window)

    if _should_add("momentum", config):
        feature_df[f"rsi_{config.rsi_window}"] = rsi(feature_df["close"], window=config.rsi_window)

    if _should_add("bollinger", config):
        width, percent_b, bb_low, bb_mid, bb_high = bollinger_bandwidth(
            feature_df["close"],
            window=config.bollinger_window,
            num_std=config.bollinger_num_std,
        )
        feature_df[f"bb_low_{config.bollinger_window}_{config.bollinger_num_std}"] = bb_low
        feature_df[f"bb_mid_{config.bollinger_window}_{config.bollinger_num_std}"] = bb_mid
        feature_df[f"bb_high_{config.bollinger_window}_{config.bollinger_num_std}"] = bb_high
        feature_df["bb_bandwidth"] = width
        feature_df["bb_percent_b"] = percent_b

    if _should_add("atr", config):
        atr, true_range = average_true_range(feature_df, window=config.atr_window)
        feature_df["true_range_norm"] = true_range / feature_df["close"]
        feature_df[f"atr_{config.atr_window}_norm"] = atr / feature_df["close"]

    if _should_add("vol_clustering", config):
        vol_df = volatility_clustering(
            feature_df["log_return_1"],
            short_window=config.short_vol_window,
            long_window=config.long_vol_window,
        )
        feature_df = pd.concat([feature_df, vol_df], axis=1)

    if _should_add("imbalance", config):
        imbalance_df = candle_imbalance_features(feature_df, smoothing=config.imbalance_smoothing)
        feature_df = pd.concat([feature_df, imbalance_df], axis=1)

    if _should_add("microstructure", config):
        from train.features.microstructure import build_microstructure_features
        feature_df = build_microstructure_features(
            feature_df,
            windows=config.microstructure_windows
        )

    if _should_add("regime", config):
        from train.features.regime_detection import integrate_regime_features
        feature_df = integrate_regime_features(feature_df, raw_df)

    if _should_add("intrinsic_time", config):
        from train.features.intrinsic_time import add_intrinsic_time_features
        feature_df = add_intrinsic_time_features(
            feature_df,
            price_col="close",
            up_threshold=config.dc_threshold_up,
            down_threshold=config.dc_threshold_down,
        )

    if _should_add("quantum_reservoir", config):
        from models.quantum_emulation.reservoir.ising_qrc import IsingQRCConfig
        from train.features.quantum_reservoir import add_quantum_reservoir_features

        qrc_cfg = IsingQRCConfig(
            n_qubits=config.qrc_num_qubits,
            n_input_qubits=config.qrc_input_qubits,
            n_memory_qubits=config.qrc_memory_qubits,
            dt=config.qrc_dt,
            steps=config.qrc_steps,
            coupling_scale=config.qrc_coupling_scale,
            field_scale=config.qrc_field_scale,
            input_scale=config.qrc_input_scale,
            measure_pairs=config.qrc_measure_pairs,
            burn_in=config.qrc_burn_in,
            seed=config.qrc_seed,
        )
        feature_df = add_quantum_reservoir_features(feature_df, qrc_cfg)

    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df

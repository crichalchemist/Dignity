"""
FX-specific trading signals and chart patterns.

This module extends the base technical indicators with forex-specific features:
- Trading session effects (London, NY, Tokyo)
- Support/resistance levels
- Trend strength (ADX)
- Classic price action patterns
"""

import numpy as np
import pandas as pd


def add_fx_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add FX trading session indicators based on UTC time.

    Sessions have different characteristics:
    - Tokyo (23:00-08:00 UTC): Lower volatility, range-bound
    - London (07:00-16:00 UTC): High volatility, trending
    - New York (12:00-21:00 UTC): High volume
    - London/NY overlap (12:00-16:00 UTC): Highest liquidity

    Args:
        df: DataFrame with 'datetime' column

    Returns:
        DataFrame with session indicator columns
    """
    df = df.copy()
    hour = pd.to_datetime(df["datetime"]).dt.hour

    # Major sessions (binary flags)
    df["london_session"] = ((hour >= 7) & (hour < 16)).astype(np.int8)
    df["ny_session"] = ((hour >= 12) & (hour < 21)).astype(np.int8)
    df["tokyo_session"] = ((hour >= 23) | (hour < 8)).astype(np.int8)

    # High-liquidity overlap period
    df["high_liquidity"] = ((hour >= 12) & (hour < 16)).astype(np.int8)

    # Session transitions (volatility spikes often occur)
    df["london_open"] = (hour == 7).astype(np.int8)
    df["ny_open"] = (hour == 12).astype(np.int8)

    return df


def add_support_resistance_features(
        df: pd.DataFrame,
        lookback: int = 100,
        proximity_threshold: float = 0.001
) -> pd.DataFrame:
    """
    Detect support and resistance levels using rolling highs/lows.

    Key insight: Prices tend to reverse or consolidate near recent extremes.
    Proximity to these levels can be predictive of price action.

    Args:
        df: DataFrame with OHLC data
        lookback: Bars to look back for high/low detection
        proximity_threshold: Distance threshold to consider "near" a level (fraction, e.g., 0.001 = 0.1%)

    Returns:
        DataFrame with S/R feature columns
    """
    df = df.copy()

    # Rolling resistance (recent high) and support (recent low)
    df[f"resistance_{lookback}"] = df["high"].rolling(lookback, min_periods=lookback).max()
    df[f"support_{lookback}"] = df["low"].rolling(lookback, min_periods=lookback).min()

    # Normalized distance to levels (positive = below resistance, negative = above support)
    df["dist_to_resistance"] = (df[f"resistance_{lookback}"] - df["close"]) / df["close"]
    df["dist_to_support"] = (df["close"] - df[f"support_{lookback}"]) / df["close"]

    # Binary proximity flags
    df["near_resistance"] = (df["dist_to_resistance"].abs() < proximity_threshold).astype(np.int8)
    df["near_support"] = (df["dist_to_support"].abs() < proximity_threshold).astype(np.int8)

    # Percentage position in range [0, 1] where 0 = at support, 1 = at resistance
    range_size = df[f"resistance_{lookback}"] - df[f"support_{lookback}"]
    df["range_position"] = (df["close"] - df[f"support_{lookback}"]) / range_size.replace(0, np.nan)

    return df


def add_price_action_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect classic FX price action patterns.

    Patterns implemented:
    - Engulfing candles (reversal signals)
    - Pin bars / hammers (rejection wicks)
    - Inside bars (consolidation)

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with pattern detection columns
    """
    df = df.copy()

    # Candle body and wick calculations
    body_size = (df["close"] - df["open"]).abs()
    is_green = df["close"] > df["open"]
    is_red = df["close"] < df["open"]

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    # Engulfing patterns (strong reversal signal)
    # Bullish: Previous red candle fully engulfed by current green candle
    bullish_engulfing = (
            is_red.shift(1) &  # Previous candle was red
            is_green &  # Current candle is green
            (df["open"] < df["close"].shift(1)) &  # Opens below previous close
            (df["close"] > df["open"].shift(1))  # Closes above previous open
    )

    # Bearish: Previous green candle fully engulfed by current red candle
    bearish_engulfing = (
            is_green.shift(1) &
            is_red &
            (df["open"] > df["close"].shift(1)) &
            (df["close"] < df["open"].shift(1))
    )

    df["bullish_engulfing"] = bullish_engulfing.astype(np.int8)
    df["bearish_engulfing"] = bearish_engulfing.astype(np.int8)

    # Pin bars (long wick indicating rejection)
    # Bullish pin: Long lower wick (buyers rejected sellers)
    # Bearish pin: Long upper wick (sellers rejected buyers)
    wick_ratio_threshold = 2.0  # Wick should be at least 2x body size

    df["bullish_pin"] = (
            (lower_wick > wick_ratio_threshold * body_size) &
            (upper_wick < body_size)
    ).astype(np.int8)

    df["bearish_pin"] = (
            (upper_wick > wick_ratio_threshold * body_size) &
            (lower_wick < body_size)
    ).astype(np.int8)

    # Inside bars (consolidation / continuation pattern)
    # Current bar's range is completely inside previous bar's range
    df["inside_bar"] = (
            (df["high"] < df["high"].shift(1)) &
            (df["low"] > df["low"].shift(1))
    ).astype(np.int8)

    return df


# Low priority: finalize trend strength detection (e.g., ADX) in add_trend_strength_features.
def add_trend_strength_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Add Average Directional Index (ADX) for trend strength detection.

    ADX measures how strongly the market is trending (not direction, just strength):
    - ADX > 25: Strong trend (good for trend-following strategies)
    - ADX < 20: Weak trend / ranging market (good for mean-reversion)

    The ADX is calculated from:
    1. Directional movement (+DM and -DM)
    2. True range (TR)
    3. Directional indicators (+DI and -DI)
    4. Directional index (DX)
    5. ADX (smoothed DX)

    Context: I've set up the structure and calculated the initial directional movement.
    Your task is to complete the ADX calculation logic.

    Your Task: In this function, implement the ADX calculation following these steps:
    1. Calculate +DI and -DI from the smoothed +DM, -DM, and ATR
    2. Calculate DX from the difference between +DI and -DI
    3. Calculate ADX as the smoothed DX
    4. Add binary features for strong_trend (ADX > 25) and weak_trend (ADX < 20)

    Guidance: The key tradeoff is smoothing window size - larger windows give smoother signals
    but slower reaction to regime changes. Consider how responsive you want the trend detection
    to be. The standard window is 14 periods, but FX traders sometimes use 10 or 20.

    Args:
        df: DataFrame with OHLC data
        window: Smoothing window for ADX calculation (default: 14)

    Returns:
        DataFrame with ADX, +DI, -DI, and trend strength flags
    """
    df = df.copy()

    # Step 1: Calculate directional movement
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()

    # Positive directional movement (+DM)
    plus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(high_diff > low_diff) & (high_diff > 0)] = high_diff

    # Negative directional movement (-DM)
    minus_dm = pd.Series(0.0, index=df.index)
    minus_dm[(low_diff > high_diff) & (low_diff > 0)] = low_diff

    # True range (already have this, but recalculating for completeness)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Smoothed values using exponential moving average
    atr = tr.rolling(window, min_periods=window).mean()
    smoothed_plus_dm = plus_dm.rolling(window, min_periods=window).mean()
    smoothed_minus_dm = minus_dm.rolling(window, min_periods=window).mean()

    # Step 2: Calculate directional indicators (+DI and -DI)
    # These show the strength of upward vs downward movement (0-100 scale)
    # Normalized by ATR to account for volatility differences
    df["plus_di"] = 100 * (smoothed_plus_dm / atr.replace(0, np.nan))
    df["minus_di"] = 100 * (smoothed_minus_dm / atr.replace(0, np.nan))

    # Step 3: Calculate directional index (DX)
    # Measures how separated +DI and -DI are (i.e., is there a clear winner?)
    # High DX = one side clearly winning (trending)
    # Low DX = evenly matched (ranging/choppy)
    di_sum = df["plus_di"] + df["minus_di"]
    di_diff = (df["plus_di"] - df["minus_di"]).abs()
    dx = 100 * (di_diff / di_sum.replace(0, np.nan))

    # Step 4: Calculate ADX (smoothed DX)
    # Smoothing reduces noise and gives a more stable trend strength signal
    df["adx"] = dx.rolling(window, min_periods=window).mean()

    # Step 5: Add binary trend classification flags
    # These are actionable signals for strategy selection:
    # - Strong trend (ADX > 25): Use trend-following strategies
    # - Weak trend (ADX < 20): Use mean-reversion strategies
    # - Middle range (20-25): Transition zone, be cautious
    df["strong_trend"] = (df["adx"] > 25).astype(np.int8)
    df["weak_trend"] = (df["adx"] < 20).astype(np.int8)

    return df


def build_fx_feature_frame(
        df: pd.DataFrame,
        include_sessions: bool = True,
        include_support_resistance: bool = True,
        support_resistance_lookback: int = 100,
        include_trend_strength: bool = True,
        trend_strength_window: int = 14,
        include_patterns: bool = True,
) -> pd.DataFrame:
    """
    Add FX-specific signals to feature frame.

    This is the main entry point for adding forex-specific features to the
    standard technical indicator set.

    Args:
        df: DataFrame with OHLC data and datetime
        include_sessions: Add trading session features
        include_support_resistance: Add S/R levels
        support_resistance_lookback: Lookback period for S/R calculation
        include_trend_strength: Add ADX/trend features
        trend_strength_window: Window for ADX calculation
        include_patterns: Add price action patterns

    Returns:
        DataFrame with additional FX features
    """
    result = df.copy()

    if include_sessions:
        result = add_fx_session_features(result)

    if include_support_resistance:
        result = add_support_resistance_features(
            result,
            lookback=support_resistance_lookback
        )

    if include_trend_strength:
        result = add_trend_strength_features(
            result,
            window=trend_strength_window
        )

    if include_patterns:
        result = add_price_action_patterns(result)

    return result

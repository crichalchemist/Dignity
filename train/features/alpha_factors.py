"""
Session-aware alpha factors for FX trading.

This module implements alpha factors tailored to forex market microstructure:
- Session-specific momentum (London trending vs Tokyo ranging)
- Cross-session price dynamics (gaps, continuations)
- Volatility regimes by trading session
- Mean reversion strength indicators

Inspired by WorldQuant 101 Alphas, adapted for FX markets.
"""

import numpy as np
import pandas as pd


def add_session_momentum(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate momentum separately for each trading session.

    Key insight: London session exhibits stronger trends while Tokyo is more range-bound.
    Separating momentum by session improves signal quality.

    Args:
        df: DataFrame with OHLC data and session indicators
        window: Lookback period for momentum calculation

    Returns:
        DataFrame with session-specific momentum features
    """
    df = df.copy()

    # Ensure we have session features
    if "london_session" not in df.columns:
        raise ValueError("Session features required. Call add_fx_session_features() first.")

    # Calculate standard momentum (for comparison)
    df["momentum"] = df["close"].pct_change(window)

    # Session-specific momentum (only calculated during that session)
    # During London hours: measure London momentum
    # During Tokyo hours: measure Tokyo momentum
    # This captures the distinct price dynamics of each session

    # Initialize session momentum columns
    df["london_momentum"] = np.nan
    df["tokyo_momentum"] = np.nan
    df["ny_momentum"] = np.nan

    # Calculate momentum for each session independently
    # We'll use a rolling window but only update during the respective session
    for i in range(window, len(df)):
        # London momentum: return over last 'window' London session periods
        london_mask = df.iloc[max(0, i - window * 3):i + 1]["london_session"] == 1
        if london_mask.sum() >= window // 2:  # Need sufficient London data points
            london_prices = df.iloc[max(0, i - window * 3):i + 1]["close"][london_mask]
            if len(london_prices) >= 2:
                df.loc[df.index[i], "london_momentum"] = (
                        (london_prices.iloc[-1] - london_prices.iloc[0]) / london_prices.iloc[0]
                )

        # Tokyo momentum
        tokyo_mask = df.iloc[max(0, i - window * 3):i + 1]["tokyo_session"] == 1
        if tokyo_mask.sum() >= window // 2:
            tokyo_prices = df.iloc[max(0, i - window * 3):i + 1]["close"][tokyo_mask]
            if len(tokyo_prices) >= 2:
                df.loc[df.index[i], "tokyo_momentum"] = (
                        (tokyo_prices.iloc[-1] - tokyo_prices.iloc[0]) / tokyo_prices.iloc[0]
                )

        # NY momentum
        ny_mask = df.iloc[max(0, i - window * 3):i + 1]["ny_session"] == 1
        if ny_mask.sum() >= window // 2:
            ny_prices = df.iloc[max(0, i - window * 3):i + 1]["close"][ny_mask]
            if len(ny_prices) >= 2:
                df.loc[df.index[i], "ny_momentum"] = (
                        (ny_prices.iloc[-1] - ny_prices.iloc[0]) / ny_prices.iloc[0]
                )

    # Forward fill NaNs (carry forward last known momentum during non-session hours)
    df["london_momentum"] = df["london_momentum"].ffill().fillna(0)
    df["tokyo_momentum"] = df["tokyo_momentum"].ffill().fillna(0)
    df["ny_momentum"] = df["ny_momentum"].ffill().fillna(0)

    # Momentum divergence: difference between overall and session-specific momentum
    # Large divergence suggests session characteristics dominating overall trend
    df["momentum_divergence_london"] = df["momentum"] - df["london_momentum"]
    df["momentum_divergence_tokyo"] = df["momentum"] - df["tokyo_momentum"]

    return df


def add_cross_session_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect price gaps between trading sessions.

    FX insight: Gaps often occur at session transitions (e.g., Tokyo → London)
    due to news events or liquidity shifts. These gaps often get filled or extended.

    Args:
        df: DataFrame with OHLC data and session indicators

    Returns:
        DataFrame with gap features
    """
    df = df.copy()

    # Detect session transitions
    df["session_transition"] = (
            (df["london_open"] == 1) | (df["ny_open"] == 1)
    ).astype(np.int8)

    # Calculate gap: difference between current open and previous close
    df["price_gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # Gap only meaningful at session transitions
    df["session_gap"] = np.where(
        df["session_transition"] == 1,
        df["price_gap"],
        0.0
    )

    # Gap direction
    df["gap_up"] = (df["session_gap"] > 0.0001).astype(np.int8)  # >1 pip threshold
    df["gap_down"] = (df["session_gap"] < -0.0001).astype(np.int8)

    # Gap fill detection (price returns to pre-gap level within session)
    # This is predictive: gaps often get filled within a few hours
    df["gap_filled"] = 0
    for i in range(1, len(df)):
        if df.iloc[i - 1]["session_gap"] != 0:
            gap_price = df.iloc[i - 1]["close"]
            current_high = df.iloc[i]["high"]
            current_low = df.iloc[i]["low"]

            # Check if price touched the pre-gap level
            if df.iloc[i - 1]["gap_up"] and current_low <= gap_price or df.iloc[i - 1][
                "gap_down"] and current_high >= gap_price:
                df.loc[df.index[i], "gap_filled"] = 1

    return df


def add_volatility_regime(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Classify volatility regime (low, medium, high) using historical volatility.

    Different strategies work in different volatility regimes:
    - Low vol: Mean reversion works well
    - High vol: Breakout strategies work well

    Args:
        df: DataFrame with OHLC data
        window: Window for volatility calculation

    Returns:
        DataFrame with volatility regime features
    """
    df = df.copy()

    # Calculate historical volatility (annualized)
    df["historical_vol"] = df["log_returns"].rolling(window).std() * np.sqrt(252 * 390)  # 390 mins/day

    # Percentile-based regime classification
    # Use expanding window to get robust percentiles
    df["vol_percentile"] = df["historical_vol"].rank(pct=True)

    # Regime flags
    df["low_vol_regime"] = (df["vol_percentile"] < 0.33).astype(np.int8)
    df["medium_vol_regime"] = ((df["vol_percentile"] >= 0.33) & (df["vol_percentile"] < 0.67)).astype(np.int8)
    df["high_vol_regime"] = (df["vol_percentile"] >= 0.67).astype(np.int8)

    # Volatility trend: is volatility increasing or decreasing?
    df["vol_trend"] = df["historical_vol"].diff(5)  # 5-period change
    df["vol_increasing"] = (df["vol_trend"] > 0).astype(np.int8)

    return df


def add_mean_reversion_strength(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Measure mean reversion tendency using price oscillator and Bollinger Bands.

    Mean reversion strength varies by session and volatility regime.
    Strong mean reversion → fade extreme moves
    Weak mean reversion → follow trends

    Args:
        df: DataFrame with OHLC data
        window: Window for calculations

    Returns:
        DataFrame with mean reversion indicators
    """
    df = df.copy()

    # Bollinger Bands
    df["sma"] = df["close"].rolling(window).mean()
    df["bb_std"] = df["close"].rolling(window).std()
    df["bb_upper"] = df["sma"] + 2 * df["bb_std"]
    df["bb_lower"] = df["sma"] - 2 * df["bb_std"]

    # %B: Position within Bollinger Bands [0, 1]
    # 0 = at lower band, 1 = at upper band, 0.5 = at middle
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # Extreme positions (potential reversal zones)
    df["bb_overbought"] = (df["bb_pct"] > 0.9).astype(np.int8)
    df["bb_oversold"] = (df["bb_pct"] < 0.1).astype(np.int8)

    # Distance from mean (normalized by volatility)
    df["distance_from_mean"] = (df["close"] - df["sma"]) / df["bb_std"].replace(0, np.nan)

    # RSI (Relative Strength Index) - classic mean reversion indicator
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # RSI extreme levels
    df["rsi_overbought"] = (df["rsi"] > 70).astype(np.int8)
    df["rsi_oversold"] = (df["rsi"] < 30).astype(np.int8)

    return df


def build_alpha_feature_frame(
        df: pd.DataFrame,
        include_session_momentum: bool = True,
        session_momentum_window: int = 20,
        include_cross_session_gap: bool = True,
        include_volatility_regime: bool = True,
        volatility_window: int = 20,
        include_mean_reversion: bool = True,
        mean_reversion_window: int = 20,
) -> pd.DataFrame:
    """
    Main entry point for adding alpha factors to feature frame.

    Args:
        df: DataFrame with OHLC data and FX session indicators
        include_session_momentum: Add session-specific momentum
        session_momentum_window: Window for momentum calculation
        include_cross_session_gap: Add gap features
        include_volatility_regime: Add volatility regime classification
        volatility_window: Window for volatility calculation
        include_mean_reversion: Add mean reversion indicators
        mean_reversion_window: Window for mean reversion calculation

    Returns:
        DataFrame with alpha factors
    """
    result = df.copy()

    # Ensure we have log_returns for volatility calculations
    if "log_returns" not in result.columns:
        result["log_returns"] = np.log(result["close"] / result["close"].shift(1))

    if include_session_momentum:
        result = add_session_momentum(result, window=session_momentum_window)

    if include_cross_session_gap:
        result = add_cross_session_gap(result)

    if include_volatility_regime:
        result = add_volatility_regime(result, window=volatility_window)

    if include_mean_reversion:
        result = add_mean_reversion_strength(result, window=mean_reversion_window)

    return result

"""
Advanced order flow and microstructure features extracted from OHLCV data.

These features estimate market microstructure signals without needing access to
order book data (LOB). They leverage candle patterns and volume to infer order flow.

References:
  - Blume, Easley & O'Hara (1994): Market Statistics and Technical Analysis
  - Chakrabarty et al. (2007): Trade Execution Cost Analysis
"""

import numpy as np
import pandas as pd


def high_low_imbalance(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    High-Low Imbalance: Estimate directional order flow from close position.

    Intuition:
      - If close is near high: buying pressure (asks hit)
      - If close is near low: selling pressure (bids hit)

    Returns:
      imbalance: -1 (strong sell pressure) to +1 (strong buy pressure)
    """
    close_position = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)
    # Normalize to [-1, 1]: 0.5 -> 0, 1 -> 1, 0 -> -1
    imbalance = 2 * close_position - 1

    # Smooth over window
    return imbalance.rolling(window).mean()


def volume_direction_score(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Volume-Weighted Direction Score.

    Combines volume and direction: if volume high AND close near top -> strong buy signal.

    Formula:
      score = high_low_imbalance * (volume / volume_ma)

    Returns:
      score: Directional volume strength
    """
    imbalance = high_low_imbalance(df, window=1)
    volume_ma = df["volume"].rolling(window).mean()
    volume_relative = df["volume"] / (volume_ma + 1e-8)

    return imbalance * volume_relative


def order_flow_toxicity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Order Flow Toxicity: Adverse selection costs proxy.

    High toxicity = high likelihood that new orders will execute at worse prices
    (adverse selection). Estimated from:
      - Range expansion (more market orders)
      - High volume with poor execution (larger spreads)
      - Sudden reversals (stops hit, reversals common)

    Returns:
      toxicity: 0 (low toxicity) to 1 (high toxicity)
    """
    # 1. Range expansion (larger true range = more market orders)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    atr = tr.rolling(window).mean()
    range_ratio = (df["high"] - df["low"]) / (atr + 1e-8)

    # 2. Volume surge (high volume -> more competition, wider spreads)
    volume_ma = df["volume"].rolling(window).mean()
    volume_surge = df["volume"] / (volume_ma + 1e-8)

    # 3. Intrabar reversal (close far from open = orders whipped)
    intrabar_move = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)

    # Combine: all scaled to [0, 1]
    toxicity = (
            0.4 * np.clip(range_ratio / range_ratio.rolling(window).mean(), 0, 1) +
            0.4 * np.clip(volume_surge / volume_surge.rolling(window).mean(), 0, 1) +
            0.2 * intrabar_move
    )

    return toxicity


def bid_ask_spread_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Estimate bid-ask spread from high-low range.

    Intuition: Spread ≈ (high - low) / (2 * close)
    This is a crude estimate but useful for spread-aware execution.

    Returns:
      spread_pct: Spread as % of closing price
    """
    return (df["high"] - df["low"]) / (2 * df["close"]) * 100


def depth_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Proxy for market depth from volume concentration.

    Higher volume with less range = deeper market (can absorb orders).
    Lower volume with large range = shallow market (orders move price).

    Returns:
      depth: Market depth indicator (higher = deeper)
    """
    range_pct = (df["high"] - df["low"]) / df["close"]
    volume_ma = df["volume"].rolling(window).mean()

    # Depth ~ volume / range: more volume, less range -> deeper
    depth = volume_ma / (range_pct * 1000 + 1e-8)

    return depth / depth.rolling(window).mean()  # Normalize by recent mean


def vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Volume-Weighted Average Price deviation.

    VWAP = cumsum(price * volume) / cumsum(volume)
    Deviation = (close - VWAP) / VWAP

    Useful for detecting mean reversion or momentum opportunities.

    Returns:
      vwap_dev: Current price deviation from VWAP (%)
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(window).sum() / df["volume"].rolling(window).sum()

    vwap_dev = (df["close"] - vwap) / (vwap + 1e-8) * 100

    return vwap_dev


def momentum_imbalance(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Buy-Sell Momentum Imbalance.

    Estimates whether momentum is skewed to buys or sells.

    Formula:
      buy_volume = volume where close > open
      sell_volume = volume where close < open
      imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol)

    Returns:
      imbalance: -1 (sell pressure) to +1 (buy pressure)
    """
    buy_mask = df["close"] > df["open"]
    buy_volume = df["volume"] * buy_mask
    sell_volume = df["volume"] * ~buy_mask

    total_volume = buy_volume + sell_volume
    imbalance = (buy_volume - sell_volume) / (total_volume + 1e-8)

    return imbalance.rolling(window).mean()


def price_impact_proxy(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Estimate price impact from order book pressure.

    If volume increases significantly but close moves away from buy/sell pressure,
    there's likely strong opposing liquidity (deep book).

    Returns:
      impact: Price impact indicator (higher = worse execution)
    """
    volume_ma = df["volume"].rolling(window).mean()
    volume_relative = df["volume"] / (volume_ma + 1e-8)

    # Range relative to volume (smaller range per unit volume = more liquidity)
    range_pct = (df["high"] - df["low"]) / df["close"]
    range_ma = range_pct.rolling(window).mean()

    # Impact = volume surge * (current range / average range)
    impact = volume_relative * (range_pct / (range_ma + 1e-8))

    return np.clip(impact, 0, 2)  # Clip to [0, 2]


def build_microstructure_features(df: pd.DataFrame, windows: list = None) -> pd.DataFrame:
    """
    Build complete microstructure feature set.

    Args:
        df: OHLCV dataframe
        windows: List of rolling windows to use (default: [5, 10, 20])

    Returns:
        feature_df: Original df with all microstructure features appended
    """
    if windows is None:
        windows = [5, 10, 20]

    features = df.copy()

    for window in windows:
        suffix = f"_{window}" if window != windows[-1] else ""  # Default window has no suffix

        features[f"hl_imbalance{suffix}"] = high_low_imbalance(df, window)
        features[f"vol_direction{suffix}"] = volume_direction_score(df, window)
        features[f"toxicity{suffix}"] = order_flow_toxicity(df, window)
        features[f"spread_proxy{suffix}"] = bid_ask_spread_proxy(df, window)
        features[f"depth_proxy{suffix}"] = depth_proxy(df, window)
        features[f"vwap_dev{suffix}"] = vwap_deviation(df, window)
        features[f"momentum_imbalance{suffix}"] = momentum_imbalance(df, window)
        features[f"price_impact{suffix}"] = price_impact_proxy(df, window)

    # Fill any NaN values
    features = features.bfill().ffill()

    return features


# Summary of microstructure features:
"""
HIGH-PRIORITY FEATURES (best for directional prediction):
  - hl_imbalance: Order flow direction proxy
  - vol_direction: Volume-weighted directional signal
  - momentum_imbalance: Buy vs. sell pressure

EXECUTION-QUALITY FEATURES (important for execution strategy):
  - toxicity: Adverse selection risk
  - spread_proxy: Estimated spread for execution cost
  - depth_proxy: Market depth for order sizing
  - price_impact: Execution impact estimate

MEAN-REVERSION FEATURES (useful in ranging markets):
  - vwap_dev: Price deviation from volume-weighted average
  - range_relative: Candle range relative to historical average
"""

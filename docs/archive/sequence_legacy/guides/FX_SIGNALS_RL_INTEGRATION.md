# Integrating FX Signals/Patterns into RL Windows

## Overview

This guide explains how to integrate traditional FX trading signals and patterns into the reinforcement learning
pipeline. The key insight: **RL agents don't see raw prices - they see engineered features that encode market structure
**.

---

## Architecture: Data Flow

```
1. Raw OHLC Data (datetime, open, high, low, close, volume)
           ↓
2. Feature Engineering Pipeline (features/agent_features.py)
   - Technical indicators (SMA, EMA, RSI, Bollinger, ATR)
   - Volatility clustering
   - Candle imbalance
   - [YOUR FX PATTERNS HERE] ← Integration point
           ↓
3. Feature Windows (N, T, F)
   - N = number of episodes/samples
   - T = lookback window (e.g., 120 bars)
   - F = number of features (e.g., 50-200)
           ↓
4. Signal Model (models/policy.py)
   - CNN → LSTM → Attention
   - Extracts patterns from feature sequences
           ↓
5. RL Policy (models/policy.py)
   - Predicts actions: BUY/HOLD/SELL
           ↓
6. Trading Environment (execution/simulated_retail_env.py)
   - Executes trades
   - Returns PnL rewards
```

---

## Current Feature Groups

The system already implements standard technical analysis in `features/agent_features.py`:

| Group              | Features                                        | Description                        |
|--------------------|-------------------------------------------------|------------------------------------|
| **base**           | log_return_1, log_return_5, spread, wick ratios | Price changes and candle structure |
| **trend**          | SMA, EMA (multiple windows)                     | Moving averages                    |
| **momentum**       | RSI                                             | Overbought/oversold                |
| **bollinger**      | BB bands, bandwidth, %b                         | Volatility bands                   |
| **atr**            | ATR, true_range                                 | Volatility measure                 |
| **vol_clustering** | Short/long vol, vol ratio                       | Regime detection                   |
| **imbalance**      | Wick/body imbalance                             | Order flow proxy                   |

---

## FX-Specific Signals to Add

### 1. **Session-Based Features** (London, NY, Tokyo overlap)

FX markets have strong time-of-day effects:

```python
def add_fx_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FX trading session indicators."""
    df = df.copy()
    hour = df["datetime"].dt.hour

    # London session (7-16 UTC)
    df["london_session"] = ((hour >= 7) & (hour < 16)).astype(int)

    # New York session (12-21 UTC)
    df["ny_session"] = ((hour >= 12) & (hour < 21)).astype(int)

    # Tokyo session (23-8 UTC)
    df["tokyo_session"] = (((hour >= 23) | (hour < 8))).astype(int)

    # High-liquidity overlap (London + NY: 12-16 UTC)
    df["high_liquidity"] = ((hour >= 12) & (hour < 16)).astype(int)

    return df
```

### 2. **Support/Resistance Levels**

Key price levels where reversals often occur:

```python
def add_support_resistance_features(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """Detect support/resistance levels using rolling highs/lows."""
    df = df.copy()

    # Rolling highs/lows as resistance/support
    df["resistance_100"] = df["high"].rolling(lookback).max()
    df["support_100"] = df["low"].rolling(lookback).min()

    # Distance to key levels (normalized)
    df["dist_to_resistance"] = (df["resistance_100"] - df["close"]) / df["close"]
    df["dist_to_support"] = (df["close"] - df["support_100"]) / df["close"]

    # Proximity to levels (binary flags)
    threshold = 0.001  # 0.1% threshold
    df["near_resistance"] = (df["dist_to_resistance"].abs() < threshold).astype(int)
    df["near_support"] = (df["dist_to_support"].abs() < threshold).astype(int)

    return df
```

### 3. **Trend Strength** (ADX, Directional Movement)

Measures how strongly the market is trending:

```python
def add_trend_strength_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add Average Directional Index (ADX) for trend strength."""
    df = df.copy()

    # Directional movement
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()

    # Positive/negative directional movement
    plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
    minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff

    # True range
    tr = df["high"] - df["low"]

    # Smoothed values
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)

    # ADX calculation
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window).mean()

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx

    # Trend interpretation
    df["strong_trend"] = (adx > 25).astype(int)
    df["weak_trend"] = (adx < 20).astype(int)

    return df
```

### 4. **Price Action Patterns**

Classic FX chart patterns:

```python
def add_price_action_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect common FX price action patterns."""
    df = df.copy()

    # Engulfing patterns
    bullish_engulfing = (
            (df["close"].shift(1) < df["open"].shift(1)) &  # Previous red candle
            (df["close"] > df["open"]) &  # Current green candle
            (df["open"] < df["close"].shift(1)) &  # Opens below prev close
            (df["close"] > df["open"].shift(1))  # Closes above prev open
    )

    bearish_engulfing = (
            (df["close"].shift(1) > df["open"].shift(1)) &
            (df["close"] < df["open"]) &
            (df["open"] > df["close"].shift(1)) &
            (df["close"] < df["open"].shift(1))
    )

    df["bullish_engulfing"] = bullish_engulfing.astype(int)
    df["bearish_engulfing"] = bearish_engulfing.astype(int)

    # Pin bars (long wicks indicating rejection)
    body_size = (df["close"] - df["open"]).abs()
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    df["bullish_pin"] = ((lower_wick > 2 * body_size) & (upper_wick < body_size)).astype(int)
    df["bearish_pin"] = ((upper_wick > 2 * body_size) & (lower_wick < body_size)).astype(int)

    # Inside bars (consolidation)
    df["inside_bar"] = (
            (df["high"] < df["high"].shift(1)) &
            (df["low"] > df["low"].shift(1))
    ).astype(int)

    return df
```

### 5. **Currency Correlation** (Multi-Pair Features)

FX pairs often move together (EUR/USD vs GBP/USD):

```python
def add_currency_correlation_features(
    df: pd.DataFrame,
    correlated_pair_df: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """Add correlation with another currency pair."""
    df = df.copy()
    
    # Align timestamps
    merged = df.merge(
        correlated_pair_df[["datetime", "close"]],
        on="datetime",
        how="left",
        suffixes=("", "_corr")
    )
    
    # Rolling correlation
    merged["pair_correlation"] = (
        merged["close"]
        .rolling(window)
        .corr(merged["close_corr"])
    )
    
    # Spread divergence (for related pairs like EUR/USD and GBP/USD)
    merged["pair_spread"] = merged["close"] - merged["close_corr"]
    merged["spread_zscore"] = (
        (merged["pair_spread"] - merged["pair_spread"].rolling(window).mean()) /
        merged["pair_spread"].rolling(window).std()
    )
    
    return merged.drop(columns=["close_corr"])
```

### 6. **Order Flow Proxies** (Already partially implemented in microstructure.py)

Volume and price action indicate order flow:

```python
def add_order_flow_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add order flow proxy features."""
    df = df.copy()
    
    # Volume-weighted direction
    body_direction = (df["close"] > df["open"]).astype(int) * 2 - 1  # +1 or -1
    if "volume" in df.columns:
        df["vol_direction"] = (body_direction * df["volume"]).rolling(window).sum()
        df["vol_direction_norm"] = df["vol_direction"] / df["volume"].rolling(window).sum()
    
    # High-low range expansion/contraction
    df["range_expansion"] = (
        (df["high"] - df["low"]) / (df["high"] - df["low"]).rolling(window).mean()
    )
    
    return df
```

---

## Integration Steps

### Step 1: Create FX Signals Module

Create `features/fx_patterns.py`:

```python
"""FX-specific trading signals and patterns."""

import pandas as pd
from typing import List, Optional


def build_fx_feature_frame(
        df: pd.DataFrame,
        include_sessions: bool = True,
        include_support_resistance: bool = True,
        include_trend_strength: bool = True,
        include_patterns: bool = True,
) -> pd.DataFrame:
    """
    Add FX-specific signals to feature frame.
    
    Args:
        df: DataFrame with OHLC data
        include_sessions: Add session-based features
        include_support_resistance: Add S/R levels
        include_trend_strength: Add ADX/trend features
        include_patterns: Add price action patterns
        
    Returns:
        DataFrame with additional FX features
    """
    result = df.copy()

    if include_sessions:
        result = add_fx_session_features(result)

    if include_support_resistance:
        result = add_support_resistance_features(result)

    if include_trend_strength:
        result = add_trend_strength_features(result)

    if include_patterns:
        result = add_price_action_patterns(result)

    return result

# [Include all the functions defined above]
```

### Step 2: Update Feature Pipeline

Modify `features/agent_features.py`:

```python
from features.fx_patterns import build_fx_feature_frame


def build_feature_frame(df: pd.DataFrame, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """Compute a configurable set of technical features."""
    config = config or FeatureConfig()

    # Existing features
    feature_df = add_base_features(df, spread_windows=config.spread_windows)

    # ... existing feature groups ...

    # NEW: Add FX-specific patterns
    if _should_add("fx_patterns", config):
        feature_df = build_fx_feature_frame(
            feature_df,
            include_sessions=True,
            include_support_resistance=True,
            include_trend_strength=True,
            include_patterns=True,
        )

    feature_df = feature_df.dropna().reset_index(drop=True)
    return feature_df
```

### Step 3: Update FeatureConfig

Add to `config/config.py`:

```python
@dataclass
class FeatureConfig:
    # ... existing fields ...
    
    # FX-specific configuration
    include_fx_sessions: bool = True
    include_support_resistance: bool = True
    support_resistance_lookback: int = 100
    include_trend_strength: bool = True
    adx_window: int = 14
    include_price_patterns: bool = True
```

### Step 4: Use in RL Training

The features automatically flow into RL training:

```python
# In prepare_dataset.py or RL training script
feature_cfg = FeatureConfig(
    include_groups=["all"],  # Includes FX patterns
    include_fx_sessions=True,
    include_support_resistance=True,
    include_trend_strength=True,
    include_price_patterns=True,
)

# Build features
feature_df = build_feature_frame(raw_df, config=feature_cfg)

# Features now include:
# - Standard indicators (SMA, RSI, Bollinger, etc.)
# - FX sessions (london_session, ny_session, etc.)
# - S/R levels (dist_to_resistance, near_support, etc.)
# - Trend strength (adx, plus_di, minus_di, etc.)
# - Price patterns (bullish_engulfing, pin_bars, etc.)

# These feed into RL training
train_with_environment(
    signal_model=model,
    policy=policy,
    train_data=feature_windows,  # (N, T, F) with FX features
    cfg=rl_cfg,
    env_config=env_cfg,
    device=device,
)
```

---

## Example: Complete Integration

```python
# 1. Load raw data
raw_df = load_histdata("GBPUSD", "2023-01-01", "2023-12-31")

# 2. Configure features with FX signals
feature_cfg = FeatureConfig(
    # Standard indicators
    sma_windows=[10, 20, 50, 200],
    ema_windows=[12, 26],
    rsi_window=14,
    bollinger_window=20,
    
    # FX-specific
    include_fx_sessions=True,
    include_support_resistance=True,
    support_resistance_lookback=100,
    include_trend_strength=True,
    adx_window=14,
    include_price_patterns=True,
)

# 3. Build features
feature_df = build_feature_frame(raw_df, config=feature_cfg)

# 4. Feature columns now include (~80-120 features):
# - log_return_1, log_return_5, spread, ...
# - sma_10, sma_20, ema_12, ema_26, ...
# - rsi_14, bb_bandwidth, bb_percent_b, ...
# - london_session, ny_session, high_liquidity, ...
# - dist_to_resistance, near_support, ...
# - adx, plus_di, minus_di, strong_trend, ...
# - bullish_engulfing, bearish_pin, inside_bar, ...

# 5. Create windows for RL
from data.agents.single_task_agent import SingleTaskDataAgent

data_cfg = DataConfig(
    csv_path="",
    datetime_column="datetime",
    feature_columns=[c for c in feature_df.columns if c != "datetime"],
    target_type="regression",
    t_in=120,  # 2-hour lookback window
    t_out=10,
)

agent = SingleTaskDataAgent(data_cfg)
datasets = agent.build_datasets(feature_df)

# 6. Train RL policy
# The agent sees 120-bar windows of ~100 features
# Each feature encodes FX market structure
# Policy learns which patterns → profitable trades
```

---

## Key Insights

### 1. **Features = Agent's Perception**

The RL agent can only act on what it perceives. Rich features = better decisions.

### 2. **Domain Knowledge Matters**

FX-specific patterns (sessions, S/R, correlations) give the agent "market sense" that generic indicators don't provide.

### 3. **Balance Complexity vs Overfitting**

More features ≠ better performance. Focus on:

- **Signal-to-noise ratio:** Clear, consistent patterns
- **Regime awareness:** Session effects, volatility states
- **Actionable information:** Features that inform trade decisions

### 4. **Feature Engineering is Iterative**

Start with core FX signals, measure impact, refine based on what the agent uses (via attention weights).

---

## Next Steps

1. **Implement FX patterns module** (features/fx_patterns.py)
2. **Integrate into feature pipeline** (features/agent_features.py)
3. **Update config** (config/config.py)
4. **Train and evaluate** - compare RL performance with/without FX signals
5. **Analyze attention weights** - see which features the agent values most
6. **Iterate** - add/remove features based on results

---

## Advanced: Custom Pattern Detection

For complex patterns (e.g., head & shoulders, Fibonacci retracements), you can:

1. **Use template matching** on price sequences
2. **Train a pattern classifier** separately
3. **Add classifier outputs as features** to RL

Example:

```python
# Separate model detects head-shoulders pattern
pattern_detector = HeadShouldersDetector()
df["hs_pattern_score"] = pattern_detector.predict(df)

# RL agent sees this as a feature
# Learns: when hs_pattern_score is high + other conditions → SELL
```

This separates pattern recognition from action selection, making the RL problem more tractable.

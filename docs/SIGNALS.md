# Signal Processing

Dignity Core computes 32 financial and statistical features from OHLCV data via the `SignalProcessor` class in `core/signals.py`.

## Overview

`SignalProcessor` is a collection of static methods. All return `float64` arrays of the same length as the input. Warmup bars are filled with the first valid computed value rather than NaN.

The full feature set is computed by `process_sequence()`, which returns an ordered dictionary of 32 features.

## Asset Configuration

Different instrument classes need different signal parameters. `AssetConfig` calibrates thresholds per asset class:

```python
from core.signals import ASSET_CONFIGS

forex_cfg = ASSET_CONFIGS["forex"]      # DC threshold: 0.0005
crypto_cfg = ASSET_CONFIGS["crypto"]    # DC threshold: 0.005
equity_cfg = ASSET_CONFIGS["equity"]    # DC threshold: 0.001
commodity_cfg = ASSET_CONFIGS["commodity"]  # DC threshold: 0.002
```

Pass an `AssetConfig` to `process_sequence()` to use the appropriate thresholds.

## The 32 Features

`SignalProcessor.process_sequence(volumes, prices, fees, high, low, tx_count, asset_config)` returns:

### Raw Inputs (4)
`volume`, `price`, `fee_rate`, `tx_count`

### Technical Indicators (10)
`rsi`, `macd_line`, `macd_signal`, `macd_hist`, `bollinger_pct_b`, `bollinger_width`, `atr`, `stoch_k`, `stoch_d`, `adx`

### Volume Signals (3)
`obv`, `vwap`, `order_flow_imbalance`

### Rate of Change (2)
`roc_5`, `roc_20`

### Momentum (2)
`momentum_10`, `momentum_20`

### Volatility (3)
`volatility_5`, `volatility_20`, `vol_ratio`

### Directional Change (4)
`dc_direction`, `dc_overshoot`, `bars_since_event`, `dc_bars_since_event`

### Derived (2)
`volume_volatility`, `volume_entropy`

### Other (2)
`price_change`, `directional_change`

## Individual Indicators

### RSI – Relative Strength Index

```python
from core.signals import SignalProcessor
import numpy as np

prices = np.array([100.0, 102.0, 98.0, 101.0, 99.0, 103.0, 105.0, 104.0])
rsi = SignalProcessor.rsi(prices, period=14)
# Returns: values in [0, 100]
```

### MACD

```python
macd_line, signal_line, histogram = SignalProcessor.macd(
    prices, fast=12, slow=26, signal_period=9
)
```

### Bollinger Bands

```python
pct_b, bandwidth = SignalProcessor.bollinger_bands(prices, window=20, n_std=2.0)
# pct_b: where price sits within the band [0, 1]
# bandwidth: width of the band
```

### ATR – Average True Range

```python
atr = SignalProcessor.atr(high, low, close, period=14)
```

### Stochastic Oscillator

```python
k, d = SignalProcessor.stochastic(high, low, close, k_period=14, d_period=3)
# Both in [0, 100]
```

### ADX – Average Directional Index

```python
adx = SignalProcessor.adx(high, low, close, period=14)
# Returns: values in [0, 100], measures trend strength
```

### OBV – On-Balance Volume

```python
obv = SignalProcessor.obv(close, volume)
```

### VWAP – Volume-Weighted Average Price

```python
vwap = SignalProcessor.vwap(high, low, close, volume)
```

### ROC – Rate of Change

```python
roc = SignalProcessor.roc(prices, period=5)
# (price[t] - price[t-period]) / price[t-period]
```

### Realized Volatility

```python
vol = SignalProcessor.realized_volatility(prices, window=20)
# Rolling std of log returns
```

### Volatility Ratio

```python
ratio = SignalProcessor.vol_ratio(prices, short_window=5, long_window=20)
# Short-term vol / long-term vol
```

### Order Flow Imbalance

```python
ofi = SignalProcessor.order_flow_imbalance(close, volume)
# Volume signed by price direction
```

### Directional Change State Machine

```python
dc = SignalProcessor.dc_state_machine(prices, threshold=0.005)
# Returns dict: dc_direction, overshoot, bars_since_event
```

The DC state machine tracks intrinsic time: it emits an event when price moves by `threshold` from its last extreme point.

## Original Signals

These methods existed before the quant expansion and remain available:

### Volatility

```python
vol = SignalProcessor.volatility(values, window=20)
# Rolling standard deviation
```

### Entropy

```python
ent = SignalProcessor.entropy(values, bins=10)
# Returns: scalar float (Shannon entropy, base 2)
```

### Price Momentum

```python
mom = SignalProcessor.price_momentum(prices, window=10)
# Rate of change over window bars
```

### Directional Change

```python
dc = SignalProcessor.directional_change(prices, threshold=0.01)
# Returns: 1 (up), -1 (down), 0 (none)
```

### Bars Since Significant Move

```python
bars = SignalProcessor.bars_since_significant_move(prices, vol_window=20)
# Count of bars since last move exceeding vol_multiplier * rolling_std
```

### Regime Detection

```python
regime = SignalProcessor.regime_detection(volatility, vol_threshold=0.5)
# Returns: 0 (calm), 1 (normal), 2 (turbulent)
```

## Full Pipeline Example

```python
from core.signals import SignalProcessor, ASSET_CONFIGS
import numpy as np

# Assume OHLCV arrays loaded
prices = np.array([...])
high = np.array([...])
low = np.array([...])
close = np.array([...])
volumes = np.array([...])
fees = np.full_like(prices, 0.0002)

# Compute all 32 features
signals = SignalProcessor.process_sequence(
    volumes=volumes,
    prices=close,
    fees=fees,
    high=high,
    low=low,
    asset_config=ASSET_CONFIGS["forex"],
)

# signals is an ordered dict with keys matching DataConfig.features
print(list(signals.keys()))
```

## Integration with Data Pipeline

```python
from data.pipeline import TransactionPipeline
from core.config import DignityConfig
from core.signals import ASSET_CONFIGS

config = DignityConfig.from_yaml("config/base.yaml")
pipeline = TransactionPipeline(seq_len=config.data.seq_len, features=config.data.features)

# compute_signals uses SignalProcessor.process_sequence internally
df_with_signals = pipeline.compute_signals(df, asset_config=ASSET_CONFIGS["forex"])

# Fit scaler and transform
pipeline.fit(df_with_signals)
X_scaled = pipeline.transform(df_with_signals)
```

## Best Practices

1. Use asset-specific configs – A DC threshold of 0.005 works for crypto but is too large for forex.
2. Handle warmup bars – Early values use expanding windows; the first valid value fills warmup positions.
3. Avoid look-ahead bias – Only use historical data when computing signals for training.
4. Normalize features – The pipeline applies RobustScaler by default.
5. Check feature count – The default feature list has 31 entries (the 32nd is `regime`, included separately by the pipeline).

## Next Steps

- [Privacy Operations](PRIVACY.md) – Combine signals with privacy
- [Configuration Guide](CONFIGURATION.md) – Configure signal parameters
- [Architecture Overview](ARCHITECTURE.md) – Understand the full data flow

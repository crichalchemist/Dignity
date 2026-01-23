# Signal Processing

Dignity Core provides financial and statistical signal computation for transaction data.

## Overview

The `core.signals` module computes:

- **Volatility** - Rolling standard deviation and variance
- **Entropy** - Information-theoretic measures of randomness
- **Momentum** - Directional trend indicators
- **Regime Detection** - Market state classification

## Volatility Signals

### Rolling Volatility

```python
from core.signals import compute_volatility
import pandas as pd

df = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
    "amount": np.random.randn(100).cumsum() + 100
})

# Compute rolling volatility
volatility = compute_volatility(
    df,
    column="amount",
    window=20,  # 20-period rolling window
    method="std"  # Standard deviation
)

print(volatility["volatility"].head())
```

### Volatility Methods

```python
# Standard deviation (default)
vol_std = compute_volatility(df, window=20, method="std")

# Variance
vol_var = compute_volatility(df, window=20, method="variance")

# Exponentially weighted
vol_ewm = compute_volatility(
    df, 
    window=20, 
    method="ewm",
    alpha=0.1  # Decay factor
)
```

**Parameters:**
- `window` - Rolling window size
- `method` - Computation method: std, variance, ewm
- `alpha` - EWM decay factor (0 < alpha < 1)

## Entropy Signals

### Transaction Entropy

```python
from core.signals import compute_entropy

# Shannon entropy of transaction amounts
entropy = compute_entropy(
    df,
    column="amount",
    window=50,
    bins=10  # Discretize into 10 bins
)

print(f"Average entropy: {entropy['entropy'].mean():.3f}")
```

### Conditional Entropy

```python
# Entropy conditioned on time-of-day
entropy = compute_entropy(
    df,
    column="amount",
    window=50,
    bins=10,
    condition_column="hour_of_day"
)
```

**Interpretation:**
- **High entropy** - Uniform distribution, high uncertainty
- **Low entropy** - Concentrated distribution, predictable patterns

## Momentum Signals

### Price Momentum

```python
from core.signals import compute_momentum

# Compute momentum over different horizons
momentum = compute_momentum(
    df,
    column="amount",
    periods=[5, 10, 20]  # Multiple time horizons
)

# Returns: momentum_5, momentum_10, momentum_20
```

### Rate of Change

```python
# Percentage change momentum
momentum = compute_momentum(
    df,
    column="amount",
    periods=[10],
    method="pct_change"
)

# Absolute difference momentum
momentum = compute_momentum(
    df,
    column="amount",
    periods=[10],
    method="diff"
)
```

## Regime Detection

### Volatility Regime

```python
from core.signals import detect_regime

# Classify into low/medium/high volatility regimes
regimes = detect_regime(
    df,
    column="amount",
    method="volatility",
    window=20,
    thresholds=[0.33, 0.67]  # Quantile thresholds
)

# Returns: 0 (low), 1 (medium), 2 (high)
print(regimes["regime"].value_counts())
```

### Trend Regime

```python
# Classify as uptrend, sideways, downtrend
regimes = detect_regime(
    df,
    column="amount",
    method="trend",
    window=20
)

# Returns: -1 (downtrend), 0 (sideways), 1 (uptrend)
```

### Hidden Markov Model Regime

```python
# HMM-based regime detection
regimes = detect_regime(
    df,
    column="amount",
    method="hmm",
    num_states=3  # Number of hidden states
)
```

## Combined Signal Pipeline

```python
from core.signals import (
    compute_volatility,
    compute_entropy,
    compute_momentum,
    detect_regime
)

def compute_all_signals(df):
    """Compute comprehensive signal suite"""
    
    # Volatility
    df = compute_volatility(df, column="amount", window=20)
    
    # Entropy
    df = compute_entropy(df, column="amount", window=50, bins=10)
    
    # Momentum (multiple horizons)
    df = compute_momentum(df, column="amount", periods=[5, 10, 20])
    
    # Regime detection
    df = detect_regime(df, column="amount", method="volatility")
    
    return df

# Apply to data
df_with_signals = compute_all_signals(df)
print(df_with_signals.columns)
# ['timestamp', 'amount', 'volatility', 'entropy', 
#  'momentum_5', 'momentum_10', 'momentum_20', 'regime']
```

## Signal Normalization

```python
from core.signals import normalize_signals

# Z-score normalization
df_norm = normalize_signals(
    df,
    columns=["volatility", "entropy", "momentum_10"],
    method="zscore"
)

# Min-max scaling
df_norm = normalize_signals(
    df,
    columns=["volatility", "entropy"],
    method="minmax",
    feature_range=(0, 1)
)

# Robust scaling (median and IQR)
df_norm = normalize_signals(
    df,
    columns=["volatility"],
    method="robust"
)
```

## Time-Based Features

### Temporal Patterns

```python
from core.signals import extract_temporal_features

# Extract hour, day-of-week, month
df = extract_temporal_features(
    df,
    timestamp_column="timestamp",
    features=["hour", "dayofweek", "month", "is_weekend"]
)

print(df[["timestamp", "hour", "dayofweek", "is_weekend"]].head())
```

### Cyclical Encoding

```python
# Encode cyclical features (hour, month) as sin/cos
df = extract_temporal_features(
    df,
    timestamp_column="timestamp",
    features=["hour", "month"],
    cyclical=True
)

# Creates: hour_sin, hour_cos, month_sin, month_cos
```

## Cross-Sectional Signals

### Entity-Relative Signals

```python
from core.signals import compute_cross_sectional_signals

# Compute signals relative to peer group
df = compute_cross_sectional_signals(
    df,
    group_column="entity_id",
    value_column="amount",
    signals=["rank", "percentile", "zscore"]
)

# rank: within-group rank
# percentile: percentile within group
# zscore: standardized relative to group
```

## Signal Quality Metrics

```python
from core.signals import compute_signal_quality

quality = compute_signal_quality(
    df,
    signal_columns=["volatility", "momentum_10"],
    target_column="future_return"
)

print(quality)
# {
#   'volatility': {'correlation': 0.32, 'information_ratio': 0.45},
#   'momentum_10': {'correlation': 0.58, 'information_ratio': 0.71}
# }
```

## Integration with Data Pipeline

```python
from data.pipeline import TransactionPipeline
from core.config import DignityConfig

config = DignityConfig.from_yaml("config/base.yaml")

# Configure signal computation
config.signals = {
    "compute_volatility": True,
    "compute_entropy": True,
    "compute_momentum": True,
    "detect_regime": True,
    "volatility_window": 20,
    "entropy_bins": 10,
    "momentum_periods": [5, 10, 20]
}

# Pipeline automatically computes signals
pipeline = TransactionPipeline(config)
data = pipeline.load_and_process("transactions.csv")

# Signals are included in output
print(data.columns)
# Includes: volatility, entropy, momentum_5, momentum_10, momentum_20, regime
```

## Advanced: Custom Signals

```python
from core.signals import register_custom_signal

@register_custom_signal
def compute_custom_signal(df, column, window=20):
    """Custom signal function"""
    # Example: Kurtosis-based signal
    from scipy.stats import kurtosis
    
    rolling_kurtosis = df[column].rolling(window).apply(
        lambda x: kurtosis(x, fisher=True)
    )
    
    df["custom_kurtosis"] = rolling_kurtosis
    return df

# Use in pipeline
df = compute_custom_signal(df, column="amount", window=20)
```

## Performance Optimization

```python
# Vectorized computation for large datasets
import numpy as np

def fast_rolling_volatility(values, window):
    """Optimized rolling volatility"""
    n = len(values)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        result[i] = np.std(values[i-window+1:i+1])
    
    return result

df["volatility"] = fast_rolling_volatility(df["amount"].values, 20)
```

## Best Practices

1. **Choose appropriate windows** - Balance responsiveness vs. stability
2. **Normalize signals** - Use consistent scaling across features
3. **Handle missing values** - Use forward-fill or interpolation
4. **Validate signal quality** - Check correlation with targets
5. **Avoid look-ahead bias** - Only use historical data
6. **Document parameters** - Track window sizes, thresholds

## References

- Shannon, C. E. (1948). *A Mathematical Theory of Communication*
- Mandelbrot, B. (1963). *The Variation of Certain Speculative Prices*
- Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models*

## Next Steps

- **[Privacy Operations](PRIVACY.md)** - Combine signals with privacy
- **[Data Pipeline](API_REFERENCE.md#data)** - Integrate signals in pipeline
- **[Configuration Guide](CONFIGURATION.md)** - Configure signal parameters

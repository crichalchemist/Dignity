# Privacy Operations

Dignity Core provides comprehensive privacy-preserving operations for sensitive transaction data.

## Overview

The `core.privacy` module implements privacy techniques:

- **Identity Protection** - Hash identifiers to prevent re-identification
- **Data Anonymization** - Quantize or generalize sensitive values
- **Differential Privacy** - Add calibrated noise for formal privacy guarantees
- **Secure Aggregation** - Combine data without revealing individuals

## Identity Hashing

### Hash Entity Identifiers

```python
from core.privacy import hash_identifiers
import pandas as pd

df = pd.DataFrame({
    "user_id": ["alice", "bob", "charlie"],
    "merchant_id": ["shop_1", "shop_2", "shop_1"],
    "amount": [100.0, 250.0, 75.0]
})

# Hash sensitive IDs
df_hashed = hash_identifiers(
    df, 
    columns=["user_id", "merchant_id"],
    algorithm="sha256",
    salt="dignity_secret_salt"
)

print(df_hashed["user_id"].head())
# Output: ['2c26b46...', '81b637...', 'ba7816...']
```

**Parameters:**
- `columns` - List of columns to hash
- `algorithm` - Hash function: sha256, sha512, blake2b
- `salt` - Secret salt for additional security

## Amount Anonymization

### Quantization

```python
from core.privacy import anonymize_amounts

# Quantize amounts into bins
df_quantized = anonymize_amounts(
    df,
    columns=["amount"],
    method="quantize",
    bins=10  # Reduce to 10 discrete levels
)

# Original: [100.0, 250.0, 75.0]
# Quantized: [2, 8, 1]  (bin indices)
```

### Generalization

```python
# Generalize to ranges
df_generalized = anonymize_amounts(
    df,
    columns=["amount"],
    method="generalize",
    ranges=[(0, 100), (100, 500), (500, float('inf'))]
)

# Original: [100.0, 250.0, 75.0]
# Generalized: ['0-100', '100-500', '0-100']
```

### Rounding

```python
# Round to nearest value
df_rounded = anonymize_amounts(
    df,
    columns=["amount"],
    method="round",
    precision=10  # Round to nearest 10
)

# Original: [103.45, 257.89, 72.10]
# Rounded: [100.0, 260.0, 70.0]
```

## Differential Privacy

### Add Calibrated Noise

```python
from core.privacy import add_differential_privacy_noise

# Add Laplace noise for epsilon-DP
df_private = add_differential_privacy_noise(
    df,
    columns=["amount"],
    epsilon=1.0,      # Privacy budget (smaller = more private)
    sensitivity=100.0, # Maximum change from single record
    mechanism="laplace"
)

# Noise magnitude: sensitivity / epsilon = 100 / 1.0 = 100
```

### Gaussian Mechanism (for (ε,δ)-DP)

```python
df_private = add_differential_privacy_noise(
    df,
    columns=["amount"],
    epsilon=1.0,
    delta=1e-5,  # Failure probability
    sensitivity=100.0,
    mechanism="gaussian"
)
```

**Privacy Parameters:**
- `epsilon` - Privacy budget (0.1 = strong, 10.0 = weak)
- `delta` - Failure probability (typically 1e-5 to 1e-7)
- `sensitivity` - Maximum influence of single record
- `mechanism` - Noise distribution: laplace, gaussian

## Feature Clipping

```python
from core.privacy import clip_features

# Clip outliers before adding noise
df_clipped = clip_features(
    df,
    columns=["amount"],
    lower=0.0,
    upper=1000.0
)

# Values outside [0, 1000] are clipped
```

## Complete Privacy Pipeline

```python
from core.privacy import (
    hash_identifiers,
    anonymize_amounts,
    add_differential_privacy_noise,
    clip_features
)

# Step 1: Hash identifiers
df = hash_identifiers(df, ["user_id", "merchant_id"])

# Step 2: Clip outliers
df = clip_features(df, ["amount"], lower=0, upper=10000)

# Step 3: Add differential privacy noise
df = add_differential_privacy_noise(
    df, 
    ["amount"], 
    epsilon=1.0, 
    sensitivity=100.0
)

# Step 4: Quantize for additional anonymization
df = anonymize_amounts(df, ["amount"], method="quantize", bins=20)
```

## Privacy-Preserving Signal Computation

```python
from core.signals import compute_volatility
from core.privacy import add_differential_privacy_noise

# Compute signals
signals = compute_volatility(df, window=10)

# Add noise to signals for privacy
signals = add_differential_privacy_noise(
    signals,
    columns=["volatility"],
    epsilon=2.0,
    sensitivity=0.1
)
```

## Privacy Budget Management

```python
class PrivacyBudget:
    """Track cumulative privacy loss"""
    
    def __init__(self, total_epsilon=10.0):
        self.total_epsilon = total_epsilon
        self.spent_epsilon = 0.0
    
    def spend(self, epsilon):
        if self.spent_epsilon + epsilon > self.total_epsilon:
            raise ValueError("Privacy budget exceeded!")
        self.spent_epsilon += epsilon
    
    def remaining(self):
        return self.total_epsilon - self.spent_epsilon

# Usage
budget = PrivacyBudget(total_epsilon=5.0)

# Operation 1: epsilon=1.0
df1 = add_differential_privacy_noise(df, ["amount"], epsilon=1.0)
budget.spend(1.0)

# Operation 2: epsilon=2.0
df2 = add_differential_privacy_noise(df, ["balance"], epsilon=2.0)
budget.spend(2.0)

print(f"Remaining budget: {budget.remaining()}")  # 2.0
```

## Privacy Guarantees

### Epsilon-Differential Privacy

**Definition:** 
For any two datasets differing in one record, the probability ratio of any output is bounded by:

$$
\\Pr[M(D_1) \\in S] \\leq e^{\\epsilon} \\cdot \\Pr[M(D_2) \\in S]
$$

**Interpretation:**
- ε = 0.1: Very strong privacy
- ε = 1.0: Strong privacy (recommended)
- ε = 5.0: Moderate privacy
- ε = 10.0: Weak privacy

### Composition Theorems

**Sequential Composition:**
Running k mechanisms with budgets ε₁, ..., εₖ gives total privacy loss:
$$
\\epsilon_{\\text{total}} = \\sum_{i=1}^k \\epsilon_i
$$

**Parallel Composition:**
Running mechanisms on disjoint data subsets uses maximum individual budget:
$$
\\epsilon_{\\text{total}} = \\max(\\epsilon_1, ..., \\epsilon_k)
$$

## Best Practices

1. **Choose appropriate ε** - Start with ε=1.0, adjust based on privacy needs
2. **Track budget** - Monitor cumulative privacy loss across operations
3. **Minimize sensitivity** - Clip outliers before adding noise
4. **Use parallel composition** - Process disjoint subsets when possible
5. **Hash before aggregation** - Prevent identifier linkage
6. **Validate privacy** - Test re-identification resistance

## References

- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*
- Abadi, M. et al. (2016). *Deep Learning with Differential Privacy*
- McMahan, H. B. et al. (2017). *Learning Differentially Private Recurrent Language Models*

## Next Steps

- **[Signal Processing](SIGNALS.md)** - Combine privacy with signal computation
- **[Data Pipeline](API_REFERENCE.md#data)** - Use privacy in data processing
- **[Configuration Guide](CONFIGURATION.md)** - Configure privacy settings

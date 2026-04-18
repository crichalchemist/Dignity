# Privacy Operations

Dignity Core provides privacy-preserving operations for sensitive transaction data through the `PrivacyManager` class in `core/privacy.py`.

## Overview

`PrivacyManager` implements five privacy techniques:

- **Hashing** – SHA-256 identifier hashing with configurable salt
- **Anonymization** – Batch address anonymization
- **Quantization** – k-anonymity via amount binning
- **Differential Privacy** – Laplace noise injection
- **Rare Event Suppression** – k-threshold filtering

## Identity Hashing

```python
from core.privacy import PrivacyManager

# Hash a single identifier
hashed = PrivacyManager.hash_identifier("0x1234abcd5678ef90", salt="secure_salt")
# Returns: SHA-256 hex string

# Hash without salt
hashed = PrivacyManager.hash_identifier("0x1234abcd5678ef90")
```

The `salt` parameter prepends a secret string before hashing. Use a consistent salt across a dataset to preserve referential integrity while preventing re-identification.

## Address Anonymization

```python
# Batch hash a list of addresses
addresses = ["0xabc123", "0xdef456", "0xghi789"]
anonymized = PrivacyManager.anonymize_addresses(addresses, salt="secure_salt")
# Returns: list of SHA-256 hex strings
```

Each address is hashed independently via `hash_identifier`. The salt parameter is optional but recommended.

## Amount Quantization

Quantization provides k-anonymity by grouping transaction amounts into discrete bins.

```python
import numpy as np

amounts = np.array([123.456, 789.012, 456.789, 234.567])

# Quantize into 10 bins
quantized = PrivacyManager.quantize_amounts(amounts, bins=10)
# Returns: bin center values

# Custom range
quantized = PrivacyManager.quantize_amounts(
    amounts, bins=20, min_val=0.0, max_val=1000.0
)
```

The function maps each value to its bin center. Values outside the `[min_val, max_val]` range are clipped to the nearest bin.

## Differential Privacy

Add calibrated Laplace noise for formal epsilon-differential privacy guarantees.

```python
import numpy as np

amounts = np.array([100.0, 250.0, 75.0, 500.0])

# Add Laplace noise (epsilon=1.0, sensitivity=1.0)
noisy = PrivacyManager.add_noise(amounts, epsilon=1.0, sensitivity=1.0)

# Stronger privacy (smaller epsilon = more noise)
noisy = PrivacyManager.add_noise(amounts, epsilon=0.1, sensitivity=1.0)
```

**Parameters:**
- `epsilon` – Privacy budget. Smaller values add more noise (stronger privacy).
- `sensitivity` – Maximum change in the output from a single record. Controls noise scale as `sensitivity / epsilon`.

## Rare Event Suppression

Prevent identification through unique or rare transactions.

```python
import numpy as np

values = np.array([1, 1, 1, 2, 2, 3, 4, 5, 5, 5])

# Suppress values appearing fewer than 3 times
suppressed = PrivacyManager.suppress_rare_events(values, threshold=3)
# Returns: [1, 1, 1, 2, 2, -1, -1, 5, 5, 5]
# (3 and 4 replaced with -1)
```

Rare values are replaced with `-1`. Choose a threshold based on your k-anonymity requirements.

## Full Sanitization Pipeline

`sanitize_dataset` chains quantization, noise injection, and address anonymization:

```python
import numpy as np

volumes = np.array([100.0, 250.0, 75.0, 500.0])
addresses = ["0xabc", "0xdef", "0xghi", "0xjkl"]

result = PrivacyManager.sanitize_dataset(
    volumes,
    addresses=addresses,
    epsilon=0.1,
    quantize_bins=10,
)

# result = {
#     "volumes": quantized amounts,
#     "volumes_noisy": quantized + Laplace noise,
#     "addresses": hashed addresses,
# }
```

## Privacy Guarantees

### Epsilon-Differential Privacy

For any two datasets differing in one record, the probability ratio of any output is bounded by:

$$\\Pr[M(D_1) \\in S] \\leq e^{\\epsilon} \\cdot \\Pr[M(D_2) \\in S]$$

**Interpretation:**
- ε = 0.1: Very strong privacy
- ε = 1.0: Strong privacy (recommended starting point)
- ε = 5.0: Moderate privacy
- ε = 10.0: Weak privacy

### Composition

**Sequential composition:** Running k mechanisms with budgets ε₁, ..., εₖ gives total privacy loss ε_total = Σεᵢ.

**Parallel composition:** Running mechanisms on disjoint data subsets uses the maximum individual budget.

## Best Practices

1. Choose appropriate ε – Start with ε=1.0, adjust based on privacy needs and utility requirements.
2. Minimize sensitivity – Clip outliers before adding noise to reduce noise magnitude.
3. Use consistent salts – Preserve referential integrity across hashed identifiers.
4. Combine techniques – Quantization plus noise provides layered protection.
5. Suppress rare events – Set a k-threshold appropriate for your dataset size.

## References

- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*
- Abadi, M. et al. (2016). *Deep Learning with Differential Privacy*

## Next Steps

- [Signal Processing](SIGNALS.md) – Combine privacy with signal computation
- [Configuration Guide](CONFIGURATION.md) – Configure privacy parameters
- [Architecture Overview](ARCHITECTURE.md) – Understand the full data pipeline

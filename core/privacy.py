"""Privacy-preserving utilities for transaction data.

Two families of primitives, each backed by a test in tests/test_privacy.py:

- Keyed pseudonymization (HMAC-SHA256): same key -> linkable, different key ->
  unlinkable.
- Input-level differential privacy: bounded Laplace noise spent against an
  epsilon ledger, and k-anonymous generalization by quantile binning.

Nothing here claims more than its test proves. docs/PRIVACY.md and
docs/THREAT-MODEL.md carry the exact wording of each guarantee.
"""

import hashlib
import hmac
import math
import secrets

import numpy as np

_MIN_KEY_BYTES = 16


class BudgetExhausted(ValueError):  # noqa: N818
    """Raised when a spend would exceed the configured epsilon total."""


class PrivacyBudget:
    """Sequential-composition ledger for epsilon-differential privacy.

    Epsilons add across releases. A spend that would push the total past
    ``epsilon_total`` raises instead of degrading the guarantee silently.
    """

    def __init__(self, epsilon_total: float):
        if epsilon_total <= 0:
            raise ValueError("epsilon_total must be positive")
        self.epsilon_total = float(epsilon_total)
        self._spent = 0.0

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self.epsilon_total - self._spent

    def spend(self, epsilon: float) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self._spent + epsilon > self.epsilon_total + 1e-12:
            raise BudgetExhausted(
                f"spending {epsilon} would exceed budget: "
                f"{self._spent:.6f} spent of {self.epsilon_total:.6f}"
            )
        self._spent += epsilon


class PrivacyManager:
    """Privacy primitives bound to one budget, one optional key, and one RNG.

    ``rng`` is any object exposing ``random() -> float`` in [0, 1). Production
    uses ``secrets.SystemRandom``; tests inject ``random.Random(seed)`` or a
    constant for determinism without weakening the default.
    """

    def __init__(
        self,
        budget: PrivacyBudget,
        key: bytes | str | None = None,
        rng=None,
    ):
        if isinstance(key, str):
            key = key.encode("utf-8")
        if key is not None and len(key) < _MIN_KEY_BYTES:
            raise ValueError(f"key must be at least {_MIN_KEY_BYTES} bytes")
        self.budget = budget
        self._key = key
        self._rng = rng if rng is not None else secrets.SystemRandom()

    # -- keyed pseudonymization ---------------------------------------------

    def pseudonymize(self, identifier: str) -> str:
        """HMAC-SHA256 of ``identifier`` under the manager's key, as hex.

        Same key -> same pseudonym (linkable). Different key -> unrelated
        pseudonym (unlinkable). Requires a key; never falls back to unkeyed.
        """
        if self._key is None:
            raise ValueError("pseudonymize requires a key")
        return hmac.new(
            self._key, identifier.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def pseudonymize_many(self, identifiers: list[str]) -> list[str]:
        return [self.pseudonymize(i) for i in identifiers]

    # -- differential privacy -------------------------------------------------

    def add_laplace_noise(
        self,
        values: np.ndarray,
        *,
        epsilon: float,
        bounds: tuple[float, float],
    ) -> np.ndarray:
        """Clip ``values`` to ``bounds`` and add Laplace noise for epsilon-DP.

        Sensitivity is the width of ``bounds``; the caller cannot understate it
        by omission. ``epsilon`` is spent from the budget before any sampling,
        so a refused spend releases nothing. Floating-point implementation:
        see docs/THREAT-MODEL.md for the Mironov (2012) caveat.

        Returns a new array; ``values`` is not modified.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        lo, hi = bounds
        if lo >= hi:
            raise ValueError("bounds must satisfy lo < hi")
        self.budget.spend(epsilon)
        scale = (hi - lo) / epsilon
        clipped = np.clip(np.asarray(values, dtype=float), lo, hi)
        noise = np.fromiter(
            (self._laplace(scale) for _ in range(clipped.size)),
            dtype=float,
            count=clipped.size,
        ).reshape(clipped.shape)
        return clipped + noise

    def _laplace(self, scale: float) -> float:
        """One Laplace(0, scale) draw by inverse CDF from a uniform in [0, 1)."""
        while True:
            u = self._rng.random() - 0.5
            if abs(u) < 0.5:
                break
        return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    @staticmethod
    def quantize_amounts(
        amounts: np.ndarray,
        bins: int = 10,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> np.ndarray:
        """
        Quantize transaction amounts to reduce granularity.

        This provides k-anonymity by grouping similar amounts.

        Args:
            amounts: Array of transaction amounts
            bins: Number of quantization bins
            min_val: Minimum value for binning (default: array min)
            max_val: Maximum value for binning (default: array max)

        Returns:
            Quantized amounts (bin centers)
        """
        if len(amounts) == 0:
            return amounts

        if min_val is None:
            min_val = np.min(amounts)
        if max_val is None:
            max_val = np.max(amounts)

        # Create bin edges
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Digitize into bins
        bin_indices = np.digitize(amounts, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        # Map to bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        quantized = bin_centers[bin_indices]

        return quantized

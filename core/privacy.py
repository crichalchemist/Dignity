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
    """Manage privacy-preserving operations on transaction data."""

    @staticmethod
    def hash_identifier(identifier: str, salt: str | None = None) -> str:
        """
        Hash an identifier (address, user ID) using SHA-256.

        Args:
            identifier: The identifier to hash
            salt: Optional salt for additional security

        Returns:
            Hexadecimal hash string
        """
        if salt:
            identifier = f"{salt}{identifier}"

        return hashlib.sha256(identifier.encode()).hexdigest()

    @staticmethod
    def anonymize_addresses(addresses: list[str], salt: str | None = None) -> list[str]:
        """Anonymize a list of addresses."""
        return [PrivacyManager.hash_identifier(addr, salt) for addr in addresses]

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

    @staticmethod
    def add_noise(
        values: np.ndarray, epsilon: float = 0.1, sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Add Laplace noise for differential privacy.

        Args:
            values: Array to add noise to
            epsilon: Privacy parameter (smaller = more privacy)
            sensitivity: Sensitivity of the function

        Returns:
            Values with added noise
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")

        # Laplace noise scale
        scale = sensitivity / epsilon

        # Generate Laplace noise
        noise = np.random.laplace(0, scale, size=values.shape)

        return values + noise

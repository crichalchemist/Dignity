"""Privacy-preserving utilities for transaction data."""

import hashlib

import numpy as np


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

    @staticmethod
    def suppress_rare_events(values: np.ndarray, threshold: int = 5) -> np.ndarray:
        """
        Suppress rare values that appear fewer than threshold times.

        This prevents identification through unique transactions.

        Args:
            values: Array of values
            threshold: Minimum count to keep

        Returns:
            Array with rare values replaced by -1
        """
        unique, counts = np.unique(values, return_counts=True)
        rare_values = unique[counts < threshold]

        result = values.copy()
        for rare in rare_values:
            result[result == rare] = -1

        return result

    @classmethod
    def sanitize_dataset(
        cls,
        volumes: np.ndarray,
        addresses: list[str] | None = None,
        epsilon: float = 0.1,
        quantize_bins: int = 10,
    ) -> dict:
        """
        Apply multiple privacy-preserving transformations.

        Args:
            volumes: Transaction volumes
            addresses: Optional transaction addresses
            epsilon: Differential privacy parameter
            quantize_bins: Number of bins for quantization

        Returns:
            Dictionary with sanitized data
        """
        sanitized = {}

        # Quantize volumes
        sanitized["volumes"] = cls.quantize_amounts(volumes, bins=quantize_bins)

        # Add differential privacy noise
        sanitized["volumes_noisy"] = cls.add_noise(
            sanitized["volumes"], epsilon=epsilon
        )

        # Anonymize addresses if provided
        if addresses is not None:
            sanitized["addresses"] = cls.anonymize_addresses(addresses)

        return sanitized

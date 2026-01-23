"""Signal processing utilities for transaction sequences."""

import numpy as np
from scipy.stats import entropy as scipy_entropy


class SignalProcessor:
    """Process raw transaction data into meaningful signals."""

    @staticmethod
    def volatility(values: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate rolling volatility (standard deviation)."""
        if len(values) < window:
            return np.zeros_like(values)

        result = np.zeros_like(values)
        for i in range(window, len(values)):
            result[i] = np.std(values[i - window : i])

        # Fill initial values with first valid volatility
        if window < len(values):
            result[:window] = result[window]

        return result

    @staticmethod
    def entropy(values: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy of value distribution."""
        if len(values) == 0:
            return 0.0

        hist, _ = np.histogram(values, bins=bins)
        hist = hist[hist > 0]  # Remove zero bins

        if len(hist) == 0:
            return 0.0

        return scipy_entropy(hist, base=2)

    @staticmethod
    def price_momentum(prices: np.ndarray, window: int = 10) -> np.ndarray:
        """Calculate price momentum (rate of change)."""
        if len(prices) < window:
            return np.zeros_like(prices)

        result = np.zeros_like(prices)
        for i in range(window, len(prices)):
            result[i] = (prices[i] - prices[i - window]) / prices[i - window]

        return result

    @staticmethod
    def directional_change(prices: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Detect directional changes in price movement.

        Returns array of:
        - 1: Upward directional change
        - -1: Downward directional change
        - 0: No significant change
        """
        if len(prices) < 2:
            return np.zeros_like(prices)

        changes = np.diff(prices) / prices[:-1]
        directions = np.zeros(len(prices))

        # Mark significant changes
        directions[1:][changes > threshold] = 1
        directions[1:][changes < -threshold] = -1

        return directions

    @staticmethod
    def regime_detection(
        volatility: np.ndarray, vol_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Detect market regime based on volatility.

        Returns:
        - 0: Low volatility (calm)
        - 1: Normal volatility
        - 2: High volatility (turbulent)
        """
        if len(volatility) == 0:
            return np.zeros(0, dtype=int)

        # Normalize volatility
        vol_mean = np.mean(volatility)
        vol_std = np.std(volatility)

        if vol_std == 0:
            return np.ones(len(volatility), dtype=int)

        normalized_vol = (volatility - vol_mean) / vol_std

        regimes = np.ones(len(volatility), dtype=int)
        regimes[normalized_vol < -vol_threshold] = 0  # Low vol
        regimes[normalized_vol > vol_threshold] = 2  # High vol

        return regimes

    @classmethod
    def process_sequence(
        cls,
        volumes: np.ndarray,
        prices: np.ndarray | None = None,
        fees: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Process a transaction sequence into multiple signals.

        Args:
            volumes: Transaction volumes
            prices: Optional price data
            fees: Optional fee data

        Returns:
            Dictionary of computed signals
        """
        signals = {}

        # Volume-based signals
        signals["volume"] = volumes
        signals["volume_volatility"] = cls.volatility(volumes)
        signals["volume_entropy"] = np.array(
            [cls.entropy(volumes[max(0, i - 50) : i + 1]) for i in range(len(volumes))]
        )

        # Price-based signals (if available)
        if prices is not None:
            signals["price"] = prices
            signals["price_change"] = np.gradient(prices)
            signals["momentum"] = cls.price_momentum(prices)
            signals["volatility"] = cls.volatility(prices)
            signals["directional_change"] = cls.directional_change(prices)
            signals["regime"] = cls.regime_detection(cls.volatility(prices))

        # Fee-based signals (if available)
        if fees is not None:
            signals["fee_rate"] = fees
            signals["fee_volatility"] = cls.volatility(fees)

        return signals

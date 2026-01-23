"""Cryptocurrency data source interface."""

import pandas as pd


class CryptoSource:
    """
    Interface for cryptocurrency transaction data.

    Can pull from:
    - CCXT (exchange data)
    - On-chain APIs (Blockchair, Etherscan, etc.)
    - Local CSV files
    """

    def __init__(self, pair: str = "BTC/USD"):
        """
        Args:
            pair: Trading pair (e.g., 'BTC/USD', 'XMR/USD')
        """
        self.pair = pair

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """
        Load cryptocurrency data from CSV file.

        Expected columns:
        - timestamp (will be normalized to milliseconds since epoch)
        - open, high, low, close, volume
        - Optional: tx_count, fee_rate

        Args:
            path: Path to CSV file

        Returns:
            DataFrame with cryptocurrency data (timestamps normalized to milliseconds)
        """
        df = pd.read_csv(path)

        # Ensure required columns
        required = ["timestamp", "close", "volume"]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Normalize timestamps to milliseconds
        df["timestamp"] = self._normalize_timestamp(df["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw cryptocurrency data.

        Args:
            df: Raw dataframe with price/volume data

        Returns:
            DataFrame with engineered features
        """
        prepared = df.copy()

        # Price-based features
        if "close" in df.columns:
            prepared["price"] = df["close"]
            prepared["price_change"] = df["close"].pct_change().fillna(0.0)
            prepared["volatility"] = df["close"].rolling(20).std().fillna(0.0)

        # Volume features
        if "volume" in df.columns:
            prepared["volume_ma"] = df["volume"].rolling(20).mean().fillna(df["volume"])
            prepared["volume_std"] = df["volume"].rolling(20).std().fillna(0.0)

        # High-low spread (if available)
        if "high" in df.columns and "low" in df.columns:
            prepared["hl_spread"] = (df["high"] - df["low"]) / df["close"]

        return prepared

    def resample_to_blocks(
        self, df: pd.DataFrame, block_time: str = "10min"
    ) -> pd.DataFrame:
        """
        Resample exchange data to block-like intervals.

        Args:
            df: Dataframe with timestamp index
            block_time: Block time interval (e.g., '10min', '1h')

        Returns:
            Resampled DataFrame
        """
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Resample
        resampled = (
            df.resample(block_time)
            .agg({"close": "last", "volume": "sum", "high": "max", "low": "min"})
            .ffill()  # Forward fill (fillna(method="ffill") deprecated in pandas 2.0+)
        )

        return resampled.reset_index()

    def _normalize_timestamp(self, ts_series: pd.Series) -> pd.Series:
        """
        Normalize timestamps to milliseconds since epoch.

        Detects and converts from multiple formats:
        - Seconds (< 2e10): multiplied by 1000
        - Milliseconds (2e10 to 3e13): passed through unchanged
        - Nanoseconds (> 3e13): divided by 1e6
        - Datetime strings: converted via pd.to_datetime then to ms

        Args:
            ts_series: Series of timestamps in potentially mixed formats

        Returns:
            Series with all timestamps normalized to milliseconds since epoch
        """
        # Try converting from datetime string first
        if ts_series.dtype == "object":
            try:
                ts_series = pd.to_datetime(ts_series).astype("int64") // 10**6
                return ts_series
            except (ValueError, TypeError):
                pass

        # Convert numeric timestamps
        numeric_ts = pd.to_numeric(ts_series, errors="coerce")

        # Check for invalid (non-convertible) timestamp values
        if numeric_ts.isna().any():
            invalid_values = ts_series[numeric_ts.isna()]
            # Show a small sample of invalid values to aid debugging
            sample = invalid_values.head().tolist()
            raise ValueError(
                f"Invalid timestamp values encountered during normalization: {sample}. "
                "Ensure all timestamps are numeric or valid datetime-like values."
            )

        # Detect format based on magnitude
        max_val = numeric_ts.max()

        if max_val < 2e10:  # Seconds
            return (numeric_ts * 1000).astype("int64")
        elif max_val > 3e13:  # Nanoseconds
            return (numeric_ts / 10**6).astype("int64")
        else:  # Already in milliseconds
            return numeric_ts.astype("int64")


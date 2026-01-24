"""Feature engineering pipeline for technical indicators.

This module computes technical indicators (returns, volatility, RSI, MACD, volume MA)
for multi-source aligned data.
"""

from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Compute technical indicators for all assets.

    Features computed:
        - Returns: Log returns
        - Volatility: Rolling standard deviation (60-min window)
        - RSI: Relative Strength Index (14-period)
        - MACD: Moving Average Convergence Divergence (12/26/9)
        - Volume MA: Moving average of volume (20-period)

    Example:
        >>> from data.processors.alignment import TimestampAligner
        >>> from data.source.crypto_historical import generate_synthetic_crypto_data
        >>> 
        >>> # Get aligned data
        >>> df_crypto = generate_synthetic_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02')
        >>> aligner = TimestampAligner()
        >>> aligned = aligner.align({'BTC/USDT': df_crypto})
        >>> 
        >>> # Compute features
        >>> engineer = FeatureEngineer()
        >>> features = engineer.compute_all_features(aligned)
        >>> 
        >>> # Access computed features
        >>> btc_returns = features[('BTC/USDT', 'returns')]
        >>> btc_rsi = features[('BTC/USDT', 'rsi')]
    """

    def compute_all_features(
        self,
        df: pd.DataFrame,
        price_cols: Optional[list[str]] = None,
        volume_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute all technical indicators for aligned data.

        Args:
            df: Aligned DataFrame with multi-level columns (source, feature).
                Expected features: open, high, low, close, volume
            price_cols: List of feature names to use as price (default: ['close'])
            volume_cols: List of feature names to use as volume (default: ['volume'])

        Returns:
            DataFrame with original columns plus new feature columns:
                - (source, 'returns'): Log returns
                - (source, 'volatility'): Rolling std of returns (60-period)
                - (source, 'rsi'): Relative Strength Index (14-period)
                - (source, 'macd'): MACD line
                - (source, 'macd_signal'): MACD signal line
                - (source, 'volume_ma'): Volume moving average (20-period)

        Example:
            >>> engineer = FeatureEngineer()
            >>> features = engineer.compute_all_features(aligned_df)
            >>> 
            >>> # Check new columns
            >>> assert ('BTC/USDT', 'returns') in features.columns
            >>> assert ('BTC/USDT', 'rsi') in features.columns
        """
        # Default to 'close' for price and 'volume' for volume
        if price_cols is None:
            price_cols = ['close']
        if volume_cols is None:
            volume_cols = ['volume']

        # Start with copy of input
        result = df.copy()

        # Extract sources from multi-level columns
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Input DataFrame must have multi-level columns (source, feature)")

        sources = df.columns.get_level_values(0).unique()

        # Compute features for each source
        for source in sources:
            # Get price and volume data for this source
            source_data = df[source]

            # Compute features for each price column
            for price_col in price_cols:
                if price_col in source_data.columns:
                    prices = source_data[price_col]

                    # 1. Returns: log(price_t / price_{t-1})
                    returns = self._compute_returns(prices)
                    result[(source, 'returns')] = returns

                    # 2. Volatility: rolling std of returns (60-period)
                    volatility = self._compute_volatility(returns, window=60)
                    result[(source, 'volatility')] = volatility

                    # 3. RSI: Relative Strength Index (14-period)
                    rsi = self._compute_rsi(prices, period=14)
                    result[(source, 'rsi')] = rsi

                    # 4. MACD: Moving Average Convergence Divergence
                    macd, signal = self._compute_macd(prices, fast=12, slow=26, signal_period=9)
                    result[(source, 'macd')] = macd
                    result[(source, 'macd_signal')] = signal

            # Compute volume features
            for volume_col in volume_cols:
                if volume_col in source_data.columns:
                    volumes = source_data[volume_col]

                    # 5. Volume MA: moving average (20-period)
                    volume_ma = self._compute_volume_ma(volumes, window=20)
                    result[(source, 'volume_ma')] = volume_ma

        return result

    def _compute_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute log returns.

        Args:
            prices: Price series

        Returns:
            Log returns: ln(price_t / price_{t-1})
        """
        # Use numpy log for better performance
        returns = np.log(prices / prices.shift(1))
        # Forward-fill first NaN
        return returns.ffill().fillna(0)

    def _compute_volatility(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """
        Compute rolling volatility.

        Args:
            returns: Returns series
            window: Rolling window size (default: 60)

        Returns:
            Rolling standard deviation of returns
        """
        volatility = returns.rolling(window=window, min_periods=1).std()
        # Forward-fill NaN values
        return volatility.ffill().fillna(0)

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).

        Formula:
            RSI = 100 - (100 / (1 + RS))
            RS = Average Gain / Average Loss

        Args:
            prices: Price series
            period: RSI period (default: 14)

        Returns:
            RSI values (0-100)
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using EMA
        avg_gains = gains.ewm(span=period, min_periods=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, min_periods=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero (when avg_losses = 0)
        rsi = rsi.fillna(100)  # If no losses, RSI = 100

        # Forward-fill initial NaN values
        return rsi.ffill().fillna(50)  # Default to neutral 50

    def _compute_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Compute MACD (Moving Average Convergence Divergence).

        Formula:
            MACD Line = EMA(fast) - EMA(slow)
            Signal Line = EMA(MACD, signal_period)

        Args:
            prices: Price series
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)

        Returns:
            Tuple of (MACD line, Signal line)
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, min_periods=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, min_periods=slow, adjust=False).mean()

        # MACD line
        macd = ema_fast - ema_slow

        # Signal line
        signal = macd.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()

        # Forward-fill NaN values
        macd = macd.ffill().fillna(0)
        signal = signal.ffill().fillna(0)

        return macd, signal

    def _compute_volume_ma(self, volumes: pd.Series, window: int = 20) -> pd.Series:
        """
        Compute volume moving average.

        Args:
            volumes: Volume series
            window: Moving average window (default: 20)

        Returns:
            Simple moving average of volume
        """
        volume_ma = volumes.rolling(window=window, min_periods=1).mean()
        # Forward-fill NaN values
        return volume_ma.ffill().fillna(0)

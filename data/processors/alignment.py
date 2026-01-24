"""Timestamp alignment processor for multi-source data.

This module provides functionality to align data from multiple sources
(crypto, forex, etc.) to a common timestamp grid with 1-minute intervals.
"""

from typing import Literal

import numpy as np
import pandas as pd


class TimestampAligner:
    """
    Align multi-source data to common timestamp grid.

    Features:
        - Reference grid: 1-minute intervals (60000 ms)
        - Fill missing values: Forward-fill, backward-fill, or interpolate
        - Multi-source support: Handle different timestamp ranges
        - Efficient: Uses pandas reindex for performance

    Example:
        >>> from data.source.crypto_historical import generate_synthetic_crypto_data
        >>> from data.source.forex_historical import generate_synthetic_forex_data
        >>> 
        >>> # Load data from multiple sources
        >>> df_crypto = generate_synthetic_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02')
        >>> df_forex = generate_synthetic_forex_data('EUR/USD', '2024-01-01', '2024-01-02')
        >>> 
        >>> # Align to common timestamp grid
        >>> aligner = TimestampAligner()
        >>> aligned = aligner.align({
        ...     'BTC/USDT': df_crypto,
        ...     'EUR/USD': df_forex
        ... })
        >>> 
        >>> # Access aligned data with multi-level columns
        >>> btc_close = aligned[('BTC/USDT', 'close')]
        >>> eur_close = aligned[('EUR/USD', 'close')]
    """

    def align(
        self,
        data_sources: dict[str, pd.DataFrame],
        method: Literal["ffill", "bfill", "interpolate"] = "ffill",
    ) -> pd.DataFrame:
        """
        Align multiple data sources to common timestamp grid.

        Args:
            data_sources: Dictionary mapping source names to DataFrames.
                Each DataFrame must have columns: [timestamp, open, high, low, close, volume]
            method: Fill method for missing values:
                - 'ffill': Forward fill (use last valid value)
                - 'bfill': Backward fill (use next valid value)
                - 'interpolate': Linear interpolation

        Returns:
            DataFrame with:
                - Index: timestamps (1-minute intervals in milliseconds)
                - Columns: Multi-level (source, feature) tuples
                    Example: ('BTC/USDT', 'close'), ('EUR/USD', 'open')

        Raises:
            ValueError: If no data sources provided or data format is invalid

        Example:
            >>> aligner = TimestampAligner()
            >>> aligned = aligner.align(
            ...     {'BTC/USDT': df_btc, 'EUR/USD': df_eur},
            ...     method='ffill'
            ... )
            >>> print(aligned.columns)
            MultiIndex([('BTC/USDT', 'open'), ('BTC/USDT', 'high'), ...])
        """
        # Validate inputs
        if not data_sources:
            raise ValueError("No data sources provided")

        # Validate each DataFrame has required columns
        required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        for source_name, df in data_sources.items():
            if len(df) == 0:
                raise ValueError(f"Data source '{source_name}' is empty")
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValueError(
                    f"Source '{source_name}' missing 'timestamp' column or OHLCV columns: {missing}"
                )

        # Step 1: Find global min/max timestamps across all sources
        all_timestamps = []
        for df in data_sources.values():
            all_timestamps.extend(df['timestamp'].values)

        global_min = min(all_timestamps)
        global_max = max(all_timestamps)

        # Step 2: Create reference grid (60000 ms intervals = 1 minute)
        # Ensure grid starts and ends on exact boundaries
        reference_grid = np.arange(global_min, global_max + 60000, 60000, dtype=np.int64)

        # Step 3: Reindex each source to grid and collect aligned DataFrames
        aligned_sources = {}

        for source_name, df in data_sources.items():
            # Set timestamp as index
            df_indexed = df.set_index('timestamp')

            # Reindex to reference grid
            df_reindexed = df_indexed.reindex(reference_grid)

            # Step 4: Apply fill method
            if method == 'ffill':
                df_filled = df_reindexed.ffill()
            elif method == 'bfill':
                df_filled = df_reindexed.bfill()
            elif method == 'interpolate':
                df_filled = df_reindexed.interpolate(method='linear')
            else:
                raise ValueError(f"Invalid fill method: {method}. Use 'ffill', 'bfill', or 'interpolate'")

            # Store with source name
            aligned_sources[source_name] = df_filled

        # Step 5: Concatenate with multi-level columns
        # Create multi-level column structure: (source, feature)
        aligned_dfs = []
        for source_name, df in aligned_sources.items():
            # Add source name as top level to columns
            df_with_source = df.copy()
            df_with_source.columns = pd.MultiIndex.from_product(
                [[source_name], df.columns],
                names=['source', 'feature']
            )
            aligned_dfs.append(df_with_source)

        # Concatenate along columns (axis=1)
        result = pd.concat(aligned_dfs, axis=1)

        # Step 6: Set index name
        result.index.name = 'timestamp'

        return result

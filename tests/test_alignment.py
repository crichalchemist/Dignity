"""Tests for timestamp alignment pipeline."""

import numpy as np
import pandas as pd
import pytest

from data.processors.alignment import TimestampAligner
from data.source.crypto_historical import generate_synthetic_crypto_data
from data.source.forex_historical import generate_synthetic_forex_data


class TestTimestampAligner:
    """Test timestamp alignment pipeline."""

    def test_aligner_initialization(self):
        """Test aligner can be instantiated."""
        aligner = TimestampAligner()
        assert aligner is not None

    def test_align_single_source(self):
        """Test alignment with single data source."""
        # Generate test data (1 hour = 60 candles)
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        )
        # Take only first 60 rows (1 hour)
        df_crypto = df_crypto.head(60).copy()

        aligner = TimestampAligner()
        data_sources = {'BTC/USDT': df_crypto}
        aligned = aligner.align(data_sources)

        # Check output structure
        assert isinstance(aligned, pd.DataFrame)
        assert isinstance(aligned.columns, pd.MultiIndex)
        assert aligned.columns.nlevels == 2

        # Check multi-level columns
        assert ('BTC/USDT', 'open') in aligned.columns
        assert ('BTC/USDT', 'high') in aligned.columns
        assert ('BTC/USDT', 'low') in aligned.columns
        assert ('BTC/USDT', 'close') in aligned.columns
        assert ('BTC/USDT', 'volume') in aligned.columns

        # Check timestamps are 1-minute intervals (60000 ms)
        timestamps = aligned.index.values
        intervals = np.diff(timestamps)
        assert np.all(intervals == 60000), "All intervals should be exactly 60000 ms"

    def test_align_multi_source(self):
        """Test alignment with multiple data sources (crypto + forex)."""
        # Generate test data (1 hour each)
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(60).copy()

        df_forex = generate_synthetic_forex_data(
            pair='EUR/USD',
            start_date='2024-01-01',
            end_date='2024-01-02',
            interval='1min'
        ).head(60).copy()

        aligner = TimestampAligner()
        data_sources = {
            'BTC/USDT': df_crypto,
            'EUR/USD': df_forex
        }
        aligned = aligner.align(data_sources)

        # Check output structure
        assert isinstance(aligned, pd.DataFrame)
        assert isinstance(aligned.columns, pd.MultiIndex)

        # Check both sources are present
        assert ('BTC/USDT', 'close') in aligned.columns
        assert ('EUR/USD', 'close') in aligned.columns

        # Check all features for both sources
        for source in ['BTC/USDT', 'EUR/USD']:
            for feature in ['open', 'high', 'low', 'close', 'volume']:
                assert (source, feature) in aligned.columns

        # Check timestamps are aligned
        timestamps = aligned.index.values
        intervals = np.diff(timestamps)
        assert np.all(intervals == 60000)

    def test_align_different_timestamp_ranges(self):
        """Test alignment when sources have different timestamp ranges."""
        # Generate full day data
        df_crypto_full = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        )
        df_forex_full = generate_synthetic_forex_data(
            pair='EUR/USD',
            start_date='2024-01-01',
            end_date='2024-01-02',
            interval='1min'
        )

        # Crypto: first 120 candles (00:00 to 02:00)
        df_crypto = df_crypto_full.head(120).copy()
        
        # Forex: candles 30 to 90 (00:30 to 01:30)
        df_forex = df_forex_full.iloc[30:90].copy()

        aligner = TimestampAligner()
        data_sources = {
            'BTC/USDT': df_crypto,
            'EUR/USD': df_forex
        }
        aligned = aligner.align(data_sources)

        # Should align to global min/max (00:00 to 02:00)
        min_ts = aligned.index.min()
        max_ts = aligned.index.max()

        expected_min = df_crypto['timestamp'].min()
        expected_max = df_crypto['timestamp'].max()

        assert min_ts == expected_min
        assert max_ts == expected_max

        # EUR/USD should have NaN before 00:30 and after 01:30
        # (before forward fill)

    def test_fill_method_ffill(self):
        """Test forward fill for missing values."""
        # Create data with gaps
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(60).copy()

        # Remove some rows to create gaps
        df_crypto_gapped = df_crypto.iloc[::2].copy()  # Take every 2nd row

        aligner = TimestampAligner()
        data_sources = {'BTC/USDT': df_crypto_gapped}
        aligned = aligner.align(data_sources, method='ffill')

        # Should have no NaN values after ffill (except possibly first row)
        # Count NaN values
        nan_count = aligned[('BTC/USDT', 'close')].isna().sum()
        
        # First row might be NaN if no previous value to fill from
        # But all subsequent rows should be filled
        assert nan_count <= 1  # At most the first row can be NaN

    def test_fill_method_bfill(self):
        """Test backward fill for missing values."""
        # Create data with gaps
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(60).copy()

        # Remove some rows to create gaps
        df_crypto_gapped = df_crypto.iloc[::2].copy()

        aligner = TimestampAligner()
        data_sources = {'BTC/USDT': df_crypto_gapped}
        aligned = aligner.align(data_sources, method='bfill')

        # Should have no NaN values after bfill (except possibly last row)
        nan_count = aligned[('BTC/USDT', 'close')].isna().sum()
        assert nan_count <= 1  # At most the last row can be NaN

    def test_fill_method_interpolate(self):
        """Test linear interpolation for missing values."""
        # Create data with gaps
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(60).copy()

        # Remove some rows to create gaps
        df_crypto_gapped = df_crypto.iloc[::2].copy()

        aligner = TimestampAligner()
        data_sources = {'BTC/USDT': df_crypto_gapped}
        aligned = aligner.align(data_sources, method='interpolate')

        # Interpolation should fill gaps with intermediate values
        # Check that values are reasonable (no huge jumps)
        close_values = aligned[('BTC/USDT', 'close')].dropna().to_numpy()
        if len(close_values) > 1:
            diffs = np.abs(np.diff(close_values))
            # Differences should be reasonable (< 5% of price)
            assert np.all(diffs < close_values[:-1] * 0.05)

    def test_timestamp_grid_precision(self):
        """Test that timestamp grid is exactly 1-minute intervals."""
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(120).copy()  # 2 hours

        aligner = TimestampAligner()
        data_sources = {'BTC/USDT': df_crypto}
        aligned = aligner.align(data_sources)

        timestamps = aligned.index.values

        # Check first timestamp
        assert timestamps[0] == df_crypto['timestamp'].min()

        # Check all intervals are exactly 60000 ms
        intervals = np.diff(timestamps)
        assert np.all(intervals == 60000)

        # Check total duration
        total_duration = timestamps[-1] - timestamps[0]
        expected_duration = (len(timestamps) - 1) * 60000
        assert total_duration == expected_duration

    def test_empty_source_handling(self):
        """Test handling of empty data sources."""
        aligner = TimestampAligner()
        
        # Empty dictionary should raise ValueError
        with pytest.raises(ValueError, match="No data sources provided"):
            aligner.align({})

    def test_missing_timestamp_column(self):
        """Test handling of data without timestamp column."""
        # Create DataFrame without timestamp column
        df_invalid = pd.DataFrame({
            'open': [1.0, 2.0, 3.0],
            'close': [1.1, 2.1, 3.1]
        })

        aligner = TimestampAligner()
        with pytest.raises(ValueError, match="missing 'timestamp' column"):
            aligner.align({'TEST': df_invalid})

    def test_preserve_data_values(self):
        """Test that alignment preserves original data values at original timestamps."""
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(60).copy()

        aligner = TimestampAligner()
        data_sources = {'BTC/USDT': df_crypto}
        aligned = aligner.align(data_sources)

        # Check that original values are preserved
        for _, row in df_crypto.iterrows():
            ts = row['timestamp']
            if ts in aligned.index:
                assert np.isclose(
                    float(aligned.loc[ts, ('BTC/USDT', 'close')]),
                    float(row['close']),
                    rtol=1e-10
                )

    def test_multi_source_three_sources(self):
        """Test alignment with three different sources."""
        df_btc = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m',
            initial_price=40000.0
        ).head(60).copy()

        df_eth = generate_synthetic_crypto_data(
            symbol='ETH/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m',
            initial_price=2500.0
        ).head(60).copy()

        df_eur = generate_synthetic_forex_data(
            pair='EUR/USD',
            start_date='2024-01-01',
            end_date='2024-01-02',
            interval='1min'
        ).head(60).copy()

        aligner = TimestampAligner()
        data_sources = {
            'BTC/USDT': df_btc,
            'ETH/USDT': df_eth,
            'EUR/USD': df_eur
        }
        aligned = aligner.align(data_sources)

        # Check all three sources are present
        assert ('BTC/USDT', 'close') in aligned.columns
        assert ('ETH/USDT', 'close') in aligned.columns
        assert ('EUR/USD', 'close') in aligned.columns

        # Check total number of columns: 3 sources * 5 features = 15
        assert len(aligned.columns) == 15

    def test_index_name(self):
        """Test that aligned DataFrame has properly named index."""
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(60).copy()

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        # Index should be named 'timestamp'
        assert aligned.index.name == 'timestamp'

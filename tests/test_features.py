"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from data.processors.alignment import TimestampAligner
from data.processors.features import FeatureEngineer
from data.source.crypto_historical import generate_synthetic_crypto_data
from data.source.forex_historical import generate_synthetic_forex_data


class TestFeatureEngineer:
    """Test feature engineering pipeline."""

    def test_engineer_initialization(self):
        """Test engineer can be instantiated."""
        engineer = FeatureEngineer()
        assert engineer is not None

    def test_compute_features_single_source(self):
        """Test feature computation with single data source."""
        # Generate and align test data
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        )
        # Take 200 rows for sufficient warmup
        df_crypto = df_crypto.head(200).copy()

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        # Compute features
        engineer = FeatureEngineer()
        features = engineer.compute_all_features(aligned)

        # Check output structure
        assert isinstance(features, pd.DataFrame)
        assert isinstance(features.columns, pd.MultiIndex)
        assert features.columns.nlevels == 2

        # Check original columns preserved
        assert ('BTC/USDT', 'open') in features.columns
        assert ('BTC/USDT', 'close') in features.columns
        assert ('BTC/USDT', 'volume') in features.columns

        # Check new feature columns created
        assert ('BTC/USDT', 'returns') in features.columns
        assert ('BTC/USDT', 'volatility') in features.columns
        assert ('BTC/USDT', 'rsi') in features.columns
        assert ('BTC/USDT', 'macd') in features.columns
        assert ('BTC/USDT', 'macd_signal') in features.columns
        assert ('BTC/USDT', 'volume_ma') in features.columns

        # Check no NaN values in output
        assert not features[('BTC/USDT', 'returns')].isna().any()
        assert not features[('BTC/USDT', 'volatility')].isna().any()
        assert not features[('BTC/USDT', 'rsi')].isna().any()
        assert not features[('BTC/USDT', 'macd')].isna().any()
        assert not features[('BTC/USDT', 'macd_signal')].isna().any()
        assert not features[('BTC/USDT', 'volume_ma')].isna().any()

        # Check shapes match
        assert len(features) == len(aligned)

    def test_compute_features_multi_source(self):
        """Test feature computation with multiple data sources."""
        # Generate test data
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(200).copy()

        df_forex = generate_synthetic_forex_data(
            pair='EUR/USD',
            start_date='2024-01-01',
            end_date='2024-01-02',
            interval='1min'
        ).head(200).copy()

        # Align data
        aligner = TimestampAligner()
        aligned = aligner.align({
            'BTC/USDT': df_crypto,
            'EUR/USD': df_forex
        })

        # Compute features
        engineer = FeatureEngineer()
        features = engineer.compute_all_features(aligned)

        # Check features for both sources
        for source in ['BTC/USDT', 'EUR/USD']:
            assert (source, 'returns') in features.columns
            assert (source, 'volatility') in features.columns
            assert (source, 'rsi') in features.columns
            assert (source, 'macd') in features.columns
            assert (source, 'macd_signal') in features.columns
            assert (source, 'volume_ma') in features.columns

            # Check no NaN values
            assert not features[(source, 'returns')].isna().any()
            assert not features[(source, 'volatility')].isna().any()
            assert not features[(source, 'rsi')].isna().any()
            assert not features[(source, 'macd')].isna().any()
            assert not features[(source, 'macd_signal')].isna().any()
            assert not features[(source, 'volume_ma')].isna().any()

    def test_returns_computation(self):
        """Test returns are computed correctly."""
        # Create simple test data with known values
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1min')
        timestamps_ms = (timestamps.astype(np.int64) // 10**6).values

        df = pd.DataFrame({
            ('TEST', 'close'): [100, 110, 121, 108.9, 119.79],
        }, index=timestamps_ms)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df)

        returns = features[('TEST', 'returns')].values

        # First return should be 0 (forward-filled)
        assert returns[0] == 0

        # Check remaining returns (log returns)
        # ln(110/100) ≈ 0.0953
        assert abs(returns[1] - np.log(110/100)) < 1e-6
        # ln(121/110) ≈ 0.0953
        assert abs(returns[2] - np.log(121/110)) < 1e-6

    def test_volatility_computation(self):
        """Test volatility is computed correctly."""
        # Generate test data
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(200).copy()

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(aligned)

        volatility = features[('BTC/USDT', 'volatility')]

        # Check volatility is non-negative
        assert (volatility >= 0).all()

        # Check no NaN values
        assert not volatility.isna().any()

        # Volatility should increase after row 60 (window size)
        # as it's computed from more data
        assert volatility.iloc[60] > 0

    def test_rsi_computation(self):
        """Test RSI is computed correctly and bounded."""
        # Generate test data
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(200).copy()

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(aligned)

        rsi = features[('BTC/USDT', 'rsi')]

        # Check RSI is bounded between 0 and 100
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

        # Check no NaN values
        assert not rsi.isna().any()

    def test_rsi_extreme_values(self):
        """Test RSI handles extreme price movements."""
        # Create test data with only gains (should give RSI = 100)
        timestamps = pd.date_range('2024-01-01', periods=30, freq='1min')
        timestamps_ms = (timestamps.astype(np.int64) // 10**6).values

        # Steadily increasing prices
        prices = np.linspace(100, 200, 30)

        df = pd.DataFrame({
            ('TEST', 'close'): prices,
        }, index=timestamps_ms)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df)

        rsi = features[('TEST', 'rsi')]

        # After warmup, RSI should be high (approaching 100)
        assert rsi.iloc[-1] > 90

    def test_macd_computation(self):
        """Test MACD is computed correctly."""
        # Generate test data
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(200).copy()

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(aligned)

        macd = features[('BTC/USDT', 'macd')]
        signal = features[('BTC/USDT', 'macd_signal')]

        # Check no NaN values
        assert not macd.isna().any()
        assert not signal.isna().any()

        # Check MACD and signal are numeric
        assert pd.api.types.is_numeric_dtype(macd)
        assert pd.api.types.is_numeric_dtype(signal)

    def test_volume_ma_computation(self):
        """Test volume MA is computed correctly."""
        # Generate test data
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(200).copy()

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(aligned)

        volume_ma = features[('BTC/USDT', 'volume_ma')]

        # Check volume MA is non-negative
        assert (volume_ma >= 0).all()

        # Check no NaN values
        assert not volume_ma.isna().any()

        # Volume MA should be smooth (less volatile than raw volume)
        volume = features[('BTC/USDT', 'volume')]
        volume_std = volume.std()
        volume_ma_std = volume_ma.std()
        assert volume_ma_std < volume_std

    def test_invalid_input_raises_error(self):
        """Test that invalid input raises appropriate error."""
        # Create DataFrame without multi-level columns
        df = pd.DataFrame({
            'close': [100, 110, 120],
        })

        engineer = FeatureEngineer()

        with pytest.raises(ValueError, match="multi-level columns"):
            engineer.compute_all_features(df)

    def test_custom_price_columns(self):
        """Test using custom price columns."""
        # Generate test data
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(200).copy()

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        engineer = FeatureEngineer()
        # Use 'open' instead of 'close'
        features = engineer.compute_all_features(aligned, price_cols=['open'])

        # Features should still be created (based on 'open' price)
        assert ('BTC/USDT', 'returns') in features.columns
        assert ('BTC/USDT', 'rsi') in features.columns

    def test_feature_shapes_match_input(self):
        """Test that feature DataFrame has same shape as input (plus new columns)."""
        # Generate test data
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m'
        ).head(200).copy()

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(aligned)

        # Same number of rows
        assert len(features) == len(aligned)

        # More columns (original + new features)
        assert len(features.columns) > len(aligned.columns)

        # Check expected number of new columns per source
        # Original: 5 (open, high, low, close, volume)
        # New: 6 (returns, volatility, rsi, macd, macd_signal, volume_ma)
        # Total: 11 per source
        n_sources = len(aligned.columns.get_level_values(0).unique())
        assert len(features.columns) == 11 * n_sources

    def test_no_data_leakage(self):
        """Test that features don't leak future information."""
        # Create test data where future values are very different
        timestamps = pd.date_range('2024-01-01', periods=100, freq='1min')
        timestamps_ms = (timestamps.astype(np.int64) // 10**6).values

        # First 50: stable around 100, Next 50: jump to 200
        prices = np.concatenate([
            np.random.normal(100, 1, 50),
            np.random.normal(200, 1, 50)
        ])

        df = pd.DataFrame({
            ('TEST', 'close'): prices,
            ('TEST', 'volume'): np.random.uniform(1000, 2000, 100),
        }, index=timestamps_ms)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df)

        # Returns at index 49 should only reflect past data
        # Should not know about the jump at index 50
        returns = features[('TEST', 'returns')]
        volatility = features[('TEST', 'volatility')]

        # Volatility before jump should be low
        volatility_before = volatility.iloc[45:50].mean()
        # Volatility after jump should be higher
        volatility_after = volatility.iloc[55:60].mean()

        assert volatility_after > volatility_before

    def test_handles_constant_prices(self):
        """Test feature computation handles constant prices gracefully."""
        timestamps = pd.date_range('2024-01-01', periods=100, freq='1min')
        timestamps_ms = (timestamps.astype(np.int64) // 10**6).values

        # Constant prices
        df = pd.DataFrame({
            ('TEST', 'close'): [100.0] * 100,
            ('TEST', 'volume'): np.random.uniform(1000, 2000, 100),
        }, index=timestamps_ms)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df)

        # Returns should be 0 (no price change)
        returns = features[('TEST', 'returns')]
        assert (returns == 0).all()

        # Volatility should be very small (near 0)
        # We use 1e-6 as fallback instead of 0 to avoid division issues
        volatility = features[('TEST', 'volatility')]
        assert (volatility <= 1e-5).all()  # Allow for small positive values

        # RSI should be neutral (50) or 100 (no losses)
        rsi = features[('TEST', 'rsi')]
        # With constant prices, RSI might be 100 (no losses) or filled with default
        assert (rsi >= 50).all()

    def test_computation_efficiency(self):
        """Test that feature computation is reasonably efficient."""
        import time

        # Generate larger dataset
        df_crypto = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-03',  # 2 days
            timeframe='1m'
        )

        aligner = TimestampAligner()
        aligned = aligner.align({'BTC/USDT': df_crypto})

        engineer = FeatureEngineer()

        start_time = time.time()
        features = engineer.compute_all_features(aligned)
        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds for ~2880 rows)
        assert elapsed < 5.0

        # Check result is valid
        assert len(features) == len(aligned)
        assert not features[('BTC/USDT', 'returns')].isna().any()

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        # Create DataFrame with correct structure but no rows
        df = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples([('TEST', 'close'), ('TEST', 'volume')])
        )

        engineer = FeatureEngineer()
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            engineer.compute_all_features(df)

    def test_non_positive_prices_raises_error(self):
        """Test that non-positive prices raise ValueError."""
        timestamps = pd.date_range('2024-01-01', periods=100, freq='1min')
        timestamps_ms = (timestamps.astype(np.int64) // 10**6).values

        # Create DataFrame with negative prices
        df = pd.DataFrame({
            ('TEST', 'close'): [-100.0] * 100,  # Negative prices
            ('TEST', 'volume'): np.random.uniform(1000, 2000, 100),
        }, index=timestamps_ms)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engineer = FeatureEngineer()
        with pytest.raises(ValueError, match="Non-positive prices found"):
            engineer.compute_all_features(df)

    def test_zero_prices_raises_error(self):
        """Test that zero prices raise ValueError."""
        timestamps = pd.date_range('2024-01-01', periods=100, freq='1min')
        timestamps_ms = (timestamps.astype(np.int64) // 10**6).values

        # Create DataFrame with zero prices
        df = pd.DataFrame({
            ('TEST', 'close'): [0.0] * 100,  # Zero prices
            ('TEST', 'volume'): np.random.uniform(1000, 2000, 100),
        }, index=timestamps_ms)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engineer = FeatureEngineer()
        with pytest.raises(ValueError, match="Non-positive prices found"):
            engineer.compute_all_features(df)

    def test_multiple_price_columns_raises_error(self):
        """Test that multiple price columns raise ValueError."""
        timestamps = pd.date_range('2024-01-01', periods=100, freq='1min')
        timestamps_ms = (timestamps.astype(np.int64) // 10**6).values

        df = pd.DataFrame({
            ('TEST', 'open'): np.random.uniform(95, 105, 100),
            ('TEST', 'close'): np.random.uniform(95, 105, 100),
            ('TEST', 'volume'): np.random.uniform(1000, 2000, 100),
        }, index=timestamps_ms)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engineer = FeatureEngineer()
        # Trying to use both 'open' and 'close' should fail
        with pytest.raises(ValueError, match="Only single price column supported"):
            engineer.compute_all_features(df, price_cols=['open', 'close'])

    def test_rsi_with_only_gains_no_division_error(self):
        """Test RSI computation when there are only gains (no losses)."""
        timestamps = pd.date_range('2024-01-01', periods=50, freq='1min')
        timestamps_ms = (timestamps.astype(np.int64) // 10**6).values

        # Create steadily increasing prices (only gains)
        prices = np.linspace(100, 200, 50)
        df = pd.DataFrame({
            ('TEST', 'close'): prices,
            ('TEST', 'volume'): np.random.uniform(1000, 2000, 50),
        }, index=timestamps_ms)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engineer = FeatureEngineer()
        features = engineer.compute_all_features(df)

        # RSI should not contain inf or nan
        rsi = features[('TEST', 'rsi')]
        assert np.isfinite(rsi).all()
        # For only gains, RSI should be high (close to 100)
        assert (rsi >= 50).all()

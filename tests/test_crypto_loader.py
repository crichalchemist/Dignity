"""Tests for CCXT cryptocurrency historical data loader."""

from datetime import datetime

import pandas as pd
import pytest

from data.source.crypto_historical import CCXTHistoricalLoader, generate_synthetic_crypto_data


class TestCCXTHistoricalLoader:
    """Tests for CCXTHistoricalLoader class."""
    
    @pytest.fixture
    def loader(self, tmp_path):
        """Create loader with temporary cache directory."""
        return CCXTHistoricalLoader(
            exchange='binance',
            cache_dir=str(tmp_path / 'cache'),
            enable_cache=True
        )
    
    def test_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader.exchange_name == 'binance'
        assert loader.enable_cache is True
        assert loader.exchange is not None
    
    def test_cache_path_generation(self, loader):
        """Test cache file path is generated correctly."""
        path = loader._get_cache_path('BTC/USDT', '1m', '2024-01-01', '2024-01-07')
        assert 'binance_BTC_USDT_1m_2024-01-01_2024-01-07.parquet' in str(path)
    
    @pytest.mark.skip(reason="Requires real API connection")
    def test_fetch_real_data(self, loader):
        """Test fetching real data from Binance (requires internet)."""
        df = loader.fetch_ohlcv(
            'BTC/USDT',
            timeframe='1h',
            start_date='2024-01-01',
            end_date='2024-01-02',
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        assert df['timestamp'].is_monotonic_increasing
    
    def test_caching_mechanism(self, loader, tmp_path):
        """Test that caching works correctly."""
        # Create synthetic data and save to cache
        df_original = generate_synthetic_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02')
        cache_path = loader._get_cache_path('BTC/USDT', '1m', '2024-01-01', '2024-01-02')
        loader._save_to_cache(df_original, cache_path)
        
        # Load from cache
        df_cached = loader._load_from_cache(cache_path)
        assert df_cached is not None
        assert len(df_cached) == len(df_original)
        pd.testing.assert_frame_equal(df_cached, df_original)


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        df = generate_synthetic_crypto_data(
            symbol='BTC/USDT',
            start_date='2024-01-01',
            end_date='2024-01-02',
            timeframe='1m',
            initial_price=40000.0
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def test_synthetic_data_shape(self):
        """Test synthetic data has correct number of candles."""
        # 1 day of 1-minute data = 1440 candles
        df = generate_synthetic_crypto_data(
            start_date='2024-01-01',
            end_date='2024-01-02',  # One full day
            timeframe='1m'
        )
        assert len(df) >= 1440  # At least one day worth
    
    def test_synthetic_ohlc_validity(self):
        """Test synthetic OHLC data is valid (high >= low, etc.)."""
        df = generate_synthetic_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02')
        
        # High should be >= Open and Close
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        
        # Low should be <= Open and Close
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        
        # Volume should be positive
        assert (df['volume'] > 0).all()
    
    def test_synthetic_timestamps_monotonic(self):
        """Test synthetic timestamps are monotonically increasing."""
        df = generate_synthetic_crypto_data('BTC/USDT', '2024-01-01', '2024-01-02')
        assert df['timestamp'].is_monotonic_increasing
    
    def test_different_timeframes(self):
        """Test synthetic data generation with different timeframes."""
        for timeframe in ['1m', '5m', '1h', '1d']:
            df = generate_synthetic_crypto_data(
                start_date='2024-01-01',
                end_date='2024-01-02',
                timeframe=timeframe
            )
            assert len(df) > 0
            assert df['timestamp'].is_monotonic_increasing


class TestMultipleSymbols:
    """Tests for fetching multiple symbols."""
    
    @pytest.mark.skip(reason="Requires real API connection")
    def test_fetch_multiple_symbols(self, tmp_path):
        """Test fetching multiple symbols (requires internet)."""
        loader = CCXTHistoricalLoader(cache_dir=str(tmp_path / 'cache'))
        
        data = loader.fetch_multiple(
            symbols=['BTC/USDT', 'ETH/USDT'],
            timeframe='1h',
            start_date='2024-01-01',
            end_date='2024-01-02',
        )
        
        assert len(data) == 2
        assert 'BTC/USDT' in data
        assert 'ETH/USDT' in data
        assert isinstance(data['BTC/USDT'], pd.DataFrame)
        assert isinstance(data['ETH/USDT'], pd.DataFrame)

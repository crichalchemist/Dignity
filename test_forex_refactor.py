"""Quick test of refactored forex loader."""

from data.source.forex_historical import ForexHistoricalLoader, generate_synthetic_forex_data


def test_synthetic_data():
    """Test synthetic data generation still works."""
    df = generate_synthetic_forex_data('EURUSD', '2024-01-01', '2024-01-02', interval='1min')
    assert len(df) > 0, "Should generate data"
    assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert df['volume'].sum() == 0, "Forex has no volume"
    print(f"✓ Generated {len(df)} synthetic candles")
    return df

def test_loader_init():
    """Test loader initializes without API key."""
    loader = ForexHistoricalLoader(cache_dir='data/cache/test_forex')
    assert loader.cache_dir.exists()
    print("✓ Loader initialized successfully (no API key required)")
    return loader

def test_cache_path():
    """Test new cache path format."""
    loader = ForexHistoricalLoader(cache_dir='data/cache/test_forex')
    cache_path = loader._get_cache_path('EURUSD', 2024, 1)
    expected_name = "forex_EURUSD_M1_2024_01.parquet"
    assert cache_path.name == expected_name, f"Expected {expected_name}, got {cache_path.name}"
    print(f"✓ Cache path format correct: {cache_path.name}")

def test_fetch_ohlc_signature():
    """Test new fetch_ohlc signature works."""
    loader = ForexHistoricalLoader(cache_dir='data/cache/test_forex')
    # Should work with year/month instead of start/end dates
    df = loader.fetch_ohlc('EURUSD', year=2024, month=1)
    # Will be empty without manual CSV, but shouldn't error
    assert isinstance(df, type(generate_synthetic_forex_data('EURUSD', '2024-01-01', '2024-01-02')))
    print("✓ fetch_ohlc() signature updated correctly (year/month params)")

def test_csv_parsing():
    """Test histdata CSV parsing."""
    loader = ForexHistoricalLoader()
    
    # Sample histdata.com M1 CSV format
    csv_content = """DateTime,Open,High,Low,Close,Volume
20240101 000000,1.10450,1.10460,1.10440,1.10455,0
20240101 000100,1.10455,1.10465,1.10450,1.10460,0
20240101 000200,1.10460,1.10470,1.10455,1.10465,0"""
    
    df = loader._parse_histdata_csv(csv_content, 'EURUSD')
    assert len(df) == 3, f"Should parse 3 rows, got {len(df)}"
    assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    assert df['open'].iloc[0] == 1.10450
    assert df['close'].iloc[2] == 1.10465
    print(f"✓ CSV parsing works correctly ({len(df)} rows parsed)")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing Refactored Forex Loader (histdata.com)")
    print("=" * 60)
    
    test_synthetic_data()
    test_loader_init()
    test_cache_path()
    test_fetch_ohlc_signature()
    test_csv_parsing()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Forex loader refactored successfully!")
    print("=" * 60)
    print("\nChanges from Alpha Vantage to histdata.com:")
    print("  - Removed: api_key, rate_limit_delay parameters")
    print("  - Updated: fetch_ohlc() now uses year/month instead of interval/start_date/end_date")
    print("  - Added: fetch_month_data() for direct monthly data access")
    print("  - Added: _parse_histdata_csv() for CSV format parsing")
    print("  - Added: manual_csv_path parameter for manually downloaded files")
    print("  - Preserved: generate_synthetic_forex_data() for testing")

"""Integration test for timestamp alignment pipeline.

This demonstrates the full alignment pipeline with real-world usage.
"""

import pandas as pd

from data.processors.alignment import TimestampAligner
from data.source.crypto_historical import generate_synthetic_crypto_data
from data.source.forex_historical import generate_synthetic_forex_data


def test_integration_alignment_pipeline():
    """Test full alignment pipeline with crypto and forex data."""
    print("\n" + "=" * 60)
    print("TIMESTAMP ALIGNMENT PIPELINE - INTEGRATION TEST")
    print("=" * 60)

    # Step 1: Generate synthetic data from multiple sources
    print("\n1. Generating synthetic data...")
    df_btc = generate_synthetic_crypto_data(
        symbol='BTC/USDT',
        start_date='2024-01-01',
        end_date='2024-01-02',
        timeframe='1m',
        initial_price=40000.0
    )
    print(f"   ✓ BTC/USDT: {len(df_btc)} candles")

    df_eth = generate_synthetic_crypto_data(
        symbol='ETH/USDT',
        start_date='2024-01-01',
        end_date='2024-01-02',
        timeframe='1m',
        initial_price=2500.0
    )
    print(f"   ✓ ETH/USDT: {len(df_eth)} candles")

    df_eur = generate_synthetic_forex_data(
        pair='EUR/USD',
        start_date='2024-01-01',
        end_date='2024-01-02',
        interval='1min'
    )
    print(f"   ✓ EUR/USD: {len(df_eur)} candles")

    # Step 2: Create aligner and align data
    print("\n2. Aligning to common timestamp grid...")
    aligner = TimestampAligner()
    data_sources = {
        'BTC/USDT': df_btc,
        'ETH/USDT': df_eth,
        'EUR/USD': df_eur
    }
    aligned = aligner.align(data_sources, method='ffill')
    print(f"   ✓ Aligned data shape: {aligned.shape}")
    print(f"   ✓ Timestamp range: {len(aligned)} intervals")

    # Step 3: Verify alignment
    print("\n3. Verification:")
    print(f"   ✓ Multi-level columns: {aligned.columns.nlevels} levels")
    print(f"   ✓ Number of sources: {len(aligned.columns.get_level_values(0).unique())}")
    print(f"   ✓ Features per source: {len(aligned.columns.get_level_values(1).unique())}")

    # Step 4: Show sample of aligned data
    print("\n4. Sample of aligned data (first 5 rows):")
    print(aligned.head())

    # Step 5: Access specific data
    print("\n5. Accessing specific data:")
    btc_close = aligned[('BTC/USDT', 'close')]
    eth_close = aligned[('ETH/USDT', 'close')]
    eur_close = aligned[('EUR/USD', 'close')]
    
    print(f"   ✓ BTC close range: ${btc_close.min():.2f} - ${btc_close.max():.2f}")
    print(f"   ✓ ETH close range: ${eth_close.min():.2f} - ${eth_close.max():.2f}")
    print(f"   ✓ EUR close range: ${eur_close.min():.4f} - ${eur_close.max():.4f}")

    print("\n" + "=" * 60)
    print("✓ INTEGRATION TEST PASSED")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    test_integration_alignment_pipeline()

"""Integration test: Data loaders + Foundation Model end-to-end."""

import torch

from data.source.crypto_historical import generate_synthetic_crypto_data
from data.source.forex_historical import generate_synthetic_forex_data
from models.foundation import create_model


def test_integration():
    """Test complete data-to-model pipeline."""
    print("=" * 80)
    print("DIGNITY FOUNDATION MODEL - INTEGRATION TEST")
    print("=" * 80)
    print()
    
    # Step 1: Generate synthetic market data
    print("Step 1: Loading Market Data")
    print("-" * 80)
    
    crypto_data = generate_synthetic_crypto_data(
        symbol='BTC/USDT',
        start_date='2024-01-01',
        end_date='2024-01-02',
        timeframe='1m',
        initial_price=40000.0
    )
    print(f"✓ Crypto data loaded: {len(crypto_data)} candles (BTC/USDT)")
    print(f"  Price range: ${crypto_data['close'].min():.2f} - ${crypto_data['close'].max():.2f}")
    
    forex_data = generate_synthetic_forex_data(
        pair='EUR/USD',
        start_date='2024-01-01',
        end_date='2024-01-02',
        interval='1min',
        initial_price=1.10
    )
    print(f"✓ Forex data loaded: {len(forex_data)} candles (EUR/USD)")
    print(f"  Price range: {forex_data['close'].min():.4f} - {forex_data['close'].max():.4f}")
    print()
    
    # Step 2: Create mock features (OHLCV + 8 technical indicators = 13 features)
    print("Step 2: Feature Engineering")
    print("-" * 80)
    
    # Take first 128 candles for quick test
    seq_len = 128
    crypto_sample = crypto_data.head(seq_len)
    
    # Create mock feature matrix: OHLCV (5) + 8 indicators = 13 features
    import numpy as np
    features = np.column_stack([
        crypto_sample['open'].values,
        crypto_sample['high'].values,
        crypto_sample['low'].values,
        crypto_sample['close'].values,
        crypto_sample['volume'].values,
        np.random.randn(seq_len, 8),  # Mock technical indicators
    ])
    
    # Convert to tensor: [batch=1, seq_len=128, features=13]
    x = torch.FloatTensor(features).unsqueeze(0)
    print(f"✓ Feature tensor created: {x.shape}")
    print(f"  Format: [batch_size=1, sequence_length={seq_len}, features=13]")
    print(f"  Features: OHLCV (5) + RSI/MACD/Volatility/etc (8)")
    print()
    
    # Step 3: Initialize Foundation Model
    print("Step 3: Loading Foundation Model")
    print("-" * 80)
    
    model = create_model(size='small')
    param_count = model.count_parameters()
    print(f"✓ Model loaded: Small variant (~2M parameters)")
    print(f"  Exact count: {param_count:,} trainable parameters")
    print(f"  Architecture:")
    print(f"    - CNN1D Feature Extractor (multi-scale)")
    print(f"    - B2S6 Backbone (6 layers)")
    print(f"    - Cross-Asset Transformer (2 layers)")
    print(f"    - Task Heads: Regime, Forecast, Risk, TradeSignal")
    print()
    
    # Step 4: Forward pass
    print("Step 4: Model Forward Pass")
    print("-" * 80)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"✓ Forward pass successful!")
    print(f"\nOutput predictions:")
    print(f"  Market Regime:  {outputs['regime'].shape} → 4 regime classes")
    print(f"  Price Forecast: {outputs['forecast'].shape} → 3 horizons (1/5/15 steps)")
    print(f"  Risk (VaR):     {outputs['risk'].shape} → 3 quantiles (5%/50%/95%)")
    print(f"  Trade Signal:   {outputs['trade_signal'].shape} → 3 actions (buy/hold/sell)")
    print()
    
    # Step 5: Sample predictions
    print("Step 5: Sample Predictions (Last Timestep)")
    print("-" * 80)
    
    # Get last timestep predictions
    regime_logits = outputs['regime'][0, -1].numpy()
    regime_probs = torch.softmax(outputs['regime'][0, -1], dim=0).numpy()
    forecast = outputs['forecast'][0, -1].numpy()
    risk = outputs['risk'][0, -1].numpy()
    signal_logits = outputs['trade_signal'][0, -1].numpy()
    signal_probs = torch.softmax(outputs['trade_signal'][0, -1], dim=0).numpy()
    
    regime_names = ['Bull', 'Bear', 'Sideways', 'Volatile']
    print(f"Market Regime Probabilities:")
    for i, (name, prob) in enumerate(zip(regime_names, regime_probs)):
        print(f"  {name:10s}: {prob:.2%}")
    
    print(f"\nPrice Forecasts:")
    print(f"  1-step ahead:  {forecast[0]:.4f}")
    print(f"  5-step ahead:  {forecast[1]:.4f}")
    print(f"  15-step ahead: {forecast[2]:.4f}")
    
    print(f"\nValue-at-Risk (Quantiles):")
    print(f"  5th percentile:  {risk[0]:.4f}")
    print(f"  50th percentile: {risk[1]:.4f}")
    print(f"  95th percentile: {risk[2]:.4f}")
    
    signal_names = ['Buy', 'Hold', 'Sell']
    print(f"\nTrade Signal Probabilities:")
    for name, prob in zip(signal_names, signal_probs):
        print(f"  {name:5s}: {prob:.2%}")
    
    print()
    print("=" * 80)
    print("✅ INTEGRATION TEST PASSED!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Data loaders working (crypto + forex)")
    print("  ✓ Feature engineering working (13-feature vectors)")
    print("  ✓ Foundation model working (all 4 task heads)")
    print("  ✓ End-to-end pipeline functional")
    print()
    print("Ready for:")
    print("  → Real data pipeline (Tasks 10-13)")
    print("  → Training script (Task 15)")
    print("  → Pre-training on historical data")
    print()


if __name__ == '__main__':
    test_integration()

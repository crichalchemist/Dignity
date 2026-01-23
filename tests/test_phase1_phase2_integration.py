"""
Integration tests for Phase 1 and Phase 2 implementations.

Tests:
  - Phase 1.1: Sentiment pipeline integration
  - Phase 1.2: Real RL training environment
  - Phase 2.1: Microstructure features
  - Phase 2.2: Regime detection
  - Phase 2.3: Intrinsic time features
  - FX patterns (sessions, S/R, ADX, price action)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import FeatureConfig
from train.features.agent_features import build_feature_frame
from train.features.fx_patterns import build_fx_feature_frame


def create_sample_data(n_bars: int = 500) -> pd.DataFrame:
    """Create realistic sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01 00:00', periods=n_bars, freq='1min')

    # Generate price with trend + noise
    trend = np.linspace(0, 0.01, n_bars)
    noise = np.random.randn(n_bars) * 0.0005
    close_prices = 1.1000 + trend + np.cumsum(noise)

    df = pd.DataFrame({
        'datetime': dates,
        'open': close_prices + np.random.randn(n_bars) * 0.0001,
        'close': close_prices,
        'volume': np.abs(np.random.randn(n_bars) * 1000 + 5000),
    })

    # Generate high/low from open/close
    df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(n_bars) * 0.0002)
    df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(n_bars) * 0.0002)

    return df


def test_1_1_sentiment_pipeline():
    """Test Phase 1.1: GDELT sentiment integration."""
    print("\n" + "=" * 60)
    print("TEST 1.1: Sentiment Pipeline Integration")
    print("=" * 60)

    try:
        from data.gdelt_ingest import load_gdelt_gkg
        from features.agent_sentiment import aggregate_sentiment, attach_sentiment_features

        print("✓ Imports successful")

        # Test with mock GDELT data
        gdelt_df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=100, freq='15min'),
            'sentiment_score': np.random.randn(100) * 2  # GDELT tone scores
        })

        price_df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=200, freq='1min'),
            'close': 1.1 + np.random.randn(200) * 0.001,
        })

        # Aggregate sentiment to bar frequency
        sent_feats = aggregate_sentiment(
            gdelt_df,
            price_df,
            time_col="datetime",
            score_col="sentiment_score"
        )

        print(f"✓ Sentiment aggregation: {sent_feats.shape}")
        print(f"  Features: {list(sent_feats.columns)[:5]}...")

        # Attach to features
        result = attach_sentiment_features(price_df, sent_feats)

        print(f"✓ Sentiment attachment: {result.shape}")
        print(f"  Total columns: {len(result.columns)}")

        print("\n✅ Phase 1.1 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Phase 1.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_1_2_real_rl_training():
    """Test Phase 1.2: Real RL training infrastructure."""
    print("\n" + "=" * 60)
    print("TEST 1.2: Real RL Training Infrastructure")
    print("=" * 60)

    try:
        from execution.simulated_retail_env import (
            ExecutionConfig,
            OrderAction,
            SimulatedRetailExecutionEnv,
        )
        from train.core.env_based_rl_training import (
            ActionConverter,
            Episode,
            collect_episode,
            update_policy,
        )

        print("✓ Imports successful")

        # Test action converter
        converter = ActionConverter(lot_size=1.0)
        order = converter.policy_to_order(action_idx=2, mid_price=1.1, inventory=0.0)

        assert order.action_type == "market"
        assert order.side == "buy"
        print("✓ ActionConverter: converts policy actions to OrderActions")

        # Test environment
        env_cfg = ExecutionConfig(
            initial_cash=10_000.0,
            lot_size=1.0,
            spread=0.02,
            time_horizon=100,
        )
        env = SimulatedRetailExecutionEnv(env_cfg)
        obs = env.reset()

        assert "portfolio_value" in obs
        print("✓ Trading environment: initialized successfully")

        # Test episode container
        episode = Episode()
        assert len(episode.states) == 0
        print("✓ Episode container: ready for trajectory collection")

        print("\n✅ Phase 1.2 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Phase 1.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_1_microstructure_features():
    """Test Phase 2.1: Microstructure feature integration."""
    print("\n" + "=" * 60)
    print("TEST 2.1: Microstructure Features")
    print("=" * 60)

    try:
        from features.microstructure import build_microstructure_features

        print("✓ Import successful")

        df = create_sample_data(n_bars=200)

        # Build microstructure features
        result = build_microstructure_features(df, windows=[5, 10, 20])

        # Check that features were added
        micro_features = [c for c in result.columns if any(x in c for x in [
            'imbalance', 'vol_direction', 'toxicity', 'spread_proxy',
            'depth_proxy', 'vwap_dev', 'momentum_imbalance', 'price_impact'
        ])]

        print(f"✓ Microstructure features added: {len(micro_features)} features")
        print(f"  Sample features: {micro_features[:5]}")

        # Check for NaN issues
        nan_cols = result[micro_features].isna().sum()
        if nan_cols.max() > len(result) * 0.5:
            print("⚠ Warning: Some features have >50% NaN values")
        else:
            print("✓ NaN handling: acceptable (<50% NaN per feature)")

        print("\n✅ Phase 2.1 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Phase 2.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_2_regime_detection():
    """Test Phase 2.2: Regime detection integration."""
    print("\n" + "=" * 60)
    print("TEST 2.2: Regime Detection")
    print("=" * 60)

    try:
        from features.regime_detection import (
            RegimeConfig,
            RegimeDetector,
            integrate_regime_features,
        )

        print("✓ Import successful")

        df = create_sample_data(n_bars=300)

        # Test regime detector
        detector = RegimeDetector(RegimeConfig(n_regimes=4, lookback=50))
        detector.fit(df)

        regimes = detector.predict(df)
        unique_regimes = np.unique(regimes)

        print("✓ Regime detection: fitted and predicted")
        print(f"  Regimes found: {unique_regimes}")
        print(f"  Regime names: {detector.REGIME_NAMES}")

        # Test integration helper
        feature_df = df[['datetime', 'close']].copy()
        result = integrate_regime_features(feature_df, df)

        regime_cols = [c for c in result.columns if 'regime' in c]
        print(f"✓ Regime features: {len(regime_cols)} columns")
        print(f"  Columns: {regime_cols}")

        # Check one-hot encoding
        regime_flags = result[[c for c in regime_cols if c.startswith('regime_is_')]]
        assert regime_flags.sum(axis=1).max() == 1, "One-hot encoding broken"
        print("✓ One-hot encoding: correct (each row has exactly 1 active regime)")

        print("\n✅ Phase 2.2 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Phase 2.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_3_intrinsic_time_features():
    """Test Phase 2.3: Intrinsic time feature integration."""
    print("\n" + "=" * 60)
    print("TEST 2.3: Intrinsic Time Features")
    print("=" * 60)

    try:
        from features.intrinsic_time import (
            add_intrinsic_time_features,
            detect_directional_changes,
        )

        print("✓ Import successful")

        df = create_sample_data(n_bars=300)

        # Test DC detection
        events = detect_directional_changes(
            df['close'],
            up_threshold=0.001,
            down_threshold=0.001,
            timestamps=df['datetime']
        )

        print(f"✓ DC detection: {len(events)} events found")
        print(f"  Directions: {events['direction'].value_counts().to_dict()}")

        # Test feature addition
        result = add_intrinsic_time_features(
            df,
            price_col='close',
            up_threshold=0.001,
            timestamp_col='datetime'
        )

        dc_features = [c for c in result.columns if c.startswith('dc_')]
        print(f"✓ Intrinsic time features: {len(dc_features)} features")
        print(f"  Features: {dc_features}")

        # Check feature values
        print(f"  DC events marked: {result['dc_event_flag'].sum()} bars")
        print(f"  Direction range: [{result['dc_direction'].min():.1f}, {result['dc_direction'].max():.1f}]")
        print(f"  Overshoot range: [{result['dc_overshoot'].min():.4f}, {result['dc_overshoot'].max():.4f}]")

        print("\n✅ Phase 2.3 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Phase 2.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fx_patterns():
    """Test FX patterns module (sessions, S/R, ADX, price action)."""
    print("\n" + "=" * 60)
    print("TEST: FX Patterns Module")
    print("=" * 60)

    try:
        from features.fx_patterns import (
            add_fx_session_features,
            add_price_action_patterns,
            add_support_resistance_features,
            add_trend_strength_features,
            build_fx_feature_frame,
        )

        print("✓ Import successful")

        df = create_sample_data(n_bars=300)

        # Test each component
        result = add_fx_session_features(df)
        session_cols = [c for c in result.columns if 'session' in c or 'liquidity' in c]
        print(f"✓ Sessions: {len(session_cols)} features - {session_cols}")

        result = add_support_resistance_features(df, lookback=50)
        sr_cols = [c for c in result.columns if 'resistance' in c or 'support' in c or 'range' in c]
        print(f"✓ S/R levels: {len(sr_cols)} features")

        result = add_trend_strength_features(df, window=14)
        adx_cols = [c for c in result.columns if 'di' in c or 'adx' in c or 'trend' in c]
        print(f"✓ ADX/Trend: {len(adx_cols)} features - {adx_cols}")

        result = add_price_action_patterns(df)
        pattern_cols = [c for c in result.columns if 'engulfing' in c or 'pin' in c or 'inside' in c]
        print(f"✓ Price patterns: {len(pattern_cols)} features - {pattern_cols}")

        # Test full integration
        result = build_fx_feature_frame(
            df,
            include_sessions=True,
            include_support_resistance=True,
            include_trend_strength=True,
            include_patterns=True,
        )

        fx_added = len(result.columns) - len(df.columns)
        print(f"✓ Complete FX features: {fx_added} features added")

        print("\n✅ FX Patterns PASSED")
        return True

    except Exception as e:
        print(f"\n❌ FX Patterns FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test complete feature pipeline with all new features enabled."""
    print("\n" + "=" * 60)
    print("TEST: Full Integration (All Features)")
    print("=" * 60)

    try:
        df = create_sample_data(n_bars=400)

        # Configure all features
        config = FeatureConfig(
            sma_windows=[10, 20],
            ema_windows=[10, 20],
            rsi_window=14,
            bollinger_window=20,
            atr_window=14,
            microstructure_windows=[5, 10, 20],
            dc_threshold_up=0.001,
            dc_threshold_down=0.001,
            include_groups=["all"],  # Enable all feature groups
        )

        print(f"✓ Config created: {len([k for k in config.__dict__ if not k.startswith('_')])} parameters")

        # Build full feature frame
        feature_df = build_feature_frame(df, config=config)

        print(f"✓ Feature frame built: {feature_df.shape}")
        print(f"  Original columns: {len(df.columns)}")
        print(f"  Feature columns: {len(feature_df.columns)}")
        print(f"  Features added: {len(feature_df.columns) - len(df.columns)}")

        # Check feature categories
        feature_categories = {
            'base': ['log_return', 'spread'],
            'trend': ['sma', 'ema'],
            'momentum': ['rsi'],
            'bollinger': ['bb_'],
            'atr': ['atr', 'true_range'],
            'microstructure': ['imbalance', 'toxicity', 'vwap_dev'],
            'regime': ['regime'],
            'intrinsic_time': ['dc_'],
        }

        print("\n  Feature breakdown:")
        for category, keywords in feature_categories.items():
            cols = [c for c in feature_df.columns if any(kw in c for kw in keywords)]
            if cols:
                print(f"    {category}: {len(cols)} features")

        # Check for NaN issues
        total_nans = feature_df.isna().sum().sum()
        total_cells = feature_df.shape[0] * feature_df.shape[1]
        nan_pct = (total_nans / total_cells) * 100

        print("\n  Data quality:")
        print(f"    Total NaN: {total_nans:,} ({nan_pct:.2f}% of cells)")
        print(f"    Rows after dropna: {len(feature_df)}")

        if nan_pct > 20:
            print("    ⚠ Warning: High NaN percentage")
        else:
            print("    ✓ NaN percentage acceptable")

        # Add FX patterns
        feature_df = build_fx_feature_frame(
            feature_df,
            include_sessions=True,
            include_support_resistance=True,
            include_trend_strength=True,
            include_patterns=True,
        )

        print("\n✓ FX patterns added")
        print(f"  Final feature count: {len(feature_df.columns)} columns")

        print("\n✅ Full Integration PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Full Integration FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("PHASE 1 & 2 INTEGRATION TEST SUITE")
    print("=" * 60)

    tests = [
        ("Phase 1.1: Sentiment Pipeline", test_1_1_sentiment_pipeline),
        ("Phase 1.2: Real RL Training", test_1_2_real_rl_training),
        ("Phase 2.1: Microstructure", test_2_1_microstructure_features),
        ("Phase 2.2: Regime Detection", test_2_2_regime_detection),
        ("Phase 2.3: Intrinsic Time", test_2_3_intrinsic_time_features),
        ("FX Patterns", test_fx_patterns),
        ("Full Integration", test_full_integration),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Ready for Phase 3.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

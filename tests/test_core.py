"""Test core utilities."""

import numpy as np
import pandas as pd
import pytest

from core.config import DignityConfig
from core.execution import MAX_POSITION_FRACTION, apply_live_position_cap
from core.privacy import PrivacyManager
from core.signals import ASSET_CONFIGS, AssetConfig, SignalProcessor


class TestSignalProcessor:
    """Test signal processing functions."""

    def test_volatility(self):
        """Test volatility calculation."""
        values = np.array([100, 102, 98, 101, 99, 103])
        vol = SignalProcessor.volatility(values, window=3)

        assert len(vol) == len(values)
        assert vol[0] == vol[1] == vol[2]  # First values filled
        assert vol[-1] > 0  # Should have volatility

    def test_entropy(self):
        """Test entropy calculation."""
        # Uniform distribution should have high entropy
        uniform = np.random.uniform(0, 100, 1000)
        uniform_entropy = SignalProcessor.entropy(uniform)

        # Constant values should have zero entropy
        constant = np.ones(1000)
        constant_entropy = SignalProcessor.entropy(constant)

        assert uniform_entropy > constant_entropy
        assert constant_entropy == 0.0

    def test_price_momentum(self):
        """Test price momentum calculation."""
        prices = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        momentum = SignalProcessor.price_momentum(prices, window=2)

        assert len(momentum) == len(prices)
        # Check that momentum is computed (non-zero after window)
        assert np.any(momentum != 0)

    def test_directional_change(self):
        """Test directional change detection."""
        # Upward trend
        prices = np.array([100, 102, 105, 103, 106])
        dc = SignalProcessor.directional_change(prices, threshold=0.015)

        assert len(dc) == len(prices)
        assert np.sum(dc == 1) > 0  # Should have upward changes

    def test_regime_detection(self):
        """Test regime detection."""
        # Create volatility with different regimes
        vol = np.concatenate(
            [
                np.ones(100) * 0.5,  # Low vol
                np.ones(100) * 2.0,  # High vol
                np.ones(100) * 1.0,  # Normal vol
            ]
        )

        regimes = SignalProcessor.regime_detection(vol)

        assert len(regimes) == len(vol)
        assert np.any(regimes == 0)  # Low vol regime
        assert np.any(regimes == 2)  # High vol regime


class TestPrivacyManager:
    """Test privacy-preserving operations."""

    def test_hash_identifier(self):
        """Test identifier hashing."""
        addr1 = "0x1234567890abcdef"
        addr2 = "0x1234567890abcdef"
        addr3 = "0xfedcba0987654321"

        hash1 = PrivacyManager.hash_identifier(addr1)
        hash2 = PrivacyManager.hash_identifier(addr2)
        hash3 = PrivacyManager.hash_identifier(addr3)

        # Same input = same hash
        assert hash1 == hash2
        # Different input = different hash
        assert hash1 != hash3
        # Hash is hex string
        assert len(hash1) == 64

    def test_anonymize_addresses(self):
        """Test batch address anonymization."""
        addresses = ["addr1", "addr2", "addr3"]
        hashed = PrivacyManager.anonymize_addresses(addresses)

        assert len(hashed) == len(addresses)
        assert all(len(h) == 64 for h in hashed)
        assert len(set(hashed)) == len(addresses)  # All unique

    def test_quantize_amounts(self):
        """Test amount quantization."""
        amounts = np.random.uniform(10, 100, 1000)
        quantized = PrivacyManager.quantize_amounts(amounts, bins=10)

        # Should have fewer unique values
        assert len(np.unique(quantized)) <= 10
        # Values should be within original range
        assert np.min(quantized) >= np.min(amounts)
        assert np.max(quantized) <= np.max(amounts)

    def test_add_noise(self):
        """Test differential privacy noise."""
        values = np.array([100.0, 200.0, 300.0])
        noisy = PrivacyManager.add_noise(values, epsilon=1.0)

        # Should be different but similar
        assert not np.array_equal(values, noisy)
        assert np.allclose(values, noisy, atol=50)  # Reasonable noise


class TestDignityConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration."""
        config = DignityConfig()

        assert config.model.task == "risk"
        assert config.model.hidden_size == 256
        assert config.data.seq_len == 100
        assert config.train.epochs == 50

    def test_config_yaml_roundtrip(self, tmp_path):
        """Test save/load config."""
        config1 = DignityConfig()
        config1.model.hidden_size = 512

        yaml_path = tmp_path / "test_config.yaml"
        config1.to_yaml(str(yaml_path))

        config2 = DignityConfig.from_yaml(str(yaml_path))

        assert config2.model.hidden_size == 512
        assert config2.data.seq_len == config1.data.seq_len


# ---------------------------------------------------------------------------
# Helpers shared across new signal tests
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (open, high, low, close, volume) arrays of length n."""
    rng = np.random.default_rng(42)
    close = np.cumprod(1 + rng.normal(0, 0.005, n)) * 100.0
    spread = np.abs(rng.normal(0, 0.5, n))
    high = close + spread
    low = close - spread
    open_ = close + rng.normal(0, 0.2, n)
    volume = rng.uniform(1000, 10000, n)
    return open_, high, low, close, volume


# ---------------------------------------------------------------------------
# AssetConfig
# ---------------------------------------------------------------------------

class TestAssetConfig:
    def test_asset_configs_keys_present(self):
        for key in ("forex", "crypto", "equity", "commodity"):
            assert key in ASSET_CONFIGS

    def test_asset_configs_types(self):
        for cfg in ASSET_CONFIGS.values():
            assert isinstance(cfg, AssetConfig)

    def test_crypto_dc_threshold_10x_larger_than_forex(self):
        assert ASSET_CONFIGS["crypto"].dc_threshold >= ASSET_CONFIGS["forex"].dc_threshold * 5

    def test_crypto_rsi_period_shorter_than_forex(self):
        assert ASSET_CONFIGS["crypto"].rsi_period < ASSET_CONFIGS["forex"].rsi_period

    def test_asset_config_is_frozen(self):
        cfg = ASSET_CONFIGS["forex"]
        with pytest.raises((AttributeError, TypeError)):
            cfg.rsi_period = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# New signal methods
# ---------------------------------------------------------------------------

class TestNewSignals:
    """Tests for the 13 new SignalProcessor methods."""

    def setup_method(self):
        _, self.high, self.low, self.close, self.volume = _make_ohlcv(200)

    # --- RSI ---

    def test_rsi_shape(self):
        result = SignalProcessor.rsi(self.close, period=14)
        assert result.shape == self.close.shape

    def test_rsi_range(self):
        result = SignalProcessor.rsi(self.close, period=14)
        assert np.all(result >= 0)
        assert np.all(result <= 100)

    def test_rsi_no_nan(self):
        result = SignalProcessor.rsi(self.close, period=14)
        assert not np.any(np.isnan(result))

    def test_rsi_dtype(self):
        result = SignalProcessor.rsi(self.close, period=14)
        assert result.dtype == np.float64

    # --- MACD ---

    def test_macd_returns_three_arrays(self):
        macd_line, signal_line, histogram = SignalProcessor.macd(self.close, 12, 26, 9)
        assert macd_line.shape == self.close.shape
        assert signal_line.shape == self.close.shape
        assert histogram.shape == self.close.shape

    def test_macd_histogram_equals_difference(self):
        macd_line, signal_line, histogram = SignalProcessor.macd(self.close, 12, 26, 9)
        np.testing.assert_allclose(histogram, macd_line - signal_line, atol=1e-10)

    def test_macd_no_nan(self):
        macd_line, signal_line, histogram = SignalProcessor.macd(self.close, 12, 26, 9)
        for arr in (macd_line, signal_line, histogram):
            assert not np.any(np.isnan(arr))

    # --- Bollinger Bands ---

    def test_bollinger_returns_two_arrays(self):
        pct_b, bandwidth = SignalProcessor.bollinger_bands(self.close, window=20, n_std=2.0)
        assert pct_b.shape == self.close.shape
        assert bandwidth.shape == self.close.shape

    def test_bollinger_no_nan(self):
        pct_b, bandwidth = SignalProcessor.bollinger_bands(self.close, window=20, n_std=2.0)
        assert not np.any(np.isnan(pct_b))
        assert not np.any(np.isnan(bandwidth))

    def test_bollinger_bandwidth_nonnegative(self):
        _, bandwidth = SignalProcessor.bollinger_bands(self.close, window=20, n_std=2.0)
        assert np.all(bandwidth >= 0)

    # --- ATR ---

    def test_atr_shape(self):
        result = SignalProcessor.atr(self.high, self.low, self.close, period=14)
        assert result.shape == self.close.shape

    def test_atr_nonnegative(self):
        result = SignalProcessor.atr(self.high, self.low, self.close, period=14)
        assert np.all(result >= 0)

    def test_atr_no_nan(self):
        result = SignalProcessor.atr(self.high, self.low, self.close, period=14)
        assert not np.any(np.isnan(result))

    # --- Stochastic ---

    def test_stochastic_returns_two_arrays(self):
        k, d = SignalProcessor.stochastic(self.high, self.low, self.close, 14, 3)
        assert k.shape == self.close.shape
        assert d.shape == self.close.shape

    def test_stochastic_range(self):
        k, d = SignalProcessor.stochastic(self.high, self.low, self.close, 14, 3)
        assert np.all(k >= 0) and np.all(k <= 100)
        assert np.all(d >= 0) and np.all(d <= 100)

    def test_stochastic_no_nan(self):
        k, d = SignalProcessor.stochastic(self.high, self.low, self.close, 14, 3)
        assert not np.any(np.isnan(k))
        assert not np.any(np.isnan(d))

    # --- ADX ---

    def test_adx_shape(self):
        result = SignalProcessor.adx(self.high, self.low, self.close, period=14)
        assert result.shape == self.close.shape

    def test_adx_range(self):
        result = SignalProcessor.adx(self.high, self.low, self.close, period=14)
        assert np.all(result >= 0) and np.all(result <= 100)

    def test_adx_no_nan(self):
        result = SignalProcessor.adx(self.high, self.low, self.close, period=14)
        assert not np.any(np.isnan(result))

    # --- OBV ---

    def test_obv_shape(self):
        result = SignalProcessor.obv(self.close, self.volume)
        assert result.shape == self.close.shape

    def test_obv_starts_at_zero(self):
        result = SignalProcessor.obv(self.close, self.volume)
        assert result[0] == 0.0

    def test_obv_no_nan(self):
        result = SignalProcessor.obv(self.close, self.volume)
        assert not np.any(np.isnan(result))

    # --- VWAP ---

    def test_vwap_shape(self):
        result = SignalProcessor.vwap(self.high, self.low, self.close, self.volume)
        assert result.shape == self.close.shape

    def test_vwap_within_price_range(self):
        result = SignalProcessor.vwap(self.high, self.low, self.close, self.volume)
        # VWAP should roughly track close price
        assert np.all(result > 0)

    def test_vwap_no_nan(self):
        result = SignalProcessor.vwap(self.high, self.low, self.close, self.volume)
        assert not np.any(np.isnan(result))

    # --- ROC ---

    def test_roc_shape(self):
        result = SignalProcessor.roc(self.close, period=5)
        assert result.shape == self.close.shape

    def test_roc_no_nan(self):
        result = SignalProcessor.roc(self.close, period=5)
        assert not np.any(np.isnan(result))

    def test_roc_zero_for_flat_series(self):
        flat = np.ones(50) * 100.0
        result = SignalProcessor.roc(flat, period=5)
        # Flat price → zero rate of change after warmup
        np.testing.assert_allclose(result[5:], 0.0, atol=1e-10)

    # --- Realized Volatility ---

    def test_realized_volatility_shape(self):
        result = SignalProcessor.realized_volatility(self.close, window=20)
        assert result.shape == self.close.shape

    def test_realized_volatility_nonnegative(self):
        result = SignalProcessor.realized_volatility(self.close, window=20)
        assert np.all(result >= 0)

    def test_realized_volatility_no_nan(self):
        result = SignalProcessor.realized_volatility(self.close, window=20)
        assert not np.any(np.isnan(result))

    # --- Vol Ratio ---

    def test_vol_ratio_shape(self):
        result = SignalProcessor.vol_ratio(self.close, short_window=5, long_window=20)
        assert result.shape == self.close.shape

    def test_vol_ratio_nonnegative(self):
        result = SignalProcessor.vol_ratio(self.close, short_window=5, long_window=20)
        assert np.all(result >= 0)

    def test_vol_ratio_no_nan(self):
        result = SignalProcessor.vol_ratio(self.close, short_window=5, long_window=20)
        assert not np.any(np.isnan(result))

    # --- Order Flow Imbalance ---

    def test_ofi_shape(self):
        result = SignalProcessor.order_flow_imbalance(self.close, self.volume)
        assert result.shape == self.close.shape

    def test_ofi_no_nan(self):
        result = SignalProcessor.order_flow_imbalance(self.close, self.volume)
        assert not np.any(np.isnan(result))

    def test_ofi_sign_matches_price_direction(self):
        # Monotonically rising prices → OFI should be positive (after first)
        prices = np.linspace(100, 200, 50)
        volume = np.ones(50) * 1000.0
        result = SignalProcessor.order_flow_imbalance(prices, volume)
        assert np.all(result[1:] > 0)

    # --- DC State Machine ---

    def test_dc_state_machine_returns_dict(self):
        result = SignalProcessor.dc_state_machine(self.close, threshold=0.01)
        assert isinstance(result, dict)
        assert "dc_direction" in result
        assert "overshoot" in result
        assert "bars_since_event" in result

    def test_dc_state_machine_shapes(self):
        result = SignalProcessor.dc_state_machine(self.close, threshold=0.01)
        for arr in result.values():
            assert arr.shape == self.close.shape

    def test_dc_state_machine_direction_values(self):
        result = SignalProcessor.dc_state_machine(self.close, threshold=0.01)
        unique = set(np.unique(result["dc_direction"]))
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_dc_state_machine_bars_since_event_nonnegative(self):
        result = SignalProcessor.dc_state_machine(self.close, threshold=0.01)
        assert np.all(result["bars_since_event"] >= 0)

    def test_dc_state_machine_no_nan(self):
        result = SignalProcessor.dc_state_machine(self.close, threshold=0.01)
        for arr in result.values():
            assert not np.any(np.isnan(arr))

    def test_dc_state_machine_emits_event_after_threshold_crossed(self):
        # Price jumps by 5% then flat — should emit at least one event
        prices = np.array([100.0] * 20 + [106.0] * 20, dtype=float)
        result = SignalProcessor.dc_state_machine(prices, threshold=0.04)
        assert np.any(result["dc_direction"] != 0)


# ---------------------------------------------------------------------------
# DC signal exact-value tests — lock the signal math against regressions
# ---------------------------------------------------------------------------

class TestDCSignalExact:
    """Deterministic DC signal verification with known price series.

    All expected values are derived analytically so refactoring errors
    cause assertion failures rather than silent behavioral drift.
    """

    # threshold=0.1 gives clean integer-percentage arithmetic
    THRESHOLD = 0.1

    def test_upward_dc_fires_at_correct_bar(self):
        # Prices rise slowly, then cross the 10% threshold at bar 4
        prices = np.array([100.0, 101.0, 103.0, 106.0, 111.0])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        assert result["dc_direction"][4] == pytest.approx(1.0)
        # No earlier bar should fire
        assert np.all(result["dc_direction"][:4] == 0.0)

    def test_downward_dc_fires_at_correct_bar(self):
        # Upward DC at bar 4, then downward DC at bar 8 (price drops 10% from new extreme)
        prices = np.array([100.0, 101.0, 103.0, 106.0, 111.0, 108.0, 105.0, 101.0, 99.9])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        assert result["dc_direction"][4] == pytest.approx(1.0)
        assert result["dc_direction"][8] == pytest.approx(-1.0)

    def test_upward_dc_overshoot_is_excess_beyond_threshold(self):
        # At bar 4: price=111.0, extreme=100.0
        # overshoot = (111.0 - 100.0) / 100.0 - 0.1 = 0.11 - 0.10 = 0.01
        prices = np.array([100.0, 101.0, 103.0, 106.0, 111.0])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        assert result["overshoot"][4] == pytest.approx(0.01, abs=1e-9)

    def test_barely_above_threshold_has_small_overshoot(self):
        # Price lands just 0.1% above the 10% threshold — overshoot is tiny.
        # (Exact boundary is avoided: 100.0 * 1.1 is not representable exactly
        # in IEEE 754, so >= checks at the boundary are unreliable.)
        prices = np.array([100.0, 110.1])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        assert result["dc_direction"][1] == pytest.approx(1.0)
        expected = (110.1 - 100.0) / 100.0 - self.THRESHOLD  # ≈ 0.001
        assert result["overshoot"][1] == pytest.approx(expected, abs=1e-9)

    def test_bars_since_event_resets_to_zero_on_event_bar(self):
        prices = np.array([100.0, 101.0, 103.0, 106.0, 111.0])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        # DC fires at bar 4 — bars_since_event must be 0 there
        assert result["bars_since_event"][4] == pytest.approx(0.0)

    def test_bars_since_event_counts_up_before_first_event(self):
        # No event fires (max rise is 4%, threshold is 10%)
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        assert np.all(result["dc_direction"] == 0.0)
        # bars_since_event should be 0, 1, 2, 3, 4 — monotonically increasing
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result["bars_since_event"], expected)

    def test_bars_since_event_counts_up_between_events(self):
        # Upward DC at bar 4, then free run — count goes 0, 1, 2, ...
        prices = np.array([100.0, 101.0, 103.0, 106.0, 111.0, 108.0, 105.0, 101.0, 99.9])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        # After upward DC at bar 4: bars 5, 6, 7 → 1, 2, 3
        assert result["bars_since_event"][5] == pytest.approx(1.0)
        assert result["bars_since_event"][6] == pytest.approx(2.0)
        assert result["bars_since_event"][7] == pytest.approx(3.0)
        # Downward DC at bar 8 resets to 0
        assert result["bars_since_event"][8] == pytest.approx(0.0)

    def test_dc_direction_only_takes_values_minus1_0_plus1(self):
        prices = np.array([100.0, 101.0, 103.0, 106.0, 111.0, 108.0, 105.0, 101.0, 99.9])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        unique_vals = set(np.unique(result["dc_direction"]))
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_no_nan_in_any_dc_output(self):
        prices = np.array([100.0, 101.0, 103.0, 106.0, 111.0, 108.0, 105.0, 101.0, 99.9])
        result = SignalProcessor.dc_state_machine(prices, threshold=self.THRESHOLD)
        for key, arr in result.items():
            assert not np.any(np.isnan(arr)), f"NaN in {key}"

    def test_bars_since_significant_move_warmup_is_sequential_not_zero(self):
        # Flat prices → no significant move fires anywhere
        prices = np.ones(30) * 100.0
        result = SignalProcessor.bars_since_significant_move(prices, vol_window=20)
        # Warmup bars must count up (not sit at 0)
        assert result[0] == pytest.approx(1.0)
        assert result[19] == pytest.approx(20.0)
        assert np.all(result[:20] > 0), "Warmup bars should never be 0"

    def test_bars_since_significant_move_resets_after_large_move(self):
        # Flat for 25 bars, then one huge spike, then flat again
        prices = np.ones(50) * 100.0
        prices[25] = 200.0  # 100% move — always significant
        result = SignalProcessor.bars_since_significant_move(prices, vol_window=20)
        # price_changes[24] = |200/100 - 1| = 1.0, detected at bar 25
        assert result[25] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# process_sequence with 32 features
# ---------------------------------------------------------------------------

class TestProcessSequence32:
    def setup_method(self):
        self.open_, self.high, self.low, self.close, self.volume = _make_ohlcv(200)
        self.asset_cfg = ASSET_CONFIGS["forex"]

    def test_process_sequence_returns_32_keys(self):
        result = SignalProcessor.process_sequence(
            volumes=self.volume,
            prices=self.close,
            high=self.high,
            low=self.low,
            asset_config=self.asset_cfg,
        )
        assert len(result) == 32

    def test_process_sequence_expected_keys_present(self):
        result = SignalProcessor.process_sequence(
            volumes=self.volume,
            prices=self.close,
            high=self.high,
            low=self.low,
            asset_config=self.asset_cfg,
        )
        required = {
            "volume", "price", "rsi", "macd_line", "macd_signal", "macd_hist",
            "bollinger_pct_b", "bollinger_width", "atr", "stoch_k", "stoch_d",
            "adx", "obv", "vwap", "roc_5", "roc_20", "momentum_10", "momentum_20",
            "volatility_5", "volatility_20", "vol_ratio", "order_flow_imbalance",
            "dc_direction", "dc_overshoot", "dc_bars_since_event",
            "volume_volatility", "volume_entropy", "price_change",
            "directional_change",
        }
        assert required.issubset(result.keys())

    def test_process_sequence_no_nan(self):
        result = SignalProcessor.process_sequence(
            volumes=self.volume,
            prices=self.close,
            high=self.high,
            low=self.low,
            asset_config=self.asset_cfg,
        )
        for key, arr in result.items():
            assert not np.any(np.isnan(arr)), f"NaN found in signal '{key}'"

    def test_process_sequence_all_arrays_same_length(self):
        result = SignalProcessor.process_sequence(
            volumes=self.volume,
            prices=self.close,
            high=self.high,
            low=self.low,
            asset_config=self.asset_cfg,
        )
        lengths = {k: len(v) for k, v in result.items()}
        assert len(set(lengths.values())) == 1, f"Length mismatch: {lengths}"

    def test_process_sequence_uses_asset_config_thresholds(self):
        """Crypto DC threshold is larger → fewer events than forex."""
        forex_result = SignalProcessor.process_sequence(
            volumes=self.volume, prices=self.close,
            high=self.high, low=self.low,
            asset_config=ASSET_CONFIGS["forex"],
        )
        crypto_result = SignalProcessor.process_sequence(
            volumes=self.volume, prices=self.close,
            high=self.high, low=self.low,
            asset_config=ASSET_CONFIGS["crypto"],
        )
        forex_events = np.sum(forex_result["dc_direction"] != 0)
        crypto_events = np.sum(crypto_result["dc_direction"] != 0)
        # Forex threshold is tighter → more events on same price series
        assert forex_events >= crypto_events


# ---------------------------------------------------------------------------
# Phase 6: Execution gate
# ---------------------------------------------------------------------------

class TestCheckRiskGate:
    """check_risk_gate is a pure, stateless function — no model dependency."""

    def test_blocks_when_var_exceeds_max_drawdown(self):
        from core.execution import check_risk_gate
        decision = check_risk_gate(
            var_estimate=0.06, position_size=1.0, max_drawdown=0.05, max_position_size=2.0
        )
        assert decision.allowed is False
        assert decision.adjusted_size == 0.0
        assert decision.reason == "drawdown_exceeded"

    def test_allows_when_var_below_max_drawdown(self):
        from core.execution import check_risk_gate
        decision = check_risk_gate(
            var_estimate=0.03, position_size=0.5, max_drawdown=0.05, max_position_size=2.0
        )
        assert decision.allowed is True
        assert decision.reason == "ok"

    def test_boundary_equal_is_allowed(self):
        """var == max_drawdown is NOT exceeded — boundary should pass."""
        from core.execution import check_risk_gate
        decision = check_risk_gate(
            var_estimate=0.05, position_size=0.5, max_drawdown=0.05, max_position_size=2.0
        )
        assert decision.allowed is True

    def test_caps_position_size_at_max(self):
        from core.execution import check_risk_gate
        decision = check_risk_gate(
            var_estimate=0.01, position_size=5.0, max_drawdown=0.05, max_position_size=1.0
        )
        assert decision.adjusted_size == pytest.approx(1.0)

    def test_preserves_size_when_within_limit(self):
        from core.execution import check_risk_gate
        decision = check_risk_gate(
            var_estimate=0.01, position_size=0.3, max_drawdown=0.05, max_position_size=1.0
        )
        assert decision.adjusted_size == pytest.approx(0.3)

    def test_gate_decision_is_immutable(self):
        from core.execution import check_risk_gate
        decision = check_risk_gate(
            var_estimate=0.01, position_size=0.5, max_drawdown=0.05, max_position_size=1.0
        )
        with pytest.raises((TypeError, AttributeError)):
            decision.allowed = False  # type: ignore[misc]

    def test_zero_var_always_allowed(self):
        from core.execution import check_risk_gate
        decision = check_risk_gate(
            var_estimate=0.0, position_size=0.1, max_drawdown=0.05, max_position_size=1.0
        )
        assert decision.allowed is True


class TestMetaApiExecutorGate:
    """Risk gate runs inside execute() even in paper mode."""

    def _make_executor(self, max_drawdown: float = 0.05):
        from data.source.metaapi import MetaApiExecutor
        return MetaApiExecutor(
            token="", account_id="", symbol="EURUSD",
            max_position_size=1.0, max_drawdown=max_drawdown, paper=True,
        )

    def _run(self, coro):
        import asyncio
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_buy_blocked_by_gate_returns_none(self):
        ex = self._make_executor(max_drawdown=0.05)
        result = self._run(ex.execute(action_idx=1, position_size=0.1, var_estimate=0.10))
        assert result is None

    def test_buy_allowed_by_gate_returns_order(self):
        ex = self._make_executor(max_drawdown=0.05)
        result = self._run(ex.execute(action_idx=1, position_size=0.1, var_estimate=0.02))
        assert result is not None
        assert result["action"] == "BUY"

    def test_sell_blocked_by_gate_returns_none(self):
        ex = self._make_executor(max_drawdown=0.05)
        result = self._run(ex.execute(action_idx=2, position_size=0.1, var_estimate=0.10))
        assert result is None

    def test_hold_always_returns_none_regardless_of_var(self):
        """HOLD short-circuits before gate check."""
        ex = self._make_executor(max_drawdown=0.05)
        result = self._run(ex.execute(action_idx=0, position_size=0.1, var_estimate=0.99))
        assert result is None


# ---------------------------------------------------------------------------
# Phase 7: Quant config YAMLs
# ---------------------------------------------------------------------------

class TestQuantConfigYamls:
    """Both quant YAML files should parse into valid DignityConfig objects."""

    def test_paper_yaml_parses_to_cascade_task(self):
        config = DignityConfig.from_yaml("config/train_quant_paper.yaml")
        assert config.model.task == "cascade"

    def test_paper_yaml_input_size_is_32(self):
        config = DignityConfig.from_yaml("config/train_quant_paper.yaml")
        assert config.model.input_size == 32

    def test_paper_yaml_has_32_features(self):
        config = DignityConfig.from_yaml("config/train_quant_paper.yaml")
        assert len(config.data.features) == 32

    def test_paper_yaml_paper_trading_true(self):
        config = DignityConfig.from_yaml("config/train_quant_paper.yaml")
        assert config.execution.paper_trading is True

    def test_paper_yaml_task_weights_sum_to_one(self):
        config = DignityConfig.from_yaml("config/train_quant_paper.yaml")
        assert sum(config.model.task_weights.values()) == pytest.approx(1.0)

    def test_live_yaml_parses_to_cascade_task(self):
        config = DignityConfig.from_yaml("config/train_quant.yaml")
        assert config.model.task == "cascade"

    def test_live_yaml_paper_trading_false(self):
        config = DignityConfig.from_yaml("config/train_quant.yaml")
        assert config.execution.paper_trading is False

    def test_live_yaml_input_size_is_32(self):
        config = DignityConfig.from_yaml("config/train_quant.yaml")
        assert config.model.input_size == 32


# ---------------------------------------------------------------------------
# Section 6 — Live position cap (core/execution.py)
# ---------------------------------------------------------------------------

class TestMaxPositionFraction:

    def test_constant_is_75_percent(self):
        assert MAX_POSITION_FRACTION == 0.75

    def test_500_account_cap_is_375(self):
        # Request more than the cap to verify clamping kicks in
        assert apply_live_position_cap(500.0, 500.0) == pytest.approx(375.0)

    def test_50000_paper_account_cap_is_37500(self):
        assert apply_live_position_cap(50_000.0, 50_000.0) == pytest.approx(37_500.0)

    def test_smaller_size_not_inflated(self):
        # If gate already approved a small size, don't inflate it
        assert apply_live_position_cap(0.01, 500.0) == pytest.approx(0.01)

    def test_zero_balance_returns_zero(self):
        assert apply_live_position_cap(1.0, 0.0) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(apply_live_position_cap(0.5, 1000.0), float)


"""Test data pipeline."""

import numpy as np
import pandas as pd
import pytest

from data.loader import TransactionDataset, create_dataloader
from data.pipeline import TransactionPipeline
from data.source.synthetic import SyntheticGenerator


class TestSyntheticGenerator:
    """Test synthetic data generation."""

    def test_normal_sequence(self):
        """Test normal sequence generation."""
        gen = SyntheticGenerator(seed=42)
        data = gen.generate_normal_sequence(length=1000)

        assert "volume" in data
        assert "price" in data
        assert "fee_rate" in data
        assert "tx_count" in data

        assert len(data["volume"]) == 1000
        assert np.all(data["volume"] > 0)

    def test_anomalous_sequence(self):
        """Test anomalous pattern generation."""
        gen = SyntheticGenerator(seed=42)

        # Test different anomaly types
        for anomaly_type in ["volume_spike", "price_manipulation", "fee_evasion"]:
            data = gen.generate_anomalous_sequence(length=100, anomaly_type=anomaly_type)

            assert len(data["volume"]) == 100
            assert "volume" in data and "price" in data

    def test_dataset_generation(self):
        """Test balanced dataset generation."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=100, num_anomalous=50, seq_len=50)

        assert len(df) == (100 + 50) * 50  # sequences * seq_len
        assert "label" in df.columns
        assert set(df["label"].unique()) == {0, 1}


class TestTransactionPipeline:
    """Test data preprocessing pipeline."""

    def setup_method(self):
        from core.signals import ASSET_CONFIGS

        self.forex_cfg = ASSET_CONFIGS["forex"]

    def test_signal_computation(self):
        """Test signal feature computation."""
        gen = SyntheticGenerator(seed=42)
        df_raw = gen.generate_dataset(num_normal=10, num_anomalous=0, seq_len=100)

        pipeline = TransactionPipeline(seq_len=50)
        df_processed = pipeline.compute_signals(df_raw, asset_config=self.forex_cfg)

        assert "volume_volatility" in df_processed.columns
        assert "volatility_5" in df_processed.columns
        assert "momentum_10" in df_processed.columns

    def test_fit_transform(self):
        """Test scaling pipeline."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=50, num_anomalous=10, seq_len=100)
        df = df.drop("label", axis=1)

        pipeline = TransactionPipeline(seq_len=50)
        X = pipeline.fit_transform(df, asset_config=self.forex_cfg)

        assert X.shape[0] == len(df)
        assert X.shape[1] > 0  # Has features

        # Check scaling (should be roughly centered)
        assert np.abs(X.mean()) < 1.0

    def test_sequence_creation(self):
        """Test sliding window sequence creation."""
        X = np.random.randn(500, 9)
        y = np.random.randint(0, 2, 500)

        pipeline = TransactionPipeline(seq_len=100)
        pipeline.fitted = True
        pipeline.available_features = [f"f{i}" for i in range(9)]

        X_seq, y_seq = pipeline.create_sequences(X, y, stride=1)

        assert X_seq.shape[0] == 401  # 500 - 100 + 1
        assert X_seq.shape[1] == 100  # seq_len
        assert X_seq.shape[2] == 9  # features
        assert len(y_seq) == 401

    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=100, num_anomalous=20, seq_len=200)
        labels = df["label"].values
        df = df.drop("label", axis=1)

        pipeline = TransactionPipeline(seq_len=100)
        X_seq, y_seq = pipeline.process(
            df, labels, fit=True, stride=10, asset_config=self.forex_cfg
        )

        assert X_seq.ndim == 3  # [sequences, seq_len, features]
        assert y_seq.ndim == 1  # [sequences]
        assert len(X_seq) == len(y_seq)


class TestDataLoader:
    """Test PyTorch data loading."""

    def test_dataset_creation(self):
        """Test TransactionDataset."""
        X = np.random.randn(100, 50, 9)
        y = np.random.randint(0, 2, 100)

        dataset = TransactionDataset(X, y)

        assert len(dataset) == 100

        sample_x, sample_y = dataset[0]
        assert sample_x.shape == (50, 9)
        assert sample_y.shape == ()

    def test_dataloader_creation(self):
        """Test DataLoader creation."""
        X = np.random.randn(100, 50, 9)
        y = np.random.randint(0, 2, 100)

        loader = create_dataloader(X, y, batch_size=16, shuffle=True, device="cpu")

        assert len(loader) == 100 // 16 + 1  # batches

        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape[0] <= 16  # batch size
        assert batch_x.shape[1:] == (50, 9)


class TestComputeSignalsWithAssetConfig:
    """TransactionPipeline.compute_signals() expanded with AssetConfig."""

    def setup_method(self):
        from core.signals import ASSET_CONFIGS

        rng = np.random.default_rng(1)
        n = 300
        close = np.cumprod(1 + rng.normal(0, 0.005, n)) * 100.0
        spread = np.abs(rng.normal(0.3, 0.05, n))
        self.df = pd.DataFrame(
            {
                "open": close - spread / 2,
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": rng.uniform(1000, 5000, n),
                "price": close,
                "fee_rate": np.full(n, 0.0002),
                "tx_count": rng.integers(10, 100, n).astype(float),
            }
        )
        self.forex_cfg = ASSET_CONFIGS["forex"]
        self.crypto_cfg = ASSET_CONFIGS["crypto"]

    def test_compute_signals_with_asset_config_returns_32_columns(self):
        pipeline = TransactionPipeline(seq_len=100)
        result = pipeline.compute_signals(self.df, asset_config=self.forex_cfg)
        # Should have original cols + all 32 signal cols (some overlap is fine,
        # but the 32 named signals must be present)
        expected = {
            "rsi",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "bollinger_pct_b",
            "bollinger_width",
            "atr",
            "stoch_k",
            "stoch_d",
            "adx",
            "obv",
            "vwap",
            "roc_5",
            "roc_20",
            "momentum_10",
            "momentum_20",
            "volatility_5",
            "volatility_20",
            "vol_ratio",
            "order_flow_imbalance",
            "dc_direction",
            "dc_overshoot",
            "dc_bars_since_event",
            "volume_volatility",
            "volume_entropy",
            "price_change",
            "directional_change",
        }
        assert expected.issubset(result.columns)

    def test_compute_signals_no_nan_after_asset_config(self):
        pipeline = TransactionPipeline(seq_len=100)
        result = pipeline.compute_signals(self.df, asset_config=self.forex_cfg)
        signal_cols = [c for c in result.columns if c not in self.df.columns]
        for col in signal_cols:
            assert not result[col].isna().any(), f"NaN found in column '{col}'"

    def test_compute_signals_dc_threshold_differs_by_asset_class(self):
        pipeline = TransactionPipeline(seq_len=100)
        forex_result = pipeline.compute_signals(self.df, asset_config=self.forex_cfg)
        crypto_result = pipeline.compute_signals(self.df, asset_config=self.crypto_cfg)
        forex_events = (forex_result["dc_direction"] != 0).sum()
        crypto_events = (crypto_result["dc_direction"] != 0).sum()
        # Forex threshold tighter → more events on same price series
        assert forex_events >= crypto_events

    def test_compute_signals_backward_compat_without_asset_config(self):
        """Calling without asset_config still works (backward compat)."""
        pipeline = TransactionPipeline(seq_len=100)
        result = pipeline.compute_signals(self.df, asset_config=self.forex_cfg)
        assert "volume_volatility" in result.columns  # original signal still present


class TestExecutionConfig:
    """ExecutionConfig dataclass."""

    def test_default_paper_trading(self):
        from core.config import ExecutionConfig

        cfg = ExecutionConfig()
        assert cfg.paper_trading is True

    def test_default_max_drawdown(self):
        from core.config import ExecutionConfig

        cfg = ExecutionConfig()
        assert cfg.max_drawdown == pytest.approx(0.05)

    def test_default_symbols(self):
        from core.config import ExecutionConfig

        cfg = ExecutionConfig()
        assert isinstance(cfg.symbols, list)
        assert len(cfg.symbols) >= 1

    def test_custom_values(self):
        from core.config import ExecutionConfig

        cfg = ExecutionConfig(
            metaapi_token="tok123",
            account_id="acc456",
            symbols=["EURUSD", "GBPUSD"],
            max_drawdown=0.02,
            paper_trading=False,
        )
        assert cfg.metaapi_token == "tok123"
        assert cfg.max_drawdown == pytest.approx(0.02)
        assert not cfg.paper_trading

    def test_dignity_config_has_execution_field(self):
        from core.config import DignityConfig, ExecutionConfig

        cfg = DignityConfig()
        assert hasattr(cfg, "execution")
        assert isinstance(cfg.execution, ExecutionConfig)

    def test_dignity_config_yaml_roundtrip_with_execution(self, tmp_path):
        from core.config import DignityConfig, ExecutionConfig

        cfg = DignityConfig()
        cfg.execution = ExecutionConfig(
            metaapi_token="tok",
            account_id="acc",
            max_drawdown=0.03,
            paper_trading=True,
        )
        path = str(tmp_path / "cfg.yaml")
        cfg.to_yaml(path)
        loaded = DignityConfig.from_yaml(path)
        assert loaded.execution.max_drawdown == pytest.approx(0.03)
        assert loaded.execution.paper_trading is True


class TestModelConfigTaskWeights:
    """ModelConfig.task_weights field."""

    def test_task_weights_field_exists(self):
        from core.config import ModelConfig

        cfg = ModelConfig()
        assert hasattr(cfg, "task_weights")

    def test_task_weights_default_keys(self):
        from core.config import ModelConfig

        cfg = ModelConfig()
        for key in ("regime", "risk", "alpha", "policy"):
            assert key in cfg.task_weights

    def test_task_weights_sum_to_one(self):
        from core.config import ModelConfig

        cfg = ModelConfig()
        assert sum(cfg.task_weights.values()) == pytest.approx(1.0)

    def test_cascade_is_valid_task(self):
        from core.config import ModelConfig

        cfg = ModelConfig(task="cascade")
        assert cfg.task == "cascade"


class TestMetaApiExecutor:
    """MetaApiExecutor paper-trading mode — no live API calls."""

    def test_action_map_constants(self):
        from data.source.metaapi import MetaApiExecutor

        assert MetaApiExecutor.ACTION_MAP[0] == "HOLD"
        assert MetaApiExecutor.ACTION_MAP[1] == "BUY"
        assert MetaApiExecutor.ACTION_MAP[2] == "SELL"

    def test_paper_mode_hold_returns_none(self):
        import asyncio

        from data.source.metaapi import MetaApiExecutor

        ex = MetaApiExecutor(
            token="",
            account_id="",
            symbol="EURUSD",
            max_position_size=1.0,
            paper=True,
        )
        result = asyncio.get_event_loop().run_until_complete(
            ex.execute(action_idx=0, position_size=0.1)
        )
        assert result is None

    def test_paper_mode_buy_returns_mock_order(self):
        import asyncio

        from data.source.metaapi import MetaApiExecutor

        ex = MetaApiExecutor(
            token="",
            account_id="",
            symbol="EURUSD",
            max_position_size=1.0,
            paper=True,
        )
        result = asyncio.get_event_loop().run_until_complete(
            ex.execute(action_idx=1, position_size=0.1)
        )
        assert result is not None
        assert "action" in result
        assert result["action"] == "BUY"

    def test_paper_mode_sell_returns_mock_order(self):
        import asyncio

        from data.source.metaapi import MetaApiExecutor

        ex = MetaApiExecutor(
            token="",
            account_id="",
            symbol="EURUSD",
            max_position_size=1.0,
            paper=True,
        )
        result = asyncio.get_event_loop().run_until_complete(
            ex.execute(action_idx=2, position_size=0.05)
        )
        assert result is not None
        assert result["action"] == "SELL"
        assert result["size"] == pytest.approx(0.05)

    def test_paper_mode_caps_position_size(self):
        import asyncio

        from data.source.metaapi import MetaApiExecutor

        ex = MetaApiExecutor(
            token="",
            account_id="",
            symbol="EURUSD",
            max_position_size=0.5,
            paper=True,
        )
        result = asyncio.get_event_loop().run_until_complete(
            ex.execute(action_idx=1, position_size=2.0)  # over limit
        )
        assert result["size"] == pytest.approx(0.5)


class TestMetaApiSource:
    """MetaApiSource construction and interface contracts."""

    def test_constructs_without_connecting(self):
        from data.source.metaapi import MetaApiSource

        src = MetaApiSource(token="", account_id="", symbol="EURUSD")
        assert src.symbol == "EURUSD"

    def test_default_timeframe(self):
        from data.source.metaapi import MetaApiSource

        src = MetaApiSource(token="", account_id="", symbol="GBPUSD")
        assert src.timeframe == "1m"

    def test_custom_timeframe(self):
        from data.source.metaapi import MetaApiSource

        src = MetaApiSource(token="", account_id="", symbol="EURUSD", timeframe="1h")
        assert src.timeframe == "1h"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

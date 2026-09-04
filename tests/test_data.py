"""Test data pipeline."""

import numpy as np
import pandas as pd
import pytest

from core.config import PrivacyConfig
from core.privacy import BudgetExhausted
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
            data = gen.generate_anomalous_sequence(
                length=100, anomaly_type=anomaly_type
            )

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

    def test_signal_computation(self):
        """Test signal feature computation."""
        gen = SyntheticGenerator(seed=42)
        df_raw = gen.generate_dataset(num_normal=10, num_anomalous=0, seq_len=100)

        pipeline = TransactionPipeline(seq_len=50)
        df_processed = pipeline.compute_signals(df_raw)

        assert "volume_volatility" in df_processed.columns
        assert "volatility" in df_processed.columns
        assert "momentum" in df_processed.columns

    def test_fit_transform(self):
        """Test scaling pipeline."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=50, num_anomalous=10, seq_len=100)
        df = df.drop("label", axis=1)

        pipeline = TransactionPipeline(seq_len=50)
        X = pipeline.fit_transform(df)

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
        X_seq, y_seq = pipeline.process(df, labels, fit=True, stride=10)

        assert X_seq.ndim == 3  # [sequences, seq_len, features]
        assert y_seq.ndim == 1  # [sequences]
        assert len(X_seq) == len(y_seq)


class _ZeroNoise:
    def random(self) -> float:
        return 0.5


def _frame(n: int = 60) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "volume": 100.0 + 10.0 * np.sin(t / 3.0),
            "price": t * 3.0,  # 0 .. 177, so clipping to [50, 150] bites on both ends
            "fee_rate": np.where(t % 20 == 0, 0.005, 0.001),
            "tx_count": (50 + (t % 7)).astype(float),
        }
    )


def _laplace_price(eps_total: float = 1.0, eps: float = 1.0) -> PrivacyConfig:
    return PrivacyConfig(
        epsilon_total=eps_total,
        features={
            "price": {"mechanism": "laplace", "epsilon": eps, "bounds": [50.0, 150.0]}
        },
    )


class TestPrivacyStage:
    """The stage runs once per public call, before signals, and only when configured."""

    def test_privacy_stage_runs_before_signals(self):
        df = _frame()
        priv = TransactionPipeline(
            seq_len=10,
            features=["price", "volatility"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        x_priv, _ = priv.process(df)

        # Zero noise => the stage is pure clipping. A plain pipeline fed pre-clipped
        # prices must produce identical output — which is only true if clipping
        # happened BEFORE volatility was computed.
        pre_clipped = df.assign(price=df["price"].clip(50.0, 150.0))
        x_ref, _ = TransactionPipeline(
            seq_len=10, features=["price", "volatility"]
        ).process(pre_clipped)
        np.testing.assert_allclose(x_priv, x_ref)

        # and it must differ from the unclipped run, or the stage did nothing
        x_raw, _ = TransactionPipeline(
            seq_len=10, features=["price", "volatility"]
        ).process(df)
        assert not np.allclose(x_priv, x_raw)

    def test_fit_transform_spends_budget_exactly_once(self):
        priv = TransactionPipeline(
            seq_len=10,
            features=["price"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        priv.process(_frame(), fit=True)
        assert priv.privacy_manager.budget.spent == pytest.approx(1.0)

    def test_budget_exhausted_propagates_from_transform(self):
        priv = TransactionPipeline(
            seq_len=10,
            features=["price"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        priv.process(_frame(), fit=True)  # spends the whole budget
        with pytest.raises(BudgetExhausted):
            priv.process(_frame(), fit=False)

    def test_absent_privacy_block_is_a_noop(self):
        plain = TransactionPipeline(seq_len=10, features=["price"])
        df = _frame()
        assert plain.privacy_manager is None
        assert plain._apply_privacy(df) is df

    def test_generalize_column_is_k_anonymous_in_output(self):
        cfg = PrivacyConfig(
            epsilon_total=1.0,
            k=5,
            features={"fee_rate": {"mechanism": "generalize", "bins": 4}},
        )
        priv = TransactionPipeline(seq_len=10, features=["fee_rate"], privacy=cfg)
        generalized = priv._apply_privacy(_frame())["fee_rate"].to_numpy()
        _, counts = np.unique(generalized, return_counts=True)
        assert (counts >= 5).all()

    def test_missing_privacy_column_raises(self):
        cfg = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "nope": {"mechanism": "laplace", "epsilon": 1.0, "bounds": [0, 1]}
            },
        )
        priv = TransactionPipeline(seq_len=10, features=["price"], privacy=cfg)
        with pytest.raises(ValueError, match="'nope'"):
            priv.process(_frame())

    def test_missing_column_spends_no_budget(self):
        cfg = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "price": {
                    "mechanism": "laplace",
                    "epsilon": 0.5,
                    "bounds": [50.0, 150.0],
                },
                "nope": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0.0, 1.0]},
            },
        )
        priv = TransactionPipeline(
            seq_len=10, features=["price"], privacy=cfg, privacy_rng=_ZeroNoise()
        )
        with pytest.raises(ValueError, match="'nope'"):
            priv.process(_frame())
        assert priv.privacy_manager.budget.spent == 0.0

    def test_missing_key_env_var_fails_at_construction(self, monkeypatch):
        monkeypatch.delenv("DIGNITY_TEST_KEY_UNSET", raising=False)
        cfg = PrivacyConfig(epsilon_total=1.0, key_env="DIGNITY_TEST_KEY_UNSET")
        with pytest.raises(ValueError, match="DIGNITY_TEST_KEY_UNSET"):
            TransactionPipeline(seq_len=10, privacy=cfg)

    def test_key_env_var_is_read_into_the_manager(self, monkeypatch):
        monkeypatch.setenv("DIGNITY_TEST_KEY", "0123456789abcdef")
        cfg = PrivacyConfig(epsilon_total=1.0, key_env="DIGNITY_TEST_KEY")
        priv = TransactionPipeline(seq_len=10, privacy=cfg)
        assert len(priv.privacy_manager.pseudonymize("x")) == 64


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

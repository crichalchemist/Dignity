"""Pytest configuration and fixtures for Dignity tests."""

import numpy as np
import pytest
import torch
import yaml

BASE_FEATURES = [
    "volume",
    "fee_rate",
    "tx_count",
    "volatility",
    "price_change",
    "momentum",
    "directional_change",
    "volume_volatility",
    "regime",
]


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_sequence():
    """Sample transaction sequence for testing."""
    return torch.randn(4, 100, 9)  # [B, T, F]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return torch.randint(0, 2, (4,))  # Binary labels


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def tiny_train_config(tmp_path, monkeypatch):
    """Factory for a CPU training config that runs in seconds."""
    import train.cli
    from data.source.synthetic import SyntheticGenerator

    class _Small(SyntheticGenerator):
        def generate_dataset(self, num_normal=800, num_anomalous=200, seq_len=100):
            return super().generate_dataset(
                num_normal=40, num_anomalous=10, seq_len=seq_len
            )

    monkeypatch.setattr(train.cli, "SyntheticGenerator", _Small)

    def make(name="tiny", **sections):
        cfg = {
            "model": {"task": "risk", "hidden_size": 8, "n_layers": 1, "dropout": 0.0},
            "data": {
                "seq_len": 10,
                "batch_size": 64,
                "test_size": 0.2,
                "num_workers": 0,
                "features": BASE_FEATURES,
            },
            "train": {
                "epochs": 1,
                "save_interval": 1,
                "use_amp": True,
                "checkpoint_dir": str(tmp_path / "runs"),
            },
            "device": "cpu",
            "seed": 0,
        }
        for key, overrides in sections.items():
            cfg[key].update(overrides)
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        return path

    return make

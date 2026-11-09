"""Pytest configuration and fixtures for Dignity tests."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from core.config import DignityConfig

ROOT = Path(__file__).resolve().parents[1]
EASY_PRESET = ROOT / "config" / "synthetic_easy.yaml"

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
def easy_config() -> DignityConfig:
    """The shipped easy preset, resolved from the repo root, not the CWD."""
    return DignityConfig.from_yaml(str(EASY_PRESET))


@pytest.fixture(scope="session")
def easy_frame() -> pd.DataFrame:
    """Easy-preset event-record frame with 100 blocks, generated once per session.

    Shared across tests: read it, never mutate it.
    """
    from data.source.synthetic import SyntheticGenerator

    synthetic = DignityConfig.from_yaml(str(EASY_PRESET)).data.synthetic
    return SyntheticGenerator(dataclasses.replace(synthetic, n_blocks=100)).generate()


@pytest.fixture
def tiny_train_config(tmp_path):
    """Factory for a CPU training config that runs in seconds."""

    def make(name="tiny", **sections):
        cfg = {
            "model": {
                "task": "risk",
                "hidden_size": 8,
                "n_layers": 1,
                "dropout": 0.0,
                "horizon": 5,
            },
            "data": {
                "seq_len": 10,
                "batch_size": 64,
                "num_workers": 0,
                "features": BASE_FEATURES,
                "synthetic": {
                    "n_blocks": 50,
                    "block_len": 40,
                    "contamination": 0.5,
                    "duration_range": [3, 20],
                    "seed": 0,
                },
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


def _tiny_window_set(rng: np.random.Generator):
    """120 separable windows; every WindowSet field is passed by keyword."""
    from data.pipeline import WindowSet

    n, horizon = 120, 10
    y = (np.arange(n) % 2).astype(np.int8)
    X = rng.standard_normal((n, 20, 3)).astype(np.float32)
    X[y == 1, :, 0] += 2.0
    d = rng.integers(1, 21, n)
    y_forecast = np.stack([(d <= h + 1) for h in range(horizon)], axis=1)
    return WindowSet(
        X=X,
        y_risk=y,
        y_forecast=y_forecast.astype(np.float32),
        forecast_mask=y == 0,
        block_index=np.arange(n) // 6,
        onset_offset=np.where(y == 1, 0, -1),
        anomaly_type=np.where(y == 1, "wash_trading", "none").astype(object),
    )


@pytest.fixture
def tiny_split():
    """A SplitData of three 120-window splits, drawn from one rng in order."""
    from data.pipeline import SplitData

    rng = np.random.default_rng(0)
    train, val, test = (_tiny_window_set(rng) for _ in range(3))
    return SplitData(
        train=train,
        val=val,
        test=test,
        feature_names=["a", "b", "c"],
        scaler_center=np.zeros(3),
        scaler_scale=np.ones(3),
        epsilon_spent={},
    )


@pytest.fixture
def tiny_fit_config() -> DignityConfig:
    """A CPU config for fit() that finishes in seconds."""
    config = DignityConfig()
    config.device = "cpu"
    config.model.hidden_size = 16
    config.model.n_layers = 1
    config.data.batch_size = 32
    config.data.num_workers = 0
    config.train.epochs = 3
    config.train.use_amp = False
    config.train.save_interval = 1
    return config

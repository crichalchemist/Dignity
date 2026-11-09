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

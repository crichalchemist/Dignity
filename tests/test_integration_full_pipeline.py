"""Integration tests for full training pipeline end-to-end."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from config.config import DataConfig, ModelConfig, TrainingConfig
from data.agents.single_task_agent import SingleTaskDataAgent
from models.agent_hybrid import build_model
from train.core.agent_train import train_model
from utils.seed import set_seed
from utils.training_checkpoint import CheckpointManager, EarlyStopping


@pytest.fixture
def synthetic_data():
    """Generate synthetic OHLCV data for testing."""
    set_seed(42)
    n_samples = 1000

    data = {
        "datetime": pd.date_range("2020-01-01", periods=n_samples, freq="1min"),
        "open": np.random.randn(n_samples).cumsum() + 100,
        "high": np.random.randn(n_samples).cumsum() + 101,
        "low": np.random.randn(n_samples).cumsum() + 99,
        "close": np.random.randn(n_samples).cumsum() + 100,
        "volume": np.abs(np.random.randn(n_samples) * 1e6) + 1e6,
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(synthetic_data):
    """Create temporary CSV file with synthetic data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        synthetic_data.to_csv(f, index=False)
        yield f.name
    Path(f.name).unlink()


class TestIntegrationPipeline:
    """End-to-end integration tests for training pipeline."""

    def test_data_loading_to_training(self, temp_data_file):
        """Test full pipeline from data loading to model training."""
        set_seed(42)

        # Config setup
        data_cfg = DataConfig(
            csv_path=temp_data_file,
            t_in=10,
            t_out=5,
            target_type="classification",
        )

        model_cfg = ModelConfig(
            num_features=5,
            hidden_size_lstm=16,
            num_layers_lstm=1,
            cnn_num_filters=8,
            attention_dim=16,
        )

        train_cfg = TrainingConfig(
            epochs=1,
            batch_size=32,
            learning_rate=1e-3,
            device="cpu",
            log_every=10,
        )

        # Create data agent and loaders
        agent = SingleTaskDataAgent(data_cfg)
        datasets = agent.load_and_split(temp_data_file)
        loaders = agent.build_dataloaders(datasets, batch_size=32)

        # Verify data shapes
        train_loader = loaders["train"]
        batch_x, batch_y = next(iter(train_loader))
        assert batch_x.shape[0] == 32  # batch size
        assert batch_x.shape[1] == 10  # t_in
        assert batch_x.shape[2] == 5  # num_features

        # Build model
        model = build_model(model_cfg, task_type="classification")

        # Train for 1 epoch
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=loaders["val"],
            cfg=train_cfg,
            task_type="classification",
        )

        # Verify training occurred
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0
        assert history["train_loss"][0] > 0

    def test_checkpoint_management(self, tmp_path):
        """Test EarlyStopping and CheckpointManager."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Test CheckpointManager
        manager = CheckpointManager(checkpoint_dir, top_n=2, mode="min")

        # Create dummy state dicts
        state1 = {"layer1.weight": torch.randn(10, 10)}
        state2 = {"layer1.weight": torch.randn(10, 10)}
        state3 = {"layer1.weight": torch.randn(10, 10)}

        # Save checkpoints with different scores
        manager.save(state1, score=0.5, epoch=1)
        manager.save(state2, score=0.3, epoch=2)
        manager.save(state3, score=0.4, epoch=3)

        # Verify only top 2 remain
        assert len(manager.checkpoints) == 2
        assert manager.checkpoints[0][0] == 0.3  # Best score first

        # Verify best checkpoint can be loaded
        best_state = manager.load_best()
        assert best_state is not None
        assert "layer1.weight" in best_state

        # Test EarlyStopping
        early_stop = EarlyStopping(patience=3, min_delta=1e-3, mode="min")

        scores = [0.5, 0.45, 0.40, 0.401, 0.402, 0.403]
        for i, score in enumerate(scores):
            should_stop = early_stop(score)
            if i < 2:
                assert not should_stop  # First 2 improve
            elif i == 2:
                assert not should_stop  # Improvement
            elif i >= 3 and i >= 5:  # After patience exceeded
                assert should_stop

    def test_model_inference_shapes(self):
        """Test model output shapes match expected dimensions."""
        set_seed(42)

        model_cfg = ModelConfig(
            num_features=5,
            hidden_size_lstm=16,
            num_layers_lstm=1,
            cnn_num_filters=8,
            attention_dim=16,
        )

        model = build_model(model_cfg, task_type="classification")

        # Create batch
        batch = torch.randn(8, 10, 5)  # [batch, time, features]

        # Forward pass
        outputs, attn_weights = model(batch)

        # Verify outputs
        assert "direction_logits" in outputs
        assert outputs["direction_logits"].shape == (8, 3)  # [batch, num_classes]
        assert "return" in outputs
        assert outputs["return"].shape == (8, 1)

        # Verify attention weights
        assert attn_weights is not None


class TestDataSplitIntegrity:
    """Test data split logic for temporal integrity."""

    def test_time_ordered_splits(self, synthetic_data):
        """Verify train/val/test splits are time-ordered and non-overlapping."""
        agent = SingleTaskDataAgent(
            DataConfig(
                csv_path="dummy",
                t_in=10,
                t_out=5,
                train_ratio=0.6,
                val_ratio=0.2,
            )
        )

        splits = agent.split_dataframe(synthetic_data)

        # Check time ordering
        train_max = splits["train"]["datetime"].max()
        val_min = splits["val"]["datetime"].min()
        val_max = splits["val"]["datetime"].max()
        test_min = splits["test"]["datetime"].min()

        assert train_max < val_min, "Train/val overlap detected"
        assert val_max < test_min, "Val/test overlap detected"

        # Check non-overlapping indices
        train_idx = set(splits["train"].index)
        val_idx = set(splits["val"].index)
        test_idx = set(splits["test"].index)

        assert train_idx.isdisjoint(val_idx)
        assert val_idx.isdisjoint(test_idx)
        assert train_idx.isdisjoint(test_idx)


class TestCLIArgumentParsing:
    """Test CLI argument parsing for robustness."""

    def test_training_cli_args(self):
        """Test train/run_training.py argument parsing."""
        import sys

        from train.run_training import parse_args

        # Save original argv
        original_argv = sys.argv
        try:
            sys.argv = [
                "run_training.py",
                "--pairs", "gbpusd,eurusd",
                "--epochs", "5",
                "--batch-size", "64",
                "--num-workers", "4",
                "--use-amp",
            ]
            args = parse_args()

            assert args.pairs == "gbpusd,eurusd"
            assert args.epochs == 5
            assert args.batch_size == 64
            assert args.num_workers == 4
            assert args.use_amp is True
        finally:
            sys.argv = original_argv

    def test_a3c_cli_args(self):
        """Test rl/run_a3c_training.py argument parsing."""
        import sys

        from rl.run_a3c_training import parse_args

        original_argv = sys.argv
        try:
            sys.argv = [
                "run_a3c_training.py",
                "--pair", "eurusd",
                "--num-workers", "8",
                "--total-steps", "50000",
                "--use-amp",
            ]
            args = parse_args()

            assert args.pair == "eurusd"
            assert args.num_workers == 8
            assert args.total_steps == 50000
        finally:
            sys.argv = original_argv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

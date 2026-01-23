"""Root conftest for all tests - shared fixtures and configuration."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    TORCH_AVAILABLE = False

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    """Apply per-test timeouts based on markers.

    Maps:
    - @pytest.mark.fast -> timeout(5)
    - @pytest.mark.timeout_300 -> timeout(300)
    """
    try:
        import pytest
        has_timeout = hasattr(pytest.mark, "timeout")
    except Exception:
        has_timeout = False

    if not has_timeout:
        # pytest-timeout not available; skip mapping
        return

    for item in items:
        if item.get_closest_marker("fast"):
            item.add_marker(pytest.mark.timeout(5))
        if item.get_closest_marker("timeout_300"):
            item.add_marker(pytest.mark.timeout(300))


def pytest_ignore_collect(collection_path, config):
    """Skip torch-dependent tests when torch is unavailable."""
    if TORCH_AVAILABLE:
        return False
    ext = collection_path.suffix
    if ext != ".py":
        return False
    # Avoid importing torch-heavy tests by scanning for direct torch imports.
    try:
        content = collection_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "import torch" in content or "from torch" in content


@pytest.fixture(scope="session")
def device():
    """Fixture for torch device (CPU or CUDA)."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch is not available in this environment")
    return torch.device("cpu")


@pytest.fixture
def sample_batch(device):
    """Create a sample batch for testing (batch_size=4, seq_len=120, features=20)."""
    x = torch.randn(4, 120, 20, device=device)  # batch, seq_len, features
    y = {
        "primary": torch.randint(0, 3, (4,), device=device),  # 3-class classification
        "max_return": torch.randn(4, 1, device=device),
        "topk_returns": torch.randn(4, 3, device=device),
        "topk_prices": torch.randn(4, 3, device=device),
    }
    return x, y


@pytest.fixture
def sample_batch_regression(device):
    """Create a sample batch for regression tasks."""
    x = torch.randn(4, 120, 20, device=device)
    y = {
        "primary": torch.randn(4, 1, device=device),  # Regression target
        "max_return": torch.randn(4, 1, device=device),
        "topk_returns": torch.randn(4, 3, device=device),
        "topk_prices": torch.randn(4, 3, device=device),
    }
    return x, y


@pytest.fixture
def sample_dataframe():
    """Create a sample price DataFrame (120 timesteps, 20 features)."""
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    data = np.random.randn(120, 20)
    df = pd.DataFrame(data, index=dates, columns=[f"feature_{i}" for i in range(20)])
    return df


@pytest.fixture
def mock_model_config():
    """Mock ModelConfig with sensible defaults."""
    from config.config import ModelConfig
    return ModelConfig(
        num_features=20,
        num_classes=3,
        hidden_size_lstm=64,
        num_layers_lstm=1,
        cnn_num_filters=32,
        cnn_kernel_size=3,
        attention_dim=64,
        dropout=0.1,
        bidirectional=True,
        max_seq_length=120,
    )


@pytest.fixture
def mock_training_config(tmp_path):
    """Mock TrainingConfig with sensible defaults."""
    from config.config import TrainingConfig
    return TrainingConfig(
        batch_size=4,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        checkpoint_path=str(tmp_path / "checkpoints"),
        max_return_weight=0.1,
        topk_return_weight=0.1,
        topk_price_weight=0.1,
        sell_now_weight=0.0,
        grad_clip=1.0,
        log_every=10,
    )


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Fixture for temporary checkpoint directory."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir


@pytest.fixture
def minimal_dataloader(sample_batch):
    """Create a minimal DataLoader with one batch."""
    from torch.utils.data import DataLoader, TensorDataset
    x, y = sample_batch
    # Create simple dataset with mock batches - use y["primary"] directly to match batch size
    dataset = TensorDataset(x, y["primary"])
    return DataLoader(dataset, batch_size=4)

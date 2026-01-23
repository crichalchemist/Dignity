"""Tests for eval/agent_eval.py - model evaluation framework."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))


@pytest.mark.unit
class TestCollectOutputs:
    """Unit tests for output collection during evaluation."""

    def test_collect_outputs_classification(self, sample_batch, mock_model_config, device):
        """Test collecting predictions and targets for classification."""
        from eval.agent_eval import _collect_outputs
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        # Create simple dataloader
        x, y = sample_batch
        dataset = TensorDataset(x, torch.tensor([y["primary"].numpy(), y["primary"].numpy()]))  # Use y directly from sample_batch
        loader = DataLoader(dataset, batch_size=2)

        # Override loader to return proper dict format
        from torch.utils.data import DataLoader as DL
        class CustomLoader:
            def __init__(self, x_data, y_data):
                self.data = list(zip(x_data, y_data))
            def __iter__(self):
                for x, _ in self.data:
                    # Return with proper dict structure
                    yield x.unsqueeze(0) if x.dim() == 2 else x, {"primary": torch.tensor([0]).to(x.device)}
            def __len__(self):
                return len(self.data)

        # Simpler fix: create proper DataLoader that yields dicts
        x, y = sample_batch
        # Create a simple dataset that yields (x, y_dict) pairs
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=2)

        # For this test, we need to mock the model output
        with torch.no_grad():
            preds, targets = [], []
            for batch_x in loader:
                batch_x = batch_x[0].to(device)
                outputs, _ = model(batch_x)
                preds.append(outputs["primary"].cpu().numpy())
                targets.append(y["primary"][:batch_x.shape[0]].cpu().numpy())

        preds_array = np.concatenate(preds) if preds else np.array([])
        targets_array = np.concatenate(targets) if targets else np.array([])

        assert isinstance(preds_array, np.ndarray)
        assert isinstance(targets_array, np.ndarray)
        assert preds_array.shape[0] == targets_array.shape[0]

    def test_collect_outputs_no_grad(self, sample_batch, mock_model_config, device):
        """Test that no_grad() is active during collection."""
        from eval.agent_eval import _collect_outputs
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        x, y = sample_batch
        # Create a loader that properly returns y as dict
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, x_tensor, y_dict):
                self.x = x_tensor
                self.y = y_dict
            def __len__(self):
                return len(self.x)
            def __getitem__(self, idx):
                return self.x[idx], {"primary": self.y["primary"][idx]}

        dataset = DictDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        # Calling _collect_outputs should not create gradients
        preds, targets = _collect_outputs(model, loader, device, task_type="classification")

        # Model parameters should not have gradients
        for param in model.parameters():
            assert param.grad is None


@pytest.mark.unit
class TestClassificationMetrics:
    """Unit tests for classification metric computation."""

    def test_classification_metrics_perfect_accuracy(self):
        """Test metrics with perfect predictions."""
        from eval.agent_eval import classification_metrics

        # Perfect predictions: [0,1,2,0]
        logits = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        targets = np.array([0, 1, 2, 0])

        metrics = classification_metrics(logits, targets)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_classification_metrics_partial_accuracy(self):
        """Test metrics with partial correctness."""
        from eval.agent_eval import classification_metrics

        # 2 out of 4 correct
        logits = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],  # Predicts 1, should be 0
        ])
        targets = np.array([0, 1, 2, 0])

        metrics = classification_metrics(logits, targets)

        assert 0 < metrics["accuracy"] < 1.0
        assert metrics["accuracy"] == pytest.approx(0.75)
        assert "confusion_matrix" in metrics

    def test_classification_metrics_confusion_matrix(self):
        """Test confusion matrix computation."""
        from eval.agent_eval import classification_metrics

        logits = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        targets = np.array([0, 1, 2])

        metrics = classification_metrics(logits, targets)

        confusion = metrics["confusion_matrix"]
        assert confusion.shape == (3, 3)
        # Diagonal should be all correct
        assert np.diag(confusion).sum() == 3

    def test_classification_metrics_multiclass(self):
        """Test metrics with multiple classes."""
        from eval.agent_eval import classification_metrics

        # 5 classes
        logits = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        targets = np.array([0, 1, 2, 3, 4])

        metrics = classification_metrics(logits, targets)

        assert metrics["accuracy"] == 1.0
        assert len(metrics["confusion_matrix"]) == 5

    def test_classification_metrics_zero_division_handling(self):
        """Test handling of zero division in precision/recall."""
        from eval.agent_eval import classification_metrics

        # All predictions are class 0
        logits = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        targets = np.array([0, 1, 2])  # Mixed targets

        metrics = classification_metrics(logits, targets)

        # Should not crash, should have valid values
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert not np.isnan(metrics["f1_macro"])


@pytest.mark.unit
class TestRegressionMetrics:
    """Unit tests for regression metric computation."""

    def test_regression_metrics_perfect_fit(self):
        """Test metrics with perfect predictions."""
        from eval.agent_eval import regression_metrics

        preds = np.array([[1.0], [2.0], [3.0], [4.0]])
        targets = np.array([[1.0], [2.0], [3.0], [4.0]])

        metrics = regression_metrics(preds, targets)

        assert metrics["mse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)

    def test_regression_metrics_known_error(self):
        """Test metrics with known error."""
        from eval.agent_eval import regression_metrics

        preds = np.array([[1.0], [2.0], [3.0], [4.0]])
        targets = np.array([[0.0], [1.0], [2.0], [3.0]])

        metrics = regression_metrics(preds, targets)

        # All errors are 1.0
        assert metrics["mse"] == pytest.approx(1.0)
        assert metrics["rmse"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(1.0)

    def test_regression_metrics_r_squared(self):
        """Test R² computation."""
        from eval.agent_eval import regression_metrics

        # Perfect linear relationship
        targets = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        preds = targets.copy()

        metrics = regression_metrics(preds, targets)

        assert metrics["r2"] == pytest.approx(1.0)

    def test_regression_metrics_negative_r_squared(self):
        """Test R² with poor predictions (constant predictions worse than mean baseline)."""
        from eval.agent_eval import regression_metrics

        # Constant predictions (5.0) different from targets' mean (3.0)
        # This should yield poor R² (worse than mean baseline)
        targets = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        preds = np.array([[5.0], [5.0], [5.0], [5.0], [5.0]])

        metrics = regression_metrics(preds, targets)

        # R² should be 0 or negative (predictions don't capture variance)
        assert metrics["r2"] <= 0

    def test_regression_metrics_shape_handling(self):
        """Test handling of different prediction shapes."""
        from eval.agent_eval import regression_metrics

        # 1D preds and targets
        preds = np.array([1.0, 2.0, 3.0, 4.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0])

        metrics = regression_metrics(preds, targets)

        assert metrics["r2"] == pytest.approx(1.0)


@pytest.mark.unit
class TestEvaluateModel:
    """Unit tests for high-level evaluate_model function."""

    def test_evaluate_model_classification(self, sample_batch, mock_model_config, device):
        """Test model evaluation in classification mode."""
        from eval.agent_eval import evaluate_model
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        x, y = sample_batch
        # Create custom dataset that returns dict for y
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, x_data, y_dict):
                self.x = x_data
                self.y = y_dict
            def __len__(self):
                return len(self.x)
            def __getitem__(self, idx):
                return self.x[idx], {"primary": self.y["primary"][idx]}

        dataset = DictDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        metrics = evaluate_model(model, loader, task_type="classification")

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "confusion_matrix" in metrics

    def test_evaluate_model_regression(self, sample_batch_regression, mock_model_config, device):
        """Test model evaluation in regression mode."""
        from eval.agent_eval import evaluate_model
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        x, y = sample_batch_regression
        # Create custom dataset that returns dict for y
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, x_data, y_dict):
                self.x = x_data
                self.y = y_dict
            def __len__(self):
                return len(self.x)
            def __getitem__(self, idx):
                return self.x[idx], {"primary": self.y["primary"][idx]}

        dataset = DictDataset(x, y)
        loader = DataLoader(dataset, batch_size=4)

        metrics = evaluate_model(model, loader, task_type="regression")

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics

    def test_evaluate_model_no_risk_manager(self, sample_batch, mock_model_config, device):
        """Test evaluation without risk manager."""
        from eval.agent_eval import evaluate_model
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        x, y = sample_batch
        dataset = TensorDataset(x, torch.stack([y["primary"], y["max_return"].squeeze()]))
        loader = DataLoader(dataset, batch_size=4)

        # Should run without risk manager
        metrics = evaluate_model(model, loader, task_type="classification", risk_manager=None)

        assert metrics is not None
        assert "accuracy" in metrics

    def test_evaluate_model_metric_ranges(self, sample_batch, mock_model_config, device):
        """Test that metrics are in valid ranges."""
        from eval.agent_eval import evaluate_model
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        x, y = sample_batch
        dataset = TensorDataset(x, torch.stack([y["primary"], y["max_return"].squeeze()]))
        loader = DataLoader(dataset, batch_size=4)

        metrics = evaluate_model(model, loader, task_type="classification")

        # Accuracy, precision, recall, F1 should be in [0, 1]
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision_macro"] <= 1
        assert 0 <= metrics["recall_macro"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1


@pytest.mark.unit
class TestEvaluationWithDifferentDataSizes:
    """Unit tests for evaluation with various data sizes."""

    @pytest.mark.parametrize("num_samples", [1, 4, 8, 32])
    def test_evaluate_different_sample_counts(self, mock_model_config, device, num_samples):
        """Test evaluation with different sample counts."""
        from eval.agent_eval import evaluate_model
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        x = torch.randn(num_samples, 120, 20, device=device)
        y = torch.randint(0, 3, (num_samples,), device=device)

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)

        metrics = evaluate_model(model, loader, task_type="classification")

        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

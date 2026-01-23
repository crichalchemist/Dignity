"""Tests for train/core/agent_train.py - loss computation and gradient flow."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))


@pytest.mark.unit
class TestLossComputation:
    """Unit tests for loss computation in agent_train.py."""

    def test_compute_losses_classification_single_batch(self, sample_batch, mock_training_config, device):
        """Test primary loss computation for classification task."""
        from train.core.agent_train import _compute_losses

        x, y = sample_batch

        # Create mock outputs that match expected structure
        outputs = {
            "primary": torch.randn(4, 3, device=device),  # logits for 3 classes
            "max_return": torch.randn(4, 1, device=device),
            "topk_returns": torch.randn(4, 3, device=device),
            "topk_prices": torch.randn(4, 3, device=device),
        }

        total_loss, losses = _compute_losses(outputs, y, mock_training_config, "classification")

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() > 0
        assert "primary" in losses
        assert "max_return" in losses
        assert "topk_returns" in losses
        assert "topk_prices" in losses
        assert all(isinstance(v, float) for v in losses.values())

    def test_compute_losses_regression_single_batch(self, sample_batch_regression, mock_training_config, device):
        """Test primary loss computation for regression task."""
        from train.core.agent_train import _compute_losses

        x, y = sample_batch_regression

        outputs = {
            "primary": torch.randn(4, 1, device=device),  # Regression output
            "max_return": torch.randn(4, 1, device=device),
            "topk_returns": torch.randn(4, 3, device=device),
            "topk_prices": torch.randn(4, 3, device=device),
        }

        total_loss, losses = _compute_losses(outputs, y, mock_training_config, "regression")

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() > 0
        assert "primary" in losses

    def test_compute_losses_with_sell_now(self, sample_batch, mock_training_config, device):
        """Test loss computation with sell_now auxiliary task."""
        from train.core.agent_train import _compute_losses

        x, y = sample_batch
        y["sell_now"] = torch.randint(0, 2, (4,), device=device).float()

        mock_training_config.sell_now_weight = 0.5

        outputs = {
            "primary": torch.randn(4, 3, device=device),
            "max_return": torch.randn(4, 1, device=device),
            "topk_returns": torch.randn(4, 3, device=device),
            "topk_prices": torch.randn(4, 3, device=device),
            "sell_now": torch.randn(4, 1, device=device),
        }

        total_loss, losses = _compute_losses(outputs, y, mock_training_config, "classification")

        assert "sell_now" in losses
        assert total_loss.item() > 0

    def test_compute_losses_gradient_flow(self, sample_batch, mock_training_config, device):
        """Test that gradients flow through loss computation."""
        from train.core.agent_train import _compute_losses

        x, y = sample_batch

        # Create outputs with requires_grad
        outputs = {
            "primary": torch.randn(4, 3, device=device, requires_grad=True),
            "max_return": torch.randn(4, 1, device=device, requires_grad=True),
            "topk_returns": torch.randn(4, 3, device=device, requires_grad=True),
            "topk_prices": torch.randn(4, 3, device=device, requires_grad=True),
        }

        total_loss, losses = _compute_losses(outputs, y, mock_training_config, "classification")

        # Backward pass should succeed without error
        total_loss.backward()

        # Check gradients exist
        assert outputs["primary"].grad is not None
        assert outputs["max_return"].grad is not None

    def test_compute_losses_weight_scaling(self, sample_batch, mock_training_config, device):
        """Test that auxiliary losses are correctly weighted."""
        from train.core.agent_train import _compute_losses

        x, y = sample_batch

        outputs = {
            "primary": torch.randn(4, 3, device=device, requires_grad=True),
            "max_return": torch.randn(4, 1, device=device, requires_grad=True),
            "topk_returns": torch.randn(4, 3, device=device, requires_grad=True),
            "topk_prices": torch.randn(4, 3, device=device, requires_grad=True),
        }

        # Set high weights to verify they affect total loss
        mock_training_config.max_return_weight = 10.0
        mock_training_config.topk_return_weight = 5.0

        total_loss, losses = _compute_losses(outputs, y, mock_training_config, "classification")

        # Total loss should be dominated by auxiliary tasks with high weights
        assert total_loss.item() > losses["primary"]

    def test_compute_losses_handles_nan(self, sample_batch, mock_training_config, device):
        """Test that NaN/Inf values are handled gracefully."""
        from train.core.agent_train import _compute_losses

        x, y = sample_batch

        # Create outputs with extreme values that may produce NaN
        outputs = {
            "primary": torch.randn(4, 3, device=device),
            "max_return": torch.tensor([[float('inf')] for _ in range(4)], device=device),
            "topk_returns": torch.randn(4, 3, device=device),
            "topk_prices": torch.randn(4, 3, device=device),
        }

        # This should not crash and NaN/Inf should be handled
        total_loss, losses = _compute_losses(outputs, y, mock_training_config, "classification")

        assert isinstance(total_loss, torch.Tensor)
        # Verify NaN/Inf were handled - loss should be finite
        assert torch.isfinite(total_loss).all(), "total_loss should be finite after NaN/Inf handling"
        # Verify all component losses are finite
        for loss_name, loss_value in losses.items():
            assert torch.isfinite(torch.tensor(loss_value)).all(), f"{loss_name} should be finite"

    def test_align_regression(self, device):
        """Test regression output alignment."""
        from train.core.agent_train import _align_regression

        # Test with different shapes
        preds = torch.randn(4, 1, device=device)
        targets = torch.randn(4, device=device)

        aligned_preds, aligned_targets = _align_regression(preds, targets)

        assert aligned_preds.shape == aligned_targets.shape
        assert aligned_preds.shape == (4,)

    def test_select_outputs_classification(self, device):
        """Test output selection for classification."""
        from train.core.agent_train import _select_outputs

        outputs = {
            "direction_logits": torch.randn(4, 3, device=device),
            "return": torch.randn(4, 1, device=device),
        }

        logits = _select_outputs(outputs, "classification")

        assert logits is outputs["direction_logits"]
        assert logits.shape == (4, 3)

    def test_select_outputs_regression(self, device):
        """Test output selection for regression."""
        from train.core.agent_train import _select_outputs

        outputs = {
            "direction_logits": torch.randn(4, 3, device=device),
            "return": torch.randn(4, 1, device=device),
        }

        output = _select_outputs(outputs, "regression")

        assert output is outputs["return"]
        assert output.shape == (4, 1)


@pytest.mark.unit
class TestClassificationMetrics:
    """Unit tests for classification metrics."""

    def test_classification_metrics_accuracy(self, device):
        """Test classification accuracy metric."""
        from train.core.agent_train import _classification_metrics

        # All correct predictions
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [5.0, 0.0, 0.0]], device=device)
        targets = torch.tensor([0, 1, 2, 0], device=device)

        accuracy = _classification_metrics(logits, targets)

        assert accuracy == 1.0

    def test_classification_metrics_partial_correct(self, device):
        """Test classification accuracy with partial correctness."""
        from train.core.agent_train import _classification_metrics

        # 2 out of 4 correct
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.0, 10.0, 0.0]], device=device)
        targets = torch.tensor([0, 1, 2, 0], device=device)  # Last prediction wrong

        accuracy = _classification_metrics(logits, targets)

        assert accuracy == pytest.approx(0.75)

    def test_classification_metrics_all_wrong(self, device):
        """Test classification accuracy with all wrong predictions."""
        from train.core.agent_train import _classification_metrics

        logits = torch.tensor([[0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], device=device)
        targets = torch.tensor([0, 1, 2, 0], device=device)

        accuracy = _classification_metrics(logits, targets)

        assert accuracy == 0.0


@pytest.mark.unit
class TestRegressionMetrics:
    """Unit tests for regression metrics."""

    def test_regression_rmse_perfect_fit(self, device):
        """Test RMSE with perfect predictions."""
        from train.core.agent_train import _regression_rmse

        preds = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
        targets = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)

        rmse = _regression_rmse(preds, targets)

        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_regression_rmse_known_error(self, device):
        """Test RMSE with known error."""
        from train.core.agent_train import _regression_rmse

        preds = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
        targets = torch.tensor([[0.0], [1.0], [2.0], [3.0]], device=device)

        rmse = _regression_rmse(preds, targets)

        # All errors are 1.0, so RMSE should be 1.0
        assert rmse == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
class TestGradientFlow:
    """Unit tests for gradient flow through training pipeline."""

    def test_gradients_exist_after_backward(self, sample_batch, device):
        """Test that gradients exist in model parameters after backward."""
        from config.config import ModelConfig
        from models.agent_hybrid import DignityModel
        from train.core.agent_train import _compute_losses

        model = DignityModel(ModelConfig(num_features=20, num_classes=3))
        model.to(device)
        model.train()

        x, y = sample_batch
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}

        # Get actual model outputs (removed torch.no_grad to allow gradient flow)
        outputs = model(x)

        # Extract primary output and create full outputs dict
        primary_output = outputs.get("direction_logits", outputs.get("primary"))
        outputs_dict = {
            "primary": primary_output,
            "max_return": outputs.get("max_return", torch.randn(4, 1, device=device, requires_grad=True)),
            "topk_returns": outputs.get("topk_returns", torch.randn(4, 3, device=device, requires_grad=True)),
            "topk_prices": outputs.get("topk_prices", torch.randn(4, 3, device=device, requires_grad=True)),
        }

        from config.config import TrainingConfig
        cfg = TrainingConfig(
            batch_size=4, epochs=1, learning_rate=1e-3, weight_decay=0.0,
            device="cpu", checkpoint_path="./test"
        )

        loss, _ = _compute_losses(outputs_dict, y, cfg, "classification")
        loss.backward()

        # Verify gradients flow through model parameters
        param_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert param_with_grad > 0, "Gradients should flow to model parameters"

    def test_gradient_clipping(self, device):
        """Test gradient clipping functionality."""
        # Create tensor with large gradients
        tensor = torch.randn(4, 3, device=device, requires_grad=True)
        loss = (tensor ** 2).sum()
        loss.backward()

        original_norm = torch.nn.utils.clip_grad_norm_(tensor, max_norm=1.0)

        # After clipping, norm should be <= 1.0
        grad_norm = tensor.grad.norm().item()
        assert grad_norm <= 1.0 or original_norm <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

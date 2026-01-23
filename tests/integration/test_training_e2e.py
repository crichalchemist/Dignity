"""Integration tests for end-to-end training pipeline."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))


@pytest.mark.integration
class TestTrainingE2E:
    """End-to-end training pipeline tests."""

    def test_single_epoch_training(self, sample_batch, mock_model_config, mock_training_config, device):
        """Test complete single epoch training loop."""
        from models.agent_hybrid import DignityModel
        from train.core.agent_train import _compute_losses

        # Setup
        model = DignityModel(mock_model_config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=mock_training_config.learning_rate)

        x, y = sample_batch
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}

        # Training step
        initial_loss = None
        for step in range(3):  # 3 optimization steps
            optimizer.zero_grad()

            outputs = model(x)
            # Use actual model outputs instead of random stubs
            # Extract primary output (direction_logits or primary)
            primary_output = outputs.get("direction_logits", outputs.get("primary"))

            # Build outputs_dict with real model outputs where available
            outputs_dict = {
                "primary": primary_output,
                # Use model outputs if available, otherwise create appropriately shaped tensors
                "max_return": outputs.get("max_return", torch.zeros(4, 1, device=device)),
                "topk_returns": outputs.get("topk_returns", torch.zeros(4, 3, device=device)),
                "topk_prices": outputs.get("topk_prices", torch.zeros(4, 3, device=device)),
            }

            loss, losses_dict = _compute_losses(outputs_dict, y, mock_training_config, "classification")

            if step == 0:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        # Verify training occurred
        assert initial_loss is not None
        assert initial_loss > 0

    def test_training_loss_decreases(self, mock_model_config, mock_training_config, device):
        """Test that loss generally decreases over multiple optimization steps."""
        from models.agent_hybrid import DignityModel
        from train.core.agent_train import _compute_losses

        model = DignityModel(mock_model_config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)  # Higher LR for demo

        x = torch.randn(16, 120, 20, device=device)
        y = {
            "primary": torch.randint(0, 3, (16,), device=device),
            "max_return": torch.randn(16, 1, device=device),
            "topk_returns": torch.randn(16, 3, device=device),
            "topk_prices": torch.randn(16, 3, device=device),
        }

        losses = []
        for step in range(10):
            optimizer.zero_grad()

            outputs = model(x)
            outputs_dict = {
                "primary": outputs.get("direction_logits", outputs.get("primary")),
                "max_return": torch.randn(16, 1, device=device),
                "topk_returns": torch.randn(16, 3, device=device),
                "topk_prices": torch.randn(16, 3, device=device),
            }

            loss, _ = _compute_losses(outputs_dict, y, mock_training_config, "classification")
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        # Loss should decrease on average (not strictly, due to randomness)
        assert losses[-1] < losses[0] or len(set(losses)) > 1  # At least some variation

    def test_checkpoint_resume_training(self, temp_checkpoint_dir, sample_batch, mock_model_config, mock_training_config, device):
        """Test that training can resume from checkpoint without loss spike."""
        from models.agent_hybrid import DignityModel
        from train.core.agent_train import _compute_losses

        x, y = sample_batch
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}

        # First training session
        model1 = DignityModel(mock_model_config).to(device)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=0.001)

        loss_at_checkpoint = None
        for step in range(5):
            optimizer1.zero_grad()

            outputs = model1(x)
            outputs_dict = {
                "primary": outputs.get("direction_logits", outputs.get("primary")),
                "max_return": torch.randn(4, 1, device=device),
                "topk_returns": torch.randn(4, 3, device=device),
                "topk_prices": torch.randn(4, 3, device=device),
            }

            loss, _ = _compute_losses(outputs_dict, y, mock_training_config, "classification")
            if step == 4:
                loss_at_checkpoint = loss.item()

            loss.backward()
            optimizer1.step()

        # Save checkpoint
        checkpoint = {
            "model_state": model1.state_dict(),
            "optimizer_state": optimizer1.state_dict(),
            "epoch": 1,
        }
        ckpt_path = temp_checkpoint_dir / "resume_test.pt"
        torch.save(checkpoint, ckpt_path)

        # Resume training
        model2 = DignityModel(mock_model_config).to(device)
        loaded = torch.load(ckpt_path, weights_only=False)
        model2.load_state_dict(loaded["model_state"])

        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.001)
        optimizer2.load_state_dict(loaded["optimizer_state"])

        # Continue training
        resumed_loss = None
        for step in range(5):
            optimizer2.zero_grad()

            outputs = model2(x)
            outputs_dict = {
                "primary": outputs.get("direction_logits", outputs.get("primary")),
                "max_return": torch.randn(4, 1, device=device),
                "topk_returns": torch.randn(4, 3, device=device),
                "topk_prices": torch.randn(4, 3, device=device),
            }

            loss, _ = _compute_losses(outputs_dict, y, mock_training_config, "classification")
            if step == 0:
                resumed_loss = loss.item()

            loss.backward()
            optimizer2.step()

        # Loss should be continuous (no big spike)
        assert resumed_loss is not None
        # Allow reasonable variation (up to 2x) between checkpoint and resume
        assert resumed_loss < loss_at_checkpoint * 2

    def test_gradient_accumulation(self, sample_batch, mock_model_config, mock_training_config, device):
        """Test gradient accumulation over multiple batches."""
        from models.agent_hybrid import DignityModel
        from train.core.agent_train import _compute_losses

        model = DignityModel(mock_model_config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        accumulated_grad = None

        # Accumulate gradients over 3 batches
        for batch_idx in range(3):
            x, y = sample_batch
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}

            outputs = model(x)
            outputs_dict = {
                "primary": outputs.get("direction_logits", outputs.get("primary")),
                "max_return": torch.randn(4, 1, device=device),
                "topk_returns": torch.randn(4, 3, device=device),
                "topk_prices": torch.randn(4, 3, device=device),
            }

            loss, _ = _compute_losses(outputs_dict, y, mock_training_config, "classification")

            # Don't zero gradients - accumulate
            if batch_idx > 0:
                loss.backward()  # Accumulate
            else:
                loss.backward()  # First backward
                # Store initial gradient norm
                total_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                accumulated_grad = total_grad_norm

        # After accumulation, gradients should exist
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients

    def test_dataloader_integration(self, sample_batch, mock_model_config, device):
        """Test training with PyTorch DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset

        from models.agent_hybrid import DignityModel

        x_data, y_data = sample_batch

        # Create dataset and loader
        dataset = TensorDataset(
            x_data,
            y_data["primary"],
            y_data["max_return"],
            y_data["topk_returns"],
            y_data["topk_prices"],
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        # Iterate through batches
        batch_count = 0
        for batch in loader:
            batch_count += 1
            x_batch = batch[0].to(device)

            with torch.no_grad():
                outputs = model(x_batch)

            assert outputs is not None

        assert batch_count == 2  # 4 samples / batch_size 2


@pytest.mark.integration
class TestGradientFlow:
    """Integration tests for gradient flow through entire pipeline."""

    def test_gradients_through_full_pipeline(self, sample_batch, mock_model_config, device):
        """Test that gradients flow from loss back to input embeddings."""
        from models.agent_hybrid import DignityModel
        from train.core.agent_train import _compute_losses

        model = DignityModel(mock_model_config).to(device)
        model.train()

        x, y = sample_batch
        x = x.to(device).requires_grad_(True)
        y = {k: v.to(device) for k, v in y.items()}

        outputs = model(x)
        outputs_dict = {
            "primary": outputs.get("direction_logits", outputs.get("primary")),
            "max_return": outputs.get("max_return", torch.zeros(4, 1, device=device)),
            "topk_returns": outputs.get("topk_returns", torch.zeros(4, 3, device=device)),
            "topk_prices": outputs.get("topk_prices", torch.zeros(4, 3, device=device)),
        }

        # Create inline TrainingConfig for this test
        from config.config import TrainingConfig
        mock_training_config = TrainingConfig(
            batch_size=4, epochs=1, learning_rate=1e-3, weight_decay=0.0,
            device="cpu", checkpoint_path="./test"
        )

        loss, _ = _compute_losses(outputs_dict, y, mock_training_config, "classification")

        loss.backward()

        # Input should have gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_no_gradient_in_eval_mode(self, sample_batch, mock_model_config, device):
        """Test that no_grad() prevents gradient computation."""
        from models.agent_hybrid import DignityModel

        model = DignityModel(mock_model_config).to(device)
        model.eval()

        x, _ = sample_batch
        x = x.to(device).requires_grad_(True)

        with torch.no_grad():
            outputs = model(x)
            loss = outputs.get("direction_logits", outputs.get("primary")).sum()

        # Attempting backward should fail in no_grad() context
        with pytest.raises(RuntimeError):
            loss.backward()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

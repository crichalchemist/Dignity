"""Test the uncertainty weighting module."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from train.loss_weighting import UncertaintyLossWeighting


def test_basic_functionality():
    """Test basic uncertainty weighting functionality."""
    print("Testing Basic Uncertainty Weighting...")

    # Create uncertainty weighting for two tasks
    task_names = ["task1_loss", "task2_loss"]
    uncertainty_weighting = UncertaintyLossWeighting(task_names=task_names)

    # Create dummy losses
    losses = {
        "task1_loss": torch.tensor(1.0),
        "task2_loss": torch.tensor(2.0)
    }

    # Apply uncertainty weighting
    weighted_losses, uncertainty_values = uncertainty_weighting.forward(losses)

    print(f"Input losses: {losses}")
    print(f"Uncertainty values: {uncertainty_values}")
    print(f"Weighted losses: {weighted_losses}")

    # Check that uncertainty values are close to 1.0 initially (since log(1.0) = 0)
    for task, uncertainty in uncertainty_values.items():
        assert abs(uncertainty - 1.0) < 0.1, f"Uncertainty for {task} should be ~1.0, got {uncertainty}"

    print("✓ Basic functionality test passed!")


def test_uncertainty_learning():
    """Test that uncertainty parameters can be learned."""
    print("\nTesting Uncertainty Learning...")

    task_names = ["regression_loss", "classification_loss"]
    uncertainty_weighting = UncertaintyLossWeighting(task_names=task_names)

    # Create an optimizer to update uncertainty parameters
    optimizer = torch.optim.Adam(uncertainty_weighting.parameters(), lr=0.01)

    # Create dummy data where classification should be more uncertain
    for epoch in range(10):
        optimizer.zero_grad()

        losses = {
            "regression_loss": torch.tensor(0.1, requires_grad=True),  # Easy task
            "classification_loss": torch.tensor(1.0, requires_grad=True)  # Hard task
        }

        # Get total weighted loss
        total_loss, uncertainty_values, reg_loss = uncertainty_weighting.get_total_weighted_loss(losses)

        total_loss.backward()
        optimizer.step()
        uncertainty_weighting.apply_bounds()

        if epoch % 3 == 0:
            print(f"Epoch {epoch}: uncertainties={uncertainty_values}, reg_loss={reg_loss.item():.4f}")

    final_uncertainties = uncertainty_weighting.get_uncertainty_values()
    print(f"Final uncertainties: {final_uncertainties}")

    # Classification should have higher uncertainty (easier to learn)
    assert final_uncertainties["classification_loss"] > final_uncertainties["regression_loss"]

    print("✓ Uncertainty learning test passed!")


def test_regularization():
    """Test regularization to prevent unbounded growth."""
    print("\nTesting Regularization...")

    task_names = ["loss1"]
    # Set very tight bounds to test regularization
    uncertainty_weighting = UncertaintyLossWeighting(
        task_names=task_names,
        log_variance_bounds=(-1.0, 1.0),
        uncertainty_reg_strength=1.0
    )

    optimizer = torch.optim.Adam(uncertainty_weighting.parameters(), lr=0.1)

    # Try to push parameter outside bounds
    for epoch in range(20):
        optimizer.zero_grad()

        loss = torch.tensor(1.0, requires_grad=True)
        losses = {"loss1": loss}

        total_loss, uncertainty_values, reg_loss = uncertainty_weighting.get_total_weighted_loss(losses)

        # Maximize loss to push uncertainty high
        (-total_loss).backward()  # Negative because we want to maximize

        optimizer.step()
        uncertainty_weighting.apply_bounds()

        if epoch % 5 == 0:
            current_log_var = uncertainty_weighting.get_log_variance_values()["loss1"]
            print(f"Epoch {epoch}: log_var={current_log_var:.3f}, reg_loss={reg_loss.item():.4f}")

            # Should be clamped to upper bound
            assert current_log_var <= 1.0 + 1e-6, f"Log variance should be <= 1.0, got {current_log_var}"

    print("✓ Regularization test passed!")


def test_initial_weights():
    """Test initialization with custom weights."""
    print("\nTesting Custom Initial Weights...")

    task_names = ["loss1", "loss2"]
    initial_weights = {"loss1": 2.0, "loss2": 0.5}  # Custom uncertainties

    uncertainty_weighting = UncertaintyLossWeighting(
        task_names=task_names,
        initial_weights=initial_weights
    )

    uncertainties = uncertainty_weighting.get_uncertainty_values()

    print(f"Initial uncertainties: {uncertainties}")

    # Should match initial weights
    assert abs(uncertainties["loss1"] - 2.0) < 0.1, f"loss1 uncertainty should be 2.0, got {uncertainties['loss1']}"
    assert abs(uncertainties["loss2"] - 0.5) < 0.1, f"loss2 uncertainty should be 0.5, got {uncertainties['loss2']}"

    print("✓ Custom initial weights test passed!")


if __name__ == "__main__":
    test_basic_functionality()
    test_uncertainty_learning()
    test_regularization()
    test_initial_weights()
    print("\n🎉 All uncertainty weighting tests passed!")

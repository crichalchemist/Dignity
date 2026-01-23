"""Dynamic loss weighting using learnable uncertainty parameters.

Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
Geometry and Semantics" by Kendall, Gal, and Cipolla (2018).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class UncertaintyLossWeighting(nn.Module):
    """Learnable uncertainty parameters for dynamic loss weighting.

    This module implements the homoscedastic uncertainty approach from Kendall et al. (2018)
    to automatically learn optimal loss weights for multi-task learning scenarios.

    Each task i has a learnable uncertainty parameter σ_i that is used to weight the loss:
    L_weighted_i = (1/σ_i²) * L_i + log(σ_i)

    The log(σ_i) term acts as regularization to prevent unbounded growth of σ_i.
    """

    def __init__(
            self,
            task_names: list[str],
            initial_weights: dict[str, float] | None = None,
            uncertainty_reg_strength: float = 0.1,
            log_variance_bounds: tuple[float, float] = (-5.0, 5.0)
    ):
        """Initialize uncertainty weighting module.

        Parameters
        ----------
        task_names : List[str]
            Names of tasks/loss components to weight.
        initial_weights : Optional[Dict[str, float]], default=None
            Initial uncertainty values for each task. If None, defaults to 1.0.
        uncertainty_reg_strength : float, default=0.1
            Strength of regularization for uncertainty parameters.
        log_variance_bounds : Tuple[float, float], default=(-5.0, 5.0)
            Bounds for log variance parameters to ensure numerical stability.
        """
        super().__init__()

        if not task_names:
            raise ValueError("task_names cannot be empty")

        self.task_names = task_names
        self.uncertainty_reg_strength = uncertainty_reg_strength
        self.log_variance_bounds = log_variance_bounds

        # Initialize log variance parameters for each task
        self.log_variances = nn.ParameterDict()

        for task_name in task_names:
            if initial_weights and task_name in initial_weights:
                # Convert to float first to avoid nested torch.tensor warning
                weight_val = float(initial_weights[task_name])
                initial_log_var = torch.tensor(weight_val, dtype=torch.float32).log()
            else:
                initial_log_var = torch.tensor(0.0, dtype=torch.float32)  # log(1.0) = 0, so σ = 1 initially

            self.log_variances[task_name] = nn.Parameter(initial_log_var)

        logger.info(f"Initialized UncertaintyLossWeighting for tasks: {task_names}")
        logger.info(f"Initial uncertainty parameters: {self.get_uncertainty_values()}")

    def get_uncertainty_values(self) -> dict[str, float]:
        """Get current uncertainty values (σ_i) for all tasks.

        Returns
        -------
        Dict[str, float]
            Current uncertainty values for each task.
        """
        uncertainties = {}
        for task_name in self.task_names:
            uncertainties[task_name] = torch.exp(self.log_variances[task_name]).item()
        return uncertainties

    def get_log_variance_values(self) -> dict[str, float]:
        """Get current log variance values for all tasks.

        Returns
        -------
        Dict[str, float]
            Current log variance values for each task.
        """
        return {task_name: param.item() for task_name, param in self.log_variances.items()}

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Apply uncertainty weighting to losses.

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Dictionary of task losses with keys matching self.task_names.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, float]]
            Tuple of (weighted_losses, uncertainty_values) where:
            - weighted_losses: Dict[str, torch.Tensor] with weighted loss for each task
            - uncertainty_values: Dict[str, float] with current uncertainty values
        """
        if set(losses.keys()) != set(self.task_names):
            raise ValueError(
                f"Loss keys {list(losses.keys())} don't match task names {self.task_names}"
            )

        weighted_losses = {}

        for task_name in self.task_names:
            loss = losses[task_name]
            log_var = self.log_variances[task_name]

            # Uncertainty weighting: (1/σ²) * L + log(σ)
            # Where σ² = exp(log_var)
            uncertainty = torch.exp(log_var)
            weighted_loss = (1.0 / uncertainty) * loss + log_var

            weighted_losses[task_name] = weighted_loss

        uncertainty_values = self.get_uncertainty_values()

        return weighted_losses, uncertainty_values

    def get_reg_loss(self) -> torch.Tensor:
        """Get regularization loss for uncertainty parameters.

        Returns
        -------
        torch.Tensor
            Regularization loss to prevent unbounded uncertainty growth.
        """
        reg_loss = 0.0
        for task_name in self.task_names:
            log_var = self.log_variances[task_name]
            # Regularization: encourage log variances to stay within reasonable bounds
            lower_bound, upper_bound = self.log_variance_bounds

            # Penalty for going outside bounds
            lower_penalty = torch.relu(lower_bound - log_var)
            upper_penalty = torch.relu(log_var - upper_bound)

            reg_loss += self.uncertainty_reg_strength * (lower_penalty + upper_penalty)

        return reg_loss

    def apply_bounds(self):
        """Apply bounds to log variance parameters for numerical stability."""
        lower_bound, upper_bound = self.log_variance_bounds

        for task_name in self.task_names:
            with torch.no_grad():
                log_var = self.log_variances[task_name]
                log_var.clamp_(lower_bound, upper_bound)

    def get_total_weighted_loss(self, losses: dict[str, torch.Tensor]) -> tuple[
        torch.Tensor, dict[str, float], torch.Tensor]:
        """Get total weighted loss including regularization.

        Parameters
        ----------
        losses : Dict[str, torch.Tensor]
            Dictionary of task losses.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float], torch.Tensor]
            Tuple of (total_weighted_loss, uncertainty_values, reg_loss) where:
            - total_weighted_loss: Sum of all weighted losses
            - uncertainty_values: Current uncertainty values for each task
            - reg_loss: Regularization loss for uncertainty parameters
        """
        weighted_losses, uncertainty_values = self.forward(losses)
        total_weighted_loss = sum(weighted_losses.values())
        reg_loss = self.get_reg_loss()

        return total_weighted_loss, uncertainty_values, reg_loss


def create_uncertainty_weighting(
        loss_config: dict[str, any],
        task_type: str = "multitask"
) -> UncertaintyLossWeighting:
    """Create uncertainty weighting module based on configuration.

    Parameters
    ----------
    loss_config : Dict[str, any]
        Loss configuration dictionary.
    task_type : str, default="multitask"
        Type of task ("multitask", "single_classification", "single_regression").

    Returns
    -------
    UncertaintyLossWeighting
        Configured uncertainty weighting module.
    """
    if not loss_config.get("use_uncertainty_weighting", False):
        logger.info("Uncertainty weighting disabled in configuration")
        return None

    # Define tasks based on task type
    if task_type == "multitask":
        task_names = [
            "direction_loss",
            "return_loss",
            "volatility_loss",
            "max_return_loss",
            "topk_returns_loss",
            "topk_prices_loss"
        ]
    elif task_type == "single_classification":
        task_names = ["direction_loss"]
    elif task_type == "single_regression":
        task_names = ["return_loss"]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Get initial weights from config if specified
    initial_weights = loss_config.get("initial_uncertainty_weights", {})
    uncertainty_reg_strength = loss_config.get("uncertainty_reg_strength", 0.1)
    log_variance_bounds = tuple(loss_config.get("log_variance_bounds", (-5.0, 5.0)))

    return UncertaintyLossWeighting(
        task_names=task_names,
        initial_weights=initial_weights,
        uncertainty_reg_strength=uncertainty_reg_strength,
        log_variance_bounds=log_variance_bounds
    )

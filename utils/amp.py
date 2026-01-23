"""Automatic Mixed Precision (AMP) training utilities for GPU memory efficiency."""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


class AMPTrainer:
    """Unified AMP training wrapper with automatic gradient scaling.

    Uses PyTorch's automatic mixed precision (AMP) to train with float16 tensors
    where possible, reducing memory usage and accelerating computation on modern GPUs.

    Parameters
    ----------
    enabled : bool, optional
        Whether to enable AMP training. Default: True if CUDA available.
    fp16 : bool, optional
        Whether to use FP16 instead of FP32. Default: False.
    device : str, optional
        Device to use for training. Default: "cuda".
    """

    def __init__(self, enabled: bool = True, fp16: bool = False, device: str = "cuda"):
        self.enabled = enabled and torch.cuda.is_available()
        self.fp16 = fp16 and torch.cuda.is_available()
        self.device = device
        self.scaler = GradScaler() if self.enabled else None

        if self.enabled:
            logger.info("AMP training enabled with GradScaler")
        if self.fp16:
            logger.info("FP16 training enabled")

    def backward(
            self,
            loss: torch.Tensor,
            optimizer: torch.optim.Optimizer,
            max_grad_norm: float = None,
            zero_grad_first: bool = False
    ) -> None:
        """Perform backward pass with gradient scaling.

        Parameters
        ----------
        loss : torch.Tensor
            Loss scalar to backpropagate.
        optimizer : torch.optim.Optimizer
            Optimizer for step.
        max_grad_norm : float, optional
            Max gradient norm for clipping. Default: None (no clipping).
        zero_grad_first : bool, optional
            Whether to zero gradients before backward. Default: False.
        """
        if zero_grad_first:
            optimizer.zero_grad()

        if self.enabled:
            self.scaler.scale(loss).backward()
            if max_grad_norm:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'],
                    max_grad_norm
                )
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'],
                    max_grad_norm
                )
            optimizer.step()

    def autocast_context(self):
        """Get autocast context manager for forward pass.

        Returns
        -------
        context manager
            torch.autocast if AMP enabled, else nullcontext.
        """
        if self.enabled:
            return torch.autocast('cuda', dtype=torch.float16)
        return nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss for AMP training.

        Parameters
        ----------
        loss : torch.Tensor
            Loss tensor to scale.

        Returns
        -------
        torch.Tensor
            Scaled loss if AMP enabled, otherwise original loss.
        """
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Step the optimizer with proper AMP handling.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to step.
        """
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients for gradient clipping.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer whose gradients to unscale.
        """
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)

    def is_enabled(self) -> bool:
        """Check if AMP is enabled.

        Returns
        -------
        bool
            True if AMP is enabled, False otherwise.
        """
        return self.enabled

    def is_fp16_enabled(self) -> bool:
        """Check if FP16 is enabled.

        Returns
        -------
        bool
            True if FP16 is enabled, False otherwise.
        """
        return self.fp16

    def get_scaler(self) -> GradScaler | None:
        """Get the GradScaler instance.

        Returns
        -------
        Optional[GradScaler]
            GradScaler if AMP enabled, None otherwise.
        """
        return self.scaler


# Alias for backward compatibility
AMPManager = AMPTrainer


def create_amp_manager(cfg) -> AMPTrainer:
    """Create an AMP manager from training configuration.

    Parameters
    ----------
    cfg : TrainingConfig
        Training configuration with AMP settings.

    Returns
    -------
    AMPTrainer
        Configured AMP trainer instance.
    """
    return AMPTrainer(
        enabled=getattr(cfg, 'use_amp', False),
        fp16=getattr(cfg, 'fp16', False),
        device=getattr(cfg, 'device', 'cuda')
    )


def convert_model_to_fp16(model: torch.nn.Module) -> torch.nn.Module:
    """Convert model parameters to FP16.

    Parameters
    ----------
    model : torch.nn.Module
        Model to convert.

    Returns
    -------
    torch.nn.Module
        Model with FP16 parameters.
    """
    return model.half()

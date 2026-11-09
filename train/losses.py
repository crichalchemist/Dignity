"""Loss factories for the training engine."""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def risk_loss(
    pos_weight: float | None,
) -> Callable[[Tensor, dict[str, Tensor]], Tensor]:
    """Return a mean binary cross-entropy loss over risk probabilities.

    When ``pos_weight`` is set, positive labels are weighted by it and negative
    labels by 1.
    """

    def loss_fn(outputs: Tensor, batch: dict[str, Tensor]) -> Tensor:
        p = outputs.reshape(-1)
        y = batch["y"].float().reshape(-1)
        w = None
        if pos_weight is not None:
            w = torch.where(y == 1, pos_weight, 1.0)
        return F.binary_cross_entropy(p, y, weight=w)

    return loss_fn


def forecast_loss(
    pos_weight: float | None,
) -> Callable[[Tensor, dict[str, Tensor]], Tensor]:
    """Return a masked mean binary cross-entropy over per-horizon onset probabilities.

    Windows with ``mask`` False (an anomaly already active) are excluded. The mean
    is over every unmasked element across all horizons. ``pos_weight`` weights
    positive labels and leaves negatives at 1; the head emits probabilities, so
    ``BCEWithLogits``' ``pos_weight`` does not apply.
    """

    def loss_fn(outputs: Tensor, batch: dict[str, Tensor]) -> Tensor:
        m = batch["mask"]
        if m.sum() == 0:
            return outputs.sum() * 0.0
        y = batch["y"][m].float()
        w = None
        if pos_weight is not None:
            w = 1 + (pos_weight - 1) * y
        return F.binary_cross_entropy(outputs[m].float(), y, weight=w)

    return loss_fn

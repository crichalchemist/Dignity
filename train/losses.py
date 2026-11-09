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

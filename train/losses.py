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


_SHARE_EMA = 0.95  # weight of the running review share kept per training batch


class ExpectedCostLoss:
    """Mean expected triage cost under the policy's softmax, with a review multiplier.

    ``cost_matrix`` is ``[3, 2]``: rows are actions (allow, review, block), columns
    the true state (0 normal, 1 anomaly). The softmax runs in float32.

    With a ``review_budget`` the review column's cost is raised by ``multiplier``
    for the gradient, and after every batch the multiplier moves by
    ``multiplier_lr * (share_ema - review_budget)``, never below
    ``multiplier_floor``, where ``share_ema`` is a running mean of the batch's mean
    review probability. The returned value is always the plain expected cost.
    """

    def __init__(
        self,
        cost_matrix: Tensor,
        review_budget: float | None = None,
        multiplier_lr: float = 0.1,
        multiplier_floor: float = -0.5,
    ):
        self._cost_matrix = cost_matrix
        self._budget = review_budget
        self._lr = multiplier_lr
        self._floor = multiplier_floor
        self.multiplier = 0.0
        self.share_ema = 0.0 if review_budget is None else review_budget

    def __call__(self, outputs: Tensor, batch: dict[str, Tensor]) -> Tensor:
        p = F.softmax(outputs.float(), dim=-1)
        costs = self._cost_matrix.to(p)[:, batch["y"]].T
        plain = (p * costs).sum(-1).mean()
        if self._budget is None:
            return plain
        p_review = p[:, 1].mean()
        loss = plain + self.multiplier * (p_review - p_review.detach())
        with torch.no_grad():
            self.share_ema = (
                _SHARE_EMA * self.share_ema + (1 - _SHARE_EMA) * p_review.item()
            )
            self.multiplier = max(
                self._floor,
                self.multiplier + self._lr * (self.share_ema - self._budget),
            )
        return loss

    def state_dict(self) -> dict[str, float]:
        return {"multiplier": self.multiplier, "share_ema": self.share_ema}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.multiplier = float(state["multiplier"])
        self.share_ema = float(state["share_ema"])


def expected_cost_loss(
    cost_matrix: Tensor,
    review_budget: float | None = None,
    multiplier_lr: float = 0.1,
    multiplier_floor: float = -0.5,
) -> ExpectedCostLoss:
    """Return the expected triage cost loss; see :class:`ExpectedCostLoss`.

    ``review_budget=None`` gives the plain mean expected cost and no state.
    """
    return ExpectedCostLoss(cost_matrix, review_budget, multiplier_lr, multiplier_floor)

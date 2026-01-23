"""Training module for Sequence FX Forecasting Toolkit."""

try:
    from train.loss_weighting import UncertaintyLossWeighting
except ModuleNotFoundError:  # pragma: no cover - torch may be missing in minimal envs
    UncertaintyLossWeighting = None

__all__ = ["UncertaintyLossWeighting"]

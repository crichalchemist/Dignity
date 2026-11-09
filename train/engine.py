"""Training engine with AMP and DDP support."""

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    grad_scaler: GradScaler,
    use_amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    """
    Train for one epoch over dict batches.

    Args:
        model: Model to train
        loader: Training dataloader yielding ``{"x": ..., **targets}`` dicts
        optimizer: Optimizer
        loss_fn: ``loss_fn(outputs, batch) -> scalar loss``
        device: Device
        grad_scaler: Caller-owned GradScaler (the engine never builds one)
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping value

    Returns:
        ``{"loss": mean batch loss}``
    """
    model.train()

    total_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc="Training")

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast(device.type, enabled=use_amp):
            outputs, _ = model(batch["x"])

        # The loss runs outside autocast in float32: RiskHead emits sigmoid
        # probabilities scored with BCELoss, which autocast refuses ("unsafe to
        # autocast"). Casting the half-precision outputs up is the
        # documented pattern for losses that must stay in float32.
        with autocast(device.type, enabled=False):
            loss = loss_fn(outputs.float(), batch)

        # Backward pass
        grad_scaler.scale(loss).backward()

        # Gradient clipping
        if grad_clip > 0:
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # Track loss
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {"loss": total_loss / num_batches}


def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    Score every window in ``loader``, in loader order.

    Returns ``{"score": float32 [N]}`` for the risk task and
    ``{"score": float32 [N, H]}`` (one onset probability per horizon) for forecast,
    and ``{"probs": float32 [N, 3]}`` (softmax over allow/review/block) for policy.
    """
    if model.task not in ("risk", "forecast", "policy"):
        raise NotImplementedError(f"predict does not support task={model.task!r}")

    model.eval()
    scores = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            outputs, _ = model(batch["x"].to(device))
            if model.task == "risk":
                outputs = outputs.reshape(-1)
            elif model.task == "policy":
                outputs = torch.softmax(outputs.float(), dim=-1)
            scores.append(outputs.float().cpu())

    key = "probs" if model.task == "policy" else "score"
    return {key: torch.cat(scores).numpy().astype(np.float32)}

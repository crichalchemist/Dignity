"""Training engine with AMP and DDP support."""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm


def make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    T_max: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Return a CosineAnnealingLR scheduler.

    Decays LR from its initial value to eta_min over T_max steps, then
    optionally cycles back up. Stable multi-task convergence requires a
    smooth LR schedule — cosine annealing avoids the abrupt drops of step LR.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)


_GATE_SUPPRESSION_FACTOR = 0.1  # scale factor applied to non-HOLD logits under risk gate


def train_cascade_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    task_weights: dict[str, float],
    device: torch.device,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    risk_gate_training: bool = True,
    max_drawdown: float = 0.05,
) -> dict[str, float]:
    """Train one epoch using the Guided Learning cascade loss.

    Each batch calls model.cascade_loss() which attaches auxiliary supervision
    at every intermediate head — short gradient paths prevent vanishing
    gradients in the deep 4-stage cascade (vault finding: +7.31pp ARR).

    Args:
        model: Dignity model instantiated with task='cascade'.
        dataloader: Iterable of (x, labels_dict) pairs.
            labels_dict must contain keys: regime, var, alpha, action.
        optimizer: Optimizer (AdamW recommended).
        task_weights: Per-head loss weights, e.g. {"regime":0.2, ...}.
        device: Target device.
        use_amp: Enable automatic mixed precision.
        grad_clip: Max gradient norm (0 to disable).
        scheduler: Optional LR scheduler stepped once per epoch.
        risk_gate_training: When True, suppress non-HOLD action logits for
            samples where var_estimate exceeds max_drawdown. This aligns
            training with deployment constraints and prevents the model from
            learning aggressive strategies that the risk gate will block at
            inference time. Default True; set False for research runs only.
        max_drawdown: VaR threshold above which the gate fires. Should match
            ExecutionConfig.max_drawdown.

    Returns:
        Metrics dict with keys: loss, regime_loss, risk_loss, alpha_loss, policy_loss.
    """
    model.train()
    scaler = GradScaler("cuda", enabled=use_amp)

    total_loss = 0.0
    head_totals: dict[str, float] = {
        "regime_loss": 0.0,
        "risk_loss": 0.0,
        "alpha_loss": 0.0,
        "policy_loss": 0.0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc="Training Cascade")
    for x, labels in pbar:
        x = x.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        optimizer.zero_grad()

        with autocast("cuda", enabled=use_amp):
            outputs = model(x)

            if risk_gate_training:
                # Suppress BUY/SELL logits for samples where var_estimate exceeds
                # max_drawdown. The suppression is differentiable (torch.where with
                # a scalar multiplier), so the model receives a weaker gradient
                # signal for aggressive actions under high-risk conditions and
                # learns to prefer HOLD when VaR is elevated.
                gate_mask = outputs["var_estimate"] > max_drawdown  # [B, 1] bool
                non_hold = outputs["action_logits"][:, 1:]  # [B, n_actions-1]
                suppressed = torch.where(
                    gate_mask.expand_as(non_hold),
                    non_hold * _GATE_SUPPRESSION_FACTOR,
                    non_hold,
                )
                gated_logits = torch.cat([outputs["action_logits"][:, :1], suppressed], dim=-1)
                outputs = {**outputs, "action_logits": gated_logits}

            loss, per_head = model.cascade_loss(outputs, labels, task_weights)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        head_totals["regime_loss"] += per_head["regime"].item()
        head_totals["risk_loss"] += per_head["risk"].item()
        head_totals["alpha_loss"] += per_head["alpha"].item()
        head_totals["policy_loss"] += per_head["policy"].item()
        n_batches += 1

        pbar.set_postfix({"loss": f"{total_loss / n_batches:.4f}"})

    if scheduler is not None:
        scheduler.step()

    denom = max(n_batches, 1)
    return {
        "loss": total_loss / denom,
        **{k: v / denom for k, v in head_totals.items()},
    }


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    grad_clip: float = 1.0,
    log_interval: int = 10,
) -> dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device
        use_amp: Use automatic mixed precision
        grad_clip: Gradient clipping value
        log_interval: Logging interval

    Returns:
        Dictionary with training metrics
    """
    model.train()
    scaler = GradScaler("cuda", enabled=use_amp)

    total_loss = 0.0
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(pbar):
        # Unpack batch
        if len(batch) == 2:
            x, y = batch
            x, y = x.to(device), y.to(device)
        else:
            x = batch.to(device)
            y = None

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast("cuda", enabled=use_amp):
            predictions, _ = model(x)
            # Squeeze predictions if they have an extra dimension
            if predictions.dim() > y.dim():
                predictions = predictions.squeeze(-1)
            loss = criterion(predictions, y) if y is not None else predictions.mean()

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Track loss
        total_loss += loss.item()

        # Update progress bar
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return {"loss": total_loss / num_batches}


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Unpack batch
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(device), y.to(device)
            else:
                x = batch.to(device)
                y = None

            # Forward pass
            predictions, _ = model(x)

            # Compute loss
            if y is not None:
                # Squeeze predictions if they have an extra dimension
                if predictions.dim() > y.dim():
                    predictions = predictions.squeeze(-1)
                loss = criterion(predictions, y)
                total_loss += loss.item()

                # Store for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(y.cpu())

    # Compute metrics
    metrics = {"loss": total_loss / len(dataloader)}

    if all_predictions:
        predictions_cat = torch.cat(all_predictions)
        targets_cat = torch.cat(all_targets)

        # Add task-specific metrics
        if predictions_cat.size(-1) == 1 and targets_cat.max() <= 1 and targets_cat.min() >= 0:
            # Binary classification accuracy
            preds_binary = (predictions_cat > 0.5).float()
            accuracy = (preds_binary == targets_cat).float().mean().item()
            metrics["accuracy"] = accuracy

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    path: str,
) -> None:
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training metrics
        path: Save path
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    path: str,
    device: torch.device,
) -> int:
    """
    Load training checkpoint.

    Args:
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        path: Checkpoint path
        device: Device to load on

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint.get("epoch", 0)

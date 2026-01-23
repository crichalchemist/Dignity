"""Training engine with AMP and DDP support."""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


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
    scaler = GradScaler(enabled=use_amp)

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
        with autocast(enabled=use_amp):
            predictions, _ = model(x)
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
        if predictions_cat.size(-1) == 1:  # Binary classification or regression
            # For risk scoring (binary)
            if targets_cat.max() <= 1 and targets_cat.min() >= 0:
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

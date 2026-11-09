"""Command-line interface for training Dignity models."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DignityConfig
from data.loader import create_dataloader
from data.pipeline import TransactionPipeline
from data.source.synthetic import SyntheticGenerator
from train.checkpoint import (
    build_model_from_meta,
    load_checkpoint,
    make_meta,
    save_checkpoint,
)
from train.engine import predict, train_epoch
from train.losses import risk_loss


def main(argv: list[str] | None = None) -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Dignity models")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args(argv)

    # Load config
    print(f"Loading config from {args.config}")
    config = DignityConfig.from_yaml(args.config)
    print(config)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config.seed)

    task = config.model.task
    if task != "risk":
        raise NotImplementedError(
            f"task={task!r} does not train end to end until M4 (DGNTY-1, DGNTY-2)"
        )
    use_amp = config.train.use_amp and device.type == "cuda"

    # Generate/load data
    print("\nGenerating synthetic data...")
    df = SyntheticGenerator(config.data.synthetic).generate()

    # Prepare data pipeline
    print("Preprocessing data...")
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len,
        features=config.data.features,
        privacy=config.privacy,
    )

    # Split at the block level: every generated block is one independent
    # sequence, so the split, the scaler fit, signals and windows all stay
    # inside blocks. data.test stays untouched here.
    data = pipeline.process_blocks(
        df,
        config.data.split,
        config.model.horizon,
        config.data.label_overlap,
        np.random.default_rng(config.data.synthetic.seed),
    )
    y_val = data.val.y_risk

    if data.epsilon_spent:
        spent = ", ".join(f"{k} {v:.3f}" for k, v in data.epsilon_spent.items())
        print(f"Privacy: ε spent per split: {spent}")

    print(
        f"Train/Val/Test windows: {len(data.train.X)}/{len(data.val.X)}/"
        f"{len(data.test.X)}"
    )

    # Create dataloaders
    train_loader = create_dataloader(
        data.train.X,
        {"y": data.train.y_risk},
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = create_dataloader(
        data.val.X,
        {"y": y_val},
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Create model
    print("\nInitializing model...")
    meta = make_meta(
        config,
        data.feature_names,
        data.scaler_center,
        data.scaler_scale,
    )
    model = build_model_from_meta(meta).to(device)

    print(model.summary())

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    # Loss function
    loss_fn = risk_loss(None)
    grad_scaler = GradScaler("cuda", enabled=use_amp)

    # Create checkpoint directory
    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {config.train.epochs} epochs...")
    best_metric = float("inf")

    start_epoch = 1
    if args.resume:
        ck = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        # a CPU (disabled) run saves {}, which an enabled scaler refuses to load
        if ck["grad_scaler_state"]:
            grad_scaler.load_state_dict(ck["grad_scaler_state"])
        start_epoch = ck["epoch"] + 1
        best_metric = ck["best_metric"]

    for epoch in range(start_epoch, config.train.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{config.train.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_scaler=grad_scaler,
            use_amp=use_amp,
            grad_clip=config.train.gradient_clip,
        )

        # Validate
        scores = predict(model, val_loader, device)["score"]
        val_loss = loss_fn(
            torch.from_numpy(scores), {"y": torch.from_numpy(y_val)}
        ).item()

        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model first, so an epoch checkpoint carries the best so far
        metrics = {
            "train": {k: float(v) for k, v in train_metrics.items()},
            "val": {"loss": float(val_loss)},
        }
        if val_loss < best_metric:
            best_metric = float(val_loss)
            best_path = checkpoint_dir / f"dignity_{config.model.task}_best.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                grad_scaler,
                meta,
                epoch,
                metrics,
                best_metric,
            )
            print(f"Best model saved (val_loss: {best_metric:.4f})")

        if epoch % config.train.save_interval == 0:
            checkpoint_path = (
                checkpoint_dir / f"dignity_{config.model.task}_epoch{epoch}.pt"
            )
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                grad_scaler,
                meta,
                epoch,
                metrics,
                best_metric,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

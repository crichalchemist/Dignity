"""Command-line interface for training Dignity models."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DignityConfig
from data.loader import create_dataloader
from data.pipeline import TransactionPipeline
from data.source.synthetic import SyntheticGenerator
from models.dignity import Dignity
from train.engine import (
    load_checkpoint,
    save_checkpoint,
    train_epoch,
    validate_epoch,
)


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Dignity models")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config = DignityConfig.from_yaml(args.config)
    print(config)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config.seed)

    # Generate/load data
    print("\nGenerating synthetic data...")
    generator = SyntheticGenerator(seed=config.seed)
    block_len = config.data.seq_len + 20  # each block yields 21 windows
    df_train = generator.generate_dataset(
        num_normal=800, num_anomalous=200, seq_len=block_len
    )

    # Prepare data pipeline
    print("Preprocessing data...")
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len,
        features=config.data.features,
        privacy=config.privacy,
    )

    # Split at the sequence level: every generated block is one independent
    # sequence, so the split, the scaler fit, signals and windows all stay
    # inside blocks. A positional split over the concatenated frame put every
    # normal block in train and every anomalous block in val.
    (X_train, y_train), (X_val, y_val) = pipeline.process_blocks(
        df_train.drop(columns="label"),
        df_train["label"].to_numpy(),
        block_len=block_len,
        test_size=config.data.test_size,
        rng=np.random.default_rng(config.seed),
    )

    if pipeline.privacy_manager is not None:
        budget = pipeline.privacy_manager.budget
        print(f"Privacy: ε spent {budget.spent:.3f} of {budget.epsilon_total:.3f}")

    print(f"Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")

    # Create dataloaders
    train_loader = create_dataloader(
        X_train, y_train, batch_size=config.data.batch_size, shuffle=True, device=device
    )

    val_loader = create_dataloader(
        X_val, y_val, batch_size=config.data.batch_size, shuffle=False, device=device
    )

    # Create model
    print("\nInitializing model...")
    model = Dignity(
        task=config.model.task,
        input_size=len(pipeline.available_features),
        hidden_size=config.model.hidden_size,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
    ).to(device)

    print(model.summary())

    # Setup training
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    # Loss function
    if config.model.task == "risk":
        criterion = nn.BCELoss()
    elif config.model.task == "forecast":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Create checkpoint directory
    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nStarting training for {config.train.epochs} epochs...")
    best_val_loss = float("inf")

    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device) + 1

    for epoch in range(start_epoch, config.train.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{config.train.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=config.train.use_amp,
            grad_clip=config.train.gradient_clip,
            log_interval=config.train.log_interval,
        )

        # Validate
        val_metrics = validate_epoch(
            model=model, dataloader=val_loader, criterion=criterion, device=device
        )

        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        if "accuracy" in val_metrics:
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

        # Save checkpoint
        if epoch % config.train.save_interval == 0:
            checkpoint_path = (
                checkpoint_dir / f"dignity_{config.model.task}_epoch{epoch}.pt"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                path=str(checkpoint_path),
            )
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = checkpoint_dir / f"dignity_{config.model.task}_best.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                path=str(best_path),
            )
            print(f"Best model saved (val_loss: {best_val_loss:.4f})")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

"""Command-line interface for training Dignity models."""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DignityConfig
from data.loader import create_dataloader
from data.pipeline import TransactionPipeline
from data.source.synthetic import SyntheticGenerator
from models.dignity import Dignity
from train.engine import save_checkpoint, train_epoch, validate_epoch


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
    df_train = generator.generate_dataset(
        num_normal=800,
        num_anomalous=200,
        seq_len=config.data.seq_len + 20,  # Extra for validation
    )

    # Prepare data pipeline
    print("Preprocessing data...")
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len, features=config.data.features
    )

    # Process training data
    labels = df_train["label"].values if "label" in df_train.columns else None
    X_train, y_train = pipeline.process(
        df_train.drop("label", axis=1, errors="ignore"), labels=labels, fit=True
    )

    # Split train/val
    split_idx = int(len(X_train) * (1 - config.data.test_size))
    X_val, y_val = X_train[split_idx:], (
        y_train[split_idx:] if y_train is not None else None
    )
    X_train, y_train = X_train[:split_idx], (
        y_train[:split_idx] if y_train is not None else None
    )

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

    for epoch in range(1, config.train.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.train.epochs}")
        print(f"{'='*60}")

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

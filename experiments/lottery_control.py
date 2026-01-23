"""
Lottery Control Experiment for Overfitting Detection.

Trains the Sequence model on Powerball lottery data (pure randomness) and measures
accuracy against random baseline. If the model beats random chance, it indicates
the model is finding spurious patterns in noise (overfitting).

Expected Results:
    - Random baseline: ~33.3% accuracy (3-class classification)
    - Acceptable model: 30-37% accuracy (within random variation)
    - RED FLAG: >40% accuracy (model is overfitting to noise)

Usage:
    # First, download and prepare lottery data:
    python data/downloaders/lottery_downloader.py --input-file data/lottery/raw/powerball.csv
    python data/prepare_dataset.py --pairs powerball --input-root data/lottery --t-in 120 --t-out 10

    # Then run control experiment:
    python experiments/lottery_control.py \
        --pair powerball \
        --epochs 20 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import DataConfig, FeatureConfig, ModelConfig
from data.agents.single_task_agent import SingleTaskDataAgent as DataAgent
from data.iterable_dataset import IterableFXDataset
from models.agent_hybrid import build_model
from train.features.agent_features import build_feature_frame


def random_baseline_accuracy(num_samples: int, num_classes: int = 3) -> float:
    """
    Calculate expected accuracy from random guessing.

    Args:
        num_samples: Number of samples
        num_classes: Number of classes (default 3: up/flat/down)

    Returns:
        Expected accuracy (should be ~1/num_classes)
    """
    return 1.0 / num_classes


def train_lottery_model(
    train_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
) -> list[float]:
    """
    Train model on lottery data.

    Args:
        train_loader: Training data loader
        model: Model to train
        device: Device to train on
        epochs: Number of epochs
        lr: Learning rate

    Returns:
        List of training losses per epoch
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            sequences, targets_dict = batch

            sequences = sequences.to(device)
            targets = targets_dict["primary"].to(device)

            optimizer.zero_grad()

            outputs, _ = model(sequences)
            logits = outputs["logits"]

            loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return losses


def evaluate_lottery_model(
    test_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate model on lottery test data.

    Args:
        test_loader: Test data loader
        model: Trained model
        device: Device to evaluate on

    Returns:
        Dictionary with accuracy, predictions, and analysis
    """
    model = model.to(device)
    model.train(False)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            sequences, targets_dict = batch

            sequences = sequences.to(device)
            targets = targets_dict["primary"].to(device)

            outputs, _ = model(sequences)
            logits = outputs["logits"]

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate accuracy
    accuracy = np.mean(all_preds == all_targets)

    # Class distribution
    unique, counts = np.unique(all_targets, return_counts=True)
    class_distribution = dict(zip(unique, counts))

    # Per-class accuracy
    per_class_acc = {}
    for cls in unique:
        mask = all_targets == cls
        if mask.sum() > 0:
            per_class_acc[int(cls)] = np.mean(all_preds[mask] == all_targets[mask])

    return {
        "accuracy": accuracy,
        "num_samples": len(all_preds),
        "class_distribution": class_distribution,
        "per_class_accuracy": per_class_acc,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Lottery control experiment for overfitting detection"
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="powerball",
        help="Lottery pair to test (powerball, powerball_ball1, etc.)",
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="data/lottery",
        help="Root directory with lottery data",
    )
    parser.add_argument(
        "--t-in",
        type=int,
        default=120,
        help="Lookback window (number of draws)",
    )
    parser.add_argument(
        "--t-out",
        type=int,
        default=10,
        help="Forecast horizon",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 60)
    print("LOTTERY CONTROL EXPERIMENT")
    print("=" * 60)
    print(f"Pair: {args.pair}")
    print("Purpose: Test if model finds patterns in pure randomness")
    print("Expected: ~33.3% accuracy (random baseline)")
    print("Red flag: >40% accuracy (overfitting to noise)")
    print("=" * 60)
    print()

    # Load lottery data
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = (ROOT / input_root).resolve()

    # Load pair data
    from data.prepare_dataset import _load_pair_data

    print(f"Loading lottery data from {input_root}...")
    raw_df = _load_pair_data(args.pair, input_root, years=None)
    print(f"✓ Loaded {len(raw_df)} lottery draws")

    # Build features (minimal for lottery - no technical indicators needed)
    feature_cfg = FeatureConfig(
        sma_windows=[],
        ema_windows=[],
        rsi_window=0,
        bollinger_window=0,
        include_groups=["base"],  # Only returns and spreads
    )

    print("Building features...")
    feature_df = build_feature_frame(raw_df, config=feature_cfg)
    print(f"✓ Created {len(feature_df.columns)} features")

    # Create data splits
    n = len(feature_df)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)

    train_df = feature_df.iloc[:train_idx]
    test_df = feature_df.iloc[val_idx:]  # Use test set for final assessment

    print(f"✓ Train: {len(train_df)} samples")
    print(f"✓ Test: {len(test_df)} samples")
    print()

    # Setup data config
    feature_cols = [c for c in feature_df.columns if c not in {"datetime", "source_file"}]

    data_cfg = DataConfig(
        csv_path="",
        datetime_column="datetime",
        feature_columns=feature_cols,
        target_type="classification",
        t_in=args.t_in,
        t_out=args.t_out,
        train_range=None,
        val_range=None,
        test_range=None,
    )

    # Create data agent
    agent = DataAgent(data_cfg)

    # Fit normalization on train data
    agent.fit_normalization(train_df, feature_cols)

    # Build datasets
    train_dataset = IterableFXDataset(
        train_df, agent.norm_stats, data_cfg, feature_cols
    )
    test_dataset = IterableFXDataset(
        test_df, agent.norm_stats, data_cfg, feature_cols
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Build model
    num_features = len(feature_cols)
    model_cfg = ModelConfig(num_features=num_features, num_classes=3)
    model = build_model(model_cfg, task_type="classification")

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Calculate random baseline
    random_acc = random_baseline_accuracy(len(test_df), num_classes=3)
    print(f"Random Baseline: {random_acc:.1%} ± 2-3%")
    print()

    # Train model
    print("Training on lottery data...")
    print("-" * 60)
    losses = train_lottery_model(
        train_loader, model, device, epochs=args.epochs
    )
    print("-" * 60)
    print()

    # Evaluate model
    print("Evaluating on lottery test set...")
    results = evaluate_lottery_model(test_loader, model, device)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {results['accuracy']:.1%}")
    print(f"Random Baseline: {random_acc:.1%}")
    print(f"Difference: {(results['accuracy'] - random_acc):.1%}")
    print()

    print(f"Class Distribution: {results['class_distribution']}")
    print(f"Per-Class Accuracy: {results['per_class_accuracy']}")
    print()

    # Interpret results
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if results["accuracy"] < 0.30:
        print("✓ PASS: Model accuracy below random (likely underfitting)")
        print("  → Model is not overfitting to noise")
    elif results["accuracy"] <= 0.37:
        print("✓ PASS: Model accuracy within random variation")
        print("  → No evidence of overfitting to noise")
    elif results["accuracy"] <= 0.40:
        print("⚠ WARNING: Model accuracy slightly above random")
        print("  → Possible mild overfitting, monitor carefully")
    else:
        print("✗ RED FLAG: Model accuracy significantly above random!")
        print("  → Model is finding spurious patterns in lottery numbers")
        print("  → This indicates overfitting to noise in training data")
        print()
        print("Recommendations:")
        print("  1. Increase regularization (dropout, weight decay)")
        print("  2. Reduce model capacity (fewer layers/units)")
        print("  3. Add more data augmentation")
        print("  4. Check for data leakage in FX training pipeline")

    print("=" * 60)
    print()

    # Save results
    results_path = ROOT / "experiments" / "lottery_control_results.txt"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w") as f:
        f.write("Lottery Control Experiment Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Pair: {args.pair}\n")
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Random Baseline: {random_acc:.4f}\n")
        f.write(f"Num Samples: {results['num_samples']}\n")
        f.write(f"Class Distribution: {results['class_distribution']}\n")
        f.write(f"Per-Class Accuracy: {results['per_class_accuracy']}\n")

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

"""Command-line interface for training Dignity models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DignityConfig
from core.signals import ASSET_CONFIGS
from data.loader import create_dataloader
from data.pipeline import TransactionPipeline
from data.source.synthetic import SyntheticGenerator
from models.dignity import Dignity
from train.engine import (
    load_checkpoint,
    make_cosine_scheduler,
    save_checkpoint,
    train_cascade_epoch,
    train_epoch,
    validate_epoch,
)

# ---------------------------------------------------------------------------
# Cascade helpers
# ---------------------------------------------------------------------------

def _build_cascade_labels(
    prices: np.ndarray,
    regime_raw: np.ndarray,
    alpha_horizon: int = 5,
    var_window: int = 20,
) -> dict[str, np.ndarray]:
    """Derive supervised Guided Learning labels from a price series.

    Each label array has one entry per raw sample (before windowing).
    Maps regime signal → discrete class, computes rolling VaR, n-step alpha,
    and a price-direction action proxy for bootstrapping the policy head.
    """
    n = len(prices)

    # regime: bin continuous signal into 4 volatility quantile classes
    quantiles = np.quantile(regime_raw, [0.25, 0.5, 0.75])
    regime = np.digitize(regime_raw, quantiles).astype(np.int64)  # 0–3

    # var: rolling peak-to-trough drawdown fraction [0, 1]
    rolling_max = pd.Series(prices).expanding().max().values
    drawdown = np.clip((rolling_max - prices) / np.maximum(rolling_max, 1e-9), 0, 1)
    var = pd.Series(drawdown).rolling(var_window).max().fillna(0).values
    var = var.reshape(-1, 1).astype(np.float32)

    # alpha: n-step forward return tanh-normalized to [-1, 1]
    fwd = np.zeros(n, dtype=np.float32)
    if alpha_horizon < n:
        fwd[:-alpha_horizon] = (
            prices[alpha_horizon:] - prices[:-alpha_horizon]
        ) / np.maximum(prices[:-alpha_horizon], 1e-9)
    alpha = np.tanh(fwd * 20).reshape(-1, 1)

    # action: price-direction proxy (0=HOLD, 1=BUY, 2=SELL)
    thresh = 0.001
    action = np.where(fwd > thresh, 1, np.where(fwd < -thresh, 2, 0)).astype(np.int64)

    return {"regime": regime, "var": var, "alpha": alpha, "action": action}


def _make_cascade_batches(
    X_seq: np.ndarray,
    labels_seq: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Pack windowed sequences and aligned labels into batches."""
    n = len(X_seq)
    idx = np.random.permutation(n) if shuffle else np.arange(n)

    batches = []
    for start in range(0, n, batch_size):
        bi = idx[start : start + batch_size]
        x = torch.FloatTensor(X_seq[bi]).to(device)
        lbl: dict[str, torch.Tensor] = {
            "regime": torch.from_numpy(labels_seq["regime"][bi]).long().to(device),
            "var":    torch.from_numpy(labels_seq["var"][bi]).float().to(device),
            "alpha":  torch.from_numpy(labels_seq["alpha"][bi]).float().to(device),
            "action": torch.from_numpy(labels_seq["action"][bi]).long().to(device),
        }
        batches.append((x, lbl))
    return batches


def _process_cascade_data(
    df: pd.DataFrame,
    config: DignityConfig,
    asset_cfg,
    pipeline: TransactionPipeline,
    fit: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run pipeline → compute labels. Returns (X_seq, labels_seq)."""
    seq_len = config.data.seq_len

    df_signals = pipeline.compute_signals(df, asset_config=asset_cfg)
    if fit:
        X_raw = pipeline.fit_transform(df_signals)
    else:
        X_raw = pipeline.transform(df_signals)

    prices = df["close"].values if "close" in df.columns else df["price"].values
    regime_raw = (
        df_signals["regime"].values
        if "regime" in df_signals.columns
        else np.zeros(len(prices))
    )
    labels_raw = _build_cascade_labels(prices, regime_raw)

    X_seq, _ = pipeline.create_sequences(X_raw, None, stride=1)
    n_seq = len(X_seq)

    # Align labels to sequence END (stride=1: seq i ends at i + seq_len - 1)
    labels_seq = {
        k: v[seq_len - 1 : seq_len - 1 + n_seq] for k, v in labels_raw.items()
    }
    return X_seq, labels_seq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train Dignity models")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    config = DignityConfig.from_yaml(args.config)
    print(config)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(config.seed)

    # ------------------------------------------------------------------
    # Cascade training path
    # ------------------------------------------------------------------
    if config.model.task == "cascade":
        _train_cascade(config, device, args.resume)
        return

    # ------------------------------------------------------------------
    # Single-head training path (risk / forecast / policy)
    # ------------------------------------------------------------------
    _train_single_head(config, device, args.resume)


def _train_cascade(config: DignityConfig, device: torch.device, resume: str | None):
    """Cascade training: Regime → Risk → Alpha → Policy with Guided Learning."""

    # --- Data loading -------------------------------------------------
    asset_cfg = ASSET_CONFIGS.get(
        config.execution.asset_class, ASSET_CONFIGS["forex"]
    )

    if config.data.source == "metaapi" and config.execution.metaapi_token:
        import asyncio

        from data.source.metaapi import MetaApiSource

        print("\nConnecting to MetaApi...")
        symbol = config.execution.symbols[0]
        src = MetaApiSource(
            token=config.execution.metaapi_token,
            account_id=config.execution.account_id,
            symbol=symbol,
        )
        df_raw = asyncio.run(_fetch_history(src, start_time=config.data.start_date + "T00:00:00Z"))
        print(f"Fetched {len(df_raw)} bars from MetaApi ({symbol})")
    else:
        print("\nGenerating synthetic OHLCV (no MetaApi credentials)...")
        gen = SyntheticGenerator(seed=config.seed)
        df_raw = gen.generate_ohlcv(
            n_bars=2000 + config.data.seq_len,
            start_date=config.data.start_date,
        )
        print(f"Generated {len(df_raw)} synthetic bars from {config.data.start_date}")

    # --- Pipeline -----------------------------------------------------
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len,
        features=config.data.features,
    )

    X_seq, labels_seq = _process_cascade_data(
        df_raw, config, asset_cfg, pipeline, fit=True
    )

    split_idx = int(len(X_seq) * (1 - config.data.test_size))
    X_train = X_seq[:split_idx]
    X_val   = X_seq[split_idx:]
    labels_train = {k: v[:split_idx] for k, v in labels_seq.items()}
    labels_val   = {k: v[split_idx:] for k, v in labels_seq.items()}

    n_features = len(pipeline.available_features)
    print(f"Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")
    print(f"Features used: {n_features}")

    # --- Model --------------------------------------------------------
    print("\nInitializing cascade model...")
    model = Dignity(
        task="cascade",
        input_size=n_features,
        hidden_size=config.model.hidden_size,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
    ).to(device)
    print(model.summary())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = make_cosine_scheduler(optimizer, T_max=config.train.epochs)
    task_weights = config.model.task_weights

    start_epoch = 1
    if resume:
        start_epoch = load_checkpoint(model, optimizer, resume, device) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ------------------------------------------------
    print(f"\nStarting cascade training for {config.train.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.train.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{config.train.epochs}\n{'='*60}")

        train_loader = _make_cascade_batches(
            X_train, labels_train, config.data.batch_size, device, shuffle=True
        )
        train_metrics = train_cascade_epoch(
            model, train_loader, optimizer, task_weights,
            device=device, use_amp=config.train.use_amp,
            grad_clip=config.train.gradient_clip, scheduler=scheduler,
        )

        # Validation: compute cascade loss without gradient updates
        val_loader = _make_cascade_batches(
            X_val, labels_val, config.data.batch_size, device, shuffle=False
        )
        val_loss = _cascade_val_loss(model, val_loader, task_weights, device)

        print(
            f"Train loss: {train_metrics['loss']:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )
        for k in ("regime_loss", "risk_loss", "alpha_loss", "policy_loss"):
            print(f"  {k}: {train_metrics[k]:.4f}")

        if epoch % config.train.save_interval == 0:
            path = checkpoint_dir / f"dignity_cascade_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, train_metrics, str(path))
            print(f"Checkpoint → {path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "dignity_cascade_best.pt"
            save_checkpoint(model, optimizer, epoch, train_metrics, str(best_path))
            print(f"Best model saved (val_loss: {best_val_loss:.4f})")

    print("\nCascade training complete!")


def _cascade_val_loss(
    model: nn.Module,
    val_loader,
    task_weights: dict[str, float],
    device: torch.device,
) -> float:
    """Compute mean cascade loss on validation set (no gradient update)."""
    model.train(False)
    total, n = 0.0, 0
    with torch.no_grad():
        for x, labels in val_loader:
            outputs = model(x)
            loss, _ = model.cascade_loss(outputs, labels, task_weights)
            total += loss.item()
            n += 1
    model.train(True)
    return total / max(n, 1)


async def _fetch_history(src, start_time: str | None = None) -> pd.DataFrame:
    """Async helper: connect, pull history, disconnect."""
    await src.connect()
    df = await src.get_history(bars=10_000, start_time=start_time)
    await src.disconnect()
    return df


def _train_single_head(
    config: DignityConfig, device: torch.device, resume: str | None
):
    """Existing single-head training path (risk / forecast / policy)."""

    print("\nGenerating synthetic data...")
    generator = SyntheticGenerator(seed=config.seed)
    df_train = generator.generate_dataset(
        num_normal=800, num_anomalous=200,
        seq_len=config.data.seq_len + 20,
    )

    print("Preprocessing data...")
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len,
        features=config.data.features,
    )

    labels = df_train["label"].values if "label" in df_train.columns else None
    X_train, y_train = pipeline.process(
        df_train.drop("label", axis=1, errors="ignore"),
        labels=labels,
        fit=True,
    )

    split_idx = int(len(X_train) * (1 - config.data.test_size))
    X_val, y_val = X_train[split_idx:], (y_train[split_idx:] if y_train is not None else None)
    X_train, y_train = X_train[:split_idx], (y_train[:split_idx] if y_train is not None else None)

    print(f"Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")

    train_loader = create_dataloader(X_train, y_train, batch_size=config.data.batch_size, shuffle=True)
    val_loader   = create_dataloader(X_val,   y_val,   batch_size=config.data.batch_size, shuffle=False)

    print("\nInitializing model...")
    model = Dignity(
        task=config.model.task,
        input_size=len(pipeline.available_features),
        hidden_size=config.model.hidden_size,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
    ).to(device)
    print(model.summary())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay
    )

    criterion: nn.Module
    if config.model.task == "risk":
        criterion = nn.BCELoss()
    elif config.model.task == "forecast":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    if resume:
        start_epoch = load_checkpoint(model, optimizer, resume, device) + 1

    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {config.train.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.train.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{config.train.epochs}\n{'='*60}")

        train_metrics = train_epoch(
            model=model, dataloader=train_loader, optimizer=optimizer,
            criterion=criterion, device=device, use_amp=config.train.use_amp,
            grad_clip=config.train.gradient_clip, log_interval=config.train.log_interval,
        )
        val_metrics = validate_epoch(
            model=model, dataloader=val_loader, criterion=criterion, device=device
        )

        print(f"\nTrain Loss: {train_metrics['loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        if "accuracy" in val_metrics:
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

        if epoch % config.train.save_interval == 0:
            path = checkpoint_dir / f"dignity_{config.model.task}_epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, {"train": train_metrics, "val": val_metrics}, str(path))
            print(f"Checkpoint saved to {path}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = checkpoint_dir / f"dignity_{config.model.task}_best.pt"
            save_checkpoint(model, optimizer, epoch, {"train": train_metrics, "val": val_metrics}, str(best_path))
            print(f"Best model saved (val_loss: {best_val_loss:.4f})")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

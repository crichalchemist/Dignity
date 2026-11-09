"""The one training loop: the CLI, the notebook and eval.run all call ``fit``."""

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler

from core.config import DignityConfig
from data.loader import create_dataloader
from data.pipeline import SplitData, TransactionPipeline
from data.source.synthetic import SyntheticGenerator
from eval.metrics import risk_metrics
from train.checkpoint import (
    CheckpointMeta,
    build_model_from_meta,
    load_checkpoint,
    make_meta,
    save_checkpoint,
)
from train.engine import predict, train_epoch
from train.losses import risk_loss

_VAL_METRICS = ("auroc", "pr_auc", "recall_at_1pct_fpr", "brier")


class ChanceGuardWarning(RuntimeWarning):
    """Validation AUROC is still below the chance guard at ``guard_epochs``."""


@dataclass
class FitResult:
    best_path: str
    history: list[dict]  # only the epochs run by this call
    best_metric: float
    meta: CheckpointMeta


def resolve_pos_weight(config_value: float | None, y_train: np.ndarray) -> float:
    """The configured weight, else the train negative-to-positive ratio."""
    if config_value is not None:
        return config_value
    y_train = np.asarray(y_train)
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("train split has no positive windows")
    return float((y_train != 1).sum()) / n_pos


def prepare_data(config: DignityConfig) -> tuple[pd.DataFrame, SplitData]:
    """Generate the synthetic frame and split it into block-level windows."""
    df = SyntheticGenerator(config.data.synthetic).generate()
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len,
        features=config.data.features,
        privacy=config.privacy,
    )
    split_data = pipeline.process_blocks(
        df,
        config.data.split,
        config.model.horizon,
        config.data.label_overlap,
        np.random.default_rng(config.data.synthetic.seed),
    )
    return df, split_data


def fit(
    config: DignityConfig,
    data: SplitData,
    seed: int,
    run_dir: str,
    resume_from: str | None = None,
) -> FitResult:
    """Train a risk model; write best.pt, last.pt and history.json to run_dir."""
    torch.manual_seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = config.train.use_amp and device.type == "cuda"
    pin_memory = device.type == "cuda"
    train_cfg = config.train
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = make_meta(
        config, data.feature_names, data.scaler_center, data.scaler_scale, {}
    )
    model = build_model_from_meta(meta).to(device)
    loss_fn = risk_loss(resolve_pos_weight(train_cfg.pos_weight, data.train.y_risk))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    grad_scaler = GradScaler("cuda", enabled=use_amp)

    best_metric = float("-inf")
    start_epoch = 1
    if resume_from:
        ckpt = load_checkpoint(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        # a CPU (disabled) run saves {}, which an enabled scaler refuses to load
        if ckpt["grad_scaler_state"]:
            grad_scaler.load_state_dict(ckpt["grad_scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt["best_metric"]

    # Closed-form cosine keyed off each group's initial_lr (restored with the
    # optimizer state). last_epoch is one behind the epoch about to train
    # because the scheduler's constructor takes the first step itself.
    epochs = train_cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: 0.5 * (1 + math.cos(math.pi * e / epochs)),
        last_epoch=start_epoch - 2,
    )

    train_loader = create_dataloader(
        data.train.X,
        {"y": data.train.y_risk.astype(np.float32)},
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = create_dataloader(
        data.val.X,
        None,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    history: list[dict] = []
    stale = 0
    for epoch in range(start_epoch, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]  # the lr this epoch trains with
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_scaler=grad_scaler,
            use_amp=use_amp,
            grad_clip=train_cfg.gradient_clip,
        )
        score = predict(model, val_loader, device)["score"]
        val = risk_metrics(
            data.val.y_risk, score, data.val.block_index, n_boot=0, seed=seed
        )
        row = {"epoch": epoch, "lr": float(lr), "train_loss": train_metrics["loss"]}
        row.update({f"val_{k}": float(val[k]["value"]) for k in _VAL_METRICS})
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} lr={row['lr']:.3e} "
            f"train_loss={row['train_loss']:.4f} "
            + " ".join(f"val_{k}={row[f'val_{k}']:.4f}" for k in _VAL_METRICS)
        )

        selected = row[f"val_{train_cfg.select_metric}"]
        if selected > best_metric:
            best_metric = selected
            stale = 0
            save_checkpoint(
                out / "best.pt",
                model,
                optimizer,
                grad_scaler,
                meta,
                epoch,
                row,
                best_metric,
            )
        else:
            stale += 1

        stop = train_cfg.patience is not None and stale >= train_cfg.patience
        if epoch % train_cfg.save_interval == 0 or stop or epoch == epochs:
            save_checkpoint(
                out / "last.pt",
                model,
                optimizer,
                grad_scaler,
                meta,
                epoch,
                row,
                best_metric,
            )

        scheduler.step()

        guard = train_cfg.chance_guard_auroc
        if (
            epoch == train_cfg.guard_epochs
            and guard is not None
            and row["val_auroc"] < guard
        ):
            warnings.warn(
                f"val AUROC {row['val_auroc']:.3f} below chance guard "
                f"{guard} after {epoch} epochs",
                ChanceGuardWarning,
                stacklevel=2,
            )
        if stop:
            break

    previous: list[dict] = []
    history_path = out / "history.json"
    if resume_from and history_path.exists():
        previous = [
            r for r in json.loads(history_path.read_text()) if r["epoch"] < start_epoch
        ]
    history_path.write_text(json.dumps(previous + history, indent=2))

    return FitResult(
        best_path=str(out / "best.pt"),
        history=history,
        best_metric=float(best_metric),
        meta=meta,
    )

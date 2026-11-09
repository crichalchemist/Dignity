"""Self-describing checkpoints: a checkpoint alone rebuilds its model."""

import dataclasses
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.config import DignityConfig
from models.dignity import Dignity

FORMAT_VERSION = 2

_MODEL_KEYS = ("hidden_size", "n_layers", "dropout", "cnn_kernel_size")


class UnsupportedCheckpointError(ValueError):
    """Raised for checkpoints this version cannot read."""


@dataclass
class CheckpointMeta:
    format_version: int
    task: str
    input_size: int
    feature_names: list[str]
    head_kwargs: dict
    scaler_center: list[float]
    scaler_scale: list[float]
    config: dict


def make_meta(
    config: DignityConfig,
    feature_names: list[str],
    scaler_center,
    scaler_scale,
    head_kwargs: dict | None = None,
) -> CheckpointMeta:
    return CheckpointMeta(
        format_version=FORMAT_VERSION,
        task=config.model.task,
        input_size=len(feature_names),
        feature_names=list(feature_names),
        head_kwargs=head_kwargs or {},
        scaler_center=np.asarray(scaler_center).tolist(),
        scaler_scale=np.asarray(scaler_scale).tolist(),
        config=dataclasses.asdict(config),
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    grad_scaler,
    meta: CheckpointMeta,
    epoch: int,
    metrics: dict,
    best_metric: float,
) -> None:
    """Write one plain dict so torch.load(weights_only=True) can read it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "grad_scaler_state": grad_scaler.state_dict(),
            "meta": dataclasses.asdict(meta),
            "epoch": epoch,
            "metrics": metrics,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path, map_location="cpu") -> dict:
    try:
        raw = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as e:
        raise UnsupportedCheckpointError(_unsupported(path, "unreadable")) from e
    found = raw.get("meta", {}).get("format_version")
    if found != FORMAT_VERSION:
        raise UnsupportedCheckpointError(_unsupported(path, found))
    raw["meta"] = CheckpointMeta(**raw["meta"])
    return raw


def build_model_from_meta(meta: CheckpointMeta) -> Dignity:
    return Dignity(
        task=meta.task,
        input_size=meta.input_size,
        **{k: meta.config["model"][k] for k in _MODEL_KEYS},
        **meta.head_kwargs,
    )


def _unsupported(path, found) -> str:
    return (
        f"{path}: unsupported checkpoint format (found {found!r}, "
        f"need {FORMAT_VERSION}). Checkpoints written before v1.0 are not "
        "supported; retrain with dignity-train."
    )

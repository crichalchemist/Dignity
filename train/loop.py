"""The one training loop: the CLI, the notebook and eval.run all call ``fit``."""

import dataclasses
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler

from core.config import DataConfig, DignityConfig, PrivacyConfig
from data.loader import create_dataloader
from data.pipeline import SplitData, TransactionPipeline, WindowSet
from data.source.synthetic import SyntheticGenerator
from eval.metrics import lift_upper, risk_metrics
from models.head.policy import cost_matrix, decide_actions
from train.checkpoint import (
    CheckpointMeta,
    build_model_from_meta,
    load_checkpoint,
    make_meta,
    save_checkpoint,
)
from train.engine import predict, train_epoch
from train.losses import expected_cost_loss, forecast_loss, risk_loss

_VAL_METRICS = ("auroc", "pr_auc", "recall_at_1pct_fpr", "brier")
_FORECAST_VAL_METRICS = ("auroc", "pr_auc", "recall_at_1pct_fpr")


class ChanceGuardWarning(RuntimeWarning):
    """Validation AUROC is still below the chance guard at ``guard_epochs``."""


class ForecastNotEligibleError(RuntimeError):
    """No forecast epoch beat the eligibility floor on validation lift."""

    def __init__(self, run_dir: str, seed: int, message: str):
        super().__init__(message)
        self.run_dir = run_dir
        self.seed = seed


@dataclass
class FitResult:
    best_path: str
    history: list[dict]  # only the epochs run by this call
    best_metric: float
    meta: CheckpointMeta
    best_epoch: int | None = None


def resolve_pos_weight(
    config_value: float | None, y_train: np.ndarray, cap: float | None = None
) -> float:
    """The configured weight, else the train negative-to-positive ratio.

    ``cap`` limits the derived ratio only; a configured weight is never capped.
    """
    if config_value is not None:
        return config_value
    y_train = np.asarray(y_train)
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("train split has no positive windows")
    ratio = float((y_train != 1).sum()) / n_pos
    return ratio if cap is None else min(ratio, cap)


def prepare_data(
    config: DignityConfig, privacy_seed: int | None = None
) -> tuple[pd.DataFrame, SplitData]:
    """Generate the synthetic frame and split it into block-level windows.

    ``privacy_seed`` seeds only the privacy noise (``None``: the data seed); the
    generator, the block split and the labels always follow ``synthetic.seed``.
    """
    df = SyntheticGenerator(config.data.synthetic).generate()
    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len,
        features=config.data.features,
        privacy=config.privacy,
        privacy_seed=(
            config.data.synthetic.seed if privacy_seed is None else privacy_seed
        ),
    )
    split_data = pipeline.process_blocks(
        df,
        split=config.data.split,
        horizon=config.model.horizon,
        overlap=config.data.label_overlap,
        rng=np.random.default_rng(config.data.synthetic.seed),
    )
    return df, split_data


def _task_targets(task: str, ws: WindowSet) -> dict[str, np.ndarray]:
    if task == "risk":
        return {"y": ws.y_risk.astype(np.float32)}
    if task == "forecast":
        return {"y": ws.y_forecast.astype(np.float32), "mask": ws.forecast_mask}
    if task == "policy":
        return {"y": ws.y_risk.astype(np.int64)}
    raise NotImplementedError(f"task={task!r} does not train end to end yet")


def _task_loss(config: DignityConfig, train_ws: WindowSet):
    task = config.model.task
    if task == "risk":
        return risk_loss(resolve_pos_weight(config.train.pos_weight, train_ws.y_risk))
    if task == "forecast":
        labels = train_ws.y_forecast[train_ws.forecast_mask].ravel()
        return forecast_loss(
            resolve_pos_weight(
                config.train.pos_weight,
                labels,
                cap=config.train.forecast_pos_weight_cap,
            )
        )
    if task == "policy":
        return expected_cost_loss(torch.from_numpy(cost_matrix(config.policy.costs)))
    raise NotImplementedError(f"task={task!r} does not train end to end yet")


def _val_metrics(
    config: DignityConfig, val_ws: WindowSet, score: np.ndarray, seed: int
) -> tuple[dict[str, float], float | None]:
    """The row's ``val_*`` entries and the AUROC the chance guard reads."""
    if config.model.task == "policy":
        # the cost of the capped decisions; AUROC is undefined for 3 actions
        costs = cost_matrix(config.policy.costs).astype(np.float64)
        actions = decide_actions(score, config.policy.review_budget)
        neg_cost = -costs[actions, val_ws.y_risk.astype(np.int64)].mean()
        return {"val_neg_cost": float(neg_cost)}, None
    if config.model.task == "risk":
        val = risk_metrics(
            val_ws.y_risk, score, val_ws.block_index, n_boot=0, seed=seed
        )
        return {f"val_{k}": float(val[k]["value"]) for k in _VAL_METRICS}, float(
            val["auroc"]["value"]
        )
    # forecast: mean over horizons whose masked column holds both classes;
    # risk_metrics raises on a one-class column, so those are skipped
    mask = val_ws.forecast_mask
    blocks = val_ws.block_index[mask]
    columns = []
    for h in range(score.shape[1]):
        y = val_ws.y_forecast[mask, h].astype(np.int8)
        if len(np.unique(y)) < 2:
            columns.append(None)
            continue
        columns.append(risk_metrics(y, score[mask, h], blocks, n_boot=0, seed=seed))
    scored = [c for c in columns if c is not None]
    if not scored:
        row = {f"val_{k}": float("nan") for k in _FORECAST_VAL_METRICS}
        row["val_lift_upper"] = float("nan")
        return row, None
    row = {
        f"val_{k}": float(np.mean([c[k]["value"] for c in scored]))
        for k in _FORECAST_VAL_METRICS
    }
    lift = lift_upper(val_ws.y_forecast, score, mask)
    row["val_lift_upper"] = float("nan") if lift is None else float(lift)
    last = columns[-1]
    return row, None if last is None else float(last["auroc"]["value"])


def _head_kwargs(config: DignityConfig) -> dict:
    if config.model.task == "forecast":
        return {"horizon": config.model.horizon}
    if config.model.task == "policy":
        return {"n_actions": config.model.n_actions}
    return {}


def _task_selection_metric(task: str, select_metric: str) -> str:
    """Policy always selects on neg_cost, forecast on lift_upper."""
    if task == "policy":
        return "neg_cost"
    if task == "forecast":
        return "lift_upper"
    return select_metric


def _selection_metric(config: DignityConfig) -> str:
    """The val metric best.pt is kept on: policy neg_cost, forecast lift_upper."""
    return _task_selection_metric(config.model.task, config.train.select_metric)


# In system mode (the default) the privacy stage draws its noise from
# secrets.SystemRandom (deliberately unseeded, core/privacy.py), so two runs of one
# config fit scalers that differ by a few percent of scale at full size, over 20% at
# 60 blocks. (privacy.deterministic_rng seeds the noise, so those runs match; the
# tolerance is not tightened for it.) Under privacy a
# scaler counts as matching when center and scale agree within this fraction of
# the run's scale. load_backbone also requires the same data config, so this only
# has to absorb noise draws, not a different dataset. Without privacy the scaler
# is deterministic.
_DP_SCALER_TOL = 0.25


def _privacy_dict(privacy) -> dict | None:
    """A PrivacyConfig, or its asdict form from checkpoint meta, as a plain dict."""
    if privacy is None:
        return None
    if isinstance(privacy, PrivacyConfig):
        return privacy.to_dict()
    return PrivacyConfig(**privacy).to_dict()


def load_backbone(
    model,
    path: str,
    data: SplitData,
    privacy: PrivacyConfig | None,
    data_config: DataConfig,
) -> None:
    """Copy only the ``backbone.*`` tensors of a checkpoint into ``model``.

    The head is left untouched. ``data`` and ``privacy`` are the current run's
    preprocessing and ``data_config`` its data block. Raises ValueError when the checkpoint was trained on other
    features, on inputs scaled differently (scaler center/scale; within 25% of
    scale under privacy; noise is unseeded in the default system mode) or privatized
    differently (privacy block), when it was trained on another data config (so
    another train split: risk-train blocks could land in this run's val/test), when its backbone tensors are not exactly the
    model's, or when any has another shape.
    """
    ckpt = load_checkpoint(path, map_location="cpu")
    meta = ckpt["meta"]
    if meta.feature_names != list(data.feature_names):
        raise ValueError(
            f"{path}: checkpoint features {meta.feature_names} differ "
            f"from the run's features {list(data.feature_names)}"
        )
    scale_now = np.asarray(data.scaler_scale, dtype=np.float64)
    for name, saved_arr, now in (
        ("scaler_center", meta.scaler_center, data.scaler_center),
        ("scaler_scale", meta.scaler_scale, data.scaler_scale),
    ):
        if privacy is None:
            same = np.allclose(saved_arr, now, rtol=1e-6)
        else:
            gap = np.abs(np.asarray(saved_arr) - np.asarray(now))
            same = bool((gap <= _DP_SCALER_TOL * scale_now).all())
        if not same:
            raise ValueError(
                f"{path}: checkpoint {name} {list(saved_arr)} differs from the "
                f"run's {np.asarray(now).tolist()}; the backbone was fitted to "
                "differently preprocessed inputs"
            )
    saved_data = json.loads(json.dumps(meta.config["data"]))
    run_data = json.loads(json.dumps(dataclasses.asdict(data_config)))
    if saved_data != run_data:
        differ = sorted(
            k
            for k in saved_data.keys() | run_data.keys()
            if saved_data.get(k) != run_data.get(k)
        )
        raise ValueError(
            f"{path}: checkpoint data config differs from the run's in {differ}; "
            "the backbone was trained on a different train split"
        )
    if _privacy_dict(meta.config.get("privacy")) != _privacy_dict(privacy):
        raise ValueError(
            f"{path}: checkpoint privacy config differs from the run's; "
            "the backbone was fitted to differently privatized inputs"
        )
    saved = {k: v for k, v in ckpt["model_state"].items() if k.startswith("backbone.")}
    current = {k: v for k, v in model.state_dict().items() if k.startswith("backbone.")}
    if set(saved) != set(current):
        raise ValueError(
            f"{path}: backbone tensors differ from the model's; "
            f"missing {sorted(set(current) - set(saved))}, "
            f"extra {sorted(set(saved) - set(current))}"
        )
    for k, v in current.items():
        if saved[k].shape != v.shape:
            raise ValueError(
                f"{path}: backbone tensor {k} has shape {tuple(saved[k].shape)}, "
                f"the model expects {tuple(v.shape)}"
            )
    model.load_state_dict(saved, strict=False)


def fit(
    config: DignityConfig,
    data: SplitData,
    seed: int,
    run_dir: str,
    resume_from: str | None = None,
    eligibility_floor: float | None = None,
) -> FitResult:
    """Train a risk, forecast or policy model.

    Writes best.pt, last.pt and history.json to run_dir. With
    ``train.forecast_eligibility`` a forecast epoch is kept only when its val
    ``lift_upper`` is above ``eligibility_floor`` (default 1.0, chance); when no
    epoch is, ``ForecastNotEligibleError`` is raised.
    """
    torch.manual_seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    use_amp = config.train.use_amp and device.type == "cuda"
    pin_memory = device.type == "cuda"
    train_cfg = config.train
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = make_meta(
        config,
        data.feature_names,
        data.scaler_center,
        data.scaler_scale,
        _head_kwargs(config),
    )
    model = build_model_from_meta(meta).to(device)
    # resume restores the full state below, so a warm start would be overwritten
    warm = model.task in ("policy", "forecast")
    if warm and train_cfg.init_backbone_from and not resume_from:
        load_backbone(
            model, train_cfg.init_backbone_from, data, config.privacy, config.data
        )
    loss_fn = _task_loss(config, data.train)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    grad_scaler = GradScaler("cuda", enabled=use_amp)

    best_metric = float("-inf")
    start_epoch = 1
    if resume_from:
        ckpt = load_checkpoint(resume_from, map_location=device)
        saved_metric = _task_selection_metric(
            ckpt["meta"].task, ckpt["meta"].config["train"].get("select_metric")
        )
        if saved_metric != _selection_metric(config):
            raise ValueError(
                f"{resume_from} was selected on {saved_metric!r}; this run selects "
                f"on {_selection_metric(config)!r}, so best_metric is not comparable"
            )
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
        _task_targets(config.model.task, data.train),
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
    wrote_best = False
    best_epoch: int | None = None
    gate = model.task == "forecast" and train_cfg.forecast_eligibility
    floor = 1.0 if eligibility_floor is None else eligibility_floor
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
        pred = predict(model, val_loader, device)
        score = pred["probs"] if model.task == "policy" else pred["score"]
        val_row, guard_auroc = _val_metrics(config, data.val, score, seed)
        row = {"epoch": epoch, "lr": float(lr), "train_loss": train_metrics["loss"]}
        row.update(val_row)
        history.append(row)
        print(
            f"epoch {epoch}/{epochs} lr={row['lr']:.3e} "
            f"train_loss={row['train_loss']:.4f} "
            + " ".join(f"{k}={v:.4f}" for k, v in val_row.items())
        )

        selected = row[f"val_{_selection_metric(config)}"]
        eligible = not gate or row["val_lift_upper"] > floor  # NaN is not eligible
        if eligible and selected > best_metric:
            best_metric = selected
            best_epoch = epoch
            stale = 0
            wrote_best = True
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
        elif not gate or wrote_best or resume_from:
            # under the gate, patience starts once an epoch has been kept
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
            and guard_auroc is not None
            and guard_auroc < guard
        ):
            warnings.warn(
                f"val AUROC {guard_auroc:.3f} below chance guard "
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

    best_path = out / "best.pt"
    # a best.pt left by an earlier run does not count; a resumed run keeps its own
    kept = wrote_best or (resume_from and best_path.exists())
    if gate and not kept:
        lifts = [
            r["val_lift_upper"] for r in history if not math.isnan(r["val_lift_upper"])
        ]
        best_seen = f"{max(lifts):.3f}" if lifts else "n/a"
        raise ForecastNotEligibleError(
            str(out),
            seed,
            f"no forecast epoch beat the eligibility floor {floor:.3f}: "
            f"best val_lift_upper {best_seen} in run_dir {out}",
        )
    if not kept:
        raise RuntimeError(
            f"no best checkpoint was written: validation {_selection_metric(config)} "
            f"never improved (NaN or no epochs run) in run_dir {out}"
        )
    if best_epoch is None:
        best_epoch = load_checkpoint(best_path, map_location="cpu")["epoch"]
    return FitResult(
        best_path=str(best_path),
        history=history,
        best_metric=float(best_metric),
        meta=meta,
        best_epoch=best_epoch,
    )

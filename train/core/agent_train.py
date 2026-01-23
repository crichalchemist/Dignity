import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

import torch
from torch.utils.data import DataLoader

from config.config import TrainingConfig
from models.agent_hybrid import DignityModel
from models.signal_policy import SignalModel
from risk.risk_manager import RiskManager


def _to_device(batch, device):
    x, y = batch
    x = x.to(device)
    if isinstance(y, dict):
        y = {k: v.to(device) for k, v in y.items()}
    else:
        y = y.to(device)
    return x, y


def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)


def _regression_rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    preds, targets = _align_regression(preds, targets)
    mse = torch.mean((preds - targets) ** 2)
    return torch.sqrt(mse).item()


def _select_outputs(outputs: dict[str, torch.Tensor], task_type: str) -> torch.Tensor:
    if task_type == "classification":
        return outputs["direction_logits"]
    return outputs["return"]


def _align_regression(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    preds = preds.squeeze(-1)
    targets = targets.squeeze(-1)
    return preds, targets


def _compute_losses(
        outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], cfg: TrainingConfig, task_type: str
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    primary_out = outputs["primary"]
    primary_target = targets["primary"]
    if task_type == "regression":
        primary_out, primary_target = _align_regression(primary_out, primary_target)
        primary_loss = mse(primary_out, primary_target)
    else:
        primary_loss = ce(primary_out, primary_target)

    losses = {"primary": primary_loss.item()}
    total_loss = primary_loss

    max_return = outputs["max_return"].squeeze(-1)
    max_ret_loss = mse(max_return, targets["max_return"])
    total_loss = total_loss + cfg.max_return_weight * max_ret_loss
    losses["max_return"] = max_ret_loss.item()

    topk_ret_loss = mse(outputs["topk_returns"], targets["topk_returns"])
    total_loss = total_loss + cfg.topk_return_weight * topk_ret_loss
    losses["topk_returns"] = topk_ret_loss.item()

    topk_price_loss = mse(outputs["topk_prices"], targets["topk_prices"])
    total_loss = total_loss + cfg.topk_price_weight * topk_price_loss
    losses["topk_prices"] = topk_price_loss.item()

    if "sell_now" in outputs and "sell_now" in targets and cfg.sell_now_weight > 0:
        bce = torch.nn.BCEWithLogitsLoss()
        sell_logits = outputs["sell_now"].squeeze(-1)
        sell_target = targets["sell_now"].float()
        sell_loss = bce(sell_logits, sell_target)
        total_loss = total_loss + cfg.sell_now_weight * sell_loss
        losses["sell_now"] = sell_loss.item()

    return total_loss, losses


def _evaluate(
        model: DignityModel,
        loader: DataLoader,
        cfg: TrainingConfig,
        device,
        task_type: str,
        risk_manager: RiskManager | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    metric_accum = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _to_device(batch, device)
            outputs, _ = model(x)
            logits = _select_outputs(outputs, task_type)
            context = risk_manager.build_context(x=x) if risk_manager else None
            if risk_manager:
                if task_type == "classification":
                    logits, reasons = risk_manager.apply_classification_logits(logits, context)
                else:
                    logits, reasons = risk_manager.apply_regression_output(logits, context)
                risk_manager.log_events(reasons, prefix="eval")
            outputs = dict(outputs)
            outputs["primary"] = logits

            loss, _ = _compute_losses(outputs, y, cfg, task_type)
            total_loss += loss.item()

            primary_out = outputs["primary"]
            primary_target = y["primary"]
            if task_type == "regression":
                primary_out, primary_target = _align_regression(primary_out, primary_target)
                metric_accum += _regression_rmse(primary_out, primary_target)
            else:
                metric_accum += _classification_metrics(primary_out, primary_target)
            batches += 1
    return total_loss / max(1, batches), metric_accum / max(1, batches)


def train_model(
        model: DignityModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingConfig,
        scheduler=None,
        task_type: str = "classification",
        risk_manager: RiskManager | None = None,
) -> dict[str, list[float]]:
    # ------------------------------------------------------------------
    # Logging configuration and device handling
    # ------------------------------------------------------------------
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    device_str = cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    model.to(device)

    if risk_manager is None and getattr(cfg, "risk", None) and cfg.risk.enabled:
        risk_manager = RiskManager(cfg.risk)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "val_metric": []}
    best_metric = -float("inf") if task_type == "classification" else float("inf")
    best_state = None
    epochs_without_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x, y = _to_device(batch, device)
            outputs, _ = model(x)
            logits = _select_outputs(outputs, task_type)
            context = risk_manager.build_context(x=x) if risk_manager else None
            if risk_manager:
                if task_type == "classification":
                    logits, reasons = risk_manager.apply_classification_logits(logits, context)
                    actions = logits.argmax(dim=1)
                    risk_manager.record_actions(actions)
                else:
                    logits, reasons = risk_manager.apply_regression_output(logits, context)
                risk_manager.log_events(reasons, prefix=f"epoch {epoch} step {step}")
            outputs = dict(outputs)
            outputs["primary"] = logits
            loss, _ = _compute_losses(outputs, y, cfg, task_type)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            if step % cfg.log_every == 0:
                print(f"epoch {epoch} step {step} loss {running_loss / step:.4f}")

        if scheduler:
            scheduler.step()

        train_epoch_loss = running_loss / max(1, len(train_loader))
        val_loss, val_metric = _evaluate(
            model, val_loader, cfg, device, task_type, risk_manager=risk_manager
        )

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)

        is_better = val_metric > best_metric if task_type == "classification" else val_metric < best_metric
        if is_better:
            best_metric = val_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            if cfg.checkpoint_path:
                torch.save(best_state, cfg.checkpoint_path)
                logger.info(f"Saved new best checkpoint to {cfg.checkpoint_path}")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        logger.info(
            f"epoch {epoch}/{cfg.epochs} train_loss {train_epoch_loss:.4f} "
            f"val_loss {val_loss:.4f} val_metric {val_metric:.4f}"
        )

        # Early‑stopping based on patience configured in TrainingConfig.
        if getattr(cfg, "early_stop_patience", None) is not None and epochs_without_improve >= cfg.early_stop_patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch} (no improvement for {cfg.early_stop_patience} epochs)."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def _signal_losses(signal_out: dict, targets: torch.Tensor, task_type: str):
    losses = []
    metrics = {}
    aux = signal_out.get("aux", {})
    if "direction_logits" in aux and task_type == "classification":
        cls_loss = torch.nn.functional.cross_entropy(aux["direction_logits"], targets.long())
        losses.append(cls_loss)
        metrics["direction_acc"] = _classification_metrics(aux["direction_logits"], targets)
    if "forecast" in aux:
        forecast_target = targets.float().unsqueeze(-1) if targets.dim() == 1 else targets.float()
        reg_loss = torch.nn.functional.mse_loss(aux["forecast"], forecast_target)
        losses.append(reg_loss)
        metrics["forecast_rmse"] = _regression_rmse(aux["forecast"], forecast_target)
    total_loss = sum(losses) if losses else torch.tensor(0.0, device=targets.device)
    return total_loss, metrics


def pretrain_signal_model(
        signal_model: SignalModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainingConfig,
        task_type: str = "classification",
) -> dict[str, list[float]]:
    device_str = cfg.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    signal_model.to(device)

    optimizer = torch.optim.Adam(
        signal_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    history = {"train_loss": [], "val_loss": [], "val_metric": []}
    best_metric = -float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        signal_model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            x, y = _to_device(batch, device)
            signal_out = signal_model(x)
            loss, _ = _signal_losses(signal_out, y, task_type)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(signal_model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            if step % cfg.log_every == 0:
                print(f"[signal] epoch {epoch} step {step} loss {running_loss / step:.4f}")

        val_loss, val_metric = _evaluate_signal(signal_model, val_loader, device, task_type)
        history["train_loss"].append(running_loss / max(1, len(train_loader)))
        history["val_loss"].append(val_loss)
        history["val_metric"].append(val_metric)

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = {k: v.cpu() for k, v in signal_model.state_dict().items()}
            if cfg.checkpoint_path:
                torch.save(best_state, cfg.checkpoint_path)

        print(
            f"[signal] epoch {epoch}/{cfg.epochs} train_loss {history['train_loss'][-1]:.4f} "
            f"val_loss {val_loss:.4f} val_metric {val_metric:.4f}"
        )

    if best_state is not None:
        signal_model.load_state_dict(best_state)

    return history


def _evaluate_signal(signal_model: SignalModel, loader: DataLoader, device, task_type: str):
    signal_model.eval()
    total_loss = 0.0
    total_metric = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x, y = _to_device(batch, device)
            signal_out = signal_model(x)
            loss, metrics = _signal_losses(signal_out, y, task_type)
            total_loss += loss.item()
            metric_val = metrics.get("direction_acc", metrics.get("forecast_rmse", 0.0))
            total_metric += metric_val
            batches += 1
    return total_loss / max(1, batches), total_metric / max(1, batches)


# REMOVED: Fake RL training functions train_execution_policy and _prepare_actions_and_rewards
# These were supervised learning disguised as RL. For real RL, use train/core/env_based_rl_training.py

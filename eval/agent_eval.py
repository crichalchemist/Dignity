import numpy as np
import torch
from torch.utils.data import DataLoader

from models.agent_hybrid import DignityModel
from models.signal_policy import SignalPolicyAgent
from risk.risk_manager import RiskManager


def _collect_outputs(
        model: DignityModel,
        loader: DataLoader,
        device: torch.device,
        risk_manager: RiskManager | None = None,
        task_type: str = "classification",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs, _ = model(x)
            primary = outputs["primary"]
            if risk_manager:
                context = risk_manager.build_context(x=x)
                if task_type == "classification":
                    primary, reasons = risk_manager.apply_classification_logits(primary, context)
                else:
                    primary, reasons = risk_manager.apply_regression_output(primary, context)
                risk_manager.log_events(reasons, prefix="eval")
            y_primary = y["primary"].to(device) if isinstance(y, dict) else y.to(device)
            preds.append(primary.detach().cpu().numpy())
            targets.append(y_primary.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    pred_labels = logits.argmax(axis=1)
    accuracy = (pred_labels == targets).mean()

    num_classes = logits.shape[1]
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, pred_labels, strict=False):
        confusion[int(t), int(p)] += 1

    precision_list = []
    recall_list = []
    f1_list = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "accuracy": float(accuracy),
        "precision_macro": float(np.mean(precision_list)),
        "recall_macro": float(np.mean(recall_list)),
        "f1_macro": float(np.mean(f1_list)),
        "confusion_matrix": confusion,
    }


def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    preds_flat = preds.squeeze()
    targets_flat = targets.squeeze()
    mse = np.mean((preds_flat - targets_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_flat - targets_flat))
    target_mean = targets_flat.mean()
    ss_tot = np.sum((targets_flat - target_mean) ** 2)
    ss_res = np.sum((targets_flat - preds_flat) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def evaluate_model(
        model: DignityModel,
        loader: DataLoader,
        task_type: str = "classification",
        risk_manager: RiskManager | None = None,
) -> dict[str, float]:
    device = next(model.parameters()).device
    logits_or_preds, targets = _collect_outputs(
        model, loader, device, risk_manager=risk_manager, task_type=task_type
    )
    if task_type == "classification":
        return classification_metrics(logits_or_preds, targets)
    return regression_metrics(logits_or_preds, targets)


def evaluate_policy_agent(
        agent: SignalPolicyAgent, loader: DataLoader, task_type: str = "classification"
) -> dict[str, float]:
    device = next(agent.parameters()).device
    agent.eval()
    total = 0
    correct = 0
    value_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = agent(x, detach_signal=True)
            actions = out["policy_logits"].argmax(dim=1)
            if task_type == "classification":
                correct += (actions == y).sum().item()
                total += y.numel()
            else:
                target_actions = (y.squeeze(-1) > 0).long()
                correct += (actions == target_actions).sum().item()
                total += target_actions.numel()
            value_sum += out["value"].detach().cpu().sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return {"action_accuracy": accuracy, "avg_value": value_sum / max(1, total)}


def evaluate_a3c_agent(
        agent, env_factory, num_episodes: int = 100, device: str = "cpu"
) -> dict[str, float]:
    """Evaluate trained A3C agent on simulated execution environment.

    Parameters
    ----------
    agent : A3CAgent
        Trained A3C agent with global_model available.
    env_factory : callable
        Function that creates environment instances.
    num_episodes : int
        Number of episodes to run. Default: 100.
    device : str
        Device for inference. Default: 'cpu'.

    Returns
    -------
    Dict[str, float]
        Evaluation metrics including mean reward, win rate, Sharpe ratio.
    """
    agent.global_model.eval()
    episode_rewards = []
    episode_lengths = []
    winning_episodes = 0

    with torch.no_grad():
        for _ in range(num_episodes):
            env = env_factory()
            obs, _ = agent._reset_env(env)
            done = False
            episode_reward = 0.0
            steps = 0

            while not done:
                obs_tensor = agent._to_tensor(obs)
                logits, value, _ = agent.global_model.forward(obs_tensor)

                # Take greedy action during evaluation
                action = logits.argmax(dim=-1).item()
                obs, reward, terminated, truncated, _ = agent._step_env(env, action)
                done = terminated or truncated

                episode_reward += reward
                steps += 1

                if steps > 1000:  # Prevent infinite episodes
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            if episode_reward > 0:
                winning_episodes += 1

    # Compute metrics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    win_rate = winning_episodes / num_episodes if num_episodes > 0 else 0.0
    mean_episode_length = np.mean(episode_lengths)

    # Simple Sharpe ratio (assuming 252 trading days and rewards as daily returns)
    returns_array = np.array(episode_rewards)
    sharpe_ratio = (returns_array.mean() / returns_array.std()) * np.sqrt(252) if returns_array.std() > 0 else 0.0

    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "max_reward": float(max_reward),
        "min_reward": float(min_reward),
        "win_rate": float(win_rate),
        "mean_episode_length": float(mean_episode_length),
        "sharpe_ratio": float(sharpe_ratio),
        "num_episodes": num_episodes,
    }

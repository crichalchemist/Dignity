"""Run Dignity encoder + policy inference against a simulator or MetaApi.

This runner builds features from candle data (plus optional intrinsic bars),
loads the Dignity CNN-LSTM-attention encoder, and routes actions through an
execution backend selected at runtime. The same inference path is shared across
simulation and live modes through dependency injection.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import ModelConfig  # noqa: E402
from models.agent_hybrid import DignityModel, build_model  # noqa: E402
from train.features.agent_features import build_feature_frame  # noqa: E402

ACTION_NAMES = ["sell", "hold", "buy"]


@dataclass
class InferenceConfig:
    candles_path: Path
    checkpoint_path: Path
    intrinsic_bars_path: Path | None = None
    t_in: int = 120
    device: str = "cpu"
    task_type: str = "classification"


class ExecutionBackend(Protocol):
    """Backend interface so simulation and live calls share the same pipeline."""

    def execute_action(self, action: str, price: float, metadata: dict[str, float]) -> float:
        """Dispatch an action and return a reward-like scalar."""


class SimulatorBackend:
    """Minimal reward simulator tracking directional PnL across steps."""

    def __init__(self, spread: float = 0.00005, volume: float = 1.0):
        self.spread = spread
        self.volume = volume
        self.position: int = 0
        self.entry_price: float | None = None

    def execute_action(self, action: str, price: float, metadata: dict[str, float]) -> float:
        reward = 0.0
        if action == "buy":
            if self.position <= 0:
                self.entry_price = price + self.spread
            self.position = 1
        elif action == "sell":
            if self.position >= 0:
                self.entry_price = price - self.spread
            self.position = -1
        else:
            self.position = 0

        if self.entry_price is not None and self.position != 0:
            direction = 1 if self.position > 0 else -1
            reward = direction * (price - self.entry_price) * self.volume
        return reward


class MetaApiBackend:
    """Thin HTTP client for MetaApi or compatible trade gateways."""

    def __init__(
            self,
            endpoint: str,
            token: str,
            account_id: str,
            timeout: int = 10,
            session: requests.Session | None = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.account_id = account_id
        self.timeout = timeout
        self.session = session or requests.Session()

    def execute_action(self, action: str, price: float, metadata: dict[str, float]) -> float:
        url = f"{self.endpoint}/trade"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"accountId": self.account_id, "action": action, "price": price, "meta": metadata}
        response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()
        reward = float(body.get("reward", 0.0))
        return reward


def _load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required candle columns: {', '.join(sorted(missing))}")
    return df.reset_index(drop=True)


def _merge_intrinsic(feature_df: pd.DataFrame, intrinsic_path: Path) -> pd.DataFrame:
    intrinsic = pd.read_csv(intrinsic_path)
    intrinsic = intrinsic.copy()
    if "datetime" in intrinsic.columns and "datetime" in feature_df.columns:
        intrinsic["datetime"] = pd.to_datetime(intrinsic["datetime"])
        intrinsic = intrinsic.sort_values("datetime")
        merged = feature_df.merge(intrinsic, on="datetime", how="inner", suffixes=("", "_intrinsic"))
    else:
        intrinsic = intrinsic.tail(len(feature_df)).reset_index(drop=True)
        intrinsic.columns = [f"intrinsic_{col}" for col in intrinsic.columns]
        merged = pd.concat([feature_df.reset_index(drop=True), intrinsic], axis=1)
    merged = merged.dropna().reset_index(drop=True)
    return merged


def build_state(cfg: InferenceConfig) -> tuple[torch.Tensor, float, list[str]]:
    candles = _load_candles(cfg.candles_path)
    feature_df = build_feature_frame(candles)
    if cfg.intrinsic_bars_path:
        feature_df = _merge_intrinsic(feature_df, cfg.intrinsic_bars_path)

    if len(feature_df) < cfg.t_in:
        raise ValueError(f"Not enough rows to build a window of length {cfg.t_in}.")

    window = feature_df.tail(cfg.t_in)
    feature_cols = [col for col in window.columns if col != "datetime"]
    state_array = window[feature_cols].to_numpy(dtype="float32")
    latest_price = float(candles.iloc[-1]["close"])
    state_tensor = torch.tensor(state_array)
    return state_tensor, latest_price, feature_cols


def load_policy(cfg: InferenceConfig, num_features: int) -> DignityModel:
    model_cfg = ModelConfig(num_features=num_features)
    model = build_model(model_cfg, task_type=cfg.task_type)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)
    model.to(cfg.device)
    model.eval()
    return model


def telemetry_log(logger: logging.Logger, message: str, payload: dict[str, object]) -> None:
    logger.info("%s | %s", message, json.dumps(payload, default=str))


class InferenceRunner:
    def __init__(self, model: DignityModel, backend: ExecutionBackend, device: str):
        self.model = model
        self.backend = backend
        self.device = device

    def predict(self, state: torch.Tensor) -> dict[str, object]:
        with torch.no_grad():
            logits, attn = self.model(state.unsqueeze(0).to(self.device))
            probs = F.softmax(logits, dim=-1).squeeze(0)
            action_idx = int(torch.argmax(probs).item())
            action = ACTION_NAMES[action_idx]
            return {
                "action": action,
                "probs": probs.cpu().tolist(),
                "attention": attn.cpu().tolist(),
            }

    def run_once(self, state: torch.Tensor, price: float) -> dict[str, object]:
        decision = self.predict(state)
        reward = self.backend.execute_action(decision["action"], price, {"prob_buy": decision["probs"][2]})
        decision["reward"] = reward
        return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dignity encoder + policy inference runner")
    parser.add_argument("--candles", required=True, type=Path, help="CSV with OHLC candles including datetime")
    parser.add_argument("--intrinsic-bars", type=Path, help="Optional CSV of intrinsic features to append")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to model checkpoint")
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length")
    parser.add_argument("--device", default="cpu", help="Device for model execution")
    parser.add_argument("--mode", choices=["simulate", "live"], default="simulate", help="Execution backend")
    parser.add_argument("--metaapi-endpoint", default="https://metaapi.cloud/v1", help="MetaApi REST endpoint")
    parser.add_argument("--metaapi-token", help="MetaApi bearer token (do not log or hardcode)")
    parser.add_argument("--account-id", help="Target MetaApi account id")
    parser.add_argument("--spread", type=float, default=0.00005, help="Simulated spread for reward calc")
    parser.add_argument("--volume", type=float, default=1.0, help="Simulated volume unit")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def configure_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    return logging.getLogger("infer")


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.log_level)

    if args.mode == "live" and (not args.metaapi_token or not args.account_id):
        raise ValueError("Live mode requires --metaapi-token and --account-id")

    cfg = InferenceConfig(
        candles_path=args.candles,
        intrinsic_bars_path=args.intrinsic_bars,
        checkpoint_path=args.checkpoint,
        t_in=args.t_in,
        device=args.device,
    )

    state, price, feature_cols = build_state(cfg)
    model = load_policy(cfg, num_features=state.shape[-1])

    backend: ExecutionBackend
    if args.mode == "simulate":
        backend = SimulatorBackend(spread=args.spread, volume=args.volume)
    else:
        backend = MetaApiBackend(
            endpoint=args.metaapi_endpoint,
            token=args.metaapi_token,
            account_id=args.account_id,
        )

    runner = InferenceRunner(model=model, backend=backend, device=cfg.device)
    decision = runner.run_once(state, price)

    telemetry_log(
        logger,
        "decision",
        {
            "action": decision["action"],
            "reward": decision["reward"],
            "probabilities": decision["probs"],
            "last_features": state[-1].tolist(),
            "features": feature_cols,
        },
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()

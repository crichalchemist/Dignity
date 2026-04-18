"""dignity-backtest CLI — evaluate a trained checkpoint against historical data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.config import DignityConfig
from core.signals import ASSET_CONFIGS
from data.pipeline import TransactionPipeline

from .runner import BacktestConfig, align_signals, run_backtest


def _load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV CSV. Expects columns: timestamp/date, open, high, low, close, volume."""
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def _set_inference_mode(model: torch.nn.Module) -> torch.nn.Module:
    """Set model to inference mode (disables dropout/batchnorm training behaviour)."""
    # model.eval() is a PyTorch method — not Python's built-in eval()
    return model.eval()


def _inference(
    checkpoint_path: str,
    ohlcv: pd.DataFrame,
    config: DignityConfig,
) -> dict[str, np.ndarray]:
    """Run cascade model inference on historical OHLCV → per-bar signals.

    Returns signal dict aligned to ohlcv length (warmup bars are neutral).
    """
    from models.dignity import Dignity

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = Dignity(
        task="cascade",
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    _set_inference_mode(model)

    execution_cfg = getattr(config, "execution", None)
    asset_class = execution_cfg.asset_class if execution_cfg else "equity"
    asset_cfg = ASSET_CONFIGS.get(asset_class, ASSET_CONFIGS["equity"])

    pipeline = TransactionPipeline(
        seq_len=config.data.seq_len,
        features=config.data.features,
    )
    X, _ = pipeline.process(ohlcv, fit=True, stride=1)

    X_tensor = torch.FloatTensor(X).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)

    n_bars = len(ohlcv)
    raw: dict[str, np.ndarray] = {
        "action": outputs["action_logits"].argmax(dim=-1).cpu().numpy().astype(float),
        "var": outputs["var_estimate"].squeeze(-1).cpu().numpy(),
        "alpha": outputs["alpha"].squeeze(-1).cpu().numpy(),
        "regime": outputs["regime"].argmax(dim=-1).cpu().numpy().astype(float),
    }

    return align_signals(raw, n_bars=n_bars, seq_len=config.data.seq_len)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Backtest a trained Dignity cascade model against historical OHLCV data."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--config", required=True, help="Path to DignityConfig YAML")
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--commission", type=float, default=0.0002)
    parser.add_argument("--spread", type=float, default=0.0001)
    parser.add_argument("--max-drawdown", type=float, default=0.05)
    parser.add_argument("--plot", action="store_true", help="Generate interactive HTML plot")
    parser.add_argument("--plot-path", default="backtest.html", help="Output path for HTML plot")
    args = parser.parse_args(argv)

    dignity_config = DignityConfig.from_yaml(args.config)
    ohlcv = _load_ohlcv(args.data)

    print(f"Running inference on {len(ohlcv)} bars ...")
    signals = _inference(args.checkpoint, ohlcv, dignity_config)

    bt_config = BacktestConfig(
        cash=args.cash,
        commission=args.commission,
        spread=args.spread,
        max_drawdown=args.max_drawdown,
    )

    print("Running backtest ...")
    stats = run_backtest(ohlcv, signals, config=bt_config, plot=args.plot, plot_path=args.plot_path)

    keys = [
        "Return [%]", "Return (Ann.) [%]", "Sharpe Ratio", "Sortino Ratio",
        "Max. Drawdown [%]", "# Trades", "Win Rate [%]", "Profit Factor",
    ]
    print("\n-- Backtest Results ---------------------------------")
    for k in keys:
        if k in stats.index:
            v = stats[k]
            print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")
    print("-----------------------------------------------------")

    if args.plot:
        print(f"\nPlot saved to: {args.plot_path}")


if __name__ == "__main__":
    main()

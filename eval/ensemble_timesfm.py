"""
Ensemble the Dignity model with TimesFM (torch) for regression forecasting.

Steps:
  1. Build regression datasets for each pair (uses Central-time HistData zips).
  2. Load Dignity checkpoint.
  3. Run TimesFM forecast_naive on denormalized close windows.
  4. Ensemble Dignity + TimesFM (mean) and report RMSE/MAE.

Usage example:
  python eval/ensemble_timesfm.py \
    --pairs gbpusd \
    --years 2023 \
    --t-in 60 --t-out 10 \
    --checkpoint-root models \
    --device cpu
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import DataConfig, FeatureConfig, ModelConfig
from data.agents.single_task_agent import SingleTaskDataAgent as DataAgent
from data.prepare_dataset import _compute_time_ranges, _load_pair_data
from eval.timesfm_wrapper import TimesFMWrapper
from models.agent_hybrid import build_model
from train.features.agent_features import build_feature_frame


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble Dignity model with TimesFM for regression.")
    p.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes")
    p.add_argument("--years", default=None, help="Comma-separated years to include (default: all)")
    p.add_argument("--input-root", default="output_central", help="Root with Central-time HistData zips")
    p.add_argument("--t-in", type=int, default=120)
    p.add_argument("--t-out", type=int, default=10)
    p.add_argument("--flat-threshold", type=float, default=0.0001)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--checkpoint-root", default="models", help="Directory containing <pair>_best_model.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--feature-groups", default="all", help="Comma-separated feature groups to include or 'all'")
    p.add_argument("--exclude-feature-groups", default=None, help="Comma-separated feature groups to drop")
    p.add_argument("--sma-windows", default="10,20,50", help="Comma-separated SMA window lengths")
    p.add_argument("--ema-windows", default="10,20,50", help="Comma-separated EMA spans")
    p.add_argument("--rsi-window", type=int, default=14, help="Window length for RSI")
    p.add_argument("--bollinger-window", type=int, default=20, help="Window length for Bollinger bands")
    p.add_argument("--bollinger-num-std", type=float, default=2.0, help="Std dev multiplier for Bollinger bands")
    p.add_argument("--atr-window", type=int, default=14, help="Window length for ATR")
    p.add_argument("--short-vol-window", type=int, default=10, help="Short window for volatility clustering")
    p.add_argument("--long-vol-window", type=int, default=50, help="Long window for volatility clustering")
    p.add_argument("--spread-windows", default="20", help="Comma-separated windows for normalized spread stats")
    p.add_argument("--imbalance-smoothing", type=int, default=5, help="Rolling mean window for wick/body imbalance")
    return p.parse_args()


def build_test_windows(
        df,
        agent: DataAgent,
        feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    t_in, t_out = agent.cfg.t_in, agent.cfg.t_out
    norm_stats = agent.norm_stats
    assert norm_stats is not None

    features = norm_stats.apply(df[feature_cols].to_numpy(dtype=np.float32))
    closes = df["close"].to_numpy(dtype=np.float32)
    future_log_ret = np.log(closes[t_out:] / closes[:-t_out])
    # Align future_log_ret with index t_in-1 ... len(df)-t_out-1
    future_log_ret = np.concatenate([future_log_ret, np.full(t_out, np.nan)])

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    close_windows: list[np.ndarray] = []

    last_idx = len(df) - t_out
    for idx in range(t_in - 1, last_idx):
        start = idx - t_in + 1
        end = idx + 1
        target_return = future_log_ret[idx]
        if not np.isfinite(target_return):
            continue
        seq = features[start:end]
        sequences.append(seq)
        targets.append(float(target_return))
        close_windows.append(closes[start:end])

    if not sequences:
        raise ValueError("No sequences created for test split.")
    return np.stack(sequences), np.array(targets), close_windows


def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    preds_flat = preds.squeeze()
    targets_flat = targets.squeeze()
    mse = np.mean((preds_flat - targets_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_flat - targets_flat))
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae)}


def main():
    args = parse_args()
    pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    # Load TimesFM wrapper (uses separate environment)
    tfm_model = TimesFMWrapper(timesfm_env_path=".venvx_timesfm")

    results = {}
    for pair in pairs:
        try:
            # Data prep
            years = args.years.split(",") if args.years else None
            input_root = Path(args.input_root)
            if not input_root.is_absolute():
                input_root = (ROOT / input_root).resolve()
            raw_df = _load_pair_data(pair, input_root, years)
            include_groups = None if args.feature_groups.lower() == "all" else [
                g.strip() for g in args.feature_groups.split(",") if g.strip()
            ]
            exclude_groups = (
                [g.strip() for g in args.exclude_feature_groups.split(",") if g.strip()]
                if args.exclude_feature_groups
                else None
            )
            feature_cfg = FeatureConfig(
                sma_windows=[int(x) for x in args.sma_windows.split(",") if x.strip()],
                ema_windows=[int(x) for x in args.ema_windows.split(",") if x.strip()],
                rsi_window=args.rsi_window,
                bollinger_window=args.bollinger_window,
                bollinger_num_std=args.bollinger_num_std,
                atr_window=args.atr_window,
                short_vol_window=args.short_vol_window,
                long_vol_window=args.long_vol_window,
                spread_windows=[int(x) for x in args.spread_windows.split(",") if x.strip()],
                imbalance_smoothing=args.imbalance_smoothing,
                include_groups=include_groups,
                exclude_groups=exclude_groups,
            )
            feature_df = build_feature_frame(raw_df, config=feature_cfg)
            feature_df["datetime"] = pd.to_datetime(feature_df["datetime"])
            train_range, val_range, test_range = _compute_time_ranges(
                feature_df, args.train_ratio, args.val_ratio
            )
            feature_cols = [c for c in feature_df.columns if c not in {"datetime", "source_file"}]

            data_cfg = DataConfig(
                csv_path="",
                datetime_column="datetime",
                feature_columns=feature_cols,
                target_type="regression",
                t_in=args.t_in,
                t_out=args.t_out,
                train_range=train_range,
                val_range=val_range,
                test_range=test_range,
                flat_threshold=args.flat_threshold,
            )
            agent = DataAgent(data_cfg)
            splits = agent.split_dataframe(feature_df)
            agent.fit_normalization(splits["train"], feature_cols)
            test_df = splits["test"]
            sequences, targets, close_windows = build_test_windows(test_df, agent, feature_cols)

            # Dignity model
            num_features = sequences.shape[-1]
            model_cfg = ModelConfig(num_features=num_features, num_classes=None, output_dim=1)
            dignity = build_model(model_cfg, task_type="regression").to(device)
            ckpt_path = Path(args.checkpoint_root) / f"{pair}_best_model.pt"
            if not ckpt_path.exists():
                print(f"[warn] checkpoint not found for {pair}: {ckpt_path}, skipping")
                continue
            state = torch.load(ckpt_path, map_location=device)
            dignity.load_state_dict(state)
            dignity.eval()

            preds_dignity = []
            preds_tfm = []
            preds_ensemble = []

            for seq, close_seq in zip(sequences, close_windows, strict=False):
                x = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    outputs, _ = dignity(x)
                dignity_ret = outputs["return"].squeeze().item()

                tfm_forecast = tfm_model.forecast_naive(horizon=args.t_out, inputs=[close_seq])[0]
                if tfm_forecast is None or len(tfm_forecast) == 0:
                    continue
                last_input = float(close_seq[-1])
                last_forecast = float(tfm_forecast[-1])
                tfm_ret = float(np.log(max(last_forecast, 1e-8) / max(last_input, 1e-8)))

                ensemble_ret = 0.5 * (dignity_ret + tfm_ret)
                preds_dignity.append(dignity_ret)
                preds_tfm.append(tfm_ret)
                preds_ensemble.append(ensemble_ret)

            targets_arr = targets[: len(preds_ensemble)]
            metrics = {
                "dignity": regression_metrics(np.array(preds_dignity), targets_arr),
                "timesfm": regression_metrics(np.array(preds_tfm), targets_arr),
                "ensemble": regression_metrics(np.array(preds_ensemble), targets_arr),
            }
            results[pair] = metrics
            print(f"[eval] {pair} -> {metrics}")
        except Exception as exc:
            print(f"[error] Failed on pair {pair}: {exc}")

    return results


if __name__ == "__main__":
    import pandas as pd

    main()

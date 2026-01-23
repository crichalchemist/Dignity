"""
End-to-end training entrypoint for the multi-head model (direction cls, trend cls,
return reg, next-close reg, vol cls, vol regime cls, candle-pattern cls).

  python train/run_training_multitask.py \\
    --pairs eurusd,eurgbp,eurjpy,eurchf,euraud,eurcad,eurnzd,gbpusd,gbpjpy,gbpchf,gbpcad,gbpaud,gbpnzd,usdjpy,usdchf,usdcad,audusd,audjpy,audcad,audchf,audnzd,nzdusd,nzdjpy,nzdcad,nzdchf,cadchf,cadjpy,chfjpy,usdbrl,usdrub,usdinr,usdcny,usdzar,usdtry,xauusd \\
    --t-in 120 --t-out 10 \\
    --epochs 5 --batch-size 64 --learning-rate 1e-3 --weight-decay 0.0\\
    --loss-w-direction 1.0 --loss-w-return 1.0 --loss-w-next-close 1.0 --loss-w-vol 1.0 \\
    --checkpoint-path models/gbpusd_best_multitask.pt
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import MultiTaskLossWeights, MultiTaskModelConfig, TrainingConfig
from data.prepare_multitask_dataset import process_pair
from models.agent_multitask import build_multitask_model
from train.core.agent_train_multitask import train_multitask


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-task training for all pairs.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes")
    parser.add_argument("--years", default=None, help="Comma-separated years to include (default: all available)")
    parser.add_argument("--input-root", default="output_central", help="Root containing Central-time zips")
    parser.add_argument("--t-in", type=int, default=120)
    parser.add_argument("--t-out", type=int, default=10)
    parser.add_argument("--lookahead-window", type=int, default=None, help="Lookahead for auxiliary tasks")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K future returns/prices predictions")
    parser.add_argument("--predict-sell-now", action="store_true", help="Enable sell-now auxiliary head")
    parser.add_argument("--flat-threshold", type=float, default=0.0001)
    parser.add_argument("--vol-min-change", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--feature-groups", default="all", help="Comma-separated feature groups to include or 'all'")
    parser.add_argument("--exclude-feature-groups", default=None, help="Comma-separated feature groups to drop")
    parser.add_argument("--sma-windows", default="10,20,50", help="Comma-separated SMA window lengths")
    parser.add_argument("--ema-windows", default="10,20,50", help="Comma-separated EMA spans")
    parser.add_argument("--rsi-window", type=int, default=14, help="Window length for RSI")
    parser.add_argument("--bollinger-window", type=int, default=20, help="Window length for Bollinger bands")
    parser.add_argument("--bollinger-num-std", type=float, default=2.0, help="Std dev multiplier for Bollinger bands")
    parser.add_argument("--atr-window", type=int, default=14, help="Window length for ATR")
    parser.add_argument("--short-vol-window", type=int, default=10, help="Short window for volatility clustering")
    parser.add_argument("--long-vol-window", type=int, default=50, help="Long window for volatility clustering")
    parser.add_argument("--spread-windows", default="20", help="Comma-separated windows for normalized spread stats")
    parser.add_argument("--imbalance-smoothing", type=int, default=5,
                        help="Rolling mean window for wick/body imbalance")
    parser.add_argument("--gdelt-path", default=None, help="Optional path to a GDELT GKG file for sentiment features")
    parser.add_argument("--gdelt-tz", default="UTC", help="Timezone to convert GDELT timestamps into (default UTC)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint-path",
        default="models/best_multitask.pt",
        help="Path for saving checkpoints; supports {pair} placeholder.",
    )
    parser.add_argument("--loss-w-direction", type=float, default=1.0)
    parser.add_argument("--loss-w-return", type=float, default=1.0)
    parser.add_argument("--loss-w-next-close", type=float, default=1.0)
    parser.add_argument("--loss-w-vol", type=float, default=1.0)
    parser.add_argument(
        "--disable-risk",
        action="store_true",
        help="Disable risk manager gating during multi-task training.",
    )
    parser.add_argument("--loss-w-max-return", type=float, default=1.0)
    parser.add_argument("--loss-w-topk-return", type=float, default=1.0)
    parser.add_argument("--loss-w-topk-price", type=float, default=1.0)
    parser.add_argument("--loss-w-sell-now", type=float, default=1.0)
    parser.add_argument("--loss-w-trend", type=float, default=1.0)
    parser.add_argument("--loss-w-vol-regime", type=float, default=1.0)
    parser.add_argument("--loss-w-candle", type=float, default=1.0)
    return parser.parse_args()


def main():
    # Initialize tracing for observability
    try:
        from utils.tracing import setup_tracing
        setup_tracing(
            service_name="sequence-multitask-training",
            otlp_endpoint="http://localhost:4318",
            environment="development"
        )
    except ImportError:
        print("[warn] OpenTelemetry not available; running without tracing")
    except Exception as e:
        print(f"[warn] Failed to initialize tracing: {e}; continuing without tracing")

    args = parse_args()
    pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    results = {}
    for pair in pairs:
        class PrepArgs:
            pairs = pair
            input_root = args.input_root
            years = args.years
            t_in = args.t_in
            t_out = args.t_out
            lookahead_window = args.lookahead_window
            top_k = args.top_k
            predict_sell_now = args.predict_sell_now
            flat_threshold = args.flat_threshold
            vol_min_change = args.vol_min_change
            train_ratio = args.train_ratio
            val_ratio = args.val_ratio
            feature_groups = args.feature_groups
            exclude_feature_groups = args.exclude_feature_groups
            sma_windows = args.sma_windows
            ema_windows = args.ema_windows
            rsi_window = args.rsi_window
            bollinger_window = args.bollinger_window
            bollinger_num_std = args.bollinger_num_std
            atr_window = args.atr_window
            short_vol_window = args.short_vol_window
            long_vol_window = args.long_vol_window
            spread_windows = args.spread_windows
            imbalance_smoothing = args.imbalance_smoothing
            gdelt_path = args.gdelt_path
            gdelt_tz = args.gdelt_tz

        try:
            pair_name, loaders = process_pair(pair, PrepArgs)
        except Exception as exc:
            print(f"[error] data prep failed for {pair}: {exc}")
            continue

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        sample_batch = next(iter(train_loader))
        num_features = sample_batch[0].shape[-1]

        if "{pair}" in args.checkpoint_path:
            ckpt_path = Path(args.checkpoint_path.format(pair=pair))
        elif args.checkpoint_path == "models/best_multitask.pt":
            ckpt_path = Path("models") / f"{pair}_best_multitask.pt"
        else:
            ckpt_path = Path(args.checkpoint_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        model_cfg = MultiTaskModelConfig(
            num_features=num_features,
            lookahead_window=args.lookahead_window,
            top_k_predictions=args.top_k,
            predict_sell_now=args.predict_sell_now,
        )
        train_cfg = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            checkpoint_path=str(ckpt_path),
        )
        train_cfg.risk.enabled = not args.disable_risk
        loss_weights = MultiTaskLossWeights(
            direction_cls=args.loss_w_direction,
            return_reg=args.loss_w_return,
            next_close_reg=args.loss_w_next_close,
            vol_cls=args.loss_w_vol,
            max_return_reg=args.loss_w_max_return,
            topk_return_reg=args.loss_w_topk_return,
            topk_price_reg=args.loss_w_topk_price,
            sell_now_cls=args.loss_w_sell_now,
            trend_cls=args.loss_w_trend,
            vol_regime_cls=args.loss_w_vol_regime,
            candle_pattern_cls=args.loss_w_candle,
        )

        model = build_multitask_model(model_cfg)
        history = train_multitask(
            model,
            train_loader,
            val_loader,
            train_cfg,
            loss_weights,
        )
        results[pair_name] = history
        print(f"[done] {pair_name} multi-task training complete.")

    return results


if __name__ == "__main__":
    main()

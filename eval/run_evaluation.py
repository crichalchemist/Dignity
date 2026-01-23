"""
Evaluation entrypoint to score trained models on the test split.

Example:
  python eval/run_evaluation.py \\
    --pairs gbpusd \\
    --years 2023 \\
    --t-in 60 --t-out 10 \\
    --task-type classification \\
    --checkpoint-path models/best_model.pt
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

from config.arg_parser import (
    add_data_preparation_args,
    add_feature_engineering_args,
    add_intrinsic_time_args,
    add_risk_args,
)
from config.config import ModelConfig, PolicyConfig, SignalModelConfig
from data.prepare_dataset import process_pair
from eval.agent_eval import evaluate_model, evaluate_policy_agent
from models.agent_hybrid import build_model
from models.signal_policy import SignalModel, SignalPolicyAgent
from risk.risk_manager import RiskManager


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test split.")
    add_data_preparation_args(parser)
    add_feature_engineering_args(parser)
    add_intrinsic_time_args(parser)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-path", default="models/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--signal-checkpoint-path", default=None,
                        help="Optional path to signal checkpoint (format string {pair} supported)")
    parser.add_argument("--policy-checkpoint-path", default=None,
                        help="Optional path to policy checkpoint (format string {pair} supported)")
    parser.add_argument("--use-policy", action="store_true",
                        help="Load the execution policy head on top of the signal model")
    parser.add_argument("--device", default="cuda")
    add_risk_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    results = {}
    risk_manager = None if args.disable_risk else RiskManager()
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
            target_type = args.task_type
            flat_threshold = args.flat_threshold
            train_ratio = args.train_ratio
            val_ratio = args.val_ratio
            batch_size = args.batch_size
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
            intrinsic_time = args.intrinsic_time
            dc_threshold_up = args.dc_threshold_up
            dc_threshold_down = args.dc_threshold_down

        try:
            pair_name, loaders = process_pair(pair, PrepArgs, batch_size=args.batch_size)
        except Exception as exc:
            print(f"[error] data prep failed for {pair}: {exc}")
            continue

        test_loader = loaders["test"]
        num_features = next(iter(test_loader))[0].shape[-1]

        if args.use_policy:
            signal_path = (
                Path(args.signal_checkpoint_path.format(pair=pair))
                if args.signal_checkpoint_path
                else Path(f"models/signal_{pair}.pt")
            )
            policy_path = (
                Path(args.policy_checkpoint_path.format(pair=pair))
                if args.policy_checkpoint_path
                else Path(f"models/policy_{pair}.pt")
            )
            if not signal_path.exists() or not policy_path.exists():
                print(
                    f"[error] signal/policy checkpoint missing: {signal_path} or {policy_path}"
                )
                continue

            signal_cfg = SignalModelConfig(
                num_features=num_features,
                num_classes=3 if args.task_type == "classification" else None,
                output_dim=1,
            )
            signal_dim = SignalModel(signal_cfg).signal_dim
            policy_cfg = PolicyConfig(
                input_dim=signal_dim,
                num_actions=3 if args.task_type == "classification" else 2,
            )
            agent = SignalPolicyAgent.load(
                signal_cfg,
                policy_cfg,
                str(signal_path),
                str(policy_path),
                device,
            )
            metrics = evaluate_policy_agent(agent, test_loader, task_type=args.task_type)
            results[pair_name] = metrics
            print(f"[eval-policy] {pair_name}: {metrics}")
            continue

        model_cfg = ModelConfig(
            num_features=num_features,
            num_classes=3 if args.task_type == "classification" else None,
            lookahead_window=args.lookahead_window,
            top_k_predictions=args.top_k,
            predict_sell_now=args.predict_sell_now,
        )
        model = build_model(model_cfg, task_type=args.task_type).to(device)

        ckpt_path = Path(args.checkpoint_path)
        if not ckpt_path.exists():
            print(f"[error] checkpoint not found: {ckpt_path}")
            continue
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        metrics = evaluate_model(
            model, test_loader, task_type=args.task_type, risk_manager=risk_manager
        )
        results[pair_name] = metrics
        print(f"[eval] {pair_name}: {metrics}")

    return results


if __name__ == "__main__":
    main()

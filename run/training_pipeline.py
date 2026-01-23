"""
Interactive pipeline runner to prepare data, train, and optionally evaluate
for one or more FX pairs. It pauses after each training run so you can decide
whether to queue additional datasets before proceeding.

Example (with optional GDELT download first):
  python utils/run_training_pipeline.py \
    --run-gdelt-download \
    --gdelt-start-date 2024-01-01 --gdelt-end-date 2024-01-07 --gdelt-resolution daily \
    --pairs gbpusd,eurusd \
    --t-in 120 --t-out 10 \
    --epochs 3 \
    --checkpoint-dir models
"""

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add run/ for config.config imports (needed for Colab compatibility)
if str(ROOT / "run") not in sys.path:
    sys.path.insert(0, str(ROOT / "run"))

from config.config import ModelConfig, TrainingConfig  # noqa: E402
from data.prepare_dataset import process_pair  # noqa: E402
from eval.agent_eval import evaluate_model  # noqa: E402
from models.agent_hybrid import build_model  # noqa: E402
from risk.risk_manager import RiskManager  # noqa: E402
from train.core.agent_train import train_model  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger(__name__)

# Optional RL imports (lazy loaded)
A3CAgent = None
A3CConfig = None
SimulatedRetailExecutionEnv = None
BacktestingRetailExecutionEnv = None
ExecutionConfig = None


def parse_pairs(pairs: str, pairs_file: Path | None) -> list[str]:
    seeds: list[str] = []
    if pairs:
        seeds.extend([p.strip().lower() for p in pairs.split(",") if p.strip()])
    if pairs_file:
        seeds.extend([line.strip().lower() for line in pairs_file.read_text().splitlines() if line.strip()])
    # Preserve order, drop duplicates.
    seen = set()
    ordered: list[str] = []
    for p in seeds:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, train, and optionally evaluate pairs sequentially.")
    parser.add_argument("--pairs", default="gbpusd", help="Comma-separated pair codes to run first.")
    parser.add_argument("--pairs-file", type=Path, help="Optional file with one pair per line to append.")
    parser.add_argument("--years", default=None, help="Comma-separated years to include (default: all available).")
    parser.add_argument("--input-root", default="output_central", help="Root containing Central-time zips.")
    parser.add_argument("--t-in", type=int, default=120, help="Lookback window length.")
    parser.add_argument("--t-out", type=int, default=10, help="Prediction horizon.")
    parser.add_argument("--task-type", choices=["classification", "regression"], default="classification")
    parser.add_argument("--flat-threshold", type=float, default=0.0001)
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
    parser.add_argument(
        "--intrinsic-time",
        action="store_true",
        help="Convert minute bars to intrinsic-time bars via directional-change events before feature building.",
    )
    parser.add_argument(
        "--dc-threshold-up",
        type=float,
        default=0.001,
        help="Fractional increase needed to flag an upward directional change (e.g., 0.001 = 0.1 percent).",
    )
    parser.add_argument(
        "--dc-threshold-down",
        type=float,
        default=None,
        help="Fractional decrease needed to flag a downward directional change. Defaults to dc-threshold-up.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", default="models", help="Directory to write checkpoints per pair.")
    parser.add_argument(
        "--force-train",
        dest="resume_if_ckpt",
        action="store_false",
        help="Train even if a checkpoint already exists for the pair.",
    )
    parser.set_defaults(resume_if_ckpt=True)
    parser.add_argument("--skip-eval", action="store_true", help="Skip test-set evaluation after training.")
    parser.add_argument(
        "--pause",
        dest="no_pause",
        action="store_false",
        help="Enable interactive prompts after each training run.",
    )
    parser.add_argument(
        "--no-game",
        dest="offer_game",
        action="store_false",
        help="Skip offering the optional training mini-game.",
    )
    parser.set_defaults(no_pause=True, offer_game=True)
    parser.add_argument(
        "--disable-risk",
        action="store_true",
        help="Disable risk manager gating during training and evaluation.",
    )
    parser.add_argument(
        "--run-gdelt-download",
        action="store_true",
        help="Download GDELT GKG files before training.",
    )
    parser.add_argument("--gdelt-start-date", default="2016-01-01", help="Start date for GDELT download (YYYY-MM-DD).")
    parser.add_argument(
        "--gdelt-end-date",
        default=None,
        help="End date for GDELT download (YYYY-MM-DD). Defaults to today if omitted.",
    )
    parser.add_argument(
        "--gdelt-resolution",
        choices=["daily", "15min"],
        default="daily",
        help="GDELT download cadence.",
    )
    parser.add_argument(
        "--gdelt-mirror",
        default="gdelt",
        help=(
            "Mirror to use for GDELT downloads (gdelt, hf-maxlong-2022, hf-olm, hf-andreas-helgesson). "
            "Use a Hugging Face mirror if the primary endpoint returns errors."
        ),
    )
    parser.add_argument(
        "--gdelt-mirror-fallbacks",
        default="",
        help=(
            "Comma-separated list of GDELT mirrors to try if the primary source fails."
        ),
    )
    parser.add_argument(
        "--gdelt-base-url",
        default=None,
        help="Override the GDELT download base URL (https:// only). Takes precedence over --gdelt-mirror.",
    )
    parser.add_argument("--gdelt-step-minutes", type=int, default=15, help="Cadence for 15min resolution.")
    parser.add_argument("--gdelt-out-dir", default="data/gdelt", help="Destination folder for GDELT zips.")
    parser.add_argument("--gdelt-overwrite", action="store_true", help="Re-download even if files exist.")
    parser.add_argument("--gdelt-timeout", type=int, default=10, help="HTTP timeout per GDELT request (seconds).")
    parser.add_argument("--gdelt-max-retries", type=int, default=3, help="Retries per GDELT file.")
    parser.add_argument("--gdelt-retry-backoff", type=float, default=2.0, help="Backoff seconds multiplied by attempt.")
    parser.add_argument(
        "--delete-input-zips",
        action="store_true",
        help="After a pair finishes training/eval, delete its input zip files to reclaim disk.",
    )
    parser.add_argument(
        "--run-histdata-download",
        action="store_true",
        help="Download HistData zips before training using data/downloaders/histdata.py (respects pairs/years).",
    )
    parser.add_argument(
        "--histdata-output-root",
        default=None,
        help="Override output directory for HistData download (defaults to input-root).",
    )
    parser.add_argument(
        "--run-yfinance-download",
        action="store_true",
        help="Download FX data from yfinance before training (pairs and date range required).",
    )
    parser.add_argument("--yf-start", default=None, help="Start date for yfinance download (YYYY-MM-DD).")
    parser.add_argument("--yf-end", default=None, help="End date for yfinance download (YYYY-MM-DD).")
    parser.add_argument("--yf-interval", default="1m", help="yfinance interval (e.g., 1m, 5m, 1h).")
    parser.add_argument(
        "--yf-output-root",
        default=None,
        help="Override output directory for yfinance download (defaults to input-root).",
    )
    parser.add_argument(
        "--normalize-yfinance",
        action="store_true",
        help="After yfinance download, upsample to synthetic 1-minute bars.",
    )
    parser.add_argument(
        "--no-auto-download",
        dest="auto_download_missing",
        action="store_false",
        help="Disable automatic download when pair data is missing.",
    )
    parser.set_defaults(auto_download_missing=True)

    # RL Training arguments
    parser.add_argument(
        "--run-rl-training",
        action="store_true",
        help="Run RL (A3C) training after supervised training completes.",
    )
    parser.add_argument(
        "--rl-env-mode",
        type=str,
        default="simulated",
        choices=["simulated", "backtesting"],
        help="RL environment mode: 'simulated' for stochastic retail execution, 'backtesting' for deterministic historical replay",
    )
    parser.add_argument(
        "--rl-num-workers",
        type=int,
        default=4,
        help="Number of A3C workers for RL training",
    )
    parser.add_argument(
        "--rl-total-steps",
        type=int,
        default=100000,
        help="Total environment steps for RL training",
    )
    parser.add_argument(
        "--rl-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for A3C training",
    )
    parser.add_argument(
        "--rl-entropy-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient for A3C exploration",
    )
    parser.add_argument(
        "--rl-initial-balance",
        type=float,
        default=10000.0,
        help="Initial account balance for RL environment",
    )
    parser.add_argument(
        "--rl-checkpoint-dir",
        default="models/rl",
        help="Directory for RL agent checkpoints",
    )
    return parser.parse_args()


def maybe_prompt_for_more(queue: list[str]) -> list[str]:
    try:
        extra = input("Add more pairs (comma-separated) or press Enter to continue: ").strip()
    except EOFError:
        extra = ""
    if not extra:
        return queue
    for p in [part.strip().lower() for part in extra.split(",") if part.strip()]:
        if p and p not in queue:
            queue.append(p)
    return queue


def cleanup_pair_zips(pair: str, input_root: Path, years: str | None) -> None:
    root = Path(input_root)
    target_dir = root / pair
    if not target_dir.exists():
        return
    filters = None
    if years:
        filters = {y.strip() for y in years.split(",") if y.strip()}
    removed = 0
    for zp in target_dir.glob("*.zip"):
        if filters and not any(y in zp.name for y in filters):
            continue
        try:
            zp.unlink()
            removed += 1
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(f"could not delete {zp}: {exc}")
    if removed:
        log.info(f"deleted {removed} zip(s) for {pair} under {target_dir}")


def has_local_data(pair: str, input_root: Path, years: str | None) -> bool:
    pair_dir = Path(input_root) / pair
    if not pair_dir.exists():
        return False
    zips = list(pair_dir.glob("*.zip"))
    csvs = list(pair_dir.glob("*.csv"))
    if years:
        filt = {y.strip() for y in years.split(",") if y.strip()}
        zips = [z for z in zips if any(y in z.name for y in filt)]
        csvs = [c for c in csvs if any(y in c.name for y in filt)]
    return bool(zips or csvs)


def run_yfinance_download(args) -> None:
    script = ROOT / "data" / "downloaders" / "yfinance_downloader.py"
    if not script.exists():
        log.warning("yfinance downloader script not found; skipping.")
        return
    output_root = args.yf_output_root or args.input_root
    if not args.yf_start or not args.yf_end:
        log.warning("yfinance download requires --yf-start and --yf-end; skipping.")
        return
    cmd = [
        sys.executable,
        str(script),
        "--pairs",
        args.pairs,
        "--start",
        args.yf_start,
        "--end",
        args.yf_end,
        "--interval",
        args.yf_interval,
        "--output-root",
        str(output_root),
    ]
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
        # Ensure pair folders exist even if download returned empty.
        for pair in [p.strip().lower() for p in args.pairs.split(",") if p.strip()]:
            (Path(output_root) / pair).mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(f"yfinance download failed: {exc}")


def normalize_yfinance(args) -> None:
    if not args.normalize_yfinance:
        return
    script = ROOT / "data" / "downloaders" / "normalize_yfinance.py"
    if not script.exists():
        log.warning("yfinance normalization script not found; skipping.")
        return
    output_root = args.yf_output_root or args.input_root
    cmd = [
        sys.executable,
        str(script),
        "--input-root",
        str(output_root),
        "--output-root",
        str(output_root),
        "--pairs",
        args.pairs,
    ]
    try:
        subprocess.run(cmd, check=True, cwd=ROOT)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(f"yfinance normalization failed: {exc}")


def auto_download_if_missing(pair: str, args) -> None:
    if not args.auto_download_missing:
        return
    if has_local_data(pair, Path(args.input_root), args.years):
        return
    log.info(f"No local data found for {pair}; triggering downloads.")
    tmp_args = deepcopy(args)
    tmp_args.pairs = pair
    # Run histdata if requested globally.
    if args.run_histdata_download:
        run_histdata_download(tmp_args)
    # Run yfinance if requested and start/end provided.
    if args.run_yfinance_download:
        if not (args.yf_start and args.yf_end):
            log.warning("yfinance download skipped: --yf-start/--yf-end required.")
        else:
            run_yfinance_download(tmp_args)
            normalize_yfinance(tmp_args)


def run_histdata_download(args) -> None:
    script = ROOT / "data" / "download_all_fx_data.py"
    if not script.exists():
        log.warning("HistData downloader script not found; skipping.")
        return
    env = os.environ.copy()
    output_root = args.histdata_output_root or args.input_root
    env["FX_DATA_OUTPUT"] = str(output_root)
    try:
        # Respect pairs by trimming pairs.csv if pairs was provided.
        if args.pairs:
            pairs_set = {p.strip().lower() for p in args.pairs.split(",") if p.strip()}
            pairs_csv = ROOT / "pairs.csv"
            if pairs_csv.exists():
                tmp_csv = Path(env.get("TMPDIR", "/tmp")) / "pairs_filtered.csv"
                with open(pairs_csv) as src, open(tmp_csv, "w") as dst:
                    for i, line in enumerate(src):
                        if i == 0:
                            dst.write(line)
                            continue
                        parts = line.strip().split(",")
                        if len(parts) >= 2 and parts[1].lower() in pairs_set:
                            dst.write(line + "\n")
                env["PAIRS_CSV"] = str(tmp_csv)
        subprocess.run([sys.executable, str(script)], check=True, env=env, cwd=ROOT)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(f"HistData download failed: {exc}")


def maybe_offer_game(enabled: bool) -> None:
    if not enabled or not sys.stdout.isatty():
        return
    game_path = ROOT / "utils" / "play_training_game.py"
    if not game_path.exists():
        return
    try:
        resp = input(
            "Launch the mini-game during training? This may slow training. [y/N]: "
        ).strip()
    except EOFError:
        return
    if resp.lower() != "y":
        return
    log.info("Starting mini-game in this terminal. Quit with Q to return.")
    subprocess.Popen([sys.executable, str(game_path)])


def run_gdelt_download(args) -> None:
    end_date = args.gdelt_end_date or datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    cmd = [
        sys.executable,
        str(ROOT / "data" / "downloaders" / "gdelt.py"),
        "--start-date",
        args.gdelt_start_date,
        "--end-date",
        end_date,
        "--resolution",
        args.gdelt_resolution,
        "--mirror",
        args.gdelt_mirror,
        "--mirror-fallbacks",
        args.gdelt_mirror_fallbacks,
        "--step-minutes",
        str(args.gdelt_step_minutes),
        "--out-dir",
        args.gdelt_out_dir,
        "--timeout",
        str(args.gdelt_timeout),
        "--max-retries",
        str(args.gdelt_max_retries),
        "--retry-backoff",
        str(args.gdelt_retry_backoff),
    ]
    if args.gdelt_overwrite:
        cmd.append("--overwrite")
    if args.gdelt_base_url:
        cmd.extend(["--base-url", args.gdelt_base_url])
    log.info(
        f"Downloading GDELT to {args.gdelt_out_dir} "
        f"({args.gdelt_resolution}, {args.gdelt_start_date} -> {end_date})"
    )
    subprocess.run(cmd, check=True)


def load_rl_modules():
    """Lazy load RL modules to avoid import overhead when not needed."""
    global A3CAgent, A3CConfig, SimulatedRetailExecutionEnv, BacktestingRetailExecutionEnv, ExecutionConfig
    if A3CAgent is None:
        from execution.simulated_retail_env import ExecutionConfig as ExecCfg  # noqa: E402
        from execution.simulated_retail_env import SimulatedRetailExecutionEnv as SimEnv
        from rl.agents.a3c_agent import A3CAgent as A3C  # noqa: E402
        from rl.agents.a3c_agent import A3CConfig as A3CCfg
        A3CAgent = A3C
        A3CConfig = A3CCfg
        SimulatedRetailExecutionEnv = SimEnv
        ExecutionConfig = ExecCfg

        # Backtesting environment is optional
        if BacktestingRetailExecutionEnv is None:
            try:
                from execution.backtesting_env import (
                    BacktestingRetailExecutionEnv as BTEnv,  # noqa: E402
                )
                BacktestingRetailExecutionEnv = BTEnv
            except ImportError:
                pass  # Backtesting mode will fail gracefully if selected


def run_rl_training(pair: str, args, prepared_data_path: Path) -> None:
    """Run A3C RL training for a single pair."""

    load_rl_modules()

    log.info(f"\nRL: Starting RL training for {pair}")
    log.info(f"RL: Environment mode: {args.rl_env_mode}")
    log.info(f"RL: Workers: {args.rl_num_workers}")
    log.info(f"RL: Total steps: {args.rl_total_steps}")

    # Create RL checkpoint directory
    rl_ckpt_root = Path(args.rl_checkpoint_dir)
    rl_ckpt_root.mkdir(parents=True, exist_ok=True)
    rl_ckpt_path = rl_ckpt_root / f"a3c_{pair}.pt"

    # Determine device
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Create model config (reuse supervised model architecture)
    model_cfg = ModelConfig(
        num_features=20,  # Will be adjusted based on environment observations
        hidden_size_lstm=64,
        num_layers_lstm=1,
        cnn_num_filters=32,
        cnn_kernel_size=3,
        attention_dim=64,
        dropout=0.1,
    )

    # Create A3C config
    a3c_cfg = A3CConfig(
        n_workers=args.rl_num_workers,
        total_steps=args.rl_total_steps,
        rollout_length=5,
        learning_rate=args.rl_learning_rate,
        weight_decay=0.0,
        entropy_coef=args.rl_entropy_coef,
        value_loss_coef=0.5,
        gamma=0.99,
        max_grad_norm=0.5,
        checkpoint_path=str(rl_ckpt_path),
        log_interval=1000,
        device=device,
    )

    # Create environment factory based on mode
    if args.rl_env_mode == "simulated":
        # Stochastic retail simulation
        def make_env():
            exec_cfg = ExecutionConfig(initial_cash=args.rl_initial_balance)
            return SimulatedRetailExecutionEnv(
                config=exec_cfg,
                pair=pair,
                initial_balance=args.rl_initial_balance,
            )

        log.info("RL: Using SimulatedRetailExecutionEnv (stochastic)")

    else:  # backtesting mode
        if BacktestingRetailExecutionEnv is None:
            log.error("backtesting.py is required for --rl-env-mode=backtesting")
            log.error("Install with: pip install backtesting>=0.3.2")
            return

        # Load historical OHLCV data
        if not prepared_data_path.exists():
            log.error(f"Prepared data not found: {prepared_data_path}")
            log.error("Cannot run backtesting mode without historical data")
            return

        log.info(f"RL: Loading historical data from {prepared_data_path}")

        # Try NPY format first (30-50x faster), fall back to CSV
        npy_path = prepared_data_path.with_suffix('.npy')
        datetime_npy_path = prepared_data_path.parent / f"{prepared_data_path.stem}_datetime.npy"
        metadata_path = prepared_data_path.parent / f"{prepared_data_path.stem}_metadata.json"

        if npy_path.exists() and datetime_npy_path.exists() and metadata_path.exists():
            import json
            import time
            start = time.perf_counter()

            # Load metadata
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Load feature data and datetime
            features = np.load(npy_path)
            datetime_arr = np.load(datetime_npy_path)

            # Reconstruct DataFrame
            price_df = pd.DataFrame(features, columns=metadata['feature_columns'])
            price_df['datetime'] = pd.to_datetime(datetime_arr)

            elapsed = time.perf_counter() - start
            log.info(f"RL: Loaded NPY format ({len(price_df):,} rows) in {elapsed:.2f}s")
        else:
            import time
            start = time.perf_counter()
            price_df = pd.read_csv(prepared_data_path)
            elapsed = time.perf_counter() - start
            log.info(f"RL: Loaded CSV format ({len(price_df):,} rows) in {elapsed:.2f}s")

        # Ensure datetime column and required OHLCV columns
        if "datetime" in price_df.columns:
            price_df["datetime"] = pd.to_datetime(price_df["datetime"])
            price_df = price_df.set_index("datetime")

        required_cols = {"open", "high", "low", "close"}
        available_cols = {c.lower() for c in price_df.columns}
        if not required_cols.issubset(available_cols):
            missing = required_cols - available_cols
            log.error(f"Missing required OHLCV columns: {missing}")
            log.error("Cannot run backtesting mode")
            return

        log.info(f"RL: Loaded {len(price_df)} bars for backtesting")

        def make_env():
            exec_cfg = ExecutionConfig(initial_cash=args.rl_initial_balance)
            return BacktestingRetailExecutionEnv(
                price_df=price_df.copy(),
                config=exec_cfg,
            )

        log.info("RL: Using BacktestingRetailExecutionEnv (deterministic historical)")

    # Create and train agent
    try:
        log.info("RL: Initializing A3C agent...")
        agent = A3CAgent(
            model_cfg=model_cfg,
            a3c_cfg=a3c_cfg,
            action_dim=3,  # hold, buy, sell
            env_factory=make_env,
        )

        log.info("RL: Starting training...")
        agent.train()

        log.info(f"RL: Training complete! Checkpoint saved to: {rl_ckpt_path}")

    except KeyboardInterrupt:
        log.warning(f"RL training interrupted for {pair}")
    except Exception as exc:
        log.error(f"RL training failed for {pair}: {exc}")
        import traceback
        traceback.print_exc()


def main() -> None:
    args = parse_args()
    pair_queue = parse_pairs(args.pairs, args.pairs_file)
    if not pair_queue:
        raise ValueError("No pairs provided.")

    if args.run_gdelt_download:
        try:
            run_gdelt_download(args)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning(f"GDELT download failed: {exc}")
    if args.run_histdata_download:
        run_histdata_download(args)
    if args.run_yfinance_download:
        run_yfinance_download(args)
        normalize_yfinance(args)

    # Offer the optional mini-game once before training starts.
    maybe_offer_game(args.offer_game)

    # Choose device sensibly.
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    ckpt_root = Path(args.checkpoint_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    while pair_queue:
        pair = pair_queue.pop(0)
        log.info(f"\n=== Running pair: {pair} ===")

        # Auto-download if the pair folder is missing or empty.
        auto_download_if_missing(pair, args)

        class PrepArgs:
            pairs = pair
            input_root = args.input_root
            years = args.years
            t_in = args.t_in
            t_out = args.t_out
            target_type = args.task_type
            flat_threshold = args.flat_threshold
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
            intrinsic_time = args.intrinsic_time
            dc_threshold_up = args.dc_threshold_up
            dc_threshold_down = args.dc_threshold_down
            # Optional attributes with defaults
            lookahead_window = None
            top_k = 3
            predict_sell_now = False
            include_sentiment = False

        try:
            pair_name, loaders = process_pair(pair, PrepArgs)
        except Exception as exc:  # pragma: no cover - defensive logging only
            log.error(f"data prep failed for {pair}: {exc}")
            continue

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]
        num_features = next(iter(train_loader))[0].shape[-1]

        ckpt_path = ckpt_root / f"{pair_name}_best_model.pt"
        ckpt_exists = ckpt_path.exists()
        model_cfg = ModelConfig(
            num_features=num_features,
            num_classes=3 if args.task_type == "classification" else None,
        )
        train_cfg = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=str(device),
            checkpoint_path=str(ckpt_path),
        )
        train_cfg.risk.enabled = not args.disable_risk
        risk_manager = RiskManager(train_cfg.risk) if train_cfg.risk.enabled else None

        model = build_model(model_cfg, task_type=args.task_type).to(device)
        history = None

        if ckpt_exists and args.resume_if_ckpt:
            try:
                state = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state)
                log.info(f"Found checkpoint for {pair_name}, skipping training (use --force-train to retrain).")
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(f"Failed to load existing checkpoint for {pair_name}: {exc}")

        if not (ckpt_exists and args.resume_if_ckpt):
            try:
                history = train_model(
                    model,
                    train_loader,
                    val_loader,
                    train_cfg,
                    task_type=args.task_type,
                    risk_manager=risk_manager,
                )
                log.info(f"training {pair_name}; history keys: {list(history.keys())}")
            except KeyboardInterrupt:
                log.warning(f"Training interrupted for {pair_name}; keeping current model state.")
            except Exception as exc:  # pragma: no cover - defensive
                log.error(f"training failed for {pair_name}: {exc}")
                continue

        if not args.skip_eval:
            metrics = evaluate_model(
                model, test_loader, task_type=args.task_type, risk_manager=risk_manager
            )
            log.info(f"Eval: {pair_name}: {metrics}")

        # Run RL training if requested
        if args.run_rl_training:
            # Find prepared data CSV for backtesting mode
            prepared_data_path = Path(args.input_root) / pair / f"{pair}_prepared.csv"
            if not prepared_data_path.exists():
                # Try alternate locations
                alt_paths = [
                    Path("data/data") / pair / f"{pair}_prepared.csv",
                    Path("data") / pair / f"{pair}.csv",
                ]
                for alt in alt_paths:
                    if alt.exists():
                        prepared_data_path = alt
                        break

            run_rl_training(pair, args, prepared_data_path)

        if args.delete_input_zips:
            cleanup_pair_zips(pair_name, Path(args.input_root), args.years)

        if args.no_pause:
            continue
        pair_queue = maybe_prompt_for_more(pair_queue)


if __name__ == "__main__":
    main()

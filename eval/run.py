"""Evaluation run: train through ``fit``, score the test split once, write results.

python -m eval.run --config config/synthetic_hard.yaml --seed tuning
python -m eval.run --smoke --seed tuning
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from core.config import TUNING_SEEDS, DignityConfig
from data.loader import create_dataloader
from data.pipeline import TransactionPipeline
from data.source.synthetic import RAW_COLUMNS
from eval.baselines import (
    block_tree,
    majority,
    risk_score_forecast,
    threshold_policy,
    window_logistic,
)
from eval.metrics import (
    FPR_BUDGET,
    forecast_metrics,
    lead_times,
    model_gate,
    policy_metrics,
    risk_metrics,
    threshold_at_fpr,
)
from eval.plots import (
    plot_forecast_pr_auc,
    plot_lead_times,
    plot_per_type_recall,
    plot_policy_cost,
    plot_pr_curves,
)
from models.head.policy import cost_matrix, decide_actions
from train.checkpoint import build_model_from_meta, load_checkpoint
from train.engine import predict
from train.loop import FitResult, fit, prepare_data

_ROOT = Path(__file__).resolve().parent.parent
PRIVACY_TEMPLATE = _ROOT / "config" / "train_risk.yaml"
_PRINTED_METRICS = ("auroc", "pr_auc", "recall_at_1pct_fpr", "brier")


def config_hash(config: DignityConfig) -> str:
    payload = json.dumps(dataclasses.asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _train_seeds(
    config: DignityConfig,
    data,
    seed_name: str,
    init_backbones: list[str] | None = None,
) -> list[tuple[int, FitResult]]:
    """Train one model per training seed; model init is independent of the data.

    The task is part of the run directory, so tasks never share a ``best.pt``.
    ``init_backbones[k]`` warm-starts the k-th seed's backbone (policy only).
    """
    runs = []
    for k in range(config.eval.n_train_seeds):
        s = config.seed + k
        run_dir = Path(config.train.checkpoint_dir) / f"eval_{seed_name}"
        run_dir = run_dir / config.model.task / f"train_seed_{s}"
        seed_config = config
        if init_backbones is not None:
            seed_config = copy.deepcopy(config)
            seed_config.train.init_backbone_from = init_backbones[k]
        runs.append((s, fit(seed_config, data, s, str(run_dir))))
    return runs


def _task_config(config: DignityConfig, task: str) -> DignityConfig:
    """A copy of ``config`` for ``task``; the shared config is never mutated."""
    task_config = copy.deepcopy(config)
    task_config.model.task = task
    return task_config


def _predict(config: DignityConfig, result: FitResult, X: np.ndarray) -> dict:
    """Predict windows with the best (validation-selected) checkpoint."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(result.best_path, map_location=device)
    model = build_model_from_meta(result.meta).to(device)
    model.load_state_dict(ckpt["model_state"])
    loader = create_dataloader(
        X,
        None,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    return predict(model, loader, device)


def _fit_risk(config: DignityConfig, data, seed_name: str) -> tuple[list, list, list]:
    """Risk models and their val and test scores, one per training seed.

    Reported by ``run_risk`` and the baseline of every forecast, so fitted once.
    """
    risk_config = _task_config(config, "risk")
    runs = _train_seeds(risk_config, data, seed_name)
    val = [_predict(risk_config, r, data.val.X)["score"] for _, r in runs]
    test = [_predict(risk_config, r, data.test.X)["score"] for _, r in runs]
    return runs, val, test


def _payload(metrics: dict, baselines: dict, run_ctx: dict) -> dict:
    return {
        "config_hash": run_ctx["config_hash"],
        "seed": int(run_ctx["seed"]),
        "git_sha": run_ctx["git_sha"],
        "metrics": metrics,
        "baselines": baselines,
    }


def _write_result(
    config: DignityConfig, name: str, metrics: dict, baselines: dict, run_ctx: dict
) -> Path:
    """Write ``<out_dir>/<name>.json`` with the contract's five top-level keys."""
    out_dir = Path(config.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(_payload(metrics, baselines, run_ctx), f, indent=2, allow_nan=False)
    return path


def _run_ctx(config: DignityConfig) -> dict:
    return {
        "config_hash": config_hash(config),
        "seed": config.data.synthetic.seed,
        "git_sha": _git_sha(),
    }


def _fmt(entry: dict) -> str:
    return f"{entry['value']:.4f}"


def _fmt_ci(entry: dict) -> str:
    low, high = entry["ci_low"], entry["ci_high"]
    if low is None or high is None:
        return f"{_fmt(entry)} [n/a]"
    return f"{_fmt(entry)} [{low:.4f}, {high:.4f}]"


def _spread(values: list[float | None]) -> dict:
    """Mean and std over the seeds that have a value (``None`` is skipped)."""
    known = [v for v in values if v is not None]
    return {
        "values": values,
        "mean": float(np.mean(known)) if known else None,
        "std": float(np.std(known, ddof=1)) if len(known) >= 2 else None,
    }


def run_risk(
    config: DignityConfig,
    seed_name: str,
    run_ctx: dict | None = None,
    df=None,
    data=None,
    risk: tuple[list, list, list] | None = None,
) -> dict:
    """Write risk.json and charts; return payload.

    Without ``df``/``data``/``risk`` (``_fit_risk``'s result) it prepares and
    trains itself. ``main`` passes them so the risk fit is shared with forecast.
    """
    run_ctx = run_ctx or _run_ctx(config)
    seed = config.data.synthetic.seed
    n_boot = config.eval.n_bootstrap

    if data is None:
        df, data = prepare_data(config)
    train, test = data.train, data.test
    if risk is None:
        risk = _fit_risk(config, data, seed_name)
    runs, _, scores = risk
    score_0 = scores[0]  # headline model: first training seed, declared in advance

    metrics = risk_metrics(
        test.y_risk,
        score_0,
        test.block_index,
        n_boot,
        seed,
        anomaly_type=test.anomaly_type,
    )

    df_train = df[df.block_id.isin(np.unique(train.block_index))]
    df_test = df[df.block_id.isin(np.unique(test.block_index))]
    y_block, s_tree = block_tree(df_train, df_test)
    model_block = np.array(
        [score_0[test.block_index == b].max() for b in np.unique(test.block_index)]
    )
    block_ids = np.arange(len(y_block))
    metrics["block_level"] = risk_metrics(y_block, model_block, block_ids, n_boot, seed)

    point = [risk_metrics(test.y_risk, s, test.block_index, 0, seed) for s in scores]
    metrics["seed_spread"] = {
        "train_seeds": [s for s, _ in runs],
        "pr_auc": _spread([float(m["pr_auc"]["value"]) for m in point]),
        "auroc": _spread([float(m["auroc"]["value"]) for m in point]),
    }

    # the window baseline is defined on the generator's raw columns (spec 5.2)
    raw = TransactionPipeline(
        seq_len=config.data.seq_len, features=list(RAW_COLUMNS)
    ).process_blocks(
        df,
        split=config.data.split,
        horizon=config.model.horizon,
        overlap=config.data.label_overlap,
        rng=np.random.default_rng(config.data.synthetic.seed),
    )
    if not np.array_equal(raw.test.block_index, test.block_index):
        raise RuntimeError(
            "raw-column split assigned different test blocks than the model split"
        )
    if not np.array_equal(raw.test.y_risk, test.y_risk):
        raise RuntimeError(
            "raw-column split produced different test labels than the model split"
        )

    s_majority = majority(train.y_risk, len(test.y_risk))
    s_logistic = window_logistic(raw.train, raw.test)
    baselines = {
        "majority": risk_metrics(
            test.y_risk, s_majority, test.block_index, n_boot, seed
        ),
        "window_logistic": risk_metrics(
            test.y_risk, s_logistic, test.block_index, n_boot, seed
        ),
        "block_tree": risk_metrics(y_block, s_tree, block_ids, n_boot, seed),
    }

    window_gate = model_gate(
        metrics["pr_auc"],
        {
            "majority": baselines["majority"]["pr_auc"],
            "window_logistic": baselines["window_logistic"]["pr_auc"],
        },
    )
    block_gate = model_gate(
        metrics["block_level"]["pr_auc"],
        {"block_tree": baselines["block_tree"]["pr_auc"]},
    )
    metrics["model_gate"] = {
        "window": window_gate,
        "block": block_gate,
        "passed": window_gate["passed"] and block_gate["passed"],
    }

    img_dir = Path(config.eval.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    plot_pr_curves(
        {
            "model": (test.y_risk, score_0),
            "window_logistic": (test.y_risk, s_logistic),
            "majority": (test.y_risk, s_majority),
        },
        str(img_dir / "risk_pr_curve.png"),
    )
    plot_per_type_recall(metrics["per_type"], str(img_dir / "risk_per_type_recall.png"))

    path = _write_result(config, "risk", metrics, baselines, run_ctx)

    for name in _PRINTED_METRICS:
        print(
            f"{name}: model {_fmt_ci(metrics[name])} "
            f"| majority {_fmt(baselines['majority'][name])} "
            f"| window_logistic {_fmt(baselines['window_logistic'][name])}"
        )
    print(
        f"block-level pr_auc: model {_fmt(metrics['block_level']['pr_auc'])} "
        f"| block_tree {_fmt(baselines['block_tree']['pr_auc'])}"
    )
    verdict = "PASS" if metrics["model_gate"]["passed"] else "FAIL"
    print(
        f"model gate: {verdict} "
        f"(window {'pass' if window_gate['passed'] else 'fail'}, "
        f"block {'pass' if block_gate['passed'] else 'fail'})"
    )
    print(f"wrote {path}")

    return _payload(metrics, baselines, run_ctx)


def _early_warning(val, test, val_score, test_score, n_boot: int, seed: int) -> dict:
    """Per-horizon PR-AUC and lead time of ``[N, H]`` scores on the test split.

    The alarm threshold is the 1% FPR point of the last-horizon val column over
    unmasked windows; ``None`` in the result when no finite threshold fits.
    """
    last = test_score.shape[1] - 1
    m = val.forecast_mask
    y_val, s_val = val.y_forecast[m, last], val_score[m, last]
    threshold = float("inf")
    if len(np.unique(y_val)) == 2:
        threshold = threshold_at_fpr(y_val, s_val, FPR_BUDGET)
    metrics = forecast_metrics(
        test.y_forecast,
        test_score,
        test.forecast_mask,
        test.block_index,
        n_boot,
        seed,
    )
    metrics["lead_time"] = lead_times(
        test_score[:, last],
        threshold,
        test.y_forecast,
        test.forecast_mask,
        test.block_index,
        test.end_row,
    )
    metrics["threshold"] = threshold if np.isfinite(threshold) else None
    return metrics


def _mean_pr_auc(side: dict) -> float | None:
    values = [e["value"] for e in side["pr_auc_by_horizon"] if e["value"] is not None]
    return float(np.mean(values)) if values else None


def run_forecast(
    config: DignityConfig,
    seed_name: str,
    data,
    risk: tuple[list, list, list],
    run_ctx: dict | None = None,
) -> dict:
    """Fit the early-warning head per seed; compare it with today's risk score."""
    run_ctx = run_ctx or _run_ctx(config)
    seed = config.data.synthetic.seed
    n_boot = config.eval.n_bootstrap
    horizon = config.model.horizon
    forecast_config = _task_config(config, "forecast")
    runs = _train_seeds(forecast_config, data, seed_name)
    _, risk_val, risk_test = risk

    model_sides = []
    for k, (_, result) in enumerate(runs):
        boot = n_boot if k == 0 else 0  # CIs only for the headline seed
        val_score = _predict(forecast_config, result, data.val.X)["score"]
        test_score = _predict(forecast_config, result, data.test.X)["score"]
        model_sides.append(
            _early_warning(data.val, data.test, val_score, test_score, boot, seed)
        )
    # the baseline is reported for the headline (first) seed only
    baseline_side = _early_warning(
        data.val,
        data.test,
        risk_score_forecast(risk_val[0], horizon),
        risk_score_forecast(risk_test[0], horizon),
        n_boot,
        seed,
    )

    metrics = model_sides[0]  # headline model: first training seed
    metrics["seed_spread"] = {
        "train_seeds": [s for s, _ in runs],
        "mean_pr_auc": _spread([_mean_pr_auc(m) for m in model_sides]),
        "lead_time_median": _spread([m["lead_time"]["median"] for m in model_sides]),
    }
    baselines = {"risk_score_forecast": baseline_side}

    img_dir = Path(config.eval.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    result = _payload(metrics, baselines, run_ctx)
    plot_forecast_pr_auc(result, str(img_dir / "forecast_pr_auc_by_horizon.png"))
    plot_lead_times(result, str(img_dir / "forecast_lead_time.png"))

    path = _write_result(config, "forecast", metrics, baselines, run_ctx)
    base = baselines["risk_score_forecast"]
    print(
        f"forecast mean pr_auc: model {metrics['seed_spread']['mean_pr_auc']['values'][0]}"
        f" | risk_score_forecast {_mean_pr_auc(base)}"
    )
    print(
        f"lead time median: model {metrics['lead_time']['median']} "
        f"| risk_score_forecast {base['lead_time']['median']}"
    )
    print(f"wrote {path}")
    return result


def run_policy(
    config: DignityConfig,
    seed_name: str,
    data,
    risk: tuple[list, list, list],
    run_ctx: dict | None = None,
) -> dict:
    """Fit the triage head per seed; compare it with two thresholds on risk.

    Each policy seed warm-starts from the same seed's risk checkpoint. Both sides
    act under ``policy.review_budget``; the baseline's thresholds are tuned on val.
    """
    run_ctx = run_ctx or _run_ctx(config)
    seed = config.data.synthetic.seed
    n_boot = config.eval.n_bootstrap
    budget = config.policy.review_budget
    C = cost_matrix(config.policy.costs)
    test = data.test
    policy_config = _task_config(config, "policy")
    risk_runs, risk_val, risk_test = risk
    runs = _train_seeds(
        policy_config, data, seed_name, [r.best_path for _, r in risk_runs]
    )

    model_sides, baseline_sides = [], []
    for k, (_, result) in enumerate(runs):
        boot = n_boot if k == 0 else 0  # CIs only for the headline seed
        probs = _predict(policy_config, result, test.X)["probs"]
        actions = decide_actions(probs, budget)
        model_sides.append(
            policy_metrics(actions, test.y_risk, C, test.block_index, boot, seed)
        )
        baseline_actions = threshold_policy(
            risk_val[k], data.val.y_risk, risk_test[k], C, budget
        )
        baseline_sides.append(
            policy_metrics(
                baseline_actions, test.y_risk, C, test.block_index, boot, seed
            )
        )

    metrics = model_sides[0]  # headline model: first training seed
    metrics["seed_spread"] = {
        "train_seeds": [s for s, _ in runs],
        "expected_cost": _spread([m["expected_cost"]["value"] for m in model_sides]),
        "baseline_expected_cost": _spread(
            [m["expected_cost"]["value"] for m in baseline_sides]
        ),
    }
    baselines = {"threshold_policy": baseline_sides[0]}

    img_dir = Path(config.eval.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    result = _payload(metrics, baselines, run_ctx)
    plot_policy_cost(result, str(img_dir / "policy_cost.png"))

    path = _write_result(config, "policy", metrics, baselines, run_ctx)
    base = baselines["threshold_policy"]
    print(
        f"expected cost: model {_fmt_ci(metrics['expected_cost'])} "
        f"| threshold_policy {_fmt_ci(base['expected_cost'])}"
    )
    print(
        f"review rate: model {_fmt(metrics['review_rate'])} "
        f"| threshold_policy {_fmt(base['review_rate'])}"
    )
    print(f"wrote {path}")
    return result


def run_privacy_sweep(config: DignityConfig, run_ctx: dict) -> dict:
    """Sweep epsilon on the shipped privacy template; write privacy.json and the curve.

    The sweep trains its own risk models, so it needs no ``--tasks risk``. The
    generator seed is the run's (tuning or evaluation) seed; ``config_hash`` is of
    the loaded run config, so every results file of a run shares one hash.
    """
    from eval.privacy_sweep import plot_curve, sweep  # imports helpers from this module

    template = DignityConfig.from_yaml(str(PRIVACY_TEMPLATE))
    swept = dataclasses.replace(
        config,
        privacy=template.privacy,
        data=dataclasses.replace(config.data, features=template.data.features),
    )
    result = sweep(swept, config.eval.epsilons, config.data.synthetic.seed)
    path = _write_result(
        config, "privacy", result["metrics"], result["baselines"], run_ctx
    )
    img_dir = Path(config.eval.img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    plot_curve(_payload(**result, run_ctx=run_ctx), str(img_dir / "privacy_curve.png"))
    print(f"wrote {path}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Dignity evaluation.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=str, help="Path to config YAML file")
    source.add_argument(
        "--smoke", action="store_true", help="CI-sized run of config/smoke.yaml"
    )
    parser.add_argument("--seed", choices=("tuning", "eval"), required=True)
    parser.add_argument("--tasks", default="risk,forecast,policy")
    parser.add_argument(
        "--privacy-sweep",
        action="store_true",
        help="also write privacy.json and the privacy-versus-utility curve",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help=(
            "run on one tuning seed of TUNING_SEEDS (1000-1009) under its own "
            "run directory; needs --seed tuning"
        ),
    )
    args = parser.parse_args(argv)

    tasks = {t.strip() for t in args.tasks.split(",")}
    for task in sorted(tasks - {"risk", "forecast", "policy"}):
        parser.error(f"unknown task {task!r}; choose from risk, forecast, policy")
    if args.smoke and args.seed == "eval":
        parser.error("--smoke never runs on the evaluation seed")
    if args.data_seed is not None and args.seed != "tuning":
        parser.error("--data-seed needs --seed tuning")
    if args.data_seed is not None and args.data_seed not in TUNING_SEEDS:
        parser.error(
            f"--data-seed must be a tuning seed ({TUNING_SEEDS[0]}-{TUNING_SEEDS[-1]})"
        )

    config = DignityConfig.from_yaml(
        str(_ROOT / "config" / "smoke.yaml") if args.smoke else args.config
    )
    config.data.synthetic.seed = (
        args.data_seed
        if args.data_seed is not None
        else getattr(config.eval, f"{args.seed}_seed")
    )
    seed_name = args.seed if args.data_seed is None else f"tuning_{args.data_seed}"
    run_ctx = _run_ctx(config)  # once, after the seed override

    # forecast and policy baselines score with the risk models (and policy warm
    # starts from them), so risk is always fitted
    df, data = prepare_data(config)
    risk = _fit_risk(config, data, seed_name)
    if "risk" in tasks:
        run_risk(config, seed_name, run_ctx, df, data, risk)
    if "forecast" in tasks:
        run_forecast(config, seed_name, data, risk, run_ctx)
    if "policy" in tasks:
        run_policy(config, seed_name, data, risk, run_ctx)
    if args.privacy_sweep:
        run_privacy_sweep(config, run_ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

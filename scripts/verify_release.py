"""Check the v1.0 success criteria 2 to 5 against the committed results files.

Reads results/v1.0/{risk,forecast,policy,privacy}.json and prints one line per
criterion, `C<n> PASS|FAIL <detail>`. Exit 0: all pass; 1: a criterion failed
(a recorded training raise counts as a failure); 2: the results are unusable
(missing file, bad JSON, missing key, a value that is not a finite number, wrong
seed, wrong number of training seeds), in which case nothing is reported as passing.

`--seed tuning` rehearses on a run from one tuning seed (read from risk.json, every
file must carry it): it prints a banner naming the seed first and is not the
evaluation.

    python scripts/verify_release.py [--results results/v1.0] [--seed {eval,tuning}]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("risk", "forecast", "policy", "privacy")
C4_TOLERANCE = 0.05


class Criterion(NamedTuple):
    id: int
    passed: bool
    detail: str


class ResultsError(ValueError):
    """The results cannot be judged: missing file, bad JSON, missing key, wrong seed."""


def _raised(data: dict, name: str) -> dict | None:
    """The recorded training raise of a results file, or None when there is none."""
    metrics = _get(data, name, "metrics")
    if not isinstance(metrics, dict) or "raised" not in metrics:
        return None
    raised = metrics["raised"]
    if not isinstance(raised, dict) or not {"error", "train_seed"} <= raised.keys():
        raise ResultsError(
            f"{name}: metrics.raised must hold error and train_seed, found {raised!r}"
        )
    return raised


def _raised_detail(raised: dict) -> str:
    return f"training raised {raised['error']} on train seed {raised['train_seed']}"


def _load(results_dir: Path, name: str) -> dict:
    path = Path(results_dir) / f"{name}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as err:
        raise ResultsError(f"{name}: cannot read {path}: {err}") from err


def _get(data: dict, name: str, path: str):
    """Value at a dotted path. A missing key raises; a JSON null returns None."""
    node = data
    for key in path.split("."):
        if node is None:
            return None
        if not isinstance(node, dict) or key not in node:
            raise ResultsError(f"{name}: no such path {path} (missing {key})")
        node = node[key]
    return node


def _num(data: dict, name: str, path: str, where: str = ""):
    """Number at a dotted path: None for a JSON null, else a finite int or float.

    A string, a boolean, NaN or Infinity is an unusable result, never a verdict.
    """
    value = _get(data, name, path)
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise ResultsError(
            f"{name}: {where}{path} must be a finite number or null, found {value!r}"
        )
    return value


def _entries(data: dict, name: str, path: str) -> list:
    """The list at a dotted path; a null or a non-list is unusable, not a failure."""
    value = _get(data, name, path)
    if not isinstance(value, list):
        raise ResultsError(f"{name}: {path} must be a list, found {value!r}")
    return value


def _f(x, digits: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _criterion_2(risk: dict) -> Criterion:
    def g(path):
        return _num(risk, "risk", path)

    aurocs = {
        "window_logistic": g("baselines.window_logistic.auroc.value"),
        "block_tree": g("baselines.block_tree.auroc.value"),
    }

    window_baselines = [
        g("baselines.majority.pr_auc.value"),
        g("baselines.window_logistic.pr_auc.value"),
    ]
    window_best = None if None in window_baselines else max(window_baselines)
    window_low = g("metrics.pr_auc.ci_low")
    window_ok = None not in (window_low, window_best) and window_low > window_best

    tree = g("baselines.block_tree.pr_auc.value")
    block_low = g("metrics.block_level.pr_auc.ci_low")
    block_ok = None not in (block_low, tree) and block_low > tree

    detail = (
        f"baseline AUROC window_logistic {_f(aurocs['window_logistic'])}, "
        f"block_tree {_f(aurocs['block_tree'])} (reported, not judged by this "
        f"criterion); "
        f"window PR-AUC ci_low {_f(window_low)} vs best baseline {_f(window_best)}; "
        f"block PR-AUC ci_low {_f(block_low)} vs block_tree {_f(tree)}"
    )
    return Criterion(2, window_ok and block_ok, detail)


class ForecastVerdict(NamedTuple):
    passed: bool
    wins: int
    n_horizons: int
    n_missing: int
    ranking_passed: bool
    detection_rate: float | None
    random_rate: float | None
    warning_ci_low: float | None
    warning_passed: bool
    raised: dict | None


def forecast_verdict(forecast: dict, n_train_seeds: int) -> ForecastVerdict:
    """Criterion 3 as numbers: the horizons won and the warning share."""
    raised = _raised(forecast, "forecast")
    if raised is not None:
        return ForecastVerdict(False, 0, 0, 0, False, None, None, None, False, raised)
    seeds = _entries(forecast, "forecast", "metrics.seed_spread.train_seeds")
    if len(seeds) != n_train_seeds:
        raise ResultsError(
            f"forecast: metrics.seed_spread.train_seeds has {len(seeds)} seeds, "
            f"expected {n_train_seeds}"
        )
    entries = _entries(forecast, "forecast", "metrics.gain_by_horizon")
    if not entries:
        raise ResultsError("forecast: metrics.gain_by_horizon has no horizons")
    lows = [
        _num(e, "forecast", "ci_low", f"metrics.gain_by_horizon.{i}.")
        for i, e in enumerate(entries)
    ]
    wins = sum(low is not None and low > 0 for low in lows)
    ranking_passed = 2 * wins > len(lows)
    warning_low = _num(forecast, "forecast", "metrics.warning.gain.ci_low")
    warning_passed = warning_low is not None and warning_low > 0
    return ForecastVerdict(
        ranking_passed and warning_passed,
        wins,
        len(lows),
        sum(low is None for low in lows),
        ranking_passed,
        _num(forecast, "forecast", "metrics.warning.detection_rate"),
        _num(forecast, "forecast", "metrics.warning.random_rate"),
        warning_low,
        warning_passed,
        None,
    )


def _criterion_3(forecast: dict, n_train_seeds: int) -> Criterion:
    v = forecast_verdict(forecast, n_train_seeds)
    if v.raised is not None:
        return Criterion(3, False, _raised_detail(v.raised))
    median = _num(forecast, "forecast", "metrics.lead_time.median")
    detail = (
        f"{v.wins} of {v.n_horizons} horizons clear the seed-matched risk-score "
        f"baseline (needs more than half; {v.n_missing} n/a); "
        f"detection {_f(v.detection_rate)} vs random {_f(v.random_rate)} "
        f"(gain ci_low {_f(v.warning_ci_low)}); "
        f"median lead time {_f(median, 2)}"
    )
    return Criterion(3, v.passed, detail)


class PolicyVerdict(NamedTuple):
    passed: bool
    cost_passed: bool
    ratio: float | None
    per_seed_ratio: list[float | None]
    collapsed: list[int]
    over_budget: list[int]
    raised: dict | None


def _per_seed(policy: dict, field: str, n_train_seeds: int) -> list:
    """The per-seed list `metrics.seed_spread.<field>.values`, one entry per seed."""
    values = _entries(policy, "policy", f"metrics.seed_spread.{field}.values")
    if len(values) != n_train_seeds:
        raise ResultsError(
            f"policy: metrics.seed_spread.{field}.values has {len(values)} entries, "
            f"expected {n_train_seeds}"
        )
    where = f"metrics.seed_spread.{field}.values."
    return [_num({str(i): v}, "policy", str(i), where) for i, v in enumerate(values)]


def _policy_series(policy: dict, n_train_seeds: int) -> dict[str, list]:
    seeds = _entries(policy, "policy", "metrics.seed_spread.train_seeds")
    if len(seeds) != n_train_seeds:
        raise ResultsError(
            f"policy: metrics.seed_spread.train_seeds has {len(seeds)} seeds, "
            f"expected {n_train_seeds}"
        )
    fields = (
        "expected_cost",
        "baseline_expected_cost",
        "review_rate",
        "uncapped_review_share",
    )
    return {f: _per_seed(policy, f, n_train_seeds) for f in fields}


def _mean(values: list) -> float | None:
    return None if None in values else sum(values) / len(values)


def policy_verdict(
    policy: dict, review_budget: float, n_train_seeds: int, collapse_factor: float
) -> PolicyVerdict:
    """Criterion 4 as numbers: the mean-cost rule, collapsed and over-budget seeds."""
    raised = _raised(policy, "policy")
    if raised is not None:
        return PolicyVerdict(False, False, None, [], [], [], raised)
    series = _policy_series(policy, n_train_seeds)
    head, base = series["expected_cost"], series["baseline_expected_cost"]
    head_mean, base_mean = _mean(head), _mean(base)
    if base_mean is not None and base_mean <= 0:
        raise ResultsError(
            f"policy: mean baseline_expected_cost {base_mean} must be positive"
        )
    per_seed_ratio = [
        None if h is None or t is None or t <= 0 else h / t
        for h, t in zip(head, base, strict=True)
    ]
    ratio = None
    cost_passed = False
    if head_mean is not None and base_mean is not None:
        ratio = head_mean / base_mean
        limit = (1 + C4_TOLERANCE) * base_mean
        cost_passed = head_mean <= limit or math.isclose(
            head_mean, limit, rel_tol=1e-12
        )
    shares, rates = series["uncapped_review_share"], series["review_rate"]
    collapsed = [
        k
        for k, share in enumerate(shares)
        if share is not None and share > collapse_factor * review_budget
    ]
    over_budget = [
        k for k, rate in enumerate(rates) if rate is not None and rate > review_budget
    ]
    complete = None not in shares and None not in rates
    return PolicyVerdict(
        cost_passed and complete and not collapsed and not over_budget,
        cost_passed,
        ratio,
        per_seed_ratio,
        collapsed,
        over_budget,
        None,
    )


def _criterion_4(
    policy: dict, review_budget: float, n_train_seeds: int, collapse_factor: float
) -> Criterion:
    v = policy_verdict(policy, review_budget, n_train_seeds, collapse_factor)
    if v.raised is not None:
        return Criterion(4, False, _raised_detail(v.raised))
    series = _policy_series(policy, n_train_seeds)

    def listing(values):
        return "[" + ", ".join(_f(x) for x in values) + "]"

    detail = (
        f"mean cost {_f(_mean(series['expected_cost']))} vs threshold baseline "
        f"{_f(_mean(series['baseline_expected_cost']))} "
        f"(ratio {_f(v.ratio)}, limit {1 + C4_TOLERANCE:.2f}); "
        f"per-seed ratios {listing(v.per_seed_ratio)}; "
        f"uncapped review share per seed {listing(series['uncapped_review_share'])} "
        f"(limit {_f(collapse_factor * review_budget)}); "
        f"review rate per seed {listing(series['review_rate'])} "
        f"vs budget {_f(review_budget)}"
    )
    return Criterion(4, v.passed, detail)


def _criterion_5(privacy: dict, epsilons: list[float]) -> Criterion:
    points = _entries(privacy, "privacy", "metrics.points")
    got = [
        _num(p, "privacy", "epsilon", f"metrics.points.{i}.")
        for i, p in enumerate(points)
    ]
    values = [
        _num(p, "privacy", "risk.pr_auc.value", f"metrics.points.{i}.")
        for i, p in enumerate(points)
    ]
    reference = _num(privacy, "privacy", "metrics.reference.risk.pr_auc.value")
    complete = got == list(epsilons)
    scored = reference is not None and all(v is not None for v in values)
    detail = (
        f"epsilons {got} vs configured {list(epsilons)}; "
        f"every point and the reference carry a risk PR-AUC "
        f"(reference {_f(reference)}, {sum(v is None for v in values)} point(s) n/a)"
    )
    return Criterion(5, complete and scored, detail)


def verify(
    results_dir: Path,
    eval_seed: int,
    epsilons: list[float],
    review_budget: float | None = None,
    n_train_seeds: int | None = None,
    collapse_factor: float | None = None,
) -> list[Criterion]:
    """Criteria 2, 3, 4, 5 in that order; raises ResultsError if unusable."""
    if review_budget is None or collapse_factor is None:
        from core.config import PolicyConfig

        policy_cfg = PolicyConfig()
        review_budget = (
            policy_cfg.review_budget if review_budget is None else review_budget
        )
        collapse_factor = (
            policy_cfg.collapse_review_factor
            if collapse_factor is None
            else collapse_factor
        )
    if n_train_seeds is None:
        from core.config import EvalConfig

        n_train_seeds = EvalConfig().n_train_seeds
    files = {name: _load(results_dir, name) for name in NAMES}
    for name, data in files.items():
        seed = _get(data, name, "seed")
        if type(seed) is not int or seed != eval_seed:
            raise ResultsError(
                f"{name}: seed {seed!r} is not the evaluation seed {eval_seed}"
            )
    return [
        _criterion_2(files["risk"]),
        _criterion_3(files["forecast"], n_train_seeds),
        _criterion_4(files["policy"], review_budget, n_train_seeds, collapse_factor),
        _criterion_5(files["privacy"], epsilons),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "results" / "v1.0"))
    parser.add_argument("--seed", choices=("eval", "tuning"), default="eval")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from core.config import TUNING_SEEDS, EvalConfig

    cfg = EvalConfig()
    banner = None
    try:
        seed = cfg.eval_seed
        if args.seed == "tuning":
            seed = _get(_load(Path(args.results), "risk"), "risk", "seed")
            if type(seed) is not int or seed not in TUNING_SEEDS:
                raise ResultsError(
                    f"risk: seed {seed!r} is not a tuning seed "
                    f"({TUNING_SEEDS[0]}-{TUNING_SEEDS[-1]})"
                )
            banner = f"rehearsal on tuning seed {seed}: not the evaluation"
        criteria = verify(Path(args.results), seed, cfg.epsilons)
    except ResultsError as err:
        print(f"unusable results: {err}", file=sys.stderr)
        return 2
    if banner:
        print(banner)
    for c in criteria:
        print(f"C{c.id} {'PASS' if c.passed else 'FAIL'} {c.detail}")
    return 0 if all(c.passed for c in criteria) else 1


if __name__ == "__main__":
    sys.exit(main())

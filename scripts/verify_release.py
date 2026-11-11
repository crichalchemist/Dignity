"""Check the v1.0 success criteria 2 to 5 against the committed results files.

Reads results/v1.0/{risk,forecast,policy,privacy}.json and prints one line per
criterion, `C<n> PASS|FAIL <detail>`. Exit 0: all pass; 1: a criterion failed;
2: the results are unusable (missing file, bad JSON, missing key, a value that is
not a finite number, wrong seed), in which case nothing is reported as passing.

    python scripts/verify_release.py [--results results/v1.0]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
NAMES = ("risk", "forecast", "policy", "privacy")
BASELINE_AUROC_BAND = (0.55, 0.80)


class Criterion(NamedTuple):
    id: int
    passed: bool
    detail: str


class ResultsError(ValueError):
    """The results cannot be judged: missing file, bad JSON, missing key, wrong seed."""


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

    low, high = BASELINE_AUROC_BAND
    aurocs = {
        "window_logistic": g("baselines.window_logistic.auroc.value"),
        "block_tree": g("baselines.block_tree.auroc.value"),
    }
    in_band = all(v is not None and low <= v <= high for v in aurocs.values())

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
        f"block_tree {_f(aurocs['block_tree'])} (band {low:.2f}-{high:.2f}); "
        f"window PR-AUC ci_low {_f(window_low)} vs best baseline {_f(window_best)}; "
        f"block PR-AUC ci_low {_f(block_low)} vs block_tree {_f(tree)}"
    )
    return Criterion(2, in_band and window_ok and block_ok, detail)


def _criterion_3(forecast: dict) -> Criterion:
    model = _entries(forecast, "forecast", "metrics.pr_auc_by_horizon")
    base_path = "baselines.risk_score_forecast.pr_auc_by_horizon"
    base = _entries(forecast, "forecast", base_path)
    if len(model) != len(base):
        raise ResultsError(
            f"forecast: {len(model)} model horizons but {len(base)} baseline horizons"
        )
    if not model:
        raise ResultsError("forecast: metrics.pr_auc_by_horizon has no horizons")
    model_path = "metrics.pr_auc_by_horizon."
    pairs = [
        (
            _num(m, "forecast", "value", f"{model_path}{i}."),
            _num(b, "forecast", "value", f"{base_path}.{i}."),
        )
        for i, (m, b) in enumerate(zip(model, base, strict=True))
    ]
    wins = sum(m is not None and b is not None and m > b for m, b in pairs)
    missing = sum(m is None or b is None for m, b in pairs)
    median = _num(forecast, "forecast", "metrics.lead_time.median")
    detail = (
        f"beats the risk-score baseline at {wins} of {len(model)} horizons "
        f"(needs more than half; {missing} horizon(s) n/a); "
        f"median lead time {_f(median, 2)}"
    )
    return Criterion(3, 2 * wins > len(model) and median is not None, detail)


def _criterion_4(policy: dict, review_budget: float) -> Criterion:
    cost = _num(policy, "policy", "metrics.expected_cost.value")
    base = _num(policy, "policy", "baselines.threshold_policy.expected_cost.value")
    rate = _num(policy, "policy", "metrics.review_rate.value")
    cheaper = cost is not None and base is not None and cost < base
    within = rate is not None and rate <= review_budget
    detail = (
        f"expected cost {_f(cost)} vs threshold baseline {_f(base)}; "
        f"review rate {_f(rate)} vs budget {_f(review_budget)}"
    )
    return Criterion(4, cheaper and within, detail)


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
) -> list[Criterion]:
    """Criteria 2, 3, 4, 5 in that order; raises ResultsError if unusable."""
    if review_budget is None:
        from core.config import PolicyConfig

        review_budget = PolicyConfig().review_budget
    files = {name: _load(results_dir, name) for name in NAMES}
    for name, data in files.items():
        seed = _get(data, name, "seed")
        if type(seed) is not int or seed != eval_seed:
            raise ResultsError(
                f"{name}: seed {seed!r} is not the evaluation seed {eval_seed}"
            )
    return [
        _criterion_2(files["risk"]),
        _criterion_3(files["forecast"]),
        _criterion_4(files["policy"], review_budget),
        _criterion_5(files["privacy"], epsilons),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(ROOT / "results" / "v1.0"))
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from core.config import EvalConfig

    cfg = EvalConfig()
    try:
        criteria = verify(Path(args.results), cfg.eval_seed, cfg.epsilons)
    except ResultsError as err:
        print(f"unusable results: {err}", file=sys.stderr)
        return 2
    for c in criteria:
        print(f"C{c.id} {'PASS' if c.passed else 'FAIL'} {c.detail}")
    return 0 if all(c.passed for c in criteria) else 1


if __name__ == "__main__":
    sys.exit(main())

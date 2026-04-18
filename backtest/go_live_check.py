"""Live activation gate enforcement — called by scripts/go_live.sh.

Checks all three programmatic gates before live trading is permitted:
  1. Backtest report exists and all metrics pass
  2. Paper trading log spans ≥ 30 calendar days
  3. All soak gate criteria pass

Exits 0 on full pass, 1 on any failure.
"""

from __future__ import annotations

import contextlib
import glob
import json
import sys
from pathlib import Path

from backtest.paper_runner import evaluate_soak_gate
from backtest.runner import BacktestGateError, validate_backtest_results

_REPORTS_DIR = Path("reports")


def _find_latest_backtest_report(reports_dir: Path = _REPORTS_DIR) -> Path | None:
    matches = sorted(glob.glob(str(reports_dir / "backtest_report_*.json")))
    return Path(matches[-1]) if matches else None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                entries.append(json.loads(line))
    return entries


def run_checks(reports_dir: Path = _REPORTS_DIR) -> bool:
    """Run all go-live gate checks. Returns True if all pass, False otherwise.

    Prints a pass/fail summary for each check to stdout.
    """
    passed = True

    # Gate 1 — Backtest report
    report_path = _find_latest_backtest_report(reports_dir)
    if report_path is None:
        print("FAIL  backtest_report: no report found in reports/")
        passed = False
    else:
        report = json.loads(report_path.read_text())
        metrics = report.get("metrics", {})
        try:
            validate_backtest_results(metrics)
            print(f"PASS  backtest_report: {report_path.name}")
        except BacktestGateError as exc:
            print(f"FAIL  backtest_report:\n{exc}")
            passed = False

    # Gate 2 + 3 — Paper soak log
    soak_log = reports_dir / "paper_trading_log.jsonl"
    entries = _load_jsonl(soak_log)

    if not entries:
        print("FAIL  paper_trading_log: file missing or empty")
        passed = False
    else:
        backtest_gate_rate = metrics.get("gate_trigger_rate", 0.0) if "metrics" in locals() else 0.0
        soak = evaluate_soak_gate(entries, backtest_gate_rate=backtest_gate_rate)

        for criterion, ok in soak.items():
            if criterion == "all_passed":
                continue
            status = "PASS" if ok else "FAIL"
            print(f"{status}  soak.{criterion}")
            if not ok:
                passed = False

    return passed


def main() -> None:
    ok = run_checks()
    if ok:
        print("\nAll gates passed. Live trading is authorized.")
        sys.exit(0)
    else:
        print("\nGate failures above must be resolved before going live.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Dignity Core — Production Path Plan
**Date:** 2026-04-13
**Status:** Active

## Overview

Comprehensive plan to take Dignity Core from ~85% complete to production-ready live execution.
Three sequential gates control progression: backtest thresholds → 30-day paper soak → live deployment.

Architecture note from second-brain research: the risk gate is embedded **inside** the training loop
(not just at inference), eliminating the train/deploy distribution mismatch that causes RL trading
agents to fail when constraints are imposed at deployment.

---

## Section 1 — Stabilization

*Prerequisite for everything. Complete before any other section.*

### 1.1 — `capped_size` NameError (`data/source/metaapi.py:232,237`)

Live mode crashes with `NameError` because `capped_size` is never defined. Paper mode bypasses
this path, which is why it has not surfaced in testing.

**Fix:** Replace both `capped_size` references with `gate.adjusted_size`.

```python
# metaapi.py:232
result = await self._connection.create_market_buy_order(self.symbol, gate.adjusted_size)
# metaapi.py:237
result = await self._connection.create_market_sell_order(self.symbol, gate.adjusted_size)
```

### 1.2 — Dead code block (`export/to_onnx.py:45-53`)

Second `return` tuple is unreachable (first return exits at line 44) and uses stale key names
(`regime`, `alpha`, `attention`) that do not match actual output dict keys
(`regime_probs`, `alpha_score`, `attention_weights`).

**Fix:** Delete lines 45-53 entirely.

### 1.3 — Feature dimension mismatch (`core/signals.py` 33 features vs `input_size: 32`)

`regime` appears in `feature_order` but is the target label — a circular input that inflates
the feature count by 1. The ONNX export shape `(1, 100, 32)` and both YAML configs declare 32.

**Fix:** Remove `regime` from `feature_order` in `core/signals.py`. Verify output length is 32.

**Gate:** All existing tests pass with no shape errors before proceeding to Section 2.

---

## Section 2 — Signal Integrity Verification

*Ensures every feature in the 32-feature vector is correctly computed before training.*

### 2.1 — Audit DC signal computation

The four DC features (`dc_direction`, `dc_overshoot`, `bars_since_event`, `dc_bars_since_event`)
are in `feature_order` and referenced in configs. Verify in `core/signals.py`:

- `bars_since_event` when no DC event has occurred yet in a window → full window length, not 0
- `dc_overshoot` correctly measures price continuation beyond the threshold, not price change itself
- No NaN padding or stale carry-forward values across windows
- `dc_direction` is ±1, not 0 (0 would be absorbed silently by the model)

### 2.2 — Synthetic data DC coverage

`data/source/synthetic.py` must generate data that exercises DC events. If synthetic data never
triggers a directional change, the model trains DC features on near-zero values and learns to
ignore them.

**Add:** `dc_event_frequency: float = 0.1` parameter to the synthetic generator controlling the
probability of a DC event per bar. Test suite uses `dc_event_frequency=0.3` to stress-test DC
signal paths.

### 2.3 — DC signal unit tests (`tests/test_core.py`)

Add targeted tests: given a known price series with a DC event at a known bar, assert exact
expected values for all four DC features. These tests lock the signal math and catch refactoring
regressions.

**Gate:** All 32 features produce non-NaN, non-constant values on synthetic data before
proceeding to Section 3.

---

## Section 3 — Training Pipeline & Guided Learning Validation

*Verifies the cascade actually converges, with risk constraints active during training.*

### 3.1 — Risk gate in training loop (`train/engine.py`)

**Key insight from second-brain research:** Models trained without position constraints learn
aggressive strategies that get clipped at deployment, creating a train/deploy distribution
mismatch. The fix is to apply the risk gate during training.

In `train_cascade_epoch()`, after each forward pass:
- If the risk gate would block a trade (VaR exceeds `max_drawdown` threshold), suppress
  non-hold action logits by scaling them down (e.g., multiply by 0.1)
- The model then learns strategies compatible with the risk envelope from epoch 1
- Controlled by `risk_gate_training: true` in the YAML config — default true for paper/live,
  can be disabled for research runs

### 3.2 — Per-head loss logging

`train_cascade_epoch()` currently reports total loss only. Add per-head tracking:
`regime_loss`, `risk_loss`, `alpha_loss`, `policy_loss` logged separately per epoch.

Without this, a well-converging total loss can mask one head dominating while others stagnate.

### 3.3 — Convergence smoke test (`tests/test_train.py`)

Run 10 epochs on synthetic data and assert:
- (a) Total loss decreases monotonically
- (b) Regime classification accuracy > 25% (above chance on 4 classes) by epoch 10
- (c) No NaN gradients detected via `torch.isnan(p.grad).any()` check

This is a regression gate — catches gradient flow breakage from future changes.

### 3.4 — Checkpoint round-trip test

Save checkpoint → reload → assert `torch.allclose()` on identical input for all 7 output tensors.
Protects the ONNX export path which depends on correctly loaded checkpoints.

### 3.5 — Canonical training config

All development training runs use `config/train_quant_paper.yaml`. The live config
(`train_quant.yaml`) is gated behind Section 5 completion. Document in CLAUDE.md.

**Gate:** Convergence smoke test passes, checkpoint round-trip passes before proceeding to Section 4.

---

## Section 4 — Backtest Validation Harness & Gate Criteria

*First quantitative gate. Model does not proceed to paper trading until all thresholds pass.*

### 4.1 — Temporal data split

**Key insight from second-brain research:** Random splits leak future behavioral context into
training via rolling features (DC `bars_since_event`, momentum windows, volatility). Models
that look worse under temporal splitting are exactly the ones that fail in production.

Fixed monthly windows in strict temporal order:
- First N months → training
- Next M months → validation
- Final K months → held-out test (never touched until final evaluation)

All rolling features computed using only data prior to each window's timestamp. No transaction
from a later period appears in an earlier split.

For `MetaApiSource`: add `date_range: tuple[str, str]` parameter to enforce the boundary.

### 4.2 — Gate criteria (named constants in `backtest/runner.py`)

```python
BACKTEST_MIN_ARR = 0.15          # 15% annualized return
BACKTEST_MIN_SHARPE = 1.0        # Sharpe ratio
BACKTEST_MAX_DRAWDOWN = 0.20     # 20% maximum drawdown
BACKTEST_MIN_WIN_RATE = 0.52     # 52% win rate
BACKTEST_MAX_GATE_TRIGGER = 0.10 # risk gate fires on ≤10% of bars
```

Thresholds are conservative (below GEMINI.md's reported 25.28% ARR) to avoid overfitting the
gate to synthetic results.

### 4.3 — Gate enforcement

`validate_backtest_results(results: dict) -> None` raises `BacktestGateError` if any threshold
is not met. `dignity-train` calls this after backtest; exits non-zero on gate failure.
Machine-enforceable, not a manual check.

### 4.4 — Backtest report artifact

On gate pass, write `reports/backtest_report_YYYYMMDD.json` capturing:
- All metric values
- Config file hash (SHA-256)
- Model checkpoint SHA
- Training data date range

This creates an audit trail before paper trading begins.

**Gate:** All 5 backtest metrics pass, report artifact written before proceeding to Section 5.

---

## Section 5 — Paper Trading Soak Infrastructure

*Second gate. 30 calendar days of live paper trading against real market data.*

### 5.1 — Paper trading run loop (`backtest/paper_runner.py`)

Long-running async loop:
```
MetaApiSource (streaming OHLCV) → signal computation → cascade forward pass
  → risk gate → MetaApiExecutor (paper=True)
```

Distinct from backtesting: runs in real time against live market data, all orders simulated.
Risk gate runs on every bar regardless of model output.

### 5.2 — Soak log (`reports/paper_trading_log.jsonl`)

One JSON line per bar (executed or blocked):
```json
{
  "timestamp": "...", "action": "BUY|SELL|HOLD|BLOCKED",
  "regime": 2, "var_estimate": 0.031, "alpha_score": 0.14,
  "gate_passed": true, "simulated_pnl": 0.0012
}
```

### 5.3 — Soak gate criteria (evaluated after 30 days)

| Metric | Threshold |
|---|---|
| Single-day drawdown | ≤ 5% on any day |
| Risk gate trigger rate | Within ±5% of backtest rate |
| Regime distribution | No single regime > 70% of bars |
| Executor errors | Zero `RuntimeError` or `NameError` |

### 5.4 — Local process management

- `scripts/start_paper.sh` — starts run loop, writes PID to `reports/paper.pid`
- `scripts/stop_paper.sh` — reads PID, sends SIGTERM, awaits clean shutdown
- Log rotation via Python `RotatingFileHandler` (10MB max, 5 backups)

### 5.5 — Local alerting

On single-day drawdown > 3%: append to `reports/alerts.log` + macOS notification via `osascript`.
On single-day drawdown > 5%: write `reports/soak_breaker.lock`, exit process cleanly.
Lock requires manual deletion before soak can resume — not automatic.

**Gate:** 30 calendar days logged, all soak criteria pass, no lock files before proceeding to Section 6.

---

## Section 6 — Live Deployment & Ongoing Monitoring

*Third gate. Reached only after Sections 4 and 5 both pass. Manual decision point.*

### 6.1 — Live activation checklist (`scripts/go_live.sh`)

Programmatic gate enforcement before starting live execution:
1. Assert `reports/backtest_report_*.json` exists and all thresholds pass
2. Assert `reports/paper_trading_log.jsonl` spans ≥ 30 calendar days
3. Assert soak gate criteria pass (computed from log)
4. Assert `METAAPI_TOKEN` and `METAAPI_ACCOUNT_ID` env vars are set
5. Only then start live run loop with `paper=False`

No manual override path. The checklist is the gate.

### 6.2 — Position sizing hard cap (`core/execution.py`)

```python
MAX_POSITION_FRACTION = 0.02  # 2% of account balance per trade — named constant, not config
```

In live mode: `min(gate.adjusted_size, MAX_POSITION_FRACTION * account_balance)`.
Named constant deliberately, not YAML-configurable — should not be easy to change casually.

### 6.3 — Live monitoring (`scripts/monitor.sh`)

Tails `reports/live_trading_log.jsonl` (same structure as soak log, adds `realized_pnl`).
Prints rolling 7-day Sharpe and drawdown in terminal. No external dashboard required.

### 6.4 — Circuit breaker (embedded in run loop)

If 7-day realized drawdown exceeds 8%:
- Write `reports/circuit_breaker.lock`
- Exit run loop cleanly (no forced kill)
- Process does not restart until lock is manually deleted

Embed inside the run loop, not an external monitor — safety before capability.

### 6.5 — Monthly re-training cadence

1. Roll the temporal window forward one month
2. Re-train using paper config
3. Run full backtest gate — new checkpoint replaces live checkpoint only on gate pass
4. Archive old checkpoint (never delete — full audit trail)

---

## Dependency Order

```
Section 1 (bugs) → Section 2 (signals) → Section 3 (training)
  → Section 4 (backtest gate) → Section 5 (paper soak, 30 days)
    → Section 6 (live, manual decision)
```

No section begins until the previous section's gate passes. Gates are machine-enforced where
possible (test suite, `BacktestGateError`, checklist script), manual only where judgment
is required (Section 6 go/no-go).

---

## Open Items (Not In Scope)

- **Per-asset calibration / `AssetConfig`** — referenced in GEMINI.md, not implemented. Defer
  until live trading reveals asset-class-specific threshold needs.
- **CopyFactory multi-account replication** — architecture diagram only. Defer post-live.

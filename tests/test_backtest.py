"""Tests for the backtesting module."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.go_live_check import run_checks
from backtest.live_runner import (
    LIVE_ALERT_DRAWDOWN,
    LIVE_CIRCUIT_BREAKER_DRAWDOWN,
    LiveConfig,
    compute_rolling_drawdown,
)
from backtest.runner import (
    BACKTEST_MAX_DRAWDOWN,
    BACKTEST_MAX_GATE_TRIGGER,
    BACKTEST_MIN_ARR,
    BACKTEST_MIN_SHARPE,
    BACKTEST_MIN_WIN_RATE,
    BacktestConfig,
    BacktestGateError,
    _gate_trigger_rate,
    align_signals,
    compute_gate_metrics,
    prepare_ohlcv,
    run_backtest,
    validate_backtest_results,
    write_backtest_report,
)
from backtest.strategy import ACTION_BUY, ACTION_HOLD, ACTION_SELL, DignityStrategy
from data.source.metaapi import MetaApiSource, _filter_date_range

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300, trend: float = 0.001) -> pd.DataFrame:
    """Synthetic OHLCV with a gentle upward trend (lowercase columns)."""
    rng = np.random.default_rng(0)
    close = np.cumprod(1 + rng.normal(trend, 0.005, n)) * 100.0
    spread = np.abs(rng.normal(0.3, 0.05, n))
    df = pd.DataFrame(
        {
            "open": close - spread / 2,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.uniform(1000, 5000, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="1h"),
    )
    return df


def _make_signals(n: int, action: int = ACTION_BUY) -> dict[str, np.ndarray]:
    """Uniform signals — every bar the same action, near-zero VaR."""
    return {
        "action": np.full(n, action, dtype=np.float64),
        "var": np.full(n, 0.01, dtype=np.float64),
        "alpha": np.full(n, 0.5, dtype=np.float64),
        "regime": np.zeros(n, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------


class TestActionConstants:
    def test_hold_is_zero(self):
        assert ACTION_HOLD == 0

    def test_buy_is_one(self):
        assert ACTION_BUY == 1

    def test_sell_is_two(self):
        assert ACTION_SELL == 2


# ---------------------------------------------------------------------------
# prepare_ohlcv
# ---------------------------------------------------------------------------


class TestPrepareOHLCV:
    def test_renames_to_title_case(self):
        df = _make_ohlcv(50)
        result = prepare_ohlcv(df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_preserves_row_count(self):
        df = _make_ohlcv(50)
        assert len(prepare_ohlcv(df)) == 50

    def test_preserves_index(self):
        df = _make_ohlcv(50)
        pd.testing.assert_index_equal(prepare_ohlcv(df).index, df.index)

    def test_already_title_case_passes_through(self):
        df = _make_ohlcv(50)
        titled = prepare_ohlcv(df)
        result = prepare_ohlcv(titled)  # second call should be idempotent
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_raises_on_missing_close(self):
        df = _make_ohlcv(50).drop(columns=["close"])
        with pytest.raises((KeyError, ValueError)):
            prepare_ohlcv(df)


# ---------------------------------------------------------------------------
# align_signals
# ---------------------------------------------------------------------------


class TestAlignSignals:
    def test_output_length_equals_n_bars(self):
        n_bars, seq_len = 300, 100
        raw = _make_signals(n_bars - seq_len + 1)  # 201 predictions
        aligned = align_signals(raw, n_bars=n_bars, seq_len=seq_len)
        for arr in aligned.values():
            assert len(arr) == n_bars

    def test_warmup_bars_are_hold(self):
        n_bars, seq_len = 200, 100
        raw = _make_signals(n_bars - seq_len + 1, action=ACTION_BUY)
        aligned = align_signals(raw, n_bars=n_bars, seq_len=seq_len)
        # First seq_len-1 bars should be neutral HOLD
        assert np.all(aligned["action"][: seq_len - 1] == ACTION_HOLD)

    def test_warmup_var_is_zero(self):
        n_bars, seq_len = 200, 100
        raw = _make_signals(n_bars - seq_len + 1)
        aligned = align_signals(raw, n_bars=n_bars, seq_len=seq_len)
        assert np.all(aligned["var"][: seq_len - 1] == 0.0)

    def test_signal_values_preserved_after_warmup(self):
        n_bars, seq_len = 200, 100
        raw = _make_signals(n_bars - seq_len + 1, action=ACTION_SELL)
        aligned = align_signals(raw, n_bars=n_bars, seq_len=seq_len)
        assert np.all(aligned["action"][seq_len - 1 :] == ACTION_SELL)

    def test_mismatched_length_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            align_signals(_make_signals(50), n_bars=300, seq_len=100)


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def setup_method(self):
        self.n = 300
        self.seq_len = 100
        self.ohlcv = _make_ohlcv(self.n)
        self.signals = _make_signals(self.n)  # already aligned to n bars

    def test_returns_series(self):
        stats = run_backtest(self.ohlcv, self.signals)
        assert isinstance(stats, pd.Series)

    def test_standard_stats_keys_present(self):
        stats = run_backtest(self.ohlcv, self.signals)
        for key in ("Return [%]", "Sharpe Ratio", "Max. Drawdown [%]", "# Trades"):
            assert key in stats.index, f"Missing stat: {key}"

    def test_hold_signals_produce_no_trades(self):
        hold_signals = _make_signals(self.n, action=ACTION_HOLD)
        stats = run_backtest(self.ohlcv, hold_signals)
        assert stats["# Trades"] == 0

    def test_buy_signals_produce_trades(self):
        buy_signals = _make_signals(self.n, action=ACTION_BUY)
        stats = run_backtest(self.ohlcv, buy_signals)
        assert stats["# Trades"] > 0

    def test_risk_gate_blocks_all_trades_when_var_exceeds_threshold(self):
        """All var_estimates above max_drawdown → risk gate blocks everything."""
        risky_signals = _make_signals(self.n, action=ACTION_BUY)
        risky_signals["var"] = np.full(self.n, 0.99)  # always exceeds gate
        config = BacktestConfig(max_drawdown=0.05)
        stats = run_backtest(self.ohlcv, risky_signals, config=config)
        assert stats["# Trades"] == 0

    def test_custom_config_applied(self):
        config = BacktestConfig(cash=50_000.0, commission=0.001)
        stats = run_backtest(self.ohlcv, self.signals, config=config)
        assert stats["Equity Final [$]"] != 10_000.0  # different starting cash

    def test_drawdown_is_nonpositive(self):
        stats = run_backtest(self.ohlcv, self.signals)
        assert stats["Max. Drawdown [%]"] <= 0

    def test_plot_writes_html(self, tmp_path):
        plot_path = str(tmp_path / "bt.html")
        run_backtest(self.ohlcv, self.signals, plot=True, plot_path=plot_path)
        import os

        assert os.path.exists(plot_path)


# ---------------------------------------------------------------------------
# DignityStrategy (unit-level, via Backtest harness)
# ---------------------------------------------------------------------------


class TestDignityStrategy:
    def _run(self, action: int, var: float = 0.01, max_drawdown: float = 0.05) -> pd.Series:
        signals = _make_signals(150, action=action)
        signals["var"] = np.full(150, var)
        config = BacktestConfig(max_drawdown=max_drawdown)
        return run_backtest(_make_ohlcv(150), signals, config=config)

    def test_all_hold_no_trades(self):
        stats = self._run(ACTION_HOLD)
        assert stats["# Trades"] == 0

    def test_all_buy_opens_long(self):
        stats = self._run(ACTION_BUY)
        assert stats["# Trades"] >= 1

    def test_all_sell_opens_short(self):
        stats = self._run(ACTION_SELL)
        assert stats["# Trades"] >= 1

    def test_risk_gate_with_zero_var_allows_trades(self):
        stats = self._run(ACTION_BUY, var=0.0, max_drawdown=0.05)
        assert stats["# Trades"] >= 1

    def test_risk_gate_with_high_var_blocks_trades(self):
        stats = self._run(ACTION_BUY, var=0.99, max_drawdown=0.05)
        assert stats["# Trades"] == 0


# ---------------------------------------------------------------------------
# Gate constants
# ---------------------------------------------------------------------------


class TestGateConstants:
    def test_min_arr_is_15_percent(self):
        assert BACKTEST_MIN_ARR == 0.15

    def test_min_sharpe_is_1(self):
        assert BACKTEST_MIN_SHARPE == 1.0

    def test_max_drawdown_is_20_percent(self):
        assert BACKTEST_MAX_DRAWDOWN == 0.20

    def test_min_win_rate_is_52_percent(self):
        assert BACKTEST_MIN_WIN_RATE == 0.52

    def test_max_gate_trigger_is_10_percent(self):
        assert BACKTEST_MAX_GATE_TRIGGER == 0.10


# ---------------------------------------------------------------------------
# validate_backtest_results
# ---------------------------------------------------------------------------


def _passing_metrics() -> dict[str, float]:
    return {
        "arr": 0.20,
        "sharpe": 1.5,
        "max_drawdown": 0.10,
        "win_rate": 0.55,
        "gate_trigger_rate": 0.05,
    }


class TestValidateBacktestResults:
    def test_passes_when_all_metrics_meet_thresholds(self):
        validate_backtest_results(_passing_metrics())  # must not raise

    def test_raises_on_low_arr(self):
        m = {**_passing_metrics(), "arr": 0.10}
        with pytest.raises(BacktestGateError, match="ARR"):
            validate_backtest_results(m)

    def test_raises_on_low_sharpe(self):
        m = {**_passing_metrics(), "sharpe": 0.5}
        with pytest.raises(BacktestGateError, match="Sharpe"):
            validate_backtest_results(m)

    def test_raises_on_excess_drawdown(self):
        m = {**_passing_metrics(), "max_drawdown": 0.25}
        with pytest.raises(BacktestGateError, match="drawdown"):
            validate_backtest_results(m)

    def test_raises_on_low_win_rate(self):
        m = {**_passing_metrics(), "win_rate": 0.40}
        with pytest.raises(BacktestGateError, match="[Ww]in rate"):
            validate_backtest_results(m)

    def test_raises_on_excess_gate_trigger(self):
        m = {**_passing_metrics(), "gate_trigger_rate": 0.15}
        with pytest.raises(BacktestGateError, match="[Gg]ate trigger"):
            validate_backtest_results(m)

    def test_error_message_lists_all_failures_at_once(self):
        m = {
            "arr": 0.05,
            "sharpe": 0.2,
            "max_drawdown": 0.30,
            "win_rate": 0.40,
            "gate_trigger_rate": 0.20,
        }
        with pytest.raises(BacktestGateError) as exc_info:
            validate_backtest_results(m)
        msg = str(exc_info.value)
        assert "ARR" in msg
        assert "Sharpe" in msg
        assert "drawdown" in msg

    def test_backtest_gate_error_is_exception_subclass(self):
        assert issubclass(BacktestGateError, Exception)


# ---------------------------------------------------------------------------
# _gate_trigger_rate
# ---------------------------------------------------------------------------


class TestGateTriggerRate:
    def test_all_below_threshold_is_zero(self):
        signals = {"var": np.full(100, 0.01)}
        assert _gate_trigger_rate(signals, max_drawdown=0.05) == 0.0

    def test_all_above_threshold_is_one(self):
        signals = {"var": np.full(100, 0.99)}
        assert _gate_trigger_rate(signals, max_drawdown=0.05) == 1.0

    def test_half_above_is_half(self):
        var = np.concatenate([np.full(50, 0.01), np.full(50, 0.99)])
        signals = {"var": var}
        assert _gate_trigger_rate(signals, max_drawdown=0.05) == pytest.approx(0.5)

    def test_empty_signals_returns_zero(self):
        assert _gate_trigger_rate({}, max_drawdown=0.05) == 0.0


# ---------------------------------------------------------------------------
# compute_gate_metrics
# ---------------------------------------------------------------------------


class TestComputeGateMetrics:
    def _make_stats(self, **overrides) -> pd.Series:
        base = {
            "Return (Ann.) [%]": 20.0,
            "Sharpe Ratio": 1.5,
            "Max. Drawdown [%]": -10.0,
            "Win Rate [%]": 55.0,
        }
        base.update(overrides)
        return pd.Series(base)

    def test_arr_converted_from_percent(self):
        m = compute_gate_metrics(self._make_stats(), {"var": np.zeros(100)}, BacktestConfig())
        assert m["arr"] == pytest.approx(0.20)

    def test_max_drawdown_is_absolute_fraction(self):
        m = compute_gate_metrics(
            self._make_stats(**{"Max. Drawdown [%]": -15.0}),
            {"var": np.zeros(100)},
            BacktestConfig(),
        )
        assert m["max_drawdown"] == pytest.approx(0.15)

    def test_nan_sharpe_becomes_zero(self):
        m = compute_gate_metrics(
            self._make_stats(**{"Sharpe Ratio": float("nan")}),
            {"var": np.zeros(100)},
            BacktestConfig(),
        )
        assert m["sharpe"] == 0.0

    def test_nan_win_rate_becomes_zero(self):
        m = compute_gate_metrics(
            self._make_stats(**{"Win Rate [%]": float("nan")}),
            {"var": np.zeros(100)},
            BacktestConfig(),
        )
        assert m["win_rate"] == 0.0

    def test_gate_trigger_rate_computed_from_signals(self):
        var = np.concatenate([np.full(80, 0.01), np.full(20, 0.99)])
        m = compute_gate_metrics(
            self._make_stats(), {"var": var}, BacktestConfig(max_drawdown=0.05)
        )
        assert m["gate_trigger_rate"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# write_backtest_report
# ---------------------------------------------------------------------------


class TestWriteBacktestReport:
    def test_creates_file_in_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.yaml")
            ckpt = os.path.join(tmp, "model.pt")
            open(config, "w").close()
            open(ckpt, "w").close()
            path = write_backtest_report(
                _passing_metrics(),
                config,
                ckpt,
                ("2020-01-01", "2023-12-31"),
                output_dir=tmp,
            )
            assert path.exists()

    def test_report_contains_required_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.yaml")
            ckpt = os.path.join(tmp, "model.pt")
            open(config, "w").close()
            open(ckpt, "w").close()
            path = write_backtest_report(
                _passing_metrics(),
                config,
                ckpt,
                ("2020-01-01", "2023-12-31"),
                output_dir=tmp,
            )
            report = json.loads(path.read_text())
        for key in (
            "metrics",
            "config_sha256",
            "checkpoint_sha256",
            "data_date_range",
            "generated",
        ):
            assert key in report, f"report missing key: '{key}'"

    def test_metrics_preserved_in_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.yaml")
            ckpt = os.path.join(tmp, "model.pt")
            open(config, "w").close()
            open(ckpt, "w").close()
            metrics = _passing_metrics()
            path = write_backtest_report(
                metrics,
                config,
                ckpt,
                ("2020-01-01", "2023-12-31"),
                output_dir=tmp,
            )
            report = json.loads(path.read_text())
        assert report["metrics"]["arr"] == pytest.approx(metrics["arr"])

    def test_sha256_is_64_hex_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.yaml")
            ckpt = os.path.join(tmp, "model.pt")
            open(config, "w").close()
            open(ckpt, "w").close()
            path = write_backtest_report(
                _passing_metrics(),
                config,
                ckpt,
                ("2020-01-01", "2023-12-31"),
                output_dir=tmp,
            )
            report = json.loads(path.read_text())
        assert len(report["config_sha256"]) == 64
        assert len(report["checkpoint_sha256"]) == 64

    def test_data_date_range_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "config.yaml")
            ckpt = os.path.join(tmp, "model.pt")
            open(config, "w").close()
            open(ckpt, "w").close()
            path = write_backtest_report(
                _passing_metrics(),
                config,
                ckpt,
                ("2020-01-01", "2023-12-31"),
                output_dir=tmp,
            )
            report = json.loads(path.read_text())
        assert report["data_date_range"] == ["2020-01-01", "2023-12-31"]


# ---------------------------------------------------------------------------
# MetaApiSource date_range (4.1)
# ---------------------------------------------------------------------------


class TestMetaApiSourceDateRange:
    def test_date_range_stored_on_construction(self):
        src = MetaApiSource("tok", "acc", "EURUSD", date_range=("2020-01-01", "2022-12-31"))
        assert src.date_range == ("2020-01-01", "2022-12-31")

    def test_no_date_range_defaults_to_none(self):
        src = MetaApiSource("tok", "acc", "EURUSD")
        assert src.date_range is None


class TestFilterDateRange:
    def _make_df(self) -> pd.DataFrame:
        idx = pd.date_range("2020-01-01", periods=730, freq="D")
        return pd.DataFrame({"close": np.arange(730, dtype=float)}, index=idx)

    def test_none_returns_full_dataframe(self):
        df = self._make_df()
        assert len(_filter_date_range(df, None)) == len(df)

    def test_start_boundary_is_inclusive(self):
        df = self._make_df()
        result = _filter_date_range(df, ("2020-06-01", "2021-06-01"))
        assert result.index[0] == pd.Timestamp("2020-06-01")

    def test_end_boundary_is_inclusive(self):
        df = self._make_df()
        result = _filter_date_range(df, ("2020-06-01", "2021-06-01"))
        assert result.index[-1] == pd.Timestamp("2021-06-01")

    def test_range_reduces_row_count(self):
        df = self._make_df()
        result = _filter_date_range(df, ("2021-01-01", "2021-12-31"))
        assert len(result) < len(df)

    def test_range_outside_data_returns_empty(self):
        df = self._make_df()
        result = _filter_date_range(df, ("2025-01-01", "2025-12-31"))
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Section 5 — Paper soak infrastructure
# ---------------------------------------------------------------------------

from backtest.paper_runner import (
    SOAK_ALERT_DAILY_DRAWDOWN,
    SOAK_GATE_RATE_TOLERANCE,
    SOAK_MAX_DAILY_DRAWDOWN,
    SOAK_MAX_REGIME_CONCENTRATION,
    SOAK_MIN_CALENDAR_DAYS,
    append_bar_log,
    bars_to_tensor,
    check_daily_drawdown,
    evaluate_soak_gate,
    write_alert,
    write_lock,
)


def _make_log_entries(
    n_days: int = 31,
    bars_per_day: int = 24,
    action: str = "HOLD",
    gate_passed: bool = True,
    regime: int | None = None,  # None → cycle 0-3 evenly; int → force constant
    daily_pnl: float = 0.001,
    gate_rate: float = 0.0,
) -> list[dict]:
    """Synthetic soak log: n_days of bars with diverse regimes by default."""
    entries = []
    base = pd.Timestamp("2026-01-01")
    n = n_days * bars_per_day
    for i in range(n):
        ts = base + pd.Timedelta(hours=i)
        bar_blocked = i % max(1, int(1 / gate_rate)) == 0 if gate_rate > 0 else False
        bar_regime = regime if regime is not None else (i % 4)
        entries.append(
            {
                "timestamp": ts.isoformat(),
                "action": "BLOCKED" if bar_blocked else action,
                "regime": bar_regime,
                "var_estimate": 0.02,
                "alpha_score": 0.1,
                "gate_passed": not bar_blocked,
                "simulated_pnl": daily_pnl / bars_per_day,
            }
        )
    return entries


class TestSoakGateThresholds:
    def test_max_daily_drawdown_is_5_percent(self):
        assert SOAK_MAX_DAILY_DRAWDOWN == 0.05

    def test_alert_daily_drawdown_is_3_percent(self):
        assert SOAK_ALERT_DAILY_DRAWDOWN == 0.03

    def test_max_regime_concentration_is_70_percent(self):
        assert SOAK_MAX_REGIME_CONCENTRATION == 0.70

    def test_gate_rate_tolerance_is_5_percent(self):
        assert SOAK_GATE_RATE_TOLERANCE == 0.05

    def test_min_calendar_days_is_30(self):
        assert SOAK_MIN_CALENDAR_DAYS == 30


class TestCheckDailyDrawdown:
    def test_empty_entries_returns_zero(self):
        assert check_daily_drawdown([]) == 0.0

    def test_all_positive_pnl_returns_positive(self):
        entries = [{"timestamp": "2026-01-01T10:00:00", "simulated_pnl": 0.01}]
        assert check_daily_drawdown(entries) > 0

    def test_returns_worst_day(self):
        entries = [
            {"timestamp": "2026-01-01T10:00:00", "simulated_pnl": 0.05},
            {"timestamp": "2026-01-02T10:00:00", "simulated_pnl": -0.03},
            {"timestamp": "2026-01-03T10:00:00", "simulated_pnl": -0.06},
        ]
        assert check_daily_drawdown(entries) == pytest.approx(-0.06)

    def test_groups_by_calendar_day(self):
        entries = [
            {"timestamp": "2026-01-01T08:00:00", "simulated_pnl": -0.02},
            {"timestamp": "2026-01-01T16:00:00", "simulated_pnl": -0.02},
        ]
        # Both bars are on the same day — total -0.04
        assert check_daily_drawdown(entries) == pytest.approx(-0.04)


class TestEvaluateSoakGate:
    def test_passes_with_sufficient_clean_data(self):
        entries = _make_log_entries(n_days=31)
        result = evaluate_soak_gate(entries, backtest_gate_rate=0.0)
        assert result["all_passed"] is True

    def test_fails_when_fewer_than_30_days(self):
        entries = _make_log_entries(n_days=25)
        result = evaluate_soak_gate(entries, backtest_gate_rate=0.0)
        assert result["days_elapsed"] is False
        assert result["all_passed"] is False

    def test_fails_on_excess_daily_drawdown(self):
        entries = _make_log_entries(n_days=31, daily_pnl=-0.06)
        result = evaluate_soak_gate(entries, backtest_gate_rate=0.0)
        assert result["daily_drawdown"] is False

    def test_fails_on_concentrated_regime(self):
        entries = _make_log_entries(n_days=31, regime=0)  # all one regime
        result = evaluate_soak_gate(entries, backtest_gate_rate=0.0)
        assert result["regime_concentration"] is False

    def test_fails_when_gate_rate_deviates_beyond_tolerance(self):
        # backtest_gate_rate=0.0, but actual rate is ~50% (every other bar blocked)
        entries = _make_log_entries(n_days=31, gate_rate=0.5)
        result = evaluate_soak_gate(entries, backtest_gate_rate=0.0)
        assert result["gate_trigger_rate"] is False

    def test_fails_on_executor_errors(self):
        entries = _make_log_entries(n_days=31)
        entries[5]["action"] = "ERROR"
        result = evaluate_soak_gate(entries, backtest_gate_rate=0.0)
        assert result["zero_executor_errors"] is False

    def test_empty_entries_all_fail(self):
        result = evaluate_soak_gate([], backtest_gate_rate=0.0)
        assert result["all_passed"] is False

    def test_returns_all_five_criteria_plus_all_passed(self):
        entries = _make_log_entries(n_days=31)
        result = evaluate_soak_gate(entries, backtest_gate_rate=0.0)
        for key in (
            "days_elapsed",
            "daily_drawdown",
            "regime_concentration",
            "gate_trigger_rate",
            "zero_executor_errors",
            "all_passed",
        ):
            assert key in result


class TestAppendBarLog:
    def test_creates_file_if_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "soak.jsonl"
            append_bar_log(log_path, {"timestamp": "2026-01-01T00:00:00", "action": "HOLD"})
            assert log_path.exists()

    def test_each_line_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "soak.jsonl"
            for i in range(3):
                append_bar_log(log_path, {"bar": i, "action": "HOLD"})
            lines = log_path.read_text().strip().splitlines()
            assert len(lines) == 3
            for line in lines:
                json.loads(line)  # must not raise

    def test_appends_not_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "soak.jsonl"
            append_bar_log(log_path, {"n": 1})
            append_bar_log(log_path, {"n": 2})
            lines = log_path.read_text().strip().splitlines()
            assert len(lines) == 2


class TestWriteAlert:
    def test_creates_file_if_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.log"
            write_alert(path, "test alert")
            assert path.exists()

    def test_message_in_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.log"
            write_alert(path, "drawdown exceeded")
            assert "drawdown exceeded" in path.read_text()

    def test_timestamp_prefix_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.log"
            write_alert(path, "test")
            content = path.read_text()
            assert content.startswith("[20")  # ISO timestamp year prefix

    def test_appends_multiple_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alerts.log"
            write_alert(path, "first")
            write_alert(path, "second")
            assert path.read_text().count("\n") == 2


class TestWriteLock:
    def test_creates_lock_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "soak_breaker.lock"
            write_lock(lock, "test reason")
            assert lock.exists()

    def test_lock_contains_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "soak_breaker.lock"
            write_lock(lock, "daily drawdown -6.00%")
            assert "daily drawdown" in lock.read_text()

    def test_lock_contains_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "soak_breaker.lock"
            write_lock(lock, "test")
            assert "20" in lock.read_text()  # year in ISO timestamp


class TestBarsToTensor:
    def _make_bar_buffer(self, n: int = 100) -> list[pd.Series]:
        rng = np.random.default_rng(0)
        close = np.cumprod(1 + rng.normal(0, 0.005, n)) * 100.0
        spread = np.abs(rng.normal(0.1, 0.02, n))
        return [
            pd.Series(
                {
                    "open": close[i] - spread[i] / 2,
                    "high": close[i] + spread[i],
                    "low": close[i] - spread[i],
                    "close": close[i],
                    "volume": rng.uniform(1000, 5000),
                }
            )
            for i in range(n)
        ]

    def test_output_shape(self):
        import torch

        buf = self._make_bar_buffer(100)
        t = bars_to_tensor(buf, input_size=32)
        assert t.shape == (1, 100, 32)

    def test_output_dtype_is_float32(self):
        import torch

        buf = self._make_bar_buffer(100)
        t = bars_to_tensor(buf, input_size=32)
        assert t.dtype == torch.float32

    def test_no_nan_in_output(self):
        import torch

        buf = self._make_bar_buffer(100)
        t = bars_to_tensor(buf, input_size=32)
        assert not torch.isnan(t).any()


class TestSoakScripts:
    def test_start_script_exists_and_is_executable(self):
        path = Path("scripts/start_paper.sh")
        assert path.exists()
        assert os.access(path, os.X_OK)

    def test_stop_script_exists_and_is_executable(self):
        path = Path("scripts/stop_paper.sh")
        assert path.exists()
        assert os.access(path, os.X_OK)

    def test_start_script_references_pid_file(self):
        content = Path("scripts/start_paper.sh").read_text()
        assert "paper.pid" in content

    def test_stop_script_sends_sigterm(self):
        content = Path("scripts/stop_paper.sh").read_text()
        assert "SIGTERM" in content or "TERM" in content


# ---------------------------------------------------------------------------
# Section 6 — live runner and go-live gate
# ---------------------------------------------------------------------------


class TestComputeRollingDrawdown:
    def test_empty_entries_returns_zero(self):
        assert compute_rolling_drawdown([]) == 0.0

    def test_single_entry_positive(self):
        entries = [{"timestamp": "2026-01-01T12:00:00", "realized_pnl": 0.03}]
        assert compute_rolling_drawdown(entries) == pytest.approx(0.03)

    def test_single_entry_negative(self):
        entries = [{"timestamp": "2026-01-01T12:00:00", "realized_pnl": -0.05}]
        assert compute_rolling_drawdown(entries) == pytest.approx(-0.05)

    def test_sums_within_window(self):
        entries = [
            {"timestamp": "2026-01-01T00:00:00", "realized_pnl": -0.03},
            {"timestamp": "2026-01-02T00:00:00", "realized_pnl": -0.02},
            {"timestamp": "2026-01-03T00:00:00", "realized_pnl": 0.01},
        ]
        assert compute_rolling_drawdown(entries, days=7) == pytest.approx(-0.04)

    def test_excludes_entries_outside_window(self):
        # 10 days of data, window=3 — only last 3 days should count
        entries = [
            {"timestamp": f"2026-01-{i:02d}T00:00:00", "realized_pnl": -0.10} for i in range(1, 11)
        ]
        result = compute_rolling_drawdown(entries, days=3)
        assert result == pytest.approx(-0.30)

    def test_multiple_bars_same_day_aggregated(self):
        entries = [
            {"timestamp": "2026-01-01T01:00:00", "realized_pnl": -0.02},
            {"timestamp": "2026-01-01T02:00:00", "realized_pnl": -0.02},
        ]
        assert compute_rolling_drawdown(entries, days=7) == pytest.approx(-0.04)

    def test_missing_pnl_key_treated_as_zero(self):
        entries = [{"timestamp": "2026-01-01T00:00:00"}]
        assert compute_rolling_drawdown(entries) == 0.0

    def test_constants_are_sane(self):
        assert pytest.approx(0.08) == LIVE_CIRCUIT_BREAKER_DRAWDOWN
        assert pytest.approx(0.04) == LIVE_ALERT_DRAWDOWN
        assert LIVE_ALERT_DRAWDOWN < LIVE_CIRCUIT_BREAKER_DRAWDOWN


class TestLiveConfig:
    def test_default_symbol(self):
        cfg = LiveConfig(model_path="m.pt", metaapi_token="tok", account_id="acc")
        assert cfg.symbol == "EURUSD"

    def test_frozen(self):
        cfg = LiveConfig(model_path="m.pt", metaapi_token="tok", account_id="acc")
        with pytest.raises((AttributeError, TypeError)):
            cfg.symbol = "GBPUSD"  # type: ignore[misc]

    def test_custom_fields(self):
        cfg = LiveConfig(
            model_path="ckpt.pt",
            metaapi_token="tok",
            account_id="acc",
            symbol="GBPUSD",
            max_drawdown=0.03,
        )
        assert cfg.max_drawdown == pytest.approx(0.03)
        assert cfg.symbol == "GBPUSD"


class TestGoLiveCheck:
    def _write_valid_report(self, reports_dir: Path) -> None:
        metrics = {
            "arr": 0.20,
            "sharpe": 1.5,
            "max_drawdown": 0.10,
            "win_rate": 0.55,
            "gate_trigger_rate": 0.05,
        }
        report = {"metrics": metrics}
        path = reports_dir / "backtest_report_2026-01-01.json"
        path.write_text(json.dumps(report))

    def _write_valid_paper_log(self, reports_dir: Path, n_days: int = 31) -> None:
        soak_log = reports_dir / "paper_trading_log.jsonl"
        lines = []
        for i in range(n_days * 24):
            day = f"2026-01-{(i // 24) + 1:02d}"
            entry = {
                "timestamp": f"{day}T{(i % 24):02d}:00:00",
                "realized_pnl": 0.001,
                "gate_passed": True,
                "regime": i % 4,
                "error": False,
            }
            lines.append(json.dumps(entry))
        soak_log.write_text("\n".join(lines))

    def test_passes_when_all_gates_met(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_valid_report(d)
            self._write_valid_paper_log(d)
            assert run_checks(reports_dir=d) is True

    def test_fails_when_no_backtest_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_valid_paper_log(d)
            assert run_checks(reports_dir=d) is False

    def test_fails_when_no_paper_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_valid_report(d)
            assert run_checks(reports_dir=d) is False

    def test_fails_when_sharpe_too_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # Write a report with failing Sharpe
            metrics = {
                "arr": 0.20,
                "sharpe": 0.5,  # below BACKTEST_MIN_SHARPE=1.0
                "max_drawdown": 0.10,
                "win_rate": 0.55,
                "gate_trigger_rate": 0.05,
            }
            (d / "backtest_report_2026-01-01.json").write_text(json.dumps({"metrics": metrics}))
            self._write_valid_paper_log(d)
            assert run_checks(reports_dir=d) is False

    def test_fails_when_paper_log_too_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_valid_report(d)
            self._write_valid_paper_log(d, n_days=10)  # below 30-day threshold
            assert run_checks(reports_dir=d) is False


class TestLiveScripts:
    def test_go_live_script_exists_and_is_executable(self):
        path = Path("scripts/go_live.sh")
        assert path.exists()
        assert os.access(path, os.X_OK)

    def test_monitor_script_exists_and_is_executable(self):
        path = Path("scripts/monitor.sh")
        assert path.exists()
        assert os.access(path, os.X_OK)

    def test_go_live_checks_metaapi_token(self):
        content = Path("scripts/go_live.sh").read_text()
        assert "METAAPI_TOKEN" in content

    def test_go_live_checks_metaapi_account_id(self):
        content = Path("scripts/go_live.sh").read_text()
        assert "METAAPI_ACCOUNT_ID" in content

    def test_go_live_checks_checkpoint_arg(self):
        content = Path("scripts/go_live.sh").read_text()
        assert "CHECKPOINT" in content

    def test_go_live_checks_circuit_breaker_lock(self):
        content = Path("scripts/go_live.sh").read_text()
        assert "circuit_breaker.lock" in content

    def test_go_live_delegates_to_go_live_check(self):
        content = Path("scripts/go_live.sh").read_text()
        assert "go_live_check" in content

    def test_monitor_watches_live_log(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "live_trading_log" in content

    def test_monitor_computes_sharpe(self):
        content = Path("scripts/monitor.sh").read_text()
        assert "sharpe" in content.lower()

    def test_go_live_writes_pid_file(self):
        content = Path("scripts/go_live.sh").read_text()
        assert "pid" in content.lower()

"""Tests for the difficulty gate and its baselines."""

import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.config import SELECTION_SEEDS, TUNING_SEEDS, DignityConfig
from data.pipeline import TransactionPipeline, WindowSet
from data.source.synthetic import RAW_COLUMNS, SyntheticGenerator
from eval import gates
from eval.baselines import (
    _forecast_features,
    block_tree,
    majority,
    window_logistic,
    window_logistic_forecast,
)
from eval.gates import (
    DEFAULT_PRESETS,
    STRENGTH_GRID,
    data_gate,
    onset_bucket_recall,
    preset_config_hash,
)


def _windowset(**fields) -> WindowSet:
    """Build a WindowSet by keyword; only X and y_risk matter to baselines."""
    n = len(fields["y_risk"])
    defaults = {
        "y_forecast": np.zeros((n, 1), dtype=np.float32),
        "forecast_mask": np.zeros(n, dtype=bool),
        "block_index": np.arange(n, dtype=np.int64),
        "onset_offset": np.full(n, -1, dtype=np.int64),
        "end_row": np.arange(n, dtype=np.int64),
        "anomaly_type": np.full(n, "none", dtype=object),
    }
    return WindowSet(**{**defaults, **fields})


def _spiked_windows(seed: int) -> WindowSet:
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0, 1], dtype=np.int8), 100)
    X = rng.normal(size=(200, 20, 4)).astype(np.float32)
    X[y == 1, 10, 0] += 8
    return _windowset(X=X, y_risk=y)


def _blocks(seed: int, n_blocks: int = 20, rows: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_blocks * rows
    block_id = np.repeat(np.arange(n_blocks), rows)
    anomalous_block = block_id % 2 == 1
    df = pd.DataFrame({c: rng.normal(10.0, 1.0, n) for c in RAW_COLUMNS})
    df.loc[anomalous_block, "volume"] *= 10
    df["block_id"] = block_id
    df["is_anomalous"] = anomalous_block
    df["block_label"] = anomalous_block.astype(np.int8)
    return df


class TestBaselines:
    def test_majority_scores_every_window_with_train_prevalence(self):
        assert majority(np.array([0, 0, 0, 1]), 3).tolist() == [0.25, 0.25, 0.25]

    def test_window_logistic_ranks_spiked_windows_above_flat_ones(self):
        train, test = _spiked_windows(0), _spiked_windows(1)
        assert roc_auc_score(test.y_risk, window_logistic(train, test)) > 0.95

    def test_block_tree_scores_one_value_per_block(self):
        df_tr, df_te = _blocks(0), _blocks(1)
        y, s = block_tree(df_tr, df_te)
        assert len(y) == df_te["block_id"].nunique() == len(s)
        assert y.dtype == np.int8
        assert roc_auc_score(y, s) == 1.0

    def test_block_tree_is_scored_against_true_anomalies_not_noisy_labels(self):
        df_tr, df_te = _blocks(0), _blocks(1)
        y_clean, _ = block_tree(df_tr, df_te)
        df_te["block_label"] = 1 - df_te["block_label"]
        y_flipped, _ = block_tree(df_tr, df_te)
        assert y_flipped.tolist() == y_clean.tolist()
        assert y_flipped.tolist() == [b % 2 for b in range(20)]

    def test_forecast_probe_scores_every_test_window_at_every_horizon(self):
        train, test = _ramp_windows(0), _ramp_windows(1)
        scores = window_logistic_forecast(train, test)
        assert scores.shape == (len(test.y_risk), 3)
        assert ((scores > 0) & (scores < 1)).all()

    def test_forecast_probe_ranks_windows_that_end_in_a_ramp_above_flat_ones(self):
        train, test = _ramp_windows(0), _ramp_windows(1)
        scores = window_logistic_forecast(train, test)
        assert roc_auc_score(test.y_forecast[:, 2], scores[:, 2]) > 0.9

    def test_forecast_probe_learns_only_from_windows_the_forecast_mask_keeps(self):
        clean, poisoned = _ramp_windows(0), _ramp_windows(2, ramp_on_positives=False)
        train = _windowset(
            X=np.concatenate([clean.X, poisoned.X]),
            y_risk=np.concatenate([clean.y_risk, poisoned.y_risk]),
            y_forecast=np.concatenate([clean.y_forecast, poisoned.y_forecast]),
            forecast_mask=np.concatenate(
                [clean.forecast_mask, np.zeros(len(poisoned.y_risk), dtype=bool)]
            ),
        )
        test = _ramp_windows(1)
        scores = window_logistic_forecast(train, test)
        assert roc_auc_score(test.y_forecast[:, 2], scores[:, 2]) > 0.9

    def test_forecast_probe_refuses_a_horizon_with_one_class_in_its_masked_windows(
        self,
    ):
        train = _ramp_windows(0)
        train.y_forecast[:, 1] = 0.0
        with pytest.raises(ValueError, match="horizon 1: one class"):
            window_logistic_forecast(train, _ramp_windows(1))

    def test_forecast_probe_reads_the_slow_and_fast_slopes_the_design_measured(self):
        x = np.random.default_rng(0).normal(size=(4, 40, 2))
        slow, fast = np.split(_forecast_features(x)[:, -4:], 2, axis=1)
        assert slow == pytest.approx(
            x[:, -10:].mean(axis=1) - x[:, -30:-10].mean(axis=1)
        )
        assert fast == pytest.approx(x[:, -5:].mean(axis=1) - x[:, -15:-5].mean(axis=1))

    def test_forecast_features_are_eight_statistics_per_input_column(self):
        for n_features in (1, 3, 7):
            x = np.random.default_rng(0).normal(size=(5, 40, n_features))
            assert _forecast_features(x).shape == (5, 8 * n_features)

    def test_the_plain_probe_is_one_logistic_model_per_horizon_on_the_kept_windows(
        self,
    ):
        train, test = _ramp_windows(0), _ramp_windows(1)
        keep = train.forecast_mask
        expected = np.empty((len(test.X), 3))
        for h in range(3):
            model = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)
            )
            model.fit(_forecast_features(train.X[keep]), train.y_forecast[keep, h])
            expected[:, h] = model.predict_proba(_forecast_features(test.X))[:, 1]
        assert np.array_equal(window_logistic_forecast(train, test), expected)

    def test_risk_aware_probe_uses_the_risk_label_as_one_more_input(self):
        def noise_windows(seed: int):
            rng = np.random.default_rng(seed)
            y = (rng.random(400) < 0.5).astype(np.int8)
            return _windowset(
                X=rng.normal(size=(400, 40, 3)).astype(np.float32),
                y_risk=y,
                y_forecast=np.repeat(y[:, None], 3, axis=1).astype(np.float32),
                forecast_mask=np.ones(400, dtype=bool),
            )

        train, test = noise_windows(0), noise_windows(1)
        plain = window_logistic_forecast(train, test)
        aware = window_logistic_forecast(train, test, risk_aware=True)
        assert roc_auc_score(test.y_forecast[:, 2], plain[:, 2]) < 0.65
        assert roc_auc_score(test.y_forecast[:, 2], aware[:, 2]) > 0.9
        assert np.array_equal(
            plain, window_logistic_forecast(train, test, risk_aware=False)
        )


def _ramp_windows(seed: int, ramp_on_positives: bool = True, n: int = 300):
    """[n, 40, 3] noise, 3 horizons; the marked half ends in a +4 ramp on feature 0.

    The ramp spans the last 10 rows, so max, min and std alone do not give it away.
    `ramp_on_positives=False` puts the ramp on the negatives (the inverted relation).
    """
    rng = np.random.default_rng(seed)
    y = np.tile(np.array([0, 1], dtype=np.int8), n // 2)
    X = rng.normal(size=(n, 40, 3)).astype(np.float32)
    ramped = (y == 1) if ramp_on_positives else (y == 0)
    X[ramped, -10:, 0] += 4 * np.linspace(0, 1, 10, dtype=np.float32)
    y_forecast = np.repeat(y[:, None], 3, axis=1).astype(np.float32)
    return _windowset(
        X=X,
        y_risk=y,
        y_forecast=y_forecast,
        forecast_mask=np.ones(n, dtype=bool),
    )


def _positional_windows(extra_onset_zero: int = 0):
    """50 positives per onset bucket (offsets 1..99), 501 negatives, seq_len 100.

    The negatives are 500 plus one spare, so a test can overwrite the last score
    (the G3 negative) without disturbing the 500 that set the 90th percentile.
    """
    offsets = np.array([b * 20 + 1 + (i % 19) for b in range(5) for i in range(50)])
    n_pos = len(offsets) + extra_onset_zero
    onset = np.concatenate(
        [offsets, np.zeros(extra_onset_zero, dtype=np.int64), np.full(501, -1)]
    ).astype(np.int64)
    y = np.concatenate([np.ones(n_pos), np.zeros(501)]).astype(np.int8)
    ws = _windowset(
        X=np.zeros((len(y), 100, 2), dtype=np.float32), y_risk=y, onset_offset=onset
    )
    return ws, offsets


ROOT = Path(__file__).resolve().parents[1]
_FLAT = [0.46, 0.47, 0.50, 0.52, 0.55]
# (val, test) AUROC per preset: easy > medium > hard, hard inside AUROC_BAND
_WL = {"easy": (0.81, 0.81), "medium": (0.76, 0.76), "hard": (0.73, 0.73)}
_BT = {"easy": (0.74, 0.74), "medium": (0.65, 0.65), "hard": (0.60, 0.60)}


def _forbid_generation(monkeypatch) -> list:
    """Any scoring or generation fails the test; the spy records generator calls."""
    calls: list = []

    def spy(config, rng=None):
        calls.append(config.seed)
        pytest.fail("data was generated for a refused request")

    monkeypatch.setattr(gates, "SyntheticGenerator", spy)
    monkeypatch.setattr(
        gates, "_score_preset", lambda *a, **k: pytest.fail("a preset was scored")
    )
    return calls


def _good_seeds(n: int) -> list[dict]:
    """Per-seed headroom results that satisfy every part of the headroom check."""
    per_seed = []
    for _ in range(n):
        per_seed.append(
            {
                "lift_h10": 6.0,
                "pr_auc": {
                    "probe": {"h1": 0.08, "h5": 0.08, "h10": 0.08},
                    "probe_risk_aware": {"h1": 0.30, "h5": 0.30, "h10": 0.30},
                    "stand_in": {"h1": 0.10, "h5": 0.10, "h10": 0.10},
                    "oracle": {"h1": 0.12, "h5": 0.12, "h10": 0.12},
                },
                "ppv_build_up": 0.45,
                "n_onsets": 100,
                "n_ramped": 80,
                "n_shape_warned": 60,
                "arrays": {
                    "onset_block": np.repeat(np.arange(50), 2),
                    "warned_probe": np.arange(100) % 5 != 0,
                    "warned_random": np.arange(100) < 5,
                    "block_index": np.repeat(np.arange(50), 4),
                },
            }
        )
    return per_seed


def _fake_score_preset(
    monkeypatch, *, wl=None, bt=None, recall=None, onsets=250, seen=None, extra=None
):
    """Replace gates._score_preset (the boundary) with fixed numbers.

    Each value is either a constant (val, test) pair or a function of the seed.
    """
    wl, bt = wl or _WL, bt or _BT
    recall = recall or (_FLAT, _FLAT)
    by_path = {path: name for name, path in DEFAULT_PRESETS.items()}
    by_path.update(extra or {})  # preset files copied by a test

    def pick(value, seed):
        return value(seed) if callable(value) else value

    def score(path, seed, n_blocks=None, strength=None):
        preset = by_path[path]
        if seen is not None:
            seen.append((preset, seed, n_blocks, strength))
        hard = preset == "hard"
        wl_val, wl_test = pick(wl[preset], seed)
        bt_val, bt_test = pick(bt[preset], seed)
        recall_val, recall_test = pick(recall, seed)
        return {
            "auroc": {
                "window_logistic": {"val": wl_val, "test": wl_test},
                "block_tree": {"val": bt_val, "test": bt_test},
            },
            "onset_recall": (
                {"val": list(recall_val), "test": list(recall_test)} if hard else None
            ),
            "test_onsets": int(pick(onsets, seed)) if hard else None,
            "headroom": _good_seeds(1)[0] if hard else None,
        }

    monkeypatch.setattr(gates, "_score_preset", score)


class TestDataGate:
    def test_flatness_check_catches_a_positional_shortcut(self):
        ws, offsets = _positional_windows()
        scores = np.zeros(len(ws.y_risk))
        scores[: len(offsets)] = (offsets >= 80).astype(float)
        # The 90th percentile of the negatives is 0.0, so this 0.5 negative sits
        # above the threshold: it is the one false positive, 1 of 501 (< 10% FPR).
        scores[-1] = 0.5
        assert onset_bucket_recall(scores, ws) == [0.0, 0.0, 0.0, 0.0, 1.0]

    def test_flatness_ignores_windows_that_open_mid_anomaly(self):
        ws, offsets = _positional_windows(extra_onset_zero=40)
        scores = np.zeros(len(ws.y_risk))
        scores[: len(offsets)] = (offsets >= 80).astype(float)
        scores[len(offsets) : len(offsets) + 40] = 1.0
        assert onset_bucket_recall(scores, ws) == [0.0, 0.0, 0.0, 0.0, 1.0]

    def test_flatness_refuses_an_empty_bucket(self):
        ws, _ = _positional_windows()
        ws.y_risk[ws.onset_offset >= 80] = 0
        with pytest.raises(ValueError, match="onset bucket 4 has no positive windows"):
            onset_bucket_recall(np.zeros(len(ws.y_risk)), ws)

    def test_gate_refuses_every_seed_outside_the_tuning_range(
        self, monkeypatch, capsys
    ):
        _forbid_generation(monkeypatch)
        for s in (999, 1010, 2026, 2027):
            with pytest.raises(ValueError):
                data_gate(DEFAULT_PRESETS, [s])
            assert gates.main(["--seeds", str(s)]) == 2, s
        assert "2027" in capsys.readouterr().out

    def test_gate_refuses_an_empty_seed_list_before_any_generation(self, monkeypatch):
        _forbid_generation(monkeypatch)
        with pytest.raises(ValueError):
            data_gate(DEFAULT_PRESETS, [])
        assert gates.main(["--seeds", ""]) == 2

    def test_gate_refuses_duplicate_seeds_before_any_generation(self, monkeypatch):
        # One seed counted twice would inflate the seed means and the win counts.
        calls = _forbid_generation(monkeypatch)
        with pytest.raises(ValueError, match="duplicate seed 1000"):
            data_gate(DEFAULT_PRESETS, [1000, 1000, 1001])
        assert gates.main(["--seeds", "1000,1000,1001"]) == 2
        assert calls == []

    def test_gate_accepts_a_permutation_of_distinct_seeds(self, monkeypatch):
        seen: list = []
        _fake_score_preset(monkeypatch, seen=seen)
        data_gate(DEFAULT_PRESETS, [1003, 1001])
        assert sorted({s for _, s, _, _ in seen}) == [1001, 1003]

    def test_gate_refuses_a_strength_outside_the_grid(self, monkeypatch):
        calls = _forbid_generation(monkeypatch)
        with pytest.raises(ValueError, match="strength"):
            data_gate(DEFAULT_PRESETS, [1000], strength=0.5)
        assert gates.main(["--strength", "0.5"]) == 2
        assert calls == []

    def test_gate_seed_reaches_the_generator(self, monkeypatch):
        seen = []

        class _StopError(Exception): ...

        def spy(config, rng=None):
            seen.append(config.seed)
            raise _StopError

        monkeypatch.setattr(gates, "SyntheticGenerator", spy)
        with pytest.raises(_StopError):
            data_gate(DEFAULT_PRESETS, [1003], n_blocks=10)
        assert seen == [1003]  # preset files say 42; the gate seed must win

    def test_gate_means_not_single_seeds_decide_the_band(self, monkeypatch):
        def hard_wl(seed):
            return (0.40, 0.70) if seed == 1000 else (0.70, 0.70)

        _fake_score_preset(monkeypatch, wl={**_WL, "hard": hard_wl})
        result = data_gate(DEFAULT_PRESETS, [1000, 1001, 1002, 1003])
        assert result["auroc"]["hard"]["window_logistic"]["pooled_mean"] == (
            pytest.approx((7 * 0.70 + 0.40) / 8)
        )
        assert result["checks"]["window_logistic_in_band"] is True

        _fake_score_preset(monkeypatch, wl={**_WL, "hard": (0.50, 0.50)})
        low = data_gate(DEFAULT_PRESETS, [1000, 1001, 1002, 1003])
        assert low["checks"]["window_logistic_in_band"] is False

    def test_flatness_is_the_spread_of_seed_means_not_the_mean_of_spreads(
        self, monkeypatch
    ):
        rising = [0.4, 0.45, 0.5, 0.55, 0.6]
        _fake_score_preset(
            monkeypatch,
            recall=lambda seed: (rising if seed == 1000 else rising[::-1], _FLAT),
        )
        mixed = data_gate(DEFAULT_PRESETS, [1000, 1001])
        assert mixed["checks"]["positional_flatness"] is True
        assert mixed["onset_recall"]["val_spread"] == pytest.approx(0.0)

        _fake_score_preset(monkeypatch, recall=(rising, _FLAT))
        same = data_gate(DEFAULT_PRESETS, [1000, 1001])
        assert same["checks"]["positional_flatness"] is False
        assert same["onset_recall"]["val_spread"] == pytest.approx(0.2)

    def test_flatness_limit_is_015_on_the_val_split(self, monkeypatch):
        def gate_with(top: float) -> dict:
            val = [0.45, 0.5, 0.5, 0.5, top]
            _fake_score_preset(monkeypatch, recall=(val, _FLAT))
            return data_gate(DEFAULT_PRESETS, [1000])

        at_limit = gate_with(0.6)
        assert at_limit["onset_recall"]["val_spread"] == pytest.approx(0.15)
        assert at_limit["checks"]["positional_flatness"] is True
        over = gate_with(0.601)
        assert over["onset_recall"]["val_spread"] == pytest.approx(0.151)
        assert over["checks"]["positional_flatness"] is False

    def test_test_split_flatness_is_reported_but_never_decides(self, monkeypatch):
        _fake_score_preset(monkeypatch, recall=(_FLAT, [0.4, 0.5, 0.5, 0.6, 0.7]))
        result = data_gate(DEFAULT_PRESETS, [1000])
        assert result["onset_recall"]["test_spread"] == pytest.approx(0.3)
        assert result["checks"]["positional_flatness"] is True

    def test_gate_fails_when_a_seed_has_too_few_scorable_test_onsets(self, monkeypatch):
        _fake_score_preset(
            monkeypatch, onsets=lambda seed: 250 if seed == 1000 else 199
        )
        result = data_gate(DEFAULT_PRESETS, [1000, 1001])
        failing = [k for k, v in result["checks"].items() if not v]
        assert failing == ["test_onsets"]
        assert result["passed"] is False
        assert result["test_onsets"] == [250, 199]

    def test_exactly_two_hundred_scorable_test_onsets_pass_and_199_fail(
        self, monkeypatch
    ):
        _fake_score_preset(monkeypatch, onsets=lambda seed: 200)
        assert data_gate(DEFAULT_PRESETS, [1000, 1001])["checks"]["test_onsets"] is True
        _fake_score_preset(
            monkeypatch, onsets=lambda seed: 200 if seed == 1000 else 199
        )
        assert (
            data_gate(DEFAULT_PRESETS, [1000, 1001])["checks"]["test_onsets"] is False
        )

    def test_gate_passes_only_when_every_check_holds(self, monkeypatch):
        _fake_score_preset(monkeypatch)
        result = data_gate(DEFAULT_PRESETS, [1000, 1001])
        assert result["passed"] is True
        assert set(result["checks"]) == {
            "window_logistic_in_band",
            "block_tree_in_band",
            "monotone",
            "positional_flatness",
            "forecast_headroom",
            "test_onsets",
        }
        assert all(result["checks"].values())

    def test_block_tree_band_failure_fails_exactly_that_check(self, monkeypatch):
        _fake_score_preset(monkeypatch, bt={**_BT, "hard": (0.50, 0.50)})
        result = data_gate(DEFAULT_PRESETS, [1000])
        failing = [k for k, v in result["checks"].items() if not v]
        assert failing == ["block_tree_in_band"]

    def test_monotone_needs_pooled_means_and_the_strength_order(self, monkeypatch):
        _fake_score_preset(monkeypatch, wl={**_WL, "medium": (0.82, 0.82)})
        result = data_gate(DEFAULT_PRESETS, [1000, 1001])
        failing = [k for k, v in result["checks"].items() if not v]
        assert failing == ["monotone"]

        seen: list = []
        _fake_score_preset(monkeypatch, seen=seen)
        data_gate(DEFAULT_PRESETS, [1000], strength=0.6)
        strengths = {preset: s for preset, _, _, s in seen}
        assert (strengths["easy"], strengths["medium"], strengths["hard"]) == (
            0.8,
            0.7,
            0.6,
        )

    def test_the_top_grid_strength_keeps_easy_at_the_cap_and_medium_between(
        self, monkeypatch
    ):
        # 0.9 + 0.2 would exceed the generator's cap of 1.0: easy stays at the
        # cap and medium sits halfway, so the three strengths still fall in order.
        seen: list = []
        _fake_score_preset(monkeypatch, seen=seen)
        result = data_gate(DEFAULT_PRESETS, [1000], strength=0.9)
        strengths = {preset: s for preset, _, _, s in seen}
        assert (strengths["easy"], strengths["medium"], strengths["hard"]) == (
            1.0,
            0.95,
            0.9,
        )
        assert result["checks"]["monotone"] is True
        assert max(gates.STRENGTH_GRID) == 0.9

    def test_monotone_fails_when_the_presets_do_not_order_their_strengths(
        self, monkeypatch, tmp_path
    ):
        flat = tmp_path / "medium.yaml"
        text = (ROOT / "config" / "synthetic_medium.yaml").read_text()
        lowered, n = re.subn(
            r"precursor_strength: [0-9.]+", "precursor_strength: 0.5", text
        )
        assert n == 1
        flat.write_text(lowered)
        _fake_score_preset(monkeypatch, extra={str(flat): "medium"})
        result = data_gate({**DEFAULT_PRESETS, "medium": str(flat)}, [1000])
        failing = [k for k, v in result["checks"].items() if not v]
        assert failing == ["monotone"]

    def test_gate_json_has_the_documented_keys_and_pooled_means(self, monkeypatch):
        _fake_score_preset(
            monkeypatch,
            bt={**_BT, "hard": lambda seed: (0.55 + (seed - 1000) / 100, 0.62)},
        )
        result = data_gate(DEFAULT_PRESETS, [1000, 1001, 1002])
        assert set(result) == {
            "seeds",
            "n_blocks",
            "strength_override",
            "preset_config_hash",
            "auroc",
            "onset_recall",
            "test_onsets",
            "headroom",
            "checks",
            "passed",
        }
        assert result["seeds"] == [1000, 1001, 1002]
        assert result["n_blocks"] == 3600
        assert result["strength_override"] is None
        block_tree = result["auroc"]["hard"]["block_tree"]
        assert block_tree["val"] == pytest.approx([0.55, 0.56, 0.57])
        assert block_tree["test"] == pytest.approx([0.62] * 3)
        assert block_tree["pooled_mean"] == pytest.approx(
            np.mean([0.55, 0.56, 0.57, 0.62, 0.62, 0.62])
        )
        assert "check" not in result["headroom"]
        assert len(result["headroom"]["per_seed"]) == 3
        assert all("arrays" not in entry for entry in result["headroom"]["per_seed"])
        assert json.loads(json.dumps(result, allow_nan=False)) == result

    def test_preset_hash_follows_the_data_block_only(self, tmp_path):
        text = (ROOT / "config" / "synthetic_hard.yaml").read_text()

        def hashed(old: str, new: str) -> str:
            assert old in text
            path = tmp_path / f"hard_{abs(hash((old, new)))}.yaml"
            path.write_text(text.replace(old, new))
            return gates.preset_config_hash({**DEFAULT_PRESETS, "hard": str(path)})

        base = gates.preset_config_hash(DEFAULT_PRESETS)
        assert len(base) == 16
        assert hashed("decoy_rate: 0.2", "decoy_rate: 0.3") != base
        assert hashed("lr: 0.0003", "lr: 0.0005") == base

    def test_small_structural_run_scores_all_three_presets(self):
        started = time.monotonic()
        result = data_gate(DEFAULT_PRESETS, [1000], n_blocks=300)
        for preset in ("easy", "medium", "hard"):
            for baseline in ("window_logistic", "block_tree"):
                scores = result["auroc"][preset][baseline]
                for value in (*scores["val"], *scores["test"], scores["pooled_mean"]):
                    assert np.isfinite(value) and 0.0 <= value <= 1.0
        assert result["n_blocks"] == 300
        assert len(result["onset_recall"]["val"]) == 5
        assert len(result["onset_recall"]["test"]) == 5
        assert isinstance(result["test_onsets"][0], int)
        assert "arrays" not in json.dumps(result, allow_nan=False)
        assert time.monotonic() - started < 300

    def test_gate_cli_writes_the_json_and_exits_by_passed(self, monkeypatch, tmp_path):
        calls: list = []
        outcome = {"seeds": [1000], "checks": {"x": True}, "passed": True}

        def fake_gate(config_paths, seeds, n_blocks=None, strength=None):
            calls.append((config_paths, seeds, n_blocks, strength))
            return dict(outcome)

        monkeypatch.setattr(gates, "data_gate", fake_gate)
        out = tmp_path / "gate.json"
        assert gates.main(["--out", str(out)]) == 0
        assert json.loads(out.read_text())["passed"] is True
        assert calls == [(DEFAULT_PRESETS, list(TUNING_SEEDS), None, None)]

        outcome["passed"] = False
        assert (
            gates.main(
                ["--seeds", "1001,1000", "--n-blocks", "50", "--strength", "0.7"]
            )
            == 1
        )
        assert calls[-1] == (DEFAULT_PRESETS, [1001, 1000], 50, 0.7)


def _summary(per_seed: list[dict]) -> dict:
    return gates.headroom_summary(per_seed, list(range(1000, 1000 + len(per_seed))))


def _with(per_seed: list[dict], index: int, **fields) -> list[dict]:
    per_seed[index].update(fields)
    return per_seed


def _hard_copy(tmp_path: Path, **replacements: str) -> str:
    """A copy of the hard preset (the gate recognises it by file name)."""
    text = (ROOT / "config" / "synthetic_hard.yaml").read_text()
    for old, new in replacements.items():
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "synthetic_hard.yaml"
    path.write_text(text)
    return str(path)


class TestForecastHeadroom:
    def test_stratified_lift_is_scored_among_windows_with_no_earlier_anomaly(self):
        ws = _windowset(
            X=np.zeros((6, 2, 1), dtype=np.float32),
            y_risk=np.array([0, 1, 0, 0, 1, 0], dtype=np.int8),
            y_forecast=np.arange(6, dtype=np.float32)[:, None],
            forecast_mask=np.array([1, 1, 0, 1, 1, 1], dtype=bool),
        )
        kept = gates._stratum(ws)
        assert kept.y_forecast[:, 0].tolist() == [0.0, 3.0, 5.0]
        assert kept.forecast_mask.all()

    def test_shape_only_score_rises_along_a_build_up(self):
        got = gates._build_up_position(np.array([0, 0, 1, 1, 1, 1, 0, 2, 2, 0]))
        assert got.tolist() == [0, 0, 0.25, 0.5, 0.75, 1, 0, 0.5, 1, 0]

    def test_headroom_needs_the_mean_lift_in_range_and_every_seed_above_the_floor(
        self,
    ):
        def lifts(values):
            per_seed = _good_seeds(len(values))
            for entry, value in zip(per_seed, values, strict=True):
                entry["lift_h10"] = value
            return _summary(per_seed)["check"]

        assert lifts([6.0] * 10) is True
        assert lifts([2.9] * 10) is False
        assert lifts([15.1] * 10) is False
        assert lifts([1.4] + [(60 - 1.4) / 9] * 9) is False
        assert lifts([1.5] + [(60 - 1.5) / 9] * 9) is True
        assert lifts([None] + [6.0] * 9) is False

    def test_the_ratio_is_against_the_stronger_of_the_stand_in_and_the_oracle(self):
        per_seed = _good_seeds(10)
        for entry in per_seed:
            entry["pr_auc"]["probe_risk_aware"] = {"h1": 0.15, "h5": 0.15, "h10": 0.15}
            entry["pr_auc"]["stand_in"] = {"h1": 0.05, "h5": 0.05, "h10": 0.05}
            entry["pr_auc"]["oracle"] = {"h1": 0.11, "h5": 0.11, "h10": 0.11}
        summary = _summary(per_seed)
        assert summary["pr_auc_ratio"]["h10"] == pytest.approx(0.15 / 0.11)
        assert summary["pr_auc_ratio"]["h5"] == pytest.approx(1.36, abs=0.01)
        assert summary["check"] is False
        # against the stand-in alone the probe would clear 1.5 (0.15 / 0.05 = 3)
        for entry in per_seed:
            entry["pr_auc"]["oracle"] = {"h1": 0.05, "h5": 0.05, "h10": 0.05}
        assert _summary(per_seed)["check"] is True

    def test_the_ratio_and_the_wins_use_the_risk_aware_probe(self):
        per_seed = _good_seeds(10)
        for entry in per_seed:
            entry["pr_auc"]["probe"] = {"h1": 0.10, "h5": 0.10, "h10": 0.10}
            aware = dict.fromkeys(("h1", "h5", "h10"), 1.6 * 0.12)
            entry["pr_auc"]["probe_risk_aware"] = aware
        summary = _summary(per_seed)
        assert summary["pr_auc_ratio"]["h5"] == pytest.approx(1.6)
        assert summary["pr_auc_ratio"]["h10"] == pytest.approx(1.6)
        assert summary["wins"] == {"h1": 10, "h5": 10, "h10": 10}
        assert summary["pr_auc_ratio_plain"]["h5"] == pytest.approx(0.10 / 0.12)
        assert summary["pr_auc_ratio_plain"]["h10"] == pytest.approx(0.10 / 0.12)
        assert summary["pr_auc_ratio_plain"]["h10"] < 1.5
        assert summary["check"] is True
        for entry in per_seed:  # the plain probe is the strong one: no headroom
            ap = entry["pr_auc"]
            ap["probe"], ap["probe_risk_aware"] = ap["probe_risk_aware"], ap["probe"]
        swapped = _summary(per_seed)
        assert swapped["wins"] == {"h1": 0, "h5": 0, "h10": 0}
        assert swapped["pr_auc_ratio_plain"]["h10"] == pytest.approx(1.6)
        assert swapped["check"] is False

    def test_the_ratio_is_the_ratio_of_the_seed_means_not_the_mean_of_ratios(self):
        per_seed = _good_seeds(2)
        for entry, (aware, rival) in zip(
            per_seed, [(0.11, 0.10), (0.9, 0.5)], strict=True
        ):
            entry["pr_auc"]["oracle"] = dict.fromkeys(("h1", "h5", "h10"), rival)
            entry["pr_auc"]["probe_risk_aware"] = dict.fromkeys(
                ("h1", "h5", "h10"), aware
            )
        summary = _summary(per_seed)
        assert (0.11 / 0.10 + 0.9 / 0.5) / 2 < 1.5  # mean of ratios: 1.45
        assert summary["pr_auc_ratio"]["h10"] == pytest.approx(1.01 / 0.6)
        assert summary["check"] is True

    def test_a_tie_with_the_stronger_baseline_is_not_a_win(self):
        def with_ties(ties: int) -> dict:
            per_seed = _good_seeds(8)
            for entry in per_seed[:ties]:
                entry["pr_auc"]["probe_risk_aware"]["h10"] = 0.12  # the oracle's value
            return _summary(per_seed)

        one = with_ties(1)
        assert one["wins"]["h10"] == 7
        assert one["check"] is True
        two = with_ties(2)
        assert two["wins"]["h10"] == 6
        assert two["check"] is False

    @pytest.mark.parametrize("horizon", ["h1", "h5", "h10"])
    @pytest.mark.parametrize(("n", "needed"), [(8, 7), (10, 9)])
    def test_wins_must_reach_ceil_seven_eighths_of_the_seeds(self, n, needed, horizon):
        def with_losses(losses: int) -> dict:
            per_seed = _good_seeds(n)
            for entry in per_seed[:losses]:
                # below the oracle's 0.12
                entry["pr_auc"]["probe_risk_aware"][horizon] = 0.11
            return _summary(per_seed)

        passing = with_losses(n - needed)
        assert passing["wins"][horizon] == needed
        assert passing["check"] is True
        failing = with_losses(n - needed + 1)
        assert failing["wins"][horizon] == needed - 1
        assert failing["check"] is False

    def test_the_probe_must_beat_a_random_score_by_a_bootstrap_lower_bound(self):
        def one_seed(probe_warned: int, random_warned: int) -> dict:
            entry = _good_seeds(1)[0]
            entry["arrays"] = {
                "onset_block": np.arange(100),
                "warned_probe": np.arange(100) < probe_warned,
                "warned_random": np.arange(100) < random_warned,
                "block_index": np.repeat(np.arange(100), 3),
            }
            return gates.headroom_summary([entry], [1000])

        good = one_seed(40, 5)
        assert good["detection_gain"]["gain"]["ci_low"] > 0
        assert good["check"] is True
        same = one_seed(5, 5)
        assert same["detection_gain"]["gain"]["ci_low"] <= 0
        assert same["check"] is False

    def test_ppv_outside_25_to_65_percent_fails_and_none_fails(self):
        def ppv(value):
            return _summary(_with(_good_seeds(10), 0, ppv_build_up=value))

        assert ppv(0.45)["check"] is True
        assert ppv(0.45)["ppv_build_up_mean"] == pytest.approx(0.45)
        assert (
            gates.headroom_summary(
                [dict(e, ppv_build_up=0.20) for e in _good_seeds(10)],
                list(TUNING_SEEDS),
            )["check"]
            is False
        )
        assert (
            gates.headroom_summary(
                [dict(e, ppv_build_up=0.70) for e in _good_seeds(10)],
                list(TUNING_SEEDS),
            )["check"]
            is False
        )
        assert ppv(None)["check"] is False

    def test_the_pooled_counts_are_reported_not_gated(self):
        summary = _summary(_good_seeds(4))
        assert summary["ramped_onset_share"] == pytest.approx(0.8)
        assert summary["shape_only_detection_rate"] == pytest.approx(0.6)
        assert len(summary["per_seed"]) == 4
        assert all("arrays" not in entry for entry in summary["per_seed"])
        assert summary["check"] is True

    def test_without_build_ups_the_gate_fails_the_headroom_check(self):
        started = time.monotonic()
        result = gates._score_preset(
            DEFAULT_PRESETS["hard"], 1000, n_blocks=600, strength=0.0
        )["headroom"]
        assert result["ppv_build_up"] is None
        summary = _summary([result])
        assert summary["check"] is False
        assert summary["pr_auc_ratio"]["h10"] < 1.5
        assert summary["stratified_lift_h10_mean"] < 3.0
        assert time.monotonic() - started < 300

    def test_decoys_lower_the_share_of_build_up_windows_that_end_in_an_onset(
        self, tmp_path
    ):
        started = time.monotonic()
        without = _hard_copy(tmp_path, **{"decoy_rate: 0.2": "decoy_rate: 0"})
        no_decoys = gates._score_preset(without, 1000, n_blocks=600, strength=0.8)
        decoys = gates._score_preset(
            DEFAULT_PRESETS["hard"], 1000, n_blocks=600, strength=0.8
        )
        assert (
            no_decoys["headroom"]["ppv_build_up"] > decoys["headroom"]["ppv_build_up"]
        )
        assert time.monotonic() - started < 300

    def test_the_headroom_windows_hold_the_same_blocks_as_the_band_windows(self):
        cfg = DignityConfig.from_yaml(DEFAULT_PRESETS["hard"])
        synthetic = replace(cfg.data.synthetic, seed=1000, n_blocks=150)
        df = SyntheticGenerator(synthetic).generate()
        raw = TransactionPipeline(
            seq_len=cfg.data.seq_len, features=list(RAW_COLUMNS)
        ).process_blocks(
            df,
            cfg.data.split,
            cfg.model.horizon,
            cfg.data.label_overlap,
            np.random.default_rng(1000),
        )
        own = gates._feature_split(df, cfg, 1000)
        assert raw.feature_names == list(RAW_COLUMNS)
        assert "momentum" in own.feature_names  # a derived signal the model sees
        for split in ("train", "val", "test"):
            assert (
                np.unique(getattr(own, split).block_index).tolist()
                == np.unique(getattr(raw, split).block_index).tolist()
            )

    def test_the_risk_aware_probe_is_what_the_headroom_stores_under_its_key(
        self, monkeypatch
    ):
        # The ratio and the wins read `probe_risk_aware`; a swap of the two probes
        # would pass every check built from stored numbers and judge the wrong one.
        cfg = DignityConfig.from_yaml(DEFAULT_PRESETS["hard"])
        df = SyntheticGenerator(
            replace(cfg.data.synthetic, seed=1000, n_blocks=150)
        ).generate()
        d = gates._feature_split(df, cfg, 1000)

        def fake_probe(train, risk_aware=False):
            if risk_aware:
                return lambda ws: np.full(ws.y_forecast.shape, 0.5)
            return lambda ws: ws.y_forecast.astype(float)

        monkeypatch.setattr(gates, "_fit_forecast_probe", fake_probe)
        pr_auc = gates.forecast_headroom(df, d, cfg.model.horizon, 1000)["pr_auc"]
        for key in ("h1", "h5", "h10"):
            assert pr_auc["probe"][key] == pytest.approx(1.0)
            assert pr_auc["probe_risk_aware"][key] < 0.9

    def test_a_planted_build_up_gives_the_risk_aware_probe_headroom_over_the_risk_score(
        self,
    ):
        # Measured on seed 1000 at strength 0.8: from 900 blocks up the assertion
        # holds at 900 and at 1200; at 600 blocks (53 test onsets) it does not.
        started = time.monotonic()
        result = gates._score_preset(
            DEFAULT_PRESETS["hard"], 1000, n_blocks=900, strength=0.8
        )["headroom"]
        ap = result["pr_auc"]
        rival = max(ap["stand_in"]["h10"], ap["oracle"]["h10"])
        assert ap["probe_risk_aware"]["h10"] > 1.5 * rival
        assert result["lift_h10"] > 1.5
        assert time.monotonic() - started < 300


class TestCommittedGateResult:
    def test_committed_gate_result_matches_the_presets(self):
        gate = json.loads((ROOT / "results/v1.0/data_gate.json").read_text())
        assert gate["seeds"] == list(TUNING_SEEDS)
        assert gate["strength_override"] is None
        assert gate["passed"] is True and all(gate["checks"].values())
        assert gate["n_blocks"] == 3600
        assert gate["preset_config_hash"] == preset_config_hash(DEFAULT_PRESETS)

    def test_the_chosen_strength_is_the_first_grid_value_that_passed_both_gates(self):
        # The rule: the smallest grid strength whose headroom check passes on the
        # selection seeds and whose ten-seed gate then passes. A strength that
        # passed the first and missed the second leaves its ten-seed record.
        hard = DignityConfig.from_yaml(DEFAULT_PRESETS["hard"]).data.synthetic
        chosen = hard.precursor_strength
        assert chosen in STRENGTH_GRID
        search = ROOT / "results/strength-search"
        for s in STRENGTH_GRID:
            path = search / f"strength_{s}.json"
            if s > chosen:
                assert not path.exists(), s
                continue
            run = json.loads(path.read_text())
            assert run["seeds"] == list(SELECTION_SEEDS)
            assert run["strength_override"] == s
            assert run["n_blocks"] == 3600
            if s == chosen:
                assert run["checks"]["forecast_headroom"] is True
            elif run["checks"]["forecast_headroom"]:
                ten = json.loads((search / f"ten_seed_gate_{s}.json").read_text())
                assert ten["seeds"] == list(TUNING_SEEDS)
                assert ten["strength_override"] == s
                assert ten["n_blocks"] == 3600
                assert ten["passed"] is False
        shipped = {
            name: DignityConfig.from_yaml(
                DEFAULT_PRESETS[name]
            ).data.synthetic.precursor_strength
            for name in ("hard", "medium", "easy")
        }
        assert shipped == gates._preset_strengths(DEFAULT_PRESETS, chosen)

    def test_the_first_search_record_is_kept_beside_the_current_one(self):
        # The first search (3000 blocks, first rule, plain probe) passed at no
        # strength; the docs link these files as the record of that.
        first = ROOT / "results/strength-search/first-search"
        for s in (0.6, 0.7, 0.8):
            run = json.loads((first / f"strength_{s}.json").read_text())
            assert run["n_blocks"] == 3000, s
            assert run["seeds"] == list(SELECTION_SEEDS), s
            assert run["checks"]["forecast_headroom"] is False, s
            assert "pr_auc_ratio_plain" not in run["headroom"], s

    def test_the_ten_seed_miss_at_the_earlier_strength_failed_on_the_lift_floor_alone(
        self,
    ):
        ten = json.loads(
            (ROOT / "results/strength-search/ten_seed_gate_0.8.json").read_text()
        )
        assert ten["passed"] is False
        assert [k for k, v in ten["checks"].items() if not v] == ["forecast_headroom"]
        head = ten["headroom"]
        assert head["stratified_lift_h10_mean"] < gates.HEADROOM_LIFT_RANGE[0]
        for key in ("h5", "h10"):
            assert head["pr_auc_ratio"][key] >= gates.HEADROOM_PR_AUC_RATIO
        needed = math.ceil(gates.HEADROOM_MIN_WIN_SHARE * 10)
        assert min(head["wins"].values()) >= needed
        assert head["detection_gain"]["gain"]["ci_low"] > 0
        low, high = gates.HEADROOM_PPV_RANGE
        assert low <= head["ppv_build_up_mean"] <= high

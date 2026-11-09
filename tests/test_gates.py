"""Tests for the difficulty gate and its baselines."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from data.pipeline import WindowSet
from data.source.synthetic import RAW_COLUMNS
from eval import gates
from eval.baselines import block_tree, majority, window_logistic
from eval.gates import DEFAULT_PRESETS, TUNING_SEED, data_gate, onset_bucket_recall


def _windowset(**fields) -> WindowSet:
    """Build a WindowSet by keyword; only X and y_risk matter to baselines."""
    n = len(fields["y_risk"])
    defaults = {
        "y_forecast": np.zeros((n, 1), dtype=np.float32),
        "forecast_mask": np.zeros(n, dtype=bool),
        "block_index": np.arange(n, dtype=np.int64),
        "onset_offset": np.full(n, -1, dtype=np.int64),
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

    def test_gate_cli_refuses_the_evaluation_seed(self, capsys):
        assert gates.main(["--seed", "2026"]) == 2
        assert "evaluation seed" in capsys.readouterr().out

    def test_gate_seed_reaches_the_generator(self, monkeypatch):
        seen = []

        class _StopError(Exception): ...

        def spy(config, rng=None):
            seen.append(config.seed)
            raise _StopError

        monkeypatch.setattr(gates, "SyntheticGenerator", spy)
        with pytest.raises(_StopError):
            gates.data_gate(gates.DEFAULT_PRESETS, 7)
        assert seen == [7]  # preset files say 42; the gate seed must win

    def test_gate_fails_when_any_check_fails_and_passes_only_when_all_hold(
        self, monkeypatch, tmp_path, capsys
    ):
        """Aggregation only: the per-preset scoring is replaced by fixed numbers."""
        by_path = {path: name for name, path in DEFAULT_PRESETS.items()}
        state: dict = {}

        def fake_score_preset(path, seed):
            preset = by_path[path]
            wl, bt = state["auroc"][preset]
            recall = state["recall"] if preset == "hard" else None
            return {"window_logistic": wl, "block_tree": bt}, recall

        monkeypatch.setattr(gates, "_score_preset", fake_score_preset)

        def run(easy, medium, hard, onset):
            state["auroc"] = {"easy": easy, "medium": medium, "hard": hard}
            state["recall"] = onset
            return data_gate(DEFAULT_PRESETS, TUNING_SEED)

        ok = ((0.81, 0.74), (0.76, 0.65), (0.73, 0.60))
        flat = [0.46, 0.47, 0.50, 0.52, 0.55]
        skewed = [0.30, 0.47, 0.50, 0.52, 0.55]

        result = run(*ok, flat)
        assert result["passed"] is True
        assert all(result["checks"].values())

        # Each case breaks exactly one check; the other three must still hold.
        cases = {
            "window_logistic_in_band": run(ok[0], ok[1], (0.50, 0.60), flat),
            "block_tree_in_band": run(ok[0], ok[1], (0.73, 0.50), flat),
            "monotone": run(ok[0], (0.82, 0.65), ok[2], flat),
            "positional_flatness": run(*ok, skewed),
        }
        for flag, outcome in cases.items():
            assert outcome["passed"] is False, flag
            failing = [k for k, v in outcome["checks"].items() if not v]
            assert failing == [flag], flag

        out = tmp_path / "gate.json"
        run(*ok, flat)
        assert gates.main(["--seed", "1000", "--out", str(out)]) == 0
        assert json.loads(out.read_text())["passed"] is True

        run(*ok, skewed)
        assert gates.main(["--seed", "1000"]) == 1

        capsys.readouterr()
        assert gates.main(["--seed", "2026"]) == 2
        assert "evaluation seed" in capsys.readouterr().out

    @pytest.mark.slow  # marker is inert (pytest.ini); this runs in CI on purpose
    def test_shipped_presets_pass_the_data_gate_on_the_tuning_seed(self):
        result = data_gate(DEFAULT_PRESETS, TUNING_SEED)
        assert result["passed"], json.dumps(result, indent=2)

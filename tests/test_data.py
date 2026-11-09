"""Test data pipeline."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from numpy.lib.stride_tricks import sliding_window_view

from core.config import DignityConfig, PrivacyConfig, SyntheticConfig
from core.privacy import BudgetExhausted, PrivacyManager
from data.labels import window_starts
from data.loader import TransactionDataset, create_dataloader
from data.pipeline import TransactionPipeline
from data.source.anomalies import (
    ANOMALY_INJECTORS,
    ANOMALY_TYPES,
    inject_benign_spike,
)
from data.source.dynamics import (
    DEFAULT_PARAMS,
    ar1_log_volume,
    congestion_fee,
    garch_returns,
    regime_path,
    tx_counts,
)
from data.source.synthetic import RAW_COLUMNS, SyntheticGenerator
from data.stats import CALIBRATION_CHUNK_LEN, calibration_stats, load_reference_stats

ROOT = Path(__file__).resolve().parents[1]


def _per_block(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per block: ground truth (any anomalous row) and the block label."""
    return frame.groupby("block_id").agg(
        truth=("is_anomalous", "any"),
        label=("block_label", "first"),
        n_labels=("block_label", "nunique"),
    )


class TestSyntheticGenerator:
    """The seeded event-record generator: one row per step, typed anomalies."""

    def test_frame_has_exactly_the_event_record_schema(self, easy_frame):
        assert list(easy_frame.columns) == [
            "timestamp",
            "volume",
            "price",
            "fee_rate",
            "tx_count",
            "block_id",
            "is_anomalous",
            "anomaly_type",
            "anomaly_id",
            "block_label",
        ]
        assert tuple(easy_frame.columns[1:5]) == RAW_COLUMNS
        assert len(easy_frame) == 100 * 160
        dtypes = {
            "timestamp": np.int64,
            "volume": np.float64,
            "price": np.float64,
            "fee_rate": np.float64,
            "tx_count": np.int64,
            "block_id": np.int64,
            "is_anomalous": np.bool_,
            "anomaly_id": np.int64,
            "block_label": np.int8,
        }
        for column, dtype in dtypes.items():
            assert easy_frame[column].dtype == dtype, column
        # object on pandas 2 (CI 3.10), str on pandas 3: both are string dtypes
        assert pd.api.types.is_string_dtype(easy_frame["anomaly_type"])

        # blocks are contiguous and the clock is hourly in ms within every block
        block_ids = easy_frame["block_id"].to_numpy()
        assert (block_ids == np.repeat(np.arange(100), 160)).all()
        steps = easy_frame.groupby("block_id")["timestamp"].diff().dropna()
        assert (steps == 3_600_000).all()

        # downstream log/ratio signals need finite, positive raw values
        raw = easy_frame[list(RAW_COLUMNS)].to_numpy(dtype=float)
        assert np.isfinite(raw).all()
        assert (easy_frame["volume"] > 0).all() and (easy_frame["price"] > 0).all()

        none = ~easy_frame["is_anomalous"]
        assert (easy_frame.loc[none, "anomaly_type"] == "none").all()
        assert (easy_frame.loc[none, "anomaly_id"] == -1).all()
        hit = easy_frame[~none]
        assert len(hit) > 0
        assert set(hit["anomaly_type"]) <= set(ANOMALY_TYPES)
        assert (hit["anomaly_id"] >= 0).all()
        # an anomaly id names one event: it never reappears in another block
        assert (hit.groupby("anomaly_id")["block_id"].nunique() == 1).all()

    def test_same_seed_reproduces_the_frame_despite_global_rng_noise(
        self, easy_config
    ):  # DGNTY-19
        a = SyntheticGenerator(easy_config.data.synthetic).generate()
        np.random.rand(10)
        b = SyntheticGenerator(easy_config.data.synthetic).generate()
        pd.testing.assert_frame_equal(a, b)

    def test_constructing_the_generator_leaves_the_global_rng_untouched(
        self, easy_config
    ):  # DGNTY-19
        np.random.seed(123)  # not 42: a reseed with config.seed must show
        np.random.rand(7)
        before = np.random.get_state()
        SyntheticGenerator(easy_config.data.synthetic).generate()
        after = np.random.get_state()
        assert after[0] == before[0] and (after[1] == before[1]).all()
        assert after[2:] == before[2:]  # compare pos too, not only the key array

    def test_injected_rng_overrides_the_config_seed(self, easy_config):
        base = dataclasses.replace(easy_config.data.synthetic, n_blocks=20)
        seed_1 = dataclasses.replace(base, seed=1)
        seed_2 = dataclasses.replace(base, seed=2)
        a = SyntheticGenerator(seed_1, rng=np.random.default_rng(7)).generate()
        b = SyntheticGenerator(seed_2, rng=np.random.default_rng(7)).generate()
        pd.testing.assert_frame_equal(a, b)
        assert not a.equals(SyntheticGenerator(seed_1).generate())

    def test_normal_and_anomalous_blocks_are_interleaved(self):
        # the legacy generator emitted every normal block, then every anomalous one
        frame = SyntheticGenerator(
            SyntheticConfig(n_blocks=400, contamination=0.3)
        ).generate()
        truth = _per_block(frame)["truth"].to_numpy()
        for half in (truth[:200], truth[200:]):
            assert 0.15 <= half.mean() <= 0.45

    def test_anomaly_onsets_and_durations_vary_across_the_block(self):  # DGNTY-16
        cfg = SyntheticConfig(
            n_blocks=200,
            anomalies_per_block=(1, 3),
            duration_range=(3, 60),
            severity_range=(0.3, 1.0),
        )
        frame = SyntheticGenerator(cfg).generate()
        row = pd.Series(np.arange(len(frame)) % cfg.block_len, index=frame.index)
        hit = frame["is_anomalous"]
        spans = row[hit].groupby(frame.loc[hit, "anomaly_id"]).agg(["min", "max"])
        durations = spans["max"] - spans["min"] + 1
        assert spans["min"].min() < 10
        assert spans["max"].max() + 1 > 150
        assert durations.nunique() >= 20
        assert durations.between(3, 60).all()

    def test_no_anomaly_starts_at_row_zero_where_its_entry_return_is_lost(self):
        # Injectors act through the log-return INTO each span row; row 0 has none,
        # so a flash crash starting there would be labelled but never happen.
        # Spans nearly as long as the block make a row-0 start near-certain if the
        # generator allowed one.
        cfg = SyntheticConfig(
            n_blocks=200,
            contamination=1.0,
            anomalies_per_block=(1, 1),
            duration_range=(150, 159),
        )
        frame = SyntheticGenerator(cfg).generate()
        row = pd.Series(np.arange(len(frame)) % cfg.block_len, index=frame.index)
        hit = frame["is_anomalous"]
        onsets = row[hit].groupby(frame.loc[hit, "anomaly_id"]).min()
        assert len(onsets) == cfg.n_blocks
        assert onsets.min() >= 1

    def test_block_label_is_the_truth_without_noise_and_flipped_with_full_noise(
        self,
    ):
        cfg = SyntheticConfig(n_blocks=60, contamination=0.5)
        clean = _per_block(SyntheticGenerator(cfg).generate())
        assert clean["truth"].any() and not clean["truth"].all()
        assert (clean["n_labels"] == 1).all()
        assert (clean["label"] == clean["truth"].astype(int)).all()

        noisy_cfg = dataclasses.replace(cfg, label_noise=1.0)
        noisy = _per_block(SyntheticGenerator(noisy_cfg).generate())
        assert (noisy["n_labels"] == 1).all()
        assert (noisy["label"] == 1 - noisy["truth"].astype(int)).all()

    def test_benign_spikes_are_never_labelled(self):
        cfg = SyntheticConfig(n_blocks=50, contamination=0.0, benign_spike_rate=1.0)
        frame = SyntheticGenerator(cfg).generate()
        assert not frame["is_anomalous"].any()
        assert (frame["block_label"] == 0).all()
        assert (frame["anomaly_type"] == "none").all()
        assert (frame["anomaly_id"] == -1).all()
        # the spikes were applied. Only block 0 is compared: both runs reach its
        # spike draw with the same RNG state, but the injector consumes draws the
        # calm run does not, so every later block differs even if a spike is a no-op
        calm = SyntheticGenerator(
            dataclasses.replace(cfg, benign_spike_rate=0.0)
        ).generate()
        first = frame["block_id"] == 0
        assert not frame.loc[first, list(RAW_COLUMNS)].equals(
            calm.loc[first, list(RAW_COLUMNS)]
        )

    def test_type_weights_restrict_the_typology(self):
        cfg = SyntheticConfig(
            n_blocks=50, contamination=1.0, type_weights={"fee_spike": 1.0}
        )
        frame = SyntheticGenerator(cfg).generate()
        assert set(frame.loc[frame["is_anomalous"], "anomaly_type"]) == {"fee_spike"}

    def test_unknown_anomaly_type_is_rejected(self):
        with pytest.raises(ValueError, match="nope"):
            SyntheticGenerator(SyntheticConfig(type_weights={"nope": 1.0}))

    def test_zero_tx_median_raises_instead_of_an_infinite_fee(self, monkeypatch):
        # congestion_fee divides by capacity_ratio * median(tx_count) (M2.3 carry)
        def no_transactions(volume, params, rng):
            return np.zeros(len(volume), dtype=np.int64)

        monkeypatch.setattr("data.source.synthetic.tx_counts", no_transactions)
        with pytest.raises(ValueError, match="tx_count"):
            SyntheticGenerator(SyntheticConfig(n_blocks=2)).generate()


class TestTransactionPipeline:
    """Test data preprocessing pipeline."""

    def test_signal_computation(self, easy_frame):
        """Test signal feature computation."""
        df_raw = easy_frame[list(RAW_COLUMNS)]

        pipeline = TransactionPipeline(seq_len=50)
        df_processed = pipeline.compute_signals(df_raw)

        assert "volume_volatility" in df_processed.columns
        assert "volatility" in df_processed.columns
        assert "momentum" in df_processed.columns

    def test_fit_transform(self, easy_frame):
        """Test scaling pipeline."""
        df = easy_frame[list(RAW_COLUMNS)]

        pipeline = TransactionPipeline(seq_len=50)
        X = pipeline.fit_transform(df)

        assert X.shape[0] == len(df)
        assert X.shape[1] > 0  # Has features

        # RobustScaler centres on the median; the mean of skewed volumes is not 0
        assert np.abs(np.median(X, axis=0)).max() < 1.0

    def test_sequence_creation(self):
        """Test sliding window sequence creation."""
        X = np.random.randn(500, 9)
        y = np.random.randint(0, 2, 500)

        pipeline = TransactionPipeline(seq_len=100)
        pipeline.fitted = True
        pipeline.available_features = [f"f{i}" for i in range(9)]

        X_seq, y_seq = pipeline.create_sequences(X, y, stride=1)

        assert X_seq.shape[0] == 401  # 500 - 100 + 1
        assert X_seq.shape[1] == 100  # seq_len
        assert X_seq.shape[2] == 9  # features
        assert len(y_seq) == 401

    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        df = _frame(500)
        labels = np.zeros(500, dtype=int)

        # _frame's price is 0 at row 0, so momentum is inf there (DGNTY-46); leave
        # momentum out of the features so the scaler sees only finite values.
        pipeline = TransactionPipeline(
            seq_len=100, features=["volume", "price", "fee_rate", "tx_count"]
        )
        X_seq, y_seq = pipeline.process(df, labels, fit=True, stride=10)

        assert X_seq.ndim == 3  # [sequences, seq_len, features]
        assert y_seq.ndim == 1  # [sequences]
        assert len(X_seq) == len(y_seq)

    def test_two_dimensional_targets_are_rejected_not_misaligned(self):
        # Forecast targets only come from data.labels.forecast_labels; the old 2-D
        # branch silently took rows after the window and misaligned them.
        p = TransactionPipeline(seq_len=10)
        with pytest.raises(ValueError, match="forecast_labels"):
            p.create_sequences(np.zeros((30, 3)), np.zeros((30, 4)))


class _ZeroNoise:
    def random(self) -> float:
        return 0.5


def _frame(n: int = 60) -> pd.DataFrame:
    t = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "volume": 100.0 + 10.0 * np.sin(t / 3.0),
            "price": t * 3.0,  # 0 .. 177, so clipping to [50, 150] bites on both ends
            "fee_rate": np.where(t % 20 == 0, 0.005, 0.001),
            "tx_count": (50 + (t % 7)).astype(float),
        }
    )


def _laplace_price(eps_total: float = 1.0, eps: float = 1.0) -> PrivacyConfig:
    return PrivacyConfig(
        epsilon_total=eps_total,
        features={
            "price": {"mechanism": "laplace", "epsilon": eps, "bounds": [50.0, 150.0]}
        },
    )


class TestPrivacyStage:
    """The stage runs once per public call, before signals, and only when configured."""

    def test_privacy_stage_runs_before_signals(self):
        df = _frame()
        priv = TransactionPipeline(
            seq_len=10,
            features=["price", "volatility"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        x_priv, _ = priv.process(df)

        # Zero noise => the stage is pure clipping. A plain pipeline fed pre-clipped
        # prices must produce identical output — which is only true if clipping
        # happened BEFORE volatility was computed.
        pre_clipped = df.assign(price=df["price"].clip(50.0, 150.0))
        x_ref, _ = TransactionPipeline(
            seq_len=10, features=["price", "volatility"]
        ).process(pre_clipped)
        np.testing.assert_allclose(x_priv, x_ref)

        # and it must differ from the unclipped run, or the stage did nothing
        x_raw, _ = TransactionPipeline(
            seq_len=10, features=["price", "volatility"]
        ).process(df)
        assert not np.allclose(x_priv, x_raw)

    def test_fit_transform_spends_budget_exactly_once(self):
        priv = TransactionPipeline(
            seq_len=10,
            features=["price"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        priv.process(_frame(), fit=True)
        assert priv.privacy_manager.budget.spent == pytest.approx(1.0)

    def test_budget_exhausted_propagates_from_transform(self):
        priv = TransactionPipeline(
            seq_len=10,
            features=["price"],
            privacy=_laplace_price(),
            privacy_rng=_ZeroNoise(),
        )
        priv.process(_frame(), fit=True)  # spends the whole budget
        with pytest.raises(BudgetExhausted):
            priv.process(_frame(), fit=False)

    def test_absent_privacy_block_is_a_noop(self):
        plain = TransactionPipeline(seq_len=10, features=["price"])
        df = _frame()
        assert plain.privacy_manager is None
        assert plain._apply_privacy(df) is df

    def test_generalize_column_is_k_anonymous_in_output(self):
        cfg = PrivacyConfig(
            epsilon_total=1.0,
            k=5,
            features={"fee_rate": {"mechanism": "generalize", "bins": 4}},
        )
        priv = TransactionPipeline(seq_len=10, features=["fee_rate"], privacy=cfg)
        generalized = priv._apply_privacy(_frame())["fee_rate"].to_numpy()
        _, counts = np.unique(generalized, return_counts=True)
        assert (counts >= 5).all()

    def test_missing_privacy_column_raises(self):
        cfg = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "nope": {"mechanism": "laplace", "epsilon": 1.0, "bounds": [0, 1]}
            },
        )
        priv = TransactionPipeline(seq_len=10, features=["price"], privacy=cfg)
        with pytest.raises(ValueError, match="'nope'"):
            priv.process(_frame())

    def test_missing_column_spends_no_budget(self):
        cfg = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "price": {
                    "mechanism": "laplace",
                    "epsilon": 0.5,
                    "bounds": [50.0, 150.0],
                },
                "nope": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0.0, 1.0]},
            },
        )
        priv = TransactionPipeline(
            seq_len=10, features=["price"], privacy=cfg, privacy_rng=_ZeroNoise()
        )
        with pytest.raises(ValueError, match="'nope'"):
            priv.process(_frame())
        assert priv.privacy_manager.budget.spent == 0.0

    def test_missing_key_env_var_fails_at_construction(self, monkeypatch):
        monkeypatch.delenv("DIGNITY_TEST_KEY_UNSET", raising=False)
        cfg = PrivacyConfig(epsilon_total=1.0, key_env="DIGNITY_TEST_KEY_UNSET")
        with pytest.raises(ValueError, match="DIGNITY_TEST_KEY_UNSET"):
            TransactionPipeline(seq_len=10, privacy=cfg)

    def test_key_env_var_is_read_into_the_manager(self, monkeypatch):
        monkeypatch.setenv("DIGNITY_TEST_KEY", "0123456789abcdef")
        cfg = PrivacyConfig(epsilon_total=1.0, key_env="DIGNITY_TEST_KEY")
        priv = TransactionPipeline(seq_len=10, privacy=cfg)
        assert len(priv.privacy_manager.pseudonymize("x")) == 64


class _ReversedOrder:
    """Stand-in rng: the highest-numbered blocks come first, so they land in val."""

    def permutation(self, n: int) -> np.ndarray:
        return np.arange(n)[::-1]


def _labelled(df: pd.DataFrame, block_len: int) -> pd.DataFrame:
    """Add the event-record columns to a hand frame of equal-length blocks."""
    out = df.copy()
    out["block_id"] = np.arange(len(df)) // block_len
    out["is_anomalous"] = False
    out["anomaly_type"] = "none"
    out["anomaly_id"] = -1
    return out


def _split(pipeline, df, rng):
    return pipeline.process_blocks(df, (0.7, 0.15, 0.15), 10, 0.0, rng)


class TestSequenceAwareSplit:
    """process_blocks treats each generated block as an indivisible unit."""

    def test_each_split_sees_both_classes(self, easy_frame):
        # A positional split once produced an all-normal train set and an
        # all-anomalous val set, so the risk head learned "always 0" and
        # validation was meaningless.
        d = _split(
            TransactionPipeline(seq_len=100, features=list(RAW_COLUMNS)),
            easy_frame,
            np.random.default_rng(0),
        )
        for w in (d.train, d.val, d.test):
            assert set(np.unique(w.y_risk)) == {0, 1}

    def test_every_block_lands_in_exactly_one_split(self, easy_frame):
        d = _split(
            TransactionPipeline(seq_len=100, features=list(RAW_COLUMNS)),
            easy_frame,
            np.random.default_rng(0),
        )
        sets = [set(np.unique(w.block_index)) for w in (d.train, d.val, d.test)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        assert set.union(*sets) == set(easy_frame["block_id"])
        assert [len(s) for s in sets] == [70, 15, 15]

    def test_windows_never_span_two_blocks(self, easy_frame):
        # Each block yields exactly len(window_starts) windows; a window crossing
        # a joint would show up as an extra one.
        d = _split(
            TransactionPipeline(seq_len=100, features=list(RAW_COLUMNS)),
            easy_frame,
            np.random.default_rng(0),
        )
        assert len(d.train.X) == 70 * len(window_starts(160, 100, 10))
        assert d.train.X.shape[1:] == (100, 4)
        assert d.train.X.dtype == np.float32

    def test_rolling_signals_restart_at_every_block(self, easy_frame):
        # Raw columns cannot show a leak across a joint; the rolling signals can.
        # Every block's windows must equal the block transformed on its own with
        # the same fitted scaler, so no signal reads the previous block's rows.
        base = DignityConfig.from_yaml(str(ROOT / "config" / "base.yaml"))
        pipeline = TransactionPipeline(seq_len=100, features=base.data.features)
        d = _split(pipeline, easy_frame, np.random.default_rng(0))
        derived = {"volatility", "momentum", "directional_change", "volume_volatility"}
        assert derived <= set(d.feature_names)

        starts = window_starts(160, 100, 10)
        for w in (d.train, d.val, d.test):
            for block_id in np.unique(w.block_index):
                block = easy_frame[easy_frame["block_id"] == block_id]
                alone = sliding_window_view(pipeline._transform_on(block), 100, axis=0)[
                    starts
                ]
                np.testing.assert_array_equal(
                    w.X[w.block_index == block_id],
                    alone.transpose(0, 2, 1).astype(np.float32),
                )

    def test_every_positive_window_contains_an_anomalous_row(self, easy_frame):
        # DGNTY-18/39: a risk label with no anomalous row inside would teach noise.
        d = _split(
            TransactionPipeline(seq_len=100, features=list(RAW_COLUMNS)),
            easy_frame,
            np.random.default_rng(0),
        )
        for w in (d.train, d.val, d.test):
            assert (w.onset_offset[w.y_risk == 1] >= 0).all()
            assert (w.onset_offset[w.y_risk == 0] == -1).all()

        # Recompute every window's truth from the event record itself, not from
        # anything the pipeline returned. WindowSet carries block_index but no
        # window start; a block's windows are emitted in window_starts order.
        overlap = 0.0  # what _split passes
        starts = window_starts(160, 100, 10)
        for w in (d.train, d.val, d.test):
            for block_id in np.unique(w.block_index):
                idx = np.flatnonzero(w.block_index == block_id)
                assert len(idx) == len(starts)
                rows = easy_frame.loc[
                    easy_frame["block_id"] == block_id, "is_anomalous"
                ].to_numpy()
                windows = sliding_window_view(rows, 100)[starts]
                positive = windows.mean(axis=1) > overlap
                first = np.where(windows.any(axis=1), windows.argmax(axis=1), -1)
                np.testing.assert_array_equal(w.y_risk[idx], positive.astype(np.int8))
                np.testing.assert_array_equal(w.onset_offset[idx], first)

    def test_validation_outliers_are_not_absorbed_by_the_scaler(self):
        # Five constant blocks; the one in val and the one in test are six orders
        # of magnitude larger. If the scaler had seen them, its IQR would swallow
        # the gap and they would scale to about 1. Fit on train only, they stay
        # enormous.
        block_len = 4
        volume = np.concatenate(
            [np.full(block_len, v) for v in (1e6, 1e6, 1.0, 2.0, 3.0)]
        )
        df = _labelled(
            pd.DataFrame({"volume": volume + np.arange(len(volume)) * 0.01}),
            block_len,
        )
        pipeline = TransactionPipeline(seq_len=3, features=["volume"])
        d = pipeline.process_blocks(df, (0.6, 0.2, 0.2), 1, 0.0, _ReversedOrder())
        assert set(np.unique(d.val.block_index)) == {1}
        assert set(np.unique(d.test.block_index)) == {0}
        assert np.abs(d.train.X).max() < 10
        assert np.abs(d.val.X).min() > 100 and np.abs(d.test.X).min() > 100

    def test_single_privacy_release_covers_all_three_splits(self, easy_frame):
        # The budget equals the per-column epsilon, so a second release for any
        # split would raise BudgetExhausted. All three come from one draw.
        # Bounds sit inside the easy prices so the clip bites on both ends.
        bounds = [500.0, 3000.0]
        privacy = PrivacyConfig(
            epsilon_total=1.0,
            features={
                "price": {"mechanism": "laplace", "epsilon": 1.0, "bounds": bounds}
            },
        )
        pipeline = TransactionPipeline(
            seq_len=100,
            features=["volume", "price"],
            privacy=privacy,
            privacy_rng=_ZeroNoise(),
        )
        d = _split(pipeline, easy_frame, np.random.default_rng(0))
        assert pipeline.privacy_manager.budget.spent == pytest.approx(1.0)
        assert d.epsilon_spent == {"train": 1.0, "val": 1.0, "test": 1.0}

        # Zero noise => the release is a pure clip. Every split must carry the
        # released values, so it equals a plain run on pre-clipped prices ...
        pre_clipped = easy_frame.assign(price=easy_frame["price"].clip(*bounds))
        ref = _split(
            TransactionPipeline(seq_len=100, features=["volume", "price"]),
            pre_clipped,
            np.random.default_rng(0),
        )
        raw = _split(
            TransactionPipeline(seq_len=100, features=["volume", "price"]),
            easy_frame,
            np.random.default_rng(0),
        )
        for name in ("train", "val", "test"):
            got, want = getattr(d, name), getattr(ref, name)
            np.testing.assert_array_equal(got.block_index, want.block_index)
            np.testing.assert_allclose(got.X, want.X)
            # ... and differs from the unreleased frame, or the release did nothing
            assert not np.allclose(got.X, getattr(raw, name).X)

    def test_no_privacy_reports_no_epsilon(self, easy_frame):
        d = _split(
            TransactionPipeline(seq_len=100, features=list(RAW_COLUMNS)),
            easy_frame,
            np.random.default_rng(0),
        )
        assert d.epsilon_spent == {}

    def test_block_too_short_for_a_window_and_horizon_is_rejected(self):
        df = _labelled(_frame(60), 20)
        pipeline = TransactionPipeline(seq_len=15, features=["volume"])
        with pytest.raises(ValueError, match="block"):
            pipeline.process_blocks(
                df, (0.7, 0.15, 0.15), 10, 0.0, np.random.default_rng(0)
            )

    def test_fewer_than_three_blocks_is_rejected(self):
        df = _labelled(_frame(40), 20)
        pipeline = TransactionPipeline(seq_len=5, features=["volume"])
        with pytest.raises(ValueError, match="blocks"):
            pipeline.process_blocks(
                df, (0.7, 0.15, 0.15), 1, 0.0, np.random.default_rng(0)
            )


class TestDataLoader:
    """Test PyTorch data loading."""

    def test_samples_are_dicts_that_stay_on_cpu(self):
        X = np.random.randn(100, 50, 9)
        ds = TransactionDataset(
            X, {"y": np.random.randint(0, 2, 100), "mask": np.ones(100, bool)}
        )
        s = ds[0]
        assert set(s) == {"x", "y", "mask"}
        assert s["x"].shape == (50, 9) and s["x"].dtype == torch.float32
        assert s["x"].device.type == "cpu"
        assert s["y"].dtype == torch.int64 and s["mask"].dtype == torch.bool

    def test_label_free_dataset_yields_only_x(self):
        assert set(TransactionDataset(np.random.randn(3, 5, 2))[0]) == {"x"}

    def test_dataloader_batches_dicts_and_carries_worker_knobs(self):
        X, y = np.random.randn(100, 50, 9), np.random.randint(0, 2, 100)
        loader = create_dataloader(X, {"y": y}, batch_size=16, shuffle=False)
        batch = next(iter(loader))
        assert batch["x"].shape == (16, 50, 9) and batch["y"].shape == (16,)
        knobs = create_dataloader(X, {"y": y}, num_workers=2, pin_memory=True)
        assert knobs.num_workers == 2 and knobs.pin_memory is True

    def test_worker_processes_yield_the_same_dict_batches(self):
        # The shipped default spawns worker processes (spawn on macOS and Colab):
        # the dataset must pickle and the batches must match the in-process ones.
        X = np.random.default_rng(0).standard_normal((32, 10, 3)).astype("float32")
        t = {"y": np.arange(32) % 2, "mask": np.ones(32, bool)}
        serial = list(create_dataloader(X, t, batch_size=8, shuffle=False))
        workers = list(
            create_dataloader(X, t, batch_size=8, shuffle=False, num_workers=2)
        )
        assert len(workers) == len(serial)
        for a, b in zip(serial, workers, strict=True):
            assert a.keys() == b.keys()
            assert all(torch.equal(a[k], b[k]) for k in a)


class TestNormalDynamics:
    """Normal-process dynamics the synthetic generator is built from."""

    def test_returns_have_fat_tails_and_volatility_clustering(self):
        rng = np.random.default_rng(0)
        r = garch_returns(20_000, DEFAULT_PARAMS, rng)
        assert scipy.stats.kurtosis(r) > 1.0
        assert np.corrcoef(np.abs(r[:-1]), np.abs(r[1:]))[0, 1] > 0.1

    def test_log_volume_is_autocorrelated_and_never_floor_clipped(self):
        rng = np.random.default_rng(0)
        x = ar1_log_volume(20_000, DEFAULT_PARAMS, rng)
        assert abs(np.corrcoef(x[:-1], x[1:])[0, 1] - DEFAULT_PARAMS.ar1_phi) < 0.05
        assert (x == x.min()).sum() == 1  # no pinned floor (DGNTY-11)

    def test_tx_counts_track_volume_and_are_overdispersed(self):
        rng = np.random.default_rng(0)
        vol = np.exp(4.6 + ar1_log_volume(20_000, DEFAULT_PARAMS, rng))
        tx = tx_counts(vol, DEFAULT_PARAMS, rng)
        assert np.corrcoef(tx, vol)[0, 1] > 0.5 and tx.var() > tx.mean()

    def test_fee_rises_with_backlog_and_keeps_a_nonzero_spread(self):
        rng = np.random.default_rng(0)
        busy = np.r_[np.full(200, 50), np.full(200, 80)]  # cap = 1.1 * 65 = 71.5
        fee = congestion_fee(busy, DEFAULT_PARAMS, rng)
        assert fee[200:].mean() > fee[:200].mean()
        calm = congestion_fee(np.full(400, 10), DEFAULT_PARAMS, rng)
        q25, q75 = np.percentile(calm, [25, 75])
        assert q75 > q25  # DGNTY-17: no degenerate IQR

    def test_regime_switches_follow_the_switch_probability(self):
        rng = np.random.default_rng(0)
        p0 = dataclasses.replace(DEFAULT_PARAMS, regime_switch_prob=0.0)
        p1 = dataclasses.replace(DEFAULT_PARAMS, regime_switch_prob=1.0)
        assert (regime_path(100, p0, rng) == 1).all()
        assert (np.diff(regime_path(100, p1, rng)) != 0).all()

    def test_dynamics_never_touch_the_global_numpy_rng(self):
        rng = np.random.default_rng(0)
        np.random.seed(123)
        np.random.rand(7)  # distinct from the autouse seed 42
        before = np.random.get_state()
        garch_returns(100, DEFAULT_PARAMS, rng)
        tx_counts(np.ones(100), DEFAULT_PARAMS, rng)
        after = np.random.get_state()
        assert after[0] == before[0] and (after[1] == before[1]).all()
        assert after[2:] == before[2:]  # pos, has_gauss, cached_gaussian


def _block() -> pd.DataFrame:
    """A 160-row normal block built from the M2.3 dynamics."""
    rng = np.random.default_rng(0)
    n = 160
    volume = np.exp(
        DEFAULT_PARAMS.log_volume_mean + ar1_log_volume(n, DEFAULT_PARAMS, rng)
    )
    tx = tx_counts(volume, DEFAULT_PARAMS, rng)
    fee = congestion_fee(tx, DEFAULT_PARAMS, rng)
    r = garch_returns(n - 1, DEFAULT_PARAMS, rng)
    price = np.exp(DEFAULT_PARAMS.price_level_mean + np.r_[0.0, np.cumsum(r)])
    return pd.DataFrame(
        {"volume": volume, "price": price, "fee_rate": fee, "tx_count": tx}
    )


class TestAnomalyInjectors:
    """The six typologies and benign spikes the generator injects."""

    def test_every_typology_has_an_injector(self):
        assert set(ANOMALY_INJECTORS) == set(ANOMALY_TYPES)
        assert len(ANOMALY_TYPES) == 6

    @pytest.mark.parametrize("kind", ANOMALY_TYPES)
    def test_zero_severity_is_invisible(self, kind):
        b = _block()
        out = ANOMALY_INJECTORS[kind](b, 40, 20, 0.0, np.random.default_rng(1))
        pd.testing.assert_frame_equal(out, b)

    @pytest.mark.parametrize("kind", ANOMALY_TYPES)
    def test_effects_stay_inside_the_span(self, kind):
        b = _block()
        before = b.copy()
        out = ANOMALY_INJECTORS[kind](b, 40, 20, 1.0, np.random.default_rng(1))
        outside = np.r_[0:40, 60:160]
        for col in ("volume", "fee_rate", "tx_count"):
            np.testing.assert_allclose(
                out[col].to_numpy()[outside], b[col].to_numpy()[outside]
            )
        r_in, r_out = (np.diff(np.log(f["price"].to_numpy())) for f in (b, out))
        ret_outside = np.r_[0:39, 59:159]  # return j is the move into row j+1
        np.testing.assert_allclose(r_out[ret_outside], r_in[ret_outside])
        assert b is not out
        pd.testing.assert_frame_equal(b, before)
        assert out["tx_count"].dtype == np.int64

    def test_typologies_have_their_signature_at_full_severity(self):
        b = _block()
        inject = {
            k: ANOMALY_INJECTORS[k](b, 40, 20, 1.0, np.random.default_rng(1))
            for k in ANOMALY_TYPES
        }
        span = slice(40, 60)
        r_in = np.diff(np.log(b["price"].to_numpy()))

        def r_of(f):
            return np.diff(np.log(f["price"].to_numpy()))

        # compare against the input's same span (the counterfactual): the block's
        # own congestion can move span levels far from rows 0..39.
        w = inject["wash_trading"]
        assert (
            w["volume"].to_numpy()[span].mean()
            >= 2 * b["volume"].to_numpy()[span].mean()
        )
        assert np.abs(r_of(w)[39:59]).mean() <= 0.05 * np.abs(r_in[39:59]).mean()

        p = inject["pump_and_dump"]
        assert p["price"].to_numpy()[span].max() > b["price"].to_numpy()[40] * 1.05

        fs = inject["fee_spike"]
        assert (
            fs["fee_rate"].to_numpy()[span].mean()
            >= 4 * b["fee_rate"].to_numpy()[span].mean()
        )

        fe = inject["fee_evasion"]
        assert (
            fe["fee_rate"].to_numpy()[span].mean()
            <= 0.3 * b["fee_rate"].to_numpy()[span].mean()
        )

        fc = inject["flash_crash"]
        assert r_of(fc)[39] <= r_in[39] - 0.09

        lay = inject["layering"]
        med = np.median(b["volume"].to_numpy())
        assert (lay["volume"].to_numpy()[40::4][:5] > med + 2 * med).all()

    def test_benign_spike_raises_volume_somewhere(self):
        b = _block()
        out = inject_benign_spike(b, np.random.default_rng(3))
        assert (out["volume"] > b["volume"] * 1.9).any()


def _normal_frame(preset: str) -> pd.DataFrame:
    """The preset's normal process alone: no anomalies or spikes, 200 blocks."""
    synthetic = DignityConfig.from_yaml(str(ROOT / preset)).data.synthetic
    normal = dataclasses.replace(
        synthetic, contamination=0.0, benign_spike_rate=0.0, n_blocks=200
    )
    return SyntheticGenerator(normal).generate()


class TestGeneratorCalibration:
    """The normal process matches the committed Bitstamp reference (DGNTY-11)."""

    @pytest.mark.parametrize(
        "preset",
        [
            "config/synthetic_easy.yaml",
            "config/synthetic_medium.yaml",
            "config/synthetic_hard.yaml",
        ],
    )
    def test_normal_process_lands_inside_calibration_bands(self, preset):
        cfg = DignityConfig.from_yaml(str(ROOT / preset))
        assert cfg.data.synthetic.block_len == CALIBRATION_CHUNK_LEN  # estimator parity
        df = _normal_frame(preset)
        stats = calibration_stats(df["price"].to_numpy(), df["volume"].to_numpy())
        bands = load_reference_stats()
        outside = {
            k: (v, bands[k]["min"], bands[k]["max"])
            for k, v in stats.items()
            if not bands[k]["min"] <= v <= bands[k]["max"]
        }
        assert outside == {}


class TestNormalProcessPlausibility:
    """fee_rate and tx_count are not calibrated (v1.1), only kept plausible."""

    def test_fee_rate_has_a_nonzero_iqr(self):  # DGNTY-17
        fee = _normal_frame("config/synthetic_easy.yaml")["fee_rate"].to_numpy()
        q25, q75 = np.percentile(fee, [25, 75])
        assert q75 > q25

    def test_tx_count_tracks_volume_within_blocks(self):
        df = _normal_frame("config/synthetic_easy.yaml")
        corr = df.groupby("block_id")[["tx_count", "volume"]].apply(
            lambda b: np.corrcoef(b["tx_count"], b["volume"])[0, 1]
        )
        assert corr.median() > 0.5

    def test_tx_count_is_overdispersed(self):
        df = _normal_frame("config/synthetic_easy.yaml")
        tx = df.groupby("block_id")["tx_count"]
        assert (tx.var() / tx.mean()).median() > 1

    def test_generalized_fee_rate_keeps_at_least_eight_classes(self):  # DGNTY-17
        shipped = DignityConfig.from_yaml(str(ROOT / "config/train_risk.yaml")).privacy
        bins = shipped.features["fee_rate"].bins  # the shipped settings, not literals
        fee = _normal_frame("config/synthetic_easy.yaml")["fee_rate"].to_numpy()
        out = PrivacyManager.generalize_amounts(fee, bins=bins, k=shipped.k)
        assert len(np.unique(out)) >= bins - 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

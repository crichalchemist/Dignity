"""Test data pipeline."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch

from core.config import PrivacyConfig
from core.privacy import BudgetExhausted
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
from data.source.synthetic import SyntheticGenerator


class TestSyntheticGenerator:
    """Test synthetic data generation."""

    def test_normal_sequence(self):
        """Test normal sequence generation."""
        gen = SyntheticGenerator(seed=42)
        data = gen.generate_normal_sequence(length=1000)

        assert "volume" in data
        assert "price" in data
        assert "fee_rate" in data
        assert "tx_count" in data

        assert len(data["volume"]) == 1000
        assert np.all(data["volume"] > 0)

    def test_anomalous_sequence(self):
        """Test anomalous pattern generation."""
        gen = SyntheticGenerator(seed=42)

        # Test different anomaly types
        for anomaly_type in ["volume_spike", "price_manipulation", "fee_evasion"]:
            data = gen.generate_anomalous_sequence(
                length=100, anomaly_type=anomaly_type
            )

            assert len(data["volume"]) == 100
            assert "volume" in data and "price" in data

    def test_dataset_generation(self):
        """Test balanced dataset generation."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=100, num_anomalous=50, seq_len=50)

        assert len(df) == (100 + 50) * 50  # sequences * seq_len
        assert "label" in df.columns
        assert set(df["label"].unique()) == {0, 1}


class TestTransactionPipeline:
    """Test data preprocessing pipeline."""

    def test_signal_computation(self):
        """Test signal feature computation."""
        gen = SyntheticGenerator(seed=42)
        df_raw = gen.generate_dataset(num_normal=10, num_anomalous=0, seq_len=100)

        pipeline = TransactionPipeline(seq_len=50)
        df_processed = pipeline.compute_signals(df_raw)

        assert "volume_volatility" in df_processed.columns
        assert "volatility" in df_processed.columns
        assert "momentum" in df_processed.columns

    def test_fit_transform(self):
        """Test scaling pipeline."""
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=50, num_anomalous=10, seq_len=100)
        df = df.drop("label", axis=1)

        pipeline = TransactionPipeline(seq_len=50)
        X = pipeline.fit_transform(df)

        assert X.shape[0] == len(df)
        assert X.shape[1] > 0  # Has features

        # Check scaling (should be roughly centered)
        assert np.abs(X.mean()) < 1.0

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
        gen = SyntheticGenerator(seed=42)
        df = gen.generate_dataset(num_normal=100, num_anomalous=20, seq_len=200)
        labels = df["label"].values
        df = df.drop("label", axis=1)

        pipeline = TransactionPipeline(seq_len=100)
        X_seq, y_seq = pipeline.process(df, labels, fit=True, stride=10)

        assert X_seq.ndim == 3  # [sequences, seq_len, features]
        assert y_seq.ndim == 1  # [sequences]
        assert len(X_seq) == len(y_seq)


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


def _blocks_frame(n_normal: int, n_anomalous: int, block_len: int):
    """Synthetic dataset as (raw frame, per-row labels); one block per sequence."""
    gen = SyntheticGenerator(seed=0)
    df = gen.generate_dataset(
        num_normal=n_normal, num_anomalous=n_anomalous, seq_len=block_len
    )
    return df.drop(columns="label"), df["label"].to_numpy()


class _ReversedOrder:
    """Stand-in rng: the highest-numbered blocks come first, so they land in val."""

    def permutation(self, n: int) -> np.ndarray:
        return np.arange(n)[::-1]


class TestSequenceAwareSplit:
    """process_blocks treats each generated sequence as an indivisible unit."""

    def test_each_split_sees_both_classes(self):
        # generate_dataset emits every normal block, then every anomalous block.
        # A positional split produced an all-normal train set and an all-anomalous
        # val set, so the risk head learned "always 0" and validation was meaningless.
        df, labels = _blocks_frame(80, 20, block_len=60)
        pipeline = TransactionPipeline(
            seq_len=50, features=["volume", "price", "fee_rate"]
        )
        (X_tr, y_tr), (X_va, y_va) = pipeline.process_blocks(
            df, labels, block_len=60, test_size=0.2, rng=np.random.default_rng(0)
        )
        assert set(np.unique(y_tr)) == {0, 1}
        assert set(np.unique(y_va)) == {0, 1}
        assert len(X_tr) == len(y_tr)
        assert len(X_va) == len(y_va)

    def test_windows_never_span_two_sequences(self):
        # With the window as long as the block, each block yields exactly one
        # window; a window crossing a joint would show up as an extra sequence.
        df, labels = _blocks_frame(8, 2, block_len=50)
        pipeline = TransactionPipeline(seq_len=50, features=["volume", "price"])
        (X_tr, _), (X_va, _) = pipeline.process_blocks(
            df, labels, block_len=50, test_size=0.2, rng=np.random.default_rng(1)
        )
        assert len(X_tr) + len(X_va) == 10
        assert len(X_va) == 2
        assert X_tr.shape[1:] == (50, 2)

    def test_validation_outliers_are_not_absorbed_by_the_scaler(self):
        # Five constant blocks; the two that land in val are six orders of magnitude
        # larger. If the scaler had seen them, its IQR would swallow the gap and the
        # val rows would scale to about 1. Fit on train only, they stay enormous.
        block_len = 4
        volume = np.concatenate(
            [np.full(block_len, v) for v in (1.0, 2.0, 3.0, 1e6, 1e6)]
        )
        df = pd.DataFrame({"volume": volume + np.arange(len(volume)) * 0.01})
        labels = np.zeros(len(df), dtype=int)
        pipeline = TransactionPipeline(seq_len=block_len, features=["volume"])
        (X_tr, _), (X_va, _) = pipeline.process_blocks(
            df, labels, block_len=block_len, test_size=0.4, rng=_ReversedOrder()
        )
        assert np.abs(X_tr).max() < 10
        assert np.abs(X_va).min() > 100

    def test_single_privacy_release_covers_both_splits(self):
        # The budget equals the per-column epsilon, so a second release for the
        # val half would raise BudgetExhausted. Both halves come from one draw.
        df, labels = _blocks_frame(8, 2, block_len=60)
        pipeline = TransactionPipeline(
            seq_len=50,
            features=["volume", "price"],
            privacy=_laplace_price(eps_total=1.0, eps=1.0),
            privacy_rng=_ZeroNoise(),
        )
        pipeline.process_blocks(
            df, labels, block_len=60, test_size=0.2, rng=np.random.default_rng(0)
        )
        assert pipeline.privacy_manager.budget.spent == pytest.approx(1.0)

    def test_frame_not_divisible_into_blocks_is_rejected(self):
        df, labels = _blocks_frame(3, 0, block_len=20)
        pipeline = TransactionPipeline(seq_len=10, features=["volume"])
        with pytest.raises(ValueError, match="block_len"):
            pipeline.process_blocks(
                df.iloc[:-1],
                labels[:-1],
                block_len=20,
                test_size=0.2,
                rng=np.random.default_rng(0),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

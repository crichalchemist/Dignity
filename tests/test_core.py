"""Test core utilities."""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.config import (
    CONFIRMATION_SEEDS,
    SELECTION_SEEDS,
    TUNING_SEEDS,
    DataConfig,
    DignityConfig,
    EvalConfig,
    PolicyConfig,
    PrivacyConfig,
    PrivacyFeature,
    SyntheticConfig,
    TrainConfig,
)
from core.signals import SignalProcessor
from data.pipeline import SIGNAL_SOURCES, TransactionPipeline
from data.source.anomalies import ANOMALY_TYPES
from data.source.synthetic import (
    RAW_COLUMNS,
    SyntheticGenerator,
    public_log1p_volume_bounds,
)
from train.loop import prepare_data

ROOT = Path(__file__).resolve().parents[1]

META = {
    "timestamp",
    "block_id",
    "is_anomalous",
    "anomaly_type",
    "anomaly_id",
    "block_label",
}


def _consumed_raw_columns(features, df):
    """Raw source columns the model reads: a derived signal counts as its source."""
    raw = df.drop(columns=[c for c in META if c in df.columns])
    p = TransactionPipeline(seq_len=8, features=features).fit(raw)
    return {SIGNAL_SOURCES.get(f, f) for f in p.available_features}


class TestSignalProcessor:
    """Test signal processing functions."""

    def test_volatility(self):
        """Test volatility calculation."""
        values = np.array([100, 102, 98, 101, 99, 103])
        vol = SignalProcessor.volatility(values, window=3)

        assert len(vol) == len(values)
        assert vol[0] == 0.0
        assert vol[-1] > 0  # Should have volatility

    def test_volatility_never_reads_the_future(self):
        v = np.arange(50, dtype=float)
        base = SignalProcessor.volatility(v, window=20)
        for t in (0, 5, 19, 20, 35):
            w = v.copy()
            w[t + 1 :] += 1000.0 * np.random.default_rng(t).standard_normal(49 - t)
            assert np.array_equal(
                SignalProcessor.volatility(w, window=20)[: t + 1], base[: t + 1]
            )
        assert base[0] == 0.0
        assert base[10] == pytest.approx(np.std(v[0:10]))  # expanding warm-up
        assert base[30] == pytest.approx(np.std(v[10:30]))  # causal window unchanged

    def test_entropy(self):
        """Test entropy calculation."""
        # Uniform distribution should have high entropy
        uniform = np.random.uniform(0, 100, 1000)
        uniform_entropy = SignalProcessor.entropy(uniform)

        # Constant values should have zero entropy
        constant = np.ones(1000)
        constant_entropy = SignalProcessor.entropy(constant)

        assert uniform_entropy > constant_entropy
        assert constant_entropy == 0.0

    def test_price_momentum(self):
        """Test price momentum calculation."""
        prices = np.array([100.0, 105.0, 110.0, 115.0, 120.0])
        momentum = SignalProcessor.price_momentum(prices, window=2)

        assert len(momentum) == len(prices)
        # Check that momentum is computed (non-zero after window)
        assert np.any(momentum != 0)

    def test_directional_change(self):
        """Test directional change detection."""
        # Upward trend
        prices = np.array([100, 102, 105, 103, 106])
        dc = SignalProcessor.directional_change(prices, threshold=0.015)

        assert len(dc) == len(prices)
        assert np.sum(dc == 1) > 0  # Should have upward changes

    def test_regime_detection(self):
        """Test regime detection."""
        # Create volatility with different regimes
        vol = np.concatenate(
            [
                np.ones(100) * 0.5,  # Low vol
                np.ones(100) * 2.0,  # High vol
                np.ones(100) * 1.0,  # Normal vol
            ]
        )

        regimes = SignalProcessor.regime_detection(vol)

        assert len(regimes) == len(vol)
        assert np.any(regimes == 0)  # Low vol regime
        assert np.any(regimes == 2)  # High vol regime


class TestDignityConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration."""
        config = DignityConfig()

        assert config.model.task == "risk"
        assert config.model.hidden_size == 256
        assert config.data.seq_len == 100
        assert config.train.epochs == 50

    def test_config_yaml_roundtrip(self, tmp_path):
        """Test save/load config."""
        config1 = DignityConfig()
        config1.model.hidden_size = 512

        yaml_path = tmp_path / "test_config.yaml"
        config1.to_yaml(str(yaml_path))

        config2 = DignityConfig.from_yaml(str(yaml_path))

        assert config2.model.hidden_size == 512
        assert config2.data.seq_len == config1.data.seq_len

    def test_no_shipped_config_can_overwrite_the_january_checkpoints(self):
        # checkpoints/risk/ holds untracked, irreplaceable checkpoints named exactly
        # like what train_risk.yaml writes. Training must never resolve into it.
        root = Path(__file__).resolve().parents[1]
        guarded = (root / "checkpoints").resolve()
        dirs = {"<dataclass default>": DignityConfig().train.checkpoint_dir}
        for path in sorted((root / "config").glob("*.yaml")):
            dirs[path.name] = DignityConfig.from_yaml(str(path)).train.checkpoint_dir
        # Not an exact count: M2-M4 add presets; every *.yaml is scanned whatever it is.
        assert {"base.yaml", "colab.yaml", "train_risk.yaml"} <= set(dirs)
        for name, directory in dirs.items():
            resolved = (root / directory).resolve()  # absolute paths ignore root
            assert not resolved.is_relative_to(guarded), name
            if not Path(directory).is_absolute():
                assert resolved.is_relative_to((root / "runs").resolve()), name

    def test_runs_dir_is_gitignored(self):
        root = Path(__file__).resolve().parents[1]
        lines = (root / ".gitignore").read_text().splitlines()
        assert "runs/" in lines and "checkpoints/" in lines

    def test_unknown_top_level_key_is_rejected(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump({"model": {}, "trainer": {"epochs": 1}}))
        with pytest.raises(TypeError, match="trainer"):
            DignityConfig.from_yaml(str(p))

    # guards the unknown-key and data.source checks against every shipped preset
    def test_every_shipped_config_still_loads(self):
        paths = sorted((Path(__file__).parent.parent / "config").glob("*.yaml"))
        assert paths
        for p in paths:
            DignityConfig.from_yaml(str(p))

    def test_unsupported_data_source_is_rejected_not_silently_synthetic(self):
        with pytest.raises(ValueError, match="crypto"):
            DataConfig(source="crypto")


def _valid_privacy_block():
    return {
        "epsilon_total": 1.0,
        "k": 5,
        "key_env": None,
        "features": {
            "volume": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 1000]},
            "price": {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 500]},
            "fee_rate": {"mechanism": "generalize", "bins": 10},
        },
    }


class TestPrivacyConfig:
    """Misconfiguration fails at load time, and absence means no privacy stage."""

    def test_absent_block_yields_none(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({"model": {"task": "risk"}}))
        assert DignityConfig.from_yaml(str(path)).privacy is None

    def test_present_but_empty_block_is_rejected(self, tmp_path):
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.safe_dump({"privacy": {}}))
        with pytest.raises(TypeError):
            DignityConfig.from_yaml(str(path))

    def test_valid_block_round_trips_through_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({"privacy": _valid_privacy_block()}))
        cfg = DignityConfig.from_yaml(str(path))
        assert cfg.privacy is not None
        assert cfg.privacy.features["volume"].bounds == (0.0, 1000.0)

        out = tmp_path / "out.yaml"
        cfg.to_yaml(str(out))
        again = DignityConfig.from_yaml(str(out))
        assert again.privacy.to_dict() == cfg.privacy.to_dict()

    def test_deterministic_rng_round_trips_and_defaults_off(self, tmp_path):
        block = {**_valid_privacy_block(), "deterministic_rng": True}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump({"privacy": block}))
        cfg = DignityConfig.from_yaml(str(path))
        assert cfg.privacy.deterministic_rng is True
        cfg.to_yaml(str(tmp_path / "o.yaml"))
        again = DignityConfig.from_yaml(str(tmp_path / "o.yaml"))
        assert again.privacy.deterministic_rng is True
        assert PrivacyConfig(**_valid_privacy_block()).deterministic_rng is False

    def test_runs_differing_only_in_rng_mode_are_told_apart(self):
        # load_backbone compares to_dict(); a seeded run must not pass as a DP run.
        seeded = PrivacyConfig(**{**_valid_privacy_block(), "deterministic_rng": True})
        real = PrivacyConfig(**_valid_privacy_block())
        assert seeded.to_dict() != real.to_dict()

    def test_checkpoint_privacy_meta_without_the_key_still_loads_as_off(self):
        old_meta = _valid_privacy_block()  # written before the field existed
        assert PrivacyConfig(**old_meta).deterministic_rng is False

    def test_shipped_configs_match_spec(self):
        root = Path(__file__).resolve().parents[1] / "config"
        assert (
            DignityConfig.from_yaml(str(root / "train_risk.yaml")).privacy is not None
        )
        assert DignityConfig.from_yaml(str(root / "base.yaml")).privacy is None

    def test_shipped_configs_stay_on_system_randomness(self):
        root = Path(__file__).resolve().parents[1] / "config"
        for name in ("train_risk.yaml", "train_policy.yaml"):
            cfg = DignityConfig.from_yaml(str(root / name))
            assert cfg.privacy.deterministic_rng is False, name

    def test_shipped_risk_bounds_are_the_public_generator_bounds(self):
        root = Path(__file__).resolve().parents[1] / "config"
        cfg = DignityConfig.from_yaml(str(root / "train_risk.yaml"))
        lo, hi = public_log1p_volume_bounds(cfg.data.synthetic)
        b = cfg.privacy.features["volume"].bounds
        assert cfg.privacy.features["volume"].transform == "log1p"
        assert b[0] <= lo < b[0] + 0.01 and b[1] - 0.01 < hi <= b[1]
        assert "price" not in cfg.privacy.features

    def test_shipped_block_spends_all_epsilon_on_volume_and_generalizes_the_rest(
        self,
    ):
        root = Path(__file__).resolve().parents[1] / "config"
        cfg = DignityConfig.from_yaml(str(root / "train_risk.yaml"))
        assert cfg.data.features == [
            "volume",
            "fee_rate",
            "tx_count",
            "volume_volatility",
        ]
        f = cfg.privacy.features
        assert cfg.privacy.epsilon_total == 1.0 and cfg.privacy.k == 5
        assert f["volume"].mechanism == "laplace" and f["volume"].epsilon == 1.0
        for name in ("fee_rate", "tx_count"):
            assert f[name].mechanism == "generalize" and f[name].bins == 10

    def test_shipped_risk_config_covers_every_raw_model_input(self):
        root = Path(__file__).resolve().parents[1] / "config"
        cfg = DignityConfig.from_yaml(str(root / "train_risk.yaml"))
        df = SyntheticGenerator(
            dataclasses.replace(cfg.data.synthetic, n_blocks=2)
        ).generate()
        consumed = _consumed_raw_columns(cfg.data.features, df)
        assert consumed & set(RAW_COLUMNS), (
            "shipped features must include at least one raw source column"
        )
        assert consumed <= set(cfg.privacy.features)

    @pytest.mark.parametrize("signal", ["volatility", "momentum", "directional_change"])
    def test_coverage_check_sees_price_behind_derived_signals(self, signal):
        # DGNTY-28: a price signal consumes price even though "price" is not listed
        df = SyntheticGenerator(
            dataclasses.replace(SyntheticConfig(), n_blocks=2)
        ).generate()
        assert "price" in _consumed_raw_columns(["volume", signal], df)

    @pytest.mark.parametrize(
        "mutate,match",
        [
            (
                lambda b: b["features"]["volume"].update(mechanism="gaussian"),
                "unknown mechanism",
            ),
            (lambda b: b["features"]["volume"].update(epsilon=0.0), "epsilon > 0"),
            (lambda b: b["features"]["volume"].pop("bounds"), "requires bounds"),
            (lambda b: b["features"]["volume"].update(bounds=[10, 10]), "lo < hi"),
            (lambda b: b["features"]["fee_rate"].update(epsilon=0.1), "bins only"),
            (lambda b: b["features"]["fee_rate"].update(bounds=[0, 1]), "bins only"),
            (lambda b: b.update(k=1), "k must be"),
            (lambda b: b["features"]["fee_rate"].update(bins=1), "bins must be"),
            (
                lambda b: b["features"]["volume"].update(epsilon=0.9),
                "exceeds epsilon_total",
            ),
            (
                lambda b: b["features"]["volume"].update(transform="sqrt"),
                "unknown transform",
            ),
            (
                lambda b: b["features"]["fee_rate"].update(transform="log1p"),
                "laplace only",
            ),
            (
                lambda b: b["features"]["volume"].update(
                    transform="log1p", bounds=[-1, 5]
                ),
                "log1p bounds",
            ),
        ],
    )
    def test_rejects_bad_privacy_block(self, mutate, match):
        block = _valid_privacy_block()
        mutate(block)
        with pytest.raises(ValueError, match=match):
            PrivacyConfig(**block)

    def test_transform_is_recorded_so_runs_differing_only_in_it_are_told_apart(self):
        # load_backbone compares to_dict(); a log-space run must not warm-start
        # from a linear-space one with the same bounds.
        plain = {"mechanism": "laplace", "epsilon": 0.5, "bounds": [0, 5]}
        logged = PrivacyFeature(**plain, transform="log1p").to_dict()
        assert logged["transform"] == "log1p"
        assert logged != PrivacyFeature(**plain).to_dict()

    def test_configs_without_a_transform_key_load_as_no_transform(self):
        feat = PrivacyFeature(mechanism="laplace", epsilon=0.5, bounds=(0, 5))
        assert feat.transform is None
        assert "transform" not in feat.to_dict()


def _write(tmp_path, **sections):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(sections))
    return str(path)


class TestSyntheticConfig:
    def test_smoke_preset_exercises_build_ups_and_decoys(self):
        # CI smoke trains on build-up data only now that the strength is committed:
        # it carries the hard preset's strength and holds true build-ups and decoys.
        cfg = DignityConfig.from_yaml(str(ROOT / "config/smoke.yaml"))
        synthetic = cfg.data.synthetic
        hard = DignityConfig.from_yaml(str(ROOT / "config/synthetic_hard.yaml"))
        assert synthetic.precursor_strength == hard.data.synthetic.precursor_strength
        assert synthetic.decoy_rate == hard.data.synthetic.decoy_rate
        frame = SyntheticGenerator(synthetic).generate()
        assert set(frame["build_up"]) == {0, 1, 2}

    def test_smoke_preset_val_and_test_each_hold_both_classes(self):
        # risk_metrics raises on a single-class split (M3.2); CI must never hit that.
        cfg = DignityConfig.from_yaml(str(ROOT / "config/smoke.yaml"))
        _, d = prepare_data(cfg)
        for ws in (d.train, d.val, d.test):
            assert set(np.unique(ws.y_risk)) == {0, 1}

    def test_smoke_preset_scores_onsets_at_every_horizon_in_every_split(self):
        # block_len must leave enough scorable rows (seq_len .. L-H) for each
        # split to hold forecast positives; otherwise every smoke forecast run
        # has no val metric and fit() refuses to select a checkpoint.
        cfg = DignityConfig.from_yaml(str(ROOT / "config/smoke.yaml"))
        _, d = prepare_data(cfg)
        last = cfg.model.horizon - 1
        for ws in (d.train, d.val, d.test):
            assert ws.y_forecast[ws.forecast_mask, last].any()
        for ws in (d.val, d.test):
            assert (ws.y_forecast[:, 0] == 1).any()

    def test_synthetic_block_loads_with_tuple_ranges(self, tmp_path):
        cfg = DignityConfig.from_yaml(
            _write(tmp_path, data={"synthetic": {"duration_range": [5, 9]}})
        )
        assert isinstance(cfg.data.synthetic, SyntheticConfig)
        assert cfg.data.synthetic.duration_range == (5, 9)
        assert cfg.data.split == (0.45, 0.15, 0.40)

    def test_unknown_synthetic_key_is_rejected(self, tmp_path):
        with pytest.raises(TypeError, match="nope"):
            DignityConfig.from_yaml(_write(tmp_path, data={"synthetic": {"nope": 1}}))

    def test_precursor_fields_default_to_off_and_the_design_length_range(self):
        cfg = SyntheticConfig()
        assert cfg.precursor_strength == 0.0
        assert cfg.precursor_len_range == (10, 30)
        assert cfg.decoy_rate == 0.0

    @pytest.mark.parametrize("field_name", ["precursor_strength", "decoy_rate"])
    def test_precursor_strength_and_decoy_rate_must_lie_in_the_unit_interval(
        self, field_name
    ):
        for bad in (-0.1, 1.01):
            with pytest.raises(ValueError, match=field_name):
                SyntheticConfig(**{field_name: bad})
        for ok in (0.0, 1.0):
            assert getattr(SyntheticConfig(**{field_name: ok}), field_name) == ok

    @pytest.mark.parametrize("bad", [(3, 10), (30, 10), (10,), (10, 20, 30)])
    def test_precursor_length_range_must_be_two_ints_from_four_up_in_order(self, bad):
        with pytest.raises(ValueError, match="precursor_len_range"):
            SyntheticConfig(precursor_len_range=bad)
        assert SyntheticConfig(precursor_len_range=(4, 4)).precursor_len_range == (4, 4)

    def test_precursor_length_range_loads_from_yaml_as_a_tuple(self, tmp_path):
        cfg = DignityConfig.from_yaml(
            _write(tmp_path, data={"synthetic": {"precursor_len_range": [12, 20]}})
        )
        assert cfg.data.synthetic.precursor_len_range == (12, 20)

    def test_block_must_hold_a_window_plus_the_horizon(self):
        with pytest.raises(ValueError, match="block_len"):
            DignityConfig(
                data=DataConfig(seq_len=100, synthetic=SyntheticConfig(block_len=110))
            )

    def test_split_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="split"):
            DataConfig(split=(0.7, 0.2, 0.2))

    def test_split_must_name_train_val_and_test(self):
        # [0.8, 0.2] sums to 1 but leaves process_blocks no test fraction
        with pytest.raises(ValueError, match="data.split"):
            DataConfig(split=[0.8, 0.2])
        with pytest.raises(ValueError, match="data.split"):
            DataConfig(split=[0.9, 0.1, 0.0])

    def test_default_type_weights_cover_every_anomaly_type(self):
        assert set(SyntheticConfig().type_weights) == set(ANOMALY_TYPES)

    def test_config_with_synthetic_block_round_trips_through_yaml(self, tmp_path):
        cfg = DignityConfig()
        cfg.to_yaml(str(tmp_path / "c.yaml"))
        assert DignityConfig.from_yaml(str(tmp_path / "c.yaml")) == cfg


class TestEvalConfig:
    def test_tuning_seeds_split_into_eight_selection_and_two_confirmation_seeds(self):
        assert tuple(range(1000, 1010)) == TUNING_SEEDS
        assert SELECTION_SEEDS + CONFIRMATION_SEEDS == TUNING_SEEDS
        assert (len(SELECTION_SEEDS), CONFIRMATION_SEEDS) == (8, (1008, 1009))
        c = EvalConfig()  # the held-out seed can never be tuned on
        assert c.tuning_seed in TUNING_SEEDS and c.eval_seed not in TUNING_SEEDS

    def test_eval_seeds_default_to_frozen_tuning_and_evaluation_values(self):
        # The tuning and evaluation seeds must differ, or the generator gets tuned
        # against the numbers it is graded on (spec §2 anti-goal).
        c = DignityConfig()
        assert (c.eval.tuning_seed, c.eval.eval_seed) == (1000, 2026)
        assert c.eval.n_train_seeds == 3 and c.eval.n_bootstrap == 1000
        assert c.eval.out_dir == "results/v1.0" and c.eval.img_dir == "docs/img"
        assert c.train.select_metric == "pr_auc" and c.train.pos_weight is None
        assert (
            c.train.chance_guard_auroc,
            c.train.guard_epochs,
            c.train.patience,
        ) == (0.55, 5, 10)

    def test_eval_block_in_yaml_overrides_defaults(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("eval:\n  tuning_seed: 7\n  n_bootstrap: 50\n")
        c = DignityConfig.from_yaml(str(p))
        assert c.eval.tuning_seed == 7 and c.eval.n_bootstrap == 50

    def test_base_yaml_states_every_eval_and_selection_default(self):
        c = DignityConfig.from_yaml("config/base.yaml")
        assert c.eval == EvalConfig()
        assert c.train.patience == 10 and c.train.chance_guard_auroc == 0.55

    def test_base_yaml_spells_out_eval_block_and_selection_keys_raw(self):
        # The dataclass fills defaults, so a key missing from base.yaml would pass
        # the test above; read the file raw so every key must be written down.
        raw = yaml.safe_load(Path("config/base.yaml").read_text())
        assert set(raw["eval"]) == {
            "tuning_seed",
            "eval_seed",
            "n_train_seeds",
            "n_bootstrap",
            "epsilons",
            "out_dir",
            "img_dir",
        }
        for key in (
            "select_metric",
            "pos_weight",
            "chance_guard_auroc",
            "guard_epochs",
            "patience",
            "init_backbone_from",
        ):
            assert key in raw["train"], key

    def test_shipped_policy_config_trains_the_policy_task(self):
        root = Path(__file__).resolve().parents[1]
        c = DignityConfig.from_yaml(str(root / "config" / "train_policy.yaml"))
        assert c.model.task == "policy" and c.train.checkpoint_dir == "./runs/policy"
        assert c.train.init_backbone_from == "./runs/risk/best.pt"
        assert c.policy.review_budget == 0.05

    def test_shipped_risk_and_policy_configs_preprocess_identically(self):
        # the documented flow warm-starts policy from the risk run; load_backbone
        # refuses a checkpoint fitted to other preprocessing, so the pair must match
        root = Path(__file__).resolve().parents[1]
        risk = DignityConfig.from_yaml(str(root / "config" / "train_risk.yaml"))
        policy = DignityConfig.from_yaml(str(root / "config" / "train_policy.yaml"))
        assert risk.privacy is not None
        assert policy.privacy == risk.privacy
        assert policy.data == risk.data

    def test_eval_block_survives_a_yaml_round_trip(self, tmp_path):
        # to_yaml must write eval:, or a saved run config silently reverts the
        # seeds and output dirs to defaults on reload.
        cfg = DignityConfig(
            eval=EvalConfig(tuning_seed=7, n_bootstrap=50, epsilons=[0.5, 2.0]),
            train=TrainConfig(select_metric="auroc", patience=None),
        )
        path = str(tmp_path / "c.yaml")
        cfg.to_yaml(path)
        assert "eval" in yaml.safe_load(Path(path).read_text())
        back = DignityConfig.from_yaml(path)
        assert back == cfg
        assert back.eval.tuning_seed == 7 and back.eval.epsilons == [0.5, 2.0]

    def test_unknown_select_metric_is_rejected_before_it_picks_a_bad_checkpoint(
        self,
    ):
        # "brier" is lower-is-better; under strict > selection it would keep the
        # worst epoch, so only higher-is-better metrics are allowed.
        with pytest.raises(ValueError, match="pr_auc"):
            TrainConfig(select_metric="brier")
        for ok in ("pr_auc", "auroc", "recall_at_1pct_fpr"):
            assert TrainConfig(select_metric=ok).select_metric == ok


class TestTrainConfigForecast:
    def test_forecast_defaults_cap_nothing_warm_start_and_require_eligibility(self):
        t = TrainConfig()
        assert (
            t.forecast_pos_weight_cap,
            t.forecast_warm_start,
            t.forecast_eligibility,
        ) == (None, True, True)

    def test_a_non_positive_positive_weight_cap_is_rejected(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="forecast_pos_weight_cap"):
                TrainConfig(forecast_pos_weight_cap=bad)
        assert TrainConfig(forecast_pos_weight_cap=10.0).forecast_pos_weight_cap == 10.0

    def test_every_shipped_config_gates_forecast_epochs_except_the_smoke_preset(self):
        # a gate switched off anywhere else would ship unnoticed
        for p in sorted((ROOT / "config").glob("*.yaml")):
            cfg = DignityConfig.from_yaml(str(p))
            assert cfg.train.forecast_eligibility is (p.name != "smoke.yaml"), p.name

    def test_base_yaml_spells_out_the_forecast_keys(self):
        raw = yaml.safe_load((ROOT / "config/base.yaml").read_text())
        for key in (
            "forecast_pos_weight_cap",
            "forecast_warm_start",
            "forecast_eligibility",
        ):
            assert key in raw["train"], key

    def test_no_shipped_config_still_calls_the_warm_start_policy_only(self):
        paths = [*sorted((ROOT / "config").glob("*.yaml")), ROOT / "core/config.py"]
        for p in paths:
            assert "policy only" not in p.read_text(), p.name


class TestPolicyConfig:
    def test_default_cost_table_is_the_spec_table(self):
        c = DignityConfig().policy
        assert c.review_budget == 0.05
        assert c.costs == {
            "allow": {"anomaly": 10.0, "normal": 0.0},
            "review": {"anomaly": 1.0, "normal": 1.0},
            "block": {"anomaly": 0.5, "normal": 3.0},
        }
        assert DignityConfig().model.n_actions == 3  # == len(ACTIONS)

    def test_policy_block_round_trips_through_yaml(self, tmp_path):
        # Mutate first: the default reloads equal even if to_yaml drops policy.
        cfg = DignityConfig()
        cfg.policy.review_budget = 0.09
        cfg.policy.costs["block"]["normal"] = 2.0
        path = str(tmp_path / "c.yaml")
        cfg.to_yaml(path)
        assert "policy" in yaml.safe_load(Path(path).read_text())
        back = DignityConfig.from_yaml(path)
        assert back.policy == cfg.policy
        assert back.policy.review_budget == 0.09
        assert back.policy.costs["block"]["normal"] == 2.0

    @pytest.mark.parametrize(
        ("factor", "budget"), [(20.0, 0.05), (0.0, 0.05), (10.0, 0.2)]
    )
    def test_the_collapse_threshold_must_leave_room_below_one(self, factor, budget):
        # a threshold of 1.0 or more can never be exceeded: the guard would be dead
        with pytest.raises(ValueError, match="collapse_review_factor"):
            PolicyConfig(review_budget=budget, collapse_review_factor=factor)
        assert PolicyConfig(collapse_review_factor=19.9).collapse_review_factor == 19.9

    def test_the_review_multiplier_step_must_be_positive_and_its_floor_not_above_zero(
        self,
    ):
        with pytest.raises(ValueError, match="multiplier_lr"):
            PolicyConfig(multiplier_lr=0)
        with pytest.raises(ValueError, match="multiplier_floor"):
            PolicyConfig(multiplier_floor=0.1)
        assert PolicyConfig(multiplier_floor=0.0).multiplier_floor == 0.0

    def test_at_least_one_saturated_epoch_must_trigger_the_collapse_guard(self):
        with pytest.raises(ValueError, match="collapse_epochs"):
            PolicyConfig(collapse_epochs=0)
        assert PolicyConfig(collapse_epochs=1).collapse_epochs == 1  # one is enough
        c = PolicyConfig()
        assert (c.multiplier_lr, c.multiplier_floor, c.collapse_epochs) == (
            0.1,
            -0.5,
            3,
        )

    def test_base_yaml_spells_out_policy_block_and_n_actions_raw(self):
        # The dataclass fills defaults, so a key missing from base.yaml would
        # still load; read the raw file.
        raw = yaml.safe_load(Path("config/base.yaml").read_text())
        assert set(raw["policy"]) == {
            "costs",
            "review_budget",
            "collapse_review_factor",
            "multiplier_lr",
            "multiplier_floor",
            "collapse_epochs",
        }
        assert "n_actions" in raw["model"]
        assert DignityConfig.from_yaml("config/base.yaml").policy == PolicyConfig()

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda d: d["costs"].pop("review"), "actions"),
            (lambda d: d["costs"]["block"].pop("normal"), "anomaly"),
            (lambda d: d.update(review_budget=0.0), "review_budget"),
            (lambda d: d.update(review_budget=1.5), "review_budget"),
        ],
    )
    def test_rejects_a_cost_table_that_cannot_price_every_decision(self, mutate, match):
        d = {
            "costs": {
                "allow": {"anomaly": 10.0, "normal": 0.0},
                "review": {"anomaly": 1.0, "normal": 1.0},
                "block": {"anomaly": 0.5, "normal": 3.0},
            },
            "review_budget": 0.05,
        }
        mutate(d)
        with pytest.raises(ValueError, match=match):
            PolicyConfig(**d)


class TestShippedEvaluationSize:
    def test_every_shipped_config_splits_blocks_45_15_40(self):
        for path in sorted((ROOT / "config").glob("*.yaml")):
            split = DignityConfig.from_yaml(path).data.split
            assert split == (0.45, 0.15, 0.40), path.name

    def test_presets_hold_3600_blocks_so_every_tuning_seed_has_over_200_test_onsets(
        self,
    ):
        for name in ("easy", "medium", "hard"):
            path = ROOT / "config" / f"synthetic_{name}.yaml"
            syn = DignityConfig.from_yaml(path).data.synthetic
            assert (syn.n_blocks, syn.block_len) == (3600, 160), name
            assert (syn.decoy_rate, syn.precursor_len_range) == (0.2, (10, 30)), name

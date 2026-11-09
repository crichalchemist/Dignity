"""Test core utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core.config import (
    DataConfig,
    DignityConfig,
    EvalConfig,
    PolicyConfig,
    PrivacyConfig,
    SyntheticConfig,
    TrainConfig,
)
from core.signals import SignalProcessor
from data.source.anomalies import ANOMALY_TYPES
from data.source.synthetic import RAW_COLUMNS, SyntheticGenerator
from train.loop import prepare_data

ROOT = Path(__file__).resolve().parents[1]


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

    def test_shipped_configs_match_spec(self):
        root = Path(__file__).resolve().parents[1] / "config"
        assert (
            DignityConfig.from_yaml(str(root / "train_risk.yaml")).privacy is not None
        )
        assert DignityConfig.from_yaml(str(root / "base.yaml")).privacy is None

    def test_shipped_risk_config_covers_every_raw_model_input(self):
        root = Path(__file__).resolve().parents[1] / "config"
        cfg = DignityConfig.from_yaml(str(root / "train_risk.yaml"))
        raw_columns = set(RAW_COLUMNS)
        raw_inputs = raw_columns & set(cfg.data.features)
        assert raw_inputs, (
            "shipped features must include at least one raw source column"
        )
        assert raw_inputs <= set(cfg.privacy.features)

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
        ],
    )
    def test_rejects_bad_privacy_block(self, mutate, match):
        block = _valid_privacy_block()
        mutate(block)
        with pytest.raises(ValueError, match=match):
            PrivacyConfig(**block)


def _write(tmp_path, **sections):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(sections))
    return str(path)


class TestSyntheticConfig:
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
        assert cfg.data.split == (0.7, 0.15, 0.15)

    def test_unknown_synthetic_key_is_rejected(self, tmp_path):
        with pytest.raises(TypeError, match="nope"):
            DignityConfig.from_yaml(_write(tmp_path, data={"synthetic": {"nope": 1}}))

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
        ):
            assert key in raw["train"], key

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
        cfg.policy.review_budget = 0.1
        cfg.policy.costs["block"]["normal"] = 2.0
        path = str(tmp_path / "c.yaml")
        cfg.to_yaml(path)
        assert "policy" in yaml.safe_load(Path(path).read_text())
        back = DignityConfig.from_yaml(path)
        assert back.policy == cfg.policy
        assert back.policy.review_budget == 0.1
        assert back.policy.costs["block"]["normal"] == 2.0

    def test_base_yaml_spells_out_policy_block_and_n_actions_raw(self):
        # The dataclass fills defaults, so a key missing from base.yaml would
        # still load; read the raw file.
        raw = yaml.safe_load(Path("config/base.yaml").read_text())
        assert set(raw["policy"]) == {"costs", "review_budget"}
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

"""Configuration management for Dignity models and training."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    task: str = "risk"  # risk, forecast, policy
    input_size: int = 9
    hidden_size: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    cnn_kernel_size: int = 3
    horizon: int = 10  # forecast H; also reserved at the end of every block
    n_actions: int = 3  # policy; must equal len(ACTIONS)


_SOURCES = ("synthetic",)

# core/ must not import data/ (data/pipeline.py imports core.config), so the
# six anomaly names are repeated here; tests pin them to ANOMALY_TYPES.
_ANOMALY_NAMES = (
    "wash_trading",
    "pump_and_dump",
    "fee_spike",
    "flash_crash",
    "layering",
    "fee_evasion",
)


@dataclass
class SyntheticConfig:
    """Event-record generator parameters (data.synthetic)."""

    n_blocks: int = 1000
    block_len: int = 160  # must be >= data.seq_len + model.horizon + 1
    contamination: float = 0.3  # P(block contains >= 1 anomaly)
    anomalies_per_block: tuple[int, int] = (1, 3)  # inclusive, anomalous blocks
    duration_range: tuple[int, int] = (3, 60)  # rows, inclusive
    severity_range: tuple[float, float] = (0.3, 1.0)  # 0 invisible, 1 preset max
    type_weights: dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(_ANOMALY_NAMES, 1.0)
    )
    benign_spike_rate: float = 0.0  # P(normal block gets an unlabelled burst)
    label_noise: float = 0.0  # P(block label flipped)
    regime_switch_prob: float = 0.0  # per-row P(Markov regime change)
    precursor_strength: float = 0.0  # build-up intensity, 0 off, 1 the anomaly's own
    precursor_len_range: tuple[int, int] = (10, 30)  # rows of a build-up, inclusive
    decoy_rate: float = 0.0  # P(a block holds one build-up that nothing follows)
    seed: int = 42

    def __post_init__(self):
        self.anomalies_per_block = tuple(int(x) for x in self.anomalies_per_block)
        self.duration_range = tuple(int(x) for x in self.duration_range)
        self.severity_range = tuple(float(x) for x in self.severity_range)
        self.precursor_len_range = tuple(int(x) for x in self.precursor_len_range)
        for name in ("precursor_strength", "decoy_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name}={getattr(self, name)} must lie in [0, 1]")
        lo_hi = self.precursor_len_range
        if len(lo_hi) != 2 or not 4 <= lo_hi[0] <= lo_hi[1]:
            raise ValueError(
                f"precursor_len_range={lo_hi} must be two ints with 4 <= lo <= hi"
            )
        # only a decoy needs a whole ramp inside one block (a true ramp is clipped)
        if (
            self.precursor_strength > 0
            and self.decoy_rate > 0
            and lo_hi[1] > self.block_len
        ):
            raise ValueError(
                f"precursor_len_range={lo_hi} must not exceed block_len="
                f"{self.block_len} while decoy_rate > 0: a decoy has to fit in a block"
            )


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # only "synthetic" in v1.0; other sources are v1.1 (DGNTY-41)
    source: str = "synthetic"
    seq_len: int = 100
    batch_size: int = 64
    num_workers: int = 4
    features: list = field(
        default_factory=lambda: [
            "volume",
            "fee_rate",
            "entropy",
            "tx_count",
            "volatility",
            "price_change",
            "momentum",
            "directional_change",
            "regime",
        ]
    )
    split: tuple[float, float, float] = (0.45, 0.15, 0.40)  # train / val / test blocks
    label_overlap: float = 0.0  # risk positive if overlap fraction > this
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)

    def __post_init__(self):
        if self.source not in _SOURCES:
            raise ValueError(
                f"data.source={self.source!r} is not supported in v1.0; "
                f"expected one of {_SOURCES}"
            )
        self.split = tuple(float(x) for x in self.split)
        if len(self.split) != 3 or min(self.split) <= 0:
            raise ValueError(
                f"data.split={self.split} must be 3 positive fractions "
                "(train, val, test)"
            )
        if abs(sum(self.split) - 1) >= 1e-9:
            raise ValueError(f"data.split={self.split} must sum to 1")
        if isinstance(self.synthetic, dict):
            self.synthetic = SyntheticConfig(**self.synthetic)


# higher is better for all of these: best.pt is kept on a strict ">"
_SELECT_METRICS = ("pr_auc", "auroc", "recall_at_1pct_fpr")


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    gradient_clip: float = 1.0
    checkpoint_dir: str = "./runs"
    save_interval: int = 5
    select_metric: str = "pr_auc"  # best.pt = highest val value of this metric
    pos_weight: float | None = None  # None = n_neg / n_pos of train
    chance_guard_auroc: float | None = 0.55  # warn below this at guard_epochs
    guard_epochs: int = 5
    patience: int | None = 10  # early stopping; None disables
    # policy and forecast: risk checkpoint to warm-start the backbone from
    init_backbone_from: str | None = None
    # forecast only: upper limit of the derived positive weight; None = no limit
    forecast_pos_weight_cap: float | None = None
    # forecast only: eval.run warm-starts each seed from the same seed's risk run
    forecast_warm_start: bool = True
    # forecast only: keep an epoch only when its val lift_upper beats the floor
    forecast_eligibility: bool = True

    def __post_init__(self):
        if self.select_metric not in _SELECT_METRICS:
            raise ValueError(
                f"train.select_metric={self.select_metric!r} is not allowed; "
                f"expected one of {_SELECT_METRICS}"
            )
        if (
            self.forecast_pos_weight_cap is not None
            and self.forecast_pos_weight_cap <= 0
        ):
            raise ValueError(
                f"train.forecast_pos_weight_cap={self.forecast_pos_weight_cap} "
                "must be > 0 when set"
            )


TUNING_SEEDS: tuple[int, ...] = tuple(range(1000, 1010))
SELECTION_SEEDS: tuple[int, ...] = tuple(range(1000, 1008))
CONFIRMATION_SEEDS: tuple[int, ...] = (1008, 1009)


@dataclass
class EvalConfig:
    """The `eval:` block: seeds and output locations for the evaluation run."""

    tuning_seed: int = 1000  # frozen; the model never sees eval_seed before M7
    eval_seed: int = 2026  # frozen
    n_train_seeds: int = 3
    n_bootstrap: int = 1000
    epsilons: list[float] = field(
        default_factory=lambda: [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    )
    out_dir: str = "results/v1.0"
    img_dir: str = "docs/img"


_MECHANISMS = ("laplace", "generalize")
_TRANSFORMS = (None, "log1p")


@dataclass
class PrivacyFeature:
    """One column's privacy mechanism. Exactly one of laplace / generalize."""

    mechanism: str
    epsilon: float | None = None
    bounds: tuple[float, float] | None = None
    bins: int = 10
    transform: str | None = None  # public fixed function applied before noise

    def __post_init__(self):
        if self.mechanism not in _MECHANISMS:
            raise ValueError(
                f"unknown mechanism {self.mechanism!r}; expected one of {_MECHANISMS}"
            )
        if self.transform not in _TRANSFORMS:
            raise ValueError(
                f"unknown transform {self.transform!r}; expected one of "
                f"{_TRANSFORMS[1:]}"
            )
        if self.transform is not None and self.mechanism != "laplace":
            raise ValueError("transform applies to laplace only")
        if self.bounds is not None:
            self.bounds = (float(self.bounds[0]), float(self.bounds[1]))
        if self.transform == "log1p" and self.bounds and self.bounds[0] < 0:
            raise ValueError("log1p bounds must be >= 0 (bounds are in log1p units)")
        if self.mechanism == "laplace":
            if self.epsilon is None or self.epsilon <= 0:
                raise ValueError("laplace requires epsilon > 0")
            if self.bounds is None:
                raise ValueError("laplace requires bounds")
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError("bounds must satisfy lo < hi")
        else:
            if self.epsilon is not None or self.bounds is not None:
                raise ValueError("generalize takes bins only, not epsilon or bounds")
            if self.bins < 2:
                raise ValueError("bins must be at least 2")

    def to_dict(self) -> dict:
        d = {"mechanism": self.mechanism}
        if self.mechanism == "laplace":
            d["epsilon"] = self.epsilon
            d["bounds"] = list(self.bounds)
            if self.transform is not None:
                d["transform"] = self.transform
        else:
            d["bins"] = self.bins
        return d


@dataclass
class PrivacyConfig:
    """The `privacy:` block. Absent block => no privacy stage runs."""

    epsilon_total: float
    k: int = 5
    key_env: str | None = None
    deterministic_rng: bool = False  # True = seeded experiment mode, NOT DP-grade
    features: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.epsilon_total <= 0:
            raise ValueError("epsilon_total must be positive")
        if self.k < 2:
            raise ValueError("k must be at least 2")
        self.features = {
            name: (f if isinstance(f, PrivacyFeature) else PrivacyFeature(**f))
            for name, f in self.features.items()
        }
        total = sum(
            f.epsilon for f in self.features.values() if f.mechanism == "laplace"
        )
        if total > self.epsilon_total + 1e-12:
            raise ValueError(
                f"sum of feature epsilons {total} exceeds epsilon_total {self.epsilon_total}"
            )

    def to_dict(self) -> dict:
        return {
            "epsilon_total": self.epsilon_total,
            "k": self.k,
            "key_env": self.key_env,
            "deterministic_rng": self.deterministic_rng,
            "features": {n: f.to_dict() for n, f in self.features.items()},
        }


# core/ must not import models/, so the action names are repeated here; a test
# pins them to models.head.policy.ACTIONS.
_ACTION_NAMES = {"allow", "review", "block"}
_STATE_NAMES = {"anomaly", "normal"}


def _default_costs() -> dict:
    return {
        "allow": {"anomaly": 10.0, "normal": 0.0},
        "review": {"anomaly": 1.0, "normal": 1.0},
        "block": {"anomaly": 0.5, "normal": 3.0},
    }


@dataclass
class PolicyConfig:
    """The `policy:` block: triage costs C[action][true state] and review cap."""

    costs: dict = field(default_factory=_default_costs)
    review_budget: float = 0.05  # share of windows sent to analysts (blocks excluded)
    collapse_review_factor: float = (
        10.0  # review-share multiple of the budget that counts as collapse
    )

    def __post_init__(self):
        if set(self.costs) != _ACTION_NAMES:
            raise ValueError(
                f"costs must price exactly the actions {sorted(_ACTION_NAMES)}, "
                f"got {sorted(self.costs)}"
            )
        for action, row in self.costs.items():
            if set(row) != _STATE_NAMES:
                raise ValueError(
                    f"costs[{action!r}] must price both 'anomaly' and 'normal', "
                    f"got {sorted(row)}"
                )
        self.costs = {
            a: {s: float(c) for s, c in row.items()} for a, row in self.costs.items()
        }
        if not 0 < self.review_budget < 1:
            raise ValueError(
                f"review_budget must satisfy 0 < review_budget < 1, "
                f"got {self.review_budget}"
            )
        if not 0 < self.collapse_review_factor * self.review_budget < 1:
            raise ValueError(
                f"collapse_review_factor must satisfy 0 < collapse_review_factor "
                f"* review_budget < 1, got {self.collapse_review_factor} "
                f"* {self.review_budget}"
            )


def _yaml_safe(obj) -> dict:
    """asdict with tuples turned into lists, so safe_load can read the dump."""

    def convert(value):
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    return convert(asdict(obj))


@dataclass
class DignityConfig:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    privacy: PrivacyConfig | None = None
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        need = self.data.seq_len + self.model.horizon + 1
        if self.data.synthetic.block_len < need:
            raise ValueError(
                f"block_len={self.data.synthetic.block_len} must be >= seq_len + "
                f"horizon + 1 = {need} (at least one row of slack)"
            )

    @classmethod
    def from_yaml(cls, path: str) -> "DignityConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        unknown = sorted(set(config_dict) - {f.name for f in fields(cls)})
        if unknown:
            raise TypeError(f"unknown top-level config keys: {unknown}")

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            train=TrainConfig(**config_dict.get("train", {})),
            eval=EvalConfig(**config_dict.get("eval", {})),
            policy=PolicyConfig(**config_dict.get("policy", {})),
            privacy=(
                PrivacyConfig(**(config_dict["privacy"] or {}))
                if "privacy" in config_dict
                else None
            ),
            device=config_dict.get("device", "cuda"),
            seed=config_dict.get("seed", 42),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": _yaml_safe(self.model),
            "data": _yaml_safe(self.data),
            "train": _yaml_safe(self.train),
            "eval": _yaml_safe(self.eval),
            "policy": _yaml_safe(self.policy),
            "device": self.device,
            "seed": self.seed,
        }

        if self.privacy is not None:
            config_dict["privacy"] = self.privacy.to_dict()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def __repr__(self) -> str:
        return (
            f"DignityConfig(\n"
            f"  task={self.model.task},\n"
            f"  seq_len={self.data.seq_len},\n"
            f"  hidden_size={self.model.hidden_size},\n"
            f"  device={self.device}\n"
            f")"
        )

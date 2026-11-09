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
    seed: int = 42

    def __post_init__(self):
        self.anomalies_per_block = tuple(int(x) for x in self.anomalies_per_block)
        self.duration_range = tuple(int(x) for x in self.duration_range)
        self.severity_range = tuple(float(x) for x in self.severity_range)


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
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)  # train / val / test blocks
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

    def __post_init__(self):
        if self.select_metric not in _SELECT_METRICS:
            raise ValueError(
                f"train.select_metric={self.select_metric!r} is not allowed; "
                f"expected one of {_SELECT_METRICS}"
            )


@dataclass
class EvalConfig:
    """The `eval:` block: seeds and output locations for the evaluation run."""

    tuning_seed: int = 1000  # frozen; the model never sees eval_seed before M7
    eval_seed: int = 2026  # frozen
    n_train_seeds: int = 3
    n_bootstrap: int = 1000
    epsilons: list[float] = field(default_factory=lambda: [0.1, 0.3, 1.0, 3.0, 10.0])
    out_dir: str = "results/v1.0"
    img_dir: str = "docs/img"


_MECHANISMS = ("laplace", "generalize")


@dataclass
class PrivacyFeature:
    """One column's privacy mechanism. Exactly one of laplace / generalize."""

    mechanism: str
    epsilon: float | None = None
    bounds: tuple[float, float] | None = None
    bins: int = 10

    def __post_init__(self):
        if self.mechanism not in _MECHANISMS:
            raise ValueError(
                f"unknown mechanism {self.mechanism!r}; expected one of {_MECHANISMS}"
            )
        if self.bounds is not None:
            self.bounds = (float(self.bounds[0]), float(self.bounds[1]))
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
        else:
            d["bins"] = self.bins
        return d


@dataclass
class PrivacyConfig:
    """The `privacy:` block. Absent block => no privacy stage runs."""

    epsilon_total: float
    k: int = 5
    key_env: str | None = None
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
            "features": {n: f.to_dict() for n, f in self.features.items()},
        }


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

"""Configuration management for Dignity models and training."""

from dataclasses import dataclass, field
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


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    source: str = "synthetic"  # synthetic, crypto, gdelt
    seq_len: int = 100
    batch_size: int = 64
    test_size: float = 0.2
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


@dataclass
class TrainConfig:
    """Training configuration."""

    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    gradient_clip: float = 1.0
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 10
    save_interval: int = 5


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


@dataclass
class DignityConfig:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    privacy: PrivacyConfig | None = None
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "DignityConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            train=TrainConfig(**config_dict.get("train", {})),
            privacy=(
                PrivacyConfig(**config_dict["privacy"])
                if config_dict.get("privacy")
                else None
            ),
            device=config_dict.get("device", "cuda"),
            seed=config_dict.get("seed", 42),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "train": self.train.__dict__,
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

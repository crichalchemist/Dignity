"""Configuration management for Dignity models and training."""

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ExecutionConfig:
    """Live / paper trading execution configuration.

    Defaults to paper_trading=True — live execution requires an explicit opt-in.
    Credentials are typically provided via environment variables rather than
    stored in YAML files that may be committed to version control.
    """

    provider: str = "mock"  # 'metaapi' or 'mock'
    metaapi_token: str = ""
    account_id: str = ""
    symbols: list = field(default_factory=lambda: ["EURUSD"])
    asset_class: str = "forex"
    max_drawdown: float = 0.05
    max_position_size: float = 1.0
    risk_sdk_enabled: bool = True
    paper_trading: bool = True  # must explicitly set False to go live


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    task: str = "risk"  # risk, forecast, policy, cascade
    input_size: int = 32
    hidden_size: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    cnn_kernel_size: int = 3
    task_weights: dict = field(
        default_factory=lambda: {
            "regime": 0.2,
            "risk": 0.3,
            "alpha": 0.3,
            "policy": 0.2,
        }
    )


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    source: str = "synthetic"  # synthetic, crypto, metaapi
    seq_len: int = 100
    batch_size: int = 64
    test_size: float = 0.2
    num_workers: int = 4
    start_date: str = "2016-01-01"  # inclusive start for both synthetic and MetaApi pulls
    features: list = field(
        default_factory=lambda: [
            "volume",
            "price",
            "fee_rate",
            "tx_count",
            "rsi",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "bollinger_pct_b",
            "bollinger_width",
            "atr",
            "stoch_k",
            "stoch_d",
            "adx",
            "obv",
            "vwap",
            "roc_5",
            "roc_20",
            "momentum_10",
            "momentum_20",
            "volatility_5",
            "volatility_20",
            "vol_ratio",
            "order_flow_imbalance",
            "dc_direction",
            "dc_overshoot",
            "dc_bars_since_event",
            "volume_volatility",
            "volume_entropy",
            "price_change",
            "directional_change",
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
    # Risk gate during training — prevents train/deploy distribution mismatch.
    # When True, action logits are suppressed for batches where VaR exceeds
    # max_drawdown so the model learns strategies compatible with the risk
    # envelope from epoch 1. Set False only for unconstrained research runs.
    risk_gate_training: bool = True


@dataclass
class DignityConfig:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "DignityConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        # Handle imports
        if "imports" in config_dict:
            base_path = Path(path).parent
            base_config = {}
            for import_path in config_dict["imports"]:
                with open(base_path / import_path) as f:
                    imported_config = yaml.safe_load(f)
                    # simple merge
                    base_config.update(imported_config)
            base_config.update(config_dict)
            config_dict = base_config

        # Create a set of all valid field names for ModelConfig
        model_fields = {f.name for f in fields(ModelConfig)}
        # Filter the config_dict to only include valid fields
        filtered_model_config = {
            k: v for k, v in config_dict.get("model", {}).items() if k in model_fields
        }

        return cls(
            model=ModelConfig(**filtered_model_config),
            data=DataConfig(**config_dict.get("data", {})),
            train=TrainConfig(**config_dict.get("train", {})),
            execution=ExecutionConfig(**config_dict.get("execution", {})),
            device=config_dict.get("device", "cuda"),
            seed=config_dict.get("seed", 42),
        )

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "train": self.train.__dict__,
            "execution": self.execution.__dict__,
            "device": self.device,
            "seed": self.seed,
        }

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

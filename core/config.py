"""Configuration management for Dignity models and training."""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


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
    features: list = field(default_factory=lambda: [
        'volume', 'fee_rate', 'entropy', 'tx_count', 
        'volatility', 'price_change', 'momentum', 
        'directional_change', 'regime'
    ])


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


@dataclass
class DignityConfig:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> "DignityConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            train=TrainConfig(**config_dict.get('train', {})),
            device=config_dict.get('device', 'cuda'),
            seed=config_dict.get('seed', 42)
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'train': self.train.__dict__,
            'device': self.device,
            'seed': self.seed
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
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

"""Dignity Foundation Model - Unified architecture integrating all components."""

from typing import Dict, Literal

import torch
import torch.nn as nn

from .backbone.b2s6 import B2S6Backbone
from .backbone.cnn1d import CNN1DFeatureExtractor
from .backbone.transformer import CrossAssetTransformerLayer
from .head.forecast import ForecastHead
from .head.policy import TradeSignalHead
from .head.regime import MarketRegimeHead
from .head.risk import RiskHead


class DignityFoundationModel(nn.Module):
    """
    Dignity Foundation Model - Multi-scale temporal modeling with multi-task learning.
    
    Architecture:
        1. CNN1DFeatureExtractor: Multi-scale local pattern extraction
        2. B2S6Backbone: Efficient temporal modeling with structured state-space layers
        3. CrossAssetTransformer Stack: Cross-market attention and dependencies
        4. Task-Specific Heads: Parallel multi-task outputs
    
    Flow:
        Input: [B, T, 13]  # OHLCV + 8 technical indicators
          ↓
        CNN1DFeatureExtractor (multi-scale local patterns)
          ↓ [B, T, d_model]
        B2S6Backbone (temporal modeling, 12 layers)
          ↓ [B, T, d_model]
        CrossAssetTransformer Stack (cross-market attention, 4 layers)
          ↓ [B, T, d_model]
        Task-Specific Heads (parallel outputs):
          • MarketRegimeHead → [B, T, 4]
          • ForecastHead → [B, T, 3]
          • RiskHead → [B, T, 3]
          • TradeSignalHead → [B, T, 3]
    
    Model Variants:
        - Small (50M params): d_model=256, b2s6_layers=6, transformer_layers=2
        - Medium (300M params): d_model=512, b2s6_layers=12, transformer_layers=4
        - Large (1B params): d_model=1024, b2s6_layers=24, transformer_layers=8
    """
    
    def __init__(
        self,
        input_features: int = 13,
        d_model: int = 512,
        b2s6_layers: int = 12,
        transformer_layers: int = 4,
        num_heads: int = 8,
        state_dim: int = 64,
        num_blocks: int = 4,
        use_gradient_checkpointing: bool = False,
        dropout: float = 0.1,
    ):
        """
        Initialize Dignity Foundation Model.
        
        Args:
            input_features: Number of input features (default: 13 for OHLCV + 8 indicators)
            d_model: Model dimension (embedding size)
            b2s6_layers: Number of B2S6 backbone layers
            transformer_layers: Number of cross-asset transformer layers
            num_heads: Number of attention heads in transformer
            state_dim: Dimension of SSM state in B2S6
            num_blocks: Number of blocks per B2S6 layer
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            dropout: Dropout probability
        """
        super().__init__()
        
        # Store config for model size determination
        self._d_model = d_model
        self._b2s6_layers = b2s6_layers
        self._transformer_layers = transformer_layers
        
        # Feature extraction: Multi-scale local patterns
        self.feature_extractor = CNN1DFeatureExtractor(
            input_features=input_features,
            d_model=d_model,
            dropout=dropout,
        )
        
        # Temporal modeling: Efficient sequence processing
        self.b2s6_backbone = B2S6Backbone(
            d_model=d_model,
            num_layers=b2s6_layers,
            state_dim=state_dim,
            num_blocks=num_blocks,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        # Cross-asset attention: Market dependencies and correlations
        self.transformer_layers = nn.ModuleList([
            CrossAssetTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(transformer_layers)
        ])
        
        # Task-specific heads: Multi-task learning
        self.regime_head = MarketRegimeHead(d_model=d_model)
        self.forecast_head = ForecastHead(d_model=d_model)
        self.risk_head = RiskHead(d_model=d_model)
        self.trade_signal_head = TradeSignalHead(d_model=d_model)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the foundation model.
        
        Args:
            x: Input tensor [batch, seq_len, input_features]
        
        Returns:
            Dictionary with task outputs:
                - 'regime': [batch, seq_len, 4] - Market regime logits (bull/bear/sideways/volatile)
                - 'forecast': [batch, seq_len, 3] - Price forecasts (short/medium/long term)
                - 'risk': [batch, seq_len, 3] - VaR quantiles (5%/50%/95%)
                - 'trade_signal': [batch, seq_len, 3] - Action logits (buy/hold/sell)
        """
        # Stage 1: Extract multi-scale local features
        x = self.feature_extractor(x)  # [B, T, d_model]
        
        # Stage 2: Temporal modeling with B2S6 backbone
        x = self.b2s6_backbone(x)  # [B, T, d_model]
        
        # Stage 3: Cross-asset attention
        for transformer in self.transformer_layers:
            x = transformer(x)  # [B, T, d_model]
        
        # Stage 4: Multi-task outputs (parallel heads)
        return {
            'regime': self.regime_head(x),          # [B, T, 4]
            'forecast': self.forecast_head(x),      # [B, T, 3]
            'risk': self.risk_head(x),              # [B, T, 3]
            'trade_signal': self.trade_signal_head(x),  # [B, T, 3]
        }
    
    def count_parameters(self) -> int:
        """
        Count total trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size(self) -> Literal["small", "medium", "large", "custom"]:
        """
        Determine model size variant based on architecture configuration.
        
        Returns:
            Model size: "small", "medium", "large", or "custom"
        """
        # Small: d_model=256, b2s6_layers=6, transformer_layers=2
        if (self._d_model == 256 and 
            self._b2s6_layers == 6 and 
            self._transformer_layers == 2):
            return "small"
        
        # Medium: d_model=512, b2s6_layers=12, transformer_layers=4
        elif (self._d_model == 512 and 
              self._b2s6_layers == 12 and 
              self._transformer_layers == 4):
            return "medium"
        
        # Large: d_model=1024, b2s6_layers=24, transformer_layers=8
        elif (self._d_model == 1024 and 
              self._b2s6_layers == 24 and 
              self._transformer_layers == 8):
            return "large"
        
        # Custom configuration
        else:
            return "custom"


def create_model(
    size: Literal["small", "medium", "large"] = "medium",
    input_features: int = 13,
    use_gradient_checkpointing: bool = False,
    dropout: float = 0.1,
) -> DignityFoundationModel:
    """
    Create a Dignity Foundation Model with predefined size configurations.
    
    Args:
        size: Model size variant
            - "small": ~50M parameters (d_model=256, 6 B2S6 layers, 2 transformer layers)
            - "medium": ~300M parameters (d_model=512, 12 B2S6 layers, 4 transformer layers)
            - "large": ~1B parameters (d_model=1024, 24 B2S6 layers, 8 transformer layers)
        input_features: Number of input features (default: 13)
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        dropout: Dropout probability
    
    Returns:
        DignityFoundationModel instance
    
    Examples:
        >>> # Create small model for experimentation
        >>> model = create_model("small")
        >>> print(f"Parameters: {model.count_parameters():,}")
        
        >>> # Create large model with gradient checkpointing
        >>> model = create_model("large", use_gradient_checkpointing=True)
    """
    configs = {
        "small": {
            "d_model": 256,
            "b2s6_layers": 6,
            "transformer_layers": 2,
            "num_heads": 4,
            "state_dim": 32,
            "num_blocks": 2,
        },
        "medium": {
            "d_model": 512,
            "b2s6_layers": 12,
            "transformer_layers": 4,
            "num_heads": 8,
            "state_dim": 64,
            "num_blocks": 4,
        },
        "large": {
            "d_model": 1024,
            "b2s6_layers": 24,
            "transformer_layers": 8,
            "num_heads": 16,
            "state_dim": 128,
            "num_blocks": 8,
        },
    }
    
    if size not in configs:
        raise ValueError(f"Invalid size: {size}. Must be one of {list(configs.keys())}")
    
    config = configs[size]
    
    return DignityFoundationModel(
        input_features=input_features,
        d_model=config["d_model"],
        b2s6_layers=config["b2s6_layers"],
        transformer_layers=config["transformer_layers"],
        num_heads=config["num_heads"],
        state_dim=config["state_dim"],
        num_blocks=config["num_blocks"],
        use_gradient_checkpointing=use_gradient_checkpointing,
        dropout=dropout,
    )

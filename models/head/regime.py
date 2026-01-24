"""Market regime classification head for foundation model."""

import torch
import torch.nn as nn


class MarketRegimeHead(nn.Module):
    """
    Task head for market regime classification.
    
    Classifies market conditions into distinct regimes:
    - Trending Up (bullish trend)
    - Trending Down (bearish trend)
    - Range-bound High Volatility (choppy markets)
    - Range-bound Low Volatility (stable consolidation)
    
    This head processes the foundation model's backbone output and produces
    regime classification logits for each timestep.
    """
    
    def __init__(self, d_model: int = 512, num_regimes: int = 4):
        """
        Initialize market regime classification head.
        
        Args:
            d_model: Dimension of backbone feature vectors (default: 512)
            num_regimes: Number of market regime classes (default: 4)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_regimes = num_regimes
        
        # Normalization layer for stable training
        self.norm = nn.LayerNorm(d_model)
        
        # Two-layer projection network
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model // 2, num_regimes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for regime classification.
        
        Args:
            x: Backbone feature tensor [batch_size, seq_len, d_model]
            
        Returns:
            Regime logits [batch_size, seq_len, num_regimes]
            
        Example:
            >>> head = MarketRegimeHead(d_model=512, num_regimes=4)
            >>> x = torch.randn(8, 100, 512)  # [B, T, d_model]
            >>> logits = head(x)  # [8, 100, 4]
        """
        # Normalize input features
        x = self.norm(x)  # [B, T, d_model]
        
        # Project to intermediate dimension
        x = self.fc1(x)  # [B, T, d_model // 2]
        x = self.activation(x)
        x = self.dropout(x)
        
        # Project to regime logits
        x = self.fc2(x)  # [B, T, num_regimes]
        
        return x

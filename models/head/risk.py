"""Value-at-Risk estimation head for foundation model."""

import torch
import torch.nn as nn


class RiskHead(nn.Module):
    """
    Task head for Value-at-Risk (VaR) estimation.
    
    Estimates risk through quantile regression across multiple quantiles:
    - 5th percentile (downside risk)
    - 50th percentile (median expected value)
    - 95th percentile (upside potential)
    
    This head processes the foundation model's backbone output and produces
    quantile predictions for risk assessment at each timestep.
    """
    
    def __init__(self, d_model: int = 512, quantiles: list[float] = None):
        """
        Initialize VaR estimation head.
        
        Args:
            d_model: Dimension of backbone feature vectors (default: 512)
            quantiles: List of quantiles to predict (default: [0.05, 0.5, 0.95])
        """
        super().__init__()
        
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        
        self.d_model = d_model
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Normalization layer for stable training
        self.norm = nn.LayerNorm(d_model)
        
        # Two-layer projection network
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model // 2, self.num_quantiles)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for VaR quantile estimation.
        
        Args:
            x: Backbone feature tensor [batch_size, seq_len, d_model]
            
        Returns:
            Quantile predictions [batch_size, seq_len, num_quantiles]
            
        Example:
            >>> head = RiskHead(d_model=512, quantiles=[0.05, 0.5, 0.95])
            >>> x = torch.randn(8, 100, 512)  # [B, T, d_model]
            >>> quantiles = head(x)  # [8, 100, 3]
        """
        # Normalize input features
        x = self.norm(x)  # [B, T, d_model]
        
        # Project to intermediate dimension
        x = self.fc1(x)  # [B, T, d_model // 2]
        x = self.activation(x)
        x = self.dropout(x)
        
        # Project to quantile predictions
        x = self.fc2(x)  # [B, T, num_quantiles]
        
        return x

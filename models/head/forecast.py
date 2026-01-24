"""Multi-horizon price forecasting head for foundation model."""

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """
    Task head for multi-horizon price prediction.
    
    Forecasts price returns across multiple time horizons:
    - 1-step ahead (short-term)
    - 5-step ahead (medium-term)
    - 15-step ahead (long-term)
    
    This head processes the foundation model's backbone output and produces
    continuous price return predictions for each horizon at each timestep.
    """
    
    def __init__(self, d_model: int = 512, horizons: list[int] = None):
        """
        Initialize multi-horizon forecasting head.
        
        Args:
            d_model: Dimension of backbone feature vectors (default: 512)
            horizons: List of prediction horizons in timesteps (default: [1, 5, 15])
        """
        super().__init__()
        
        if horizons is None:
            horizons = [1, 5, 15]
        
        self.d_model = d_model
        self.horizons = horizons
        self.num_horizons = len(horizons)
        
        # Normalization layer for stable training
        self.norm = nn.LayerNorm(d_model)
        
        # Two-layer projection network
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model // 2, self.num_horizons)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-horizon forecasting.
        
        Args:
            x: Backbone feature tensor [batch_size, seq_len, d_model]
            
        Returns:
            Price return forecasts [batch_size, seq_len, num_horizons]
            
        Example:
            >>> head = ForecastHead(d_model=512, horizons=[1, 5, 15])
            >>> x = torch.randn(8, 100, 512)  # [B, T, d_model]
            >>> forecasts = head(x)  # [8, 100, 3]
        """
        # Normalize input features
        x = self.norm(x)  # [B, T, d_model]
        
        # Project to intermediate dimension
        x = self.fc1(x)  # [B, T, d_model // 2]
        x = self.activation(x)
        x = self.dropout(x)
        
        # Project to horizon predictions
        x = self.fc2(x)  # [B, T, num_horizons]
        
        return x

"""Forecasting head for future transaction prediction."""

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """
    Task head for forecasting future transaction metrics.
    
    Predicts multiple timesteps ahead:
    - Volume forecasts
    - Price forecasts (if applicable)
    - Success probability
    """
    
    def __init__(self,
                 input_size: int,
                 pred_len: int = 5,
                 num_features: int = 3,
                 hidden_size: int = 128,
                 dropout: float = 0.1):
        """
        Args:
            input_size: Size of backbone context vector
            pred_len: Number of timesteps to predict
            num_features: Number of features to forecast per timestep
            hidden_size: Size of intermediate layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.pred_len = pred_len
        self.num_features = num_features
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len * num_features)
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forecast future values from context vector.
        
        Args:
            context: Context vector [batch_size, input_size]
            
        Returns:
            Forecasts [batch_size, pred_len, num_features]
        """
        batch_size = context.size(0)
        
        # Generate flat predictions
        pred_flat = self.net(context)  # [B, pred_len * num_features]
        
        # Reshape to [B, pred_len, num_features]
        predictions = pred_flat.view(
            batch_size,
            self.pred_len,
            self.num_features
        )
        
        return predictions

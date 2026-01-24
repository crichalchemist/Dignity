"""Trade signal generation head for foundation model."""

import torch
import torch.nn as nn


class TradeSignalHead(nn.Module):
    """
    Task head for trade signal generation.
    
    Produces action probability distributions over discrete trading actions:
    - Buy (long position)
    - Hold (no position change)
    - Sell (short position)
    
    This head processes the foundation model's backbone output and produces
    action logits for categorical distribution sampling at each timestep.
    """
    
    def __init__(self, d_model: int = 512, num_actions: int = 3):
        """
        Initialize trade signal generation head.
        
        Args:
            d_model: Dimension of backbone feature vectors (default: 512)
            num_actions: Number of discrete trading actions (default: 3 for Buy/Hold/Sell)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_actions = num_actions
        
        # Normalization layer for stable training
        self.norm = nn.LayerNorm(d_model)
        
        # Two-layer projection network
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(d_model // 2, num_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for trade signal generation.
        
        Args:
            x: Backbone feature tensor [batch_size, seq_len, d_model]
            
        Returns:
            Action logits [batch_size, seq_len, num_actions]
            
        Example:
            >>> head = TradeSignalHead(d_model=512, num_actions=3)
            >>> x = torch.randn(8, 100, 512)  # [B, T, d_model]
            >>> logits = head(x)  # [8, 100, 3]
        """
        # Normalize input features
        x = self.norm(x)  # [B, T, d_model]
        
        # Project to intermediate dimension
        x = self.fc1(x)  # [B, T, d_model // 2]
        x = self.activation(x)
        x = self.dropout(x)
        
        # Project to action logits
        x = self.fc2(x)  # [B, T, num_actions]
        
        return x


# Keep old PolicyHead name as alias for backward compatibility
PolicyHead = TradeSignalHead

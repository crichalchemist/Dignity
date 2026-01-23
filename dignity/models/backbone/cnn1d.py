"""1D-CNN for local pattern extraction in transaction sequences."""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D Convolutional Network for extracting local temporal patterns.
    
    Processes sequences with multiple convolutional layers to capture
    short-term dependencies and local features.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 kernel_size: int = 3,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of CNN filters
            kernel_size: Convolution kernel size
            num_layers: Number of CNN layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        # Build CNN layers
        layers = []
        
        # First layer: input_size -> hidden_size
        layers.append(nn.Conv1d(
            in_channels=input_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        ))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Additional layers: hidden_size -> hidden_size
        for _ in range(num_layers - 1):
            layers.append(nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.cnn = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN layers.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Conv1d expects [B, C, L] format
        x = x.transpose(1, 2)  # [B, input_size, seq_len]
        
        # Apply convolutions
        x = self.cnn(x)  # [B, hidden_size, seq_len]
        
        # Transpose back to [B, seq_len, hidden_size]
        x = x.transpose(1, 2)
        
        return x

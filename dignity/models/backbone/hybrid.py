"""Hybrid CNN-LSTM-Attention backbone for sequence modeling."""

import torch
import torch.nn as nn
from .cnn1d import CNN1D
from .lstm import StackedLSTM
from .attention import AdditiveAttention


class DignityBackbone(nn.Module):
    """
    Composable backbone combining CNN, LSTM, and Attention.
    
    Architecture:
    1. 1D-CNN: Extract local temporal patterns
    2. Stacked LSTM: Model long-term dependencies
    3. Additive Attention: Focus on important timesteps
    
    This replaces the monolithic agent_hybrid.py with a clean,
    modular architecture focused solely on sequence encoding.
    """
    
    def __init__(self,
                 input_size: int = 9,
                 hidden_size: int = 256,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 cnn_kernel_size: int = 3):
        """
        Args:
            input_size: Number of input features
            hidden_size: Hidden size for CNN/LSTM
            n_layers: Number of LSTM layers
            dropout: Dropout probability
            cnn_kernel_size: CNN kernel size
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # 1. CNN for local pattern extraction
        self.cnn = CNN1D(
            input_size=input_size,
            hidden_size=hidden_size,
            kernel_size=cnn_kernel_size,
            num_layers=2,
            dropout=dropout
        )
        
        # 2. LSTM for temporal modeling
        self.lstm = StackedLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=False
        )
        
        # 3. Attention for sequence weighting
        self.attn = AdditiveAttention(hidden_size)
        
        # 4. Dropout regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None) -> tuple:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Optional mask [batch_size, seq_len]
            
        Returns:
            Tuple of:
            - context: Context vector [batch_size, hidden_size]
            - attention_weights: [batch_size, seq_len]
        """
        # 1. CNN: Extract local features
        x = self.cnn(x)  # [B, T, H]
        
        # 2. LSTM: Model temporal dependencies
        x, _ = self.lstm(x)  # [B, T, H]
        x = self.dropout(x)
        
        # 3. Attention: Compute weighted context
        context, attn_weights = self.attn(x, mask)  # [B, H], [B, T]
        
        return context, attn_weights
    
    def extract_features(self,
                        x: torch.Tensor,
                        layer: str = 'all') -> torch.Tensor:
        """
        Extract intermediate features for analysis.
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            layer: Which layer to extract from ('cnn', 'lstm', 'all')
            
        Returns:
            Features from specified layer
        """
        if layer == 'cnn':
            return self.cnn(x)
        
        elif layer == 'lstm':
            cnn_out = self.cnn(x)
            lstm_out, _ = self.lstm(cnn_out)
            return lstm_out
        
        else:  # 'all'
            context, _ = self.forward(x)
            return context

"""Stacked LSTM for sequence modeling."""

import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """
    Stacked LSTM for capturing long-term temporal dependencies.
    
    Processes sequences through multiple LSTM layers to learn
    complex temporal patterns and dependencies.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 bidirectional: bool = False):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output size adjustment for bidirectional
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
    
    def forward(self, 
                x: torch.Tensor,
                hidden: tuple = None) -> tuple:
        """
        Forward pass through LSTM layers.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            hidden: Optional initial hidden state tuple (h_0, c_0)
            
        Returns:
            Tuple of (output, (h_n, c_n))
            - output: [batch_size, seq_len, hidden_size * num_directions]
            - h_n: [num_layers * num_directions, batch_size, hidden_size]
            - c_n: [num_layers * num_directions, batch_size, hidden_size]
        """
        output, (h_n, c_n) = self.lstm(x, hidden)
        return output, (h_n, c_n)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """
        Initialize hidden state.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            
        Returns:
            Tuple of (h_0, c_0) initialized to zeros
        """
        num_directions = 2 if self.bidirectional else 1
        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_size,
            device=device
        )
        return (h_0, c_0)

"""Main Dignity model: Backbone + Task Head."""

import torch
import torch.nn as nn
from .backbone.hybrid import DignityBackbone
from .head.risk import RiskHead
from .head.forecast import ForecastHead
from .head.policy import PolicyHead


class Dignity(nn.Module):
    """
    Main Dignity model combining backbone and task-specific head.
    
    This modular design allows:
    - Single backbone training
    - Multiple task heads
    - Easy deployment (ONNX export)
    - Fast inference
    """
    
    def __init__(self,
                 task: str = 'risk',
                 input_size: int = 9,
                 hidden_size: int = 256,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 **task_kwargs):
        """
        Args:
            task: Task type ('risk', 'forecast', 'policy')
            input_size: Number of input features
            hidden_size: Backbone hidden size
            n_layers: Number of LSTM layers
            dropout: Dropout probability
            **task_kwargs: Additional task-specific arguments
        """
        super().__init__()
        
        self.task = task
        
        # Backbone (shared across all tasks)
        self.backbone = DignityBackbone(
            input_size=input_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Task-specific head
        if task == 'risk':
            self.head = RiskHead(
                input_size=hidden_size,
                hidden_size=task_kwargs.get('head_hidden', 128),
                dropout=dropout
            )
        
        elif task == 'forecast':
            self.head = ForecastHead(
                input_size=hidden_size,
                pred_len=task_kwargs.get('pred_len', 5),
                num_features=task_kwargs.get('num_features', 3),
                hidden_size=task_kwargs.get('head_hidden', 128),
                dropout=dropout
            )
        
        elif task == 'policy':
            self.head = PolicyHead(
                input_size=hidden_size,
                n_actions=task_kwargs.get('n_actions', 3),
                hidden_size=task_kwargs.get('head_hidden', 128),
                dropout=dropout
            )
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None) -> tuple:
        """
        Forward pass through model.
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Optional mask [batch_size, seq_len]
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # Get context from backbone
        context, attn_weights = self.backbone(x, mask)
        
        # Task-specific prediction
        predictions = self.head(context)
        
        return predictions, attn_weights
    
    def predict(self,
                x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Generate predictions (inference mode).
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            mask: Optional mask [batch_size, seq_len]
            
        Returns:
            Predictions only (no attention weights)
        """
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(x, mask)
        return predictions
    
    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Generate model summary string."""
        return (
            f"Dignity Model Summary\n"
            f"{'=' * 50}\n"
            f"Task: {self.task}\n"
            f"Backbone: CNN1D + StackedLSTM + Attention\n"
            f"Input size: {self.backbone.input_size}\n"
            f"Hidden size: {self.backbone.hidden_size}\n"
            f"LSTM layers: {self.backbone.n_layers}\n"
            f"Total parameters: {self.num_parameters:,}\n"
            f"{'=' * 50}"
        )

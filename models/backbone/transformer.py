"""Cross-Asset Transformer Layer for multi-asset correlation modeling.

Implements multi-head self-attention for learning cross-asset dependencies
(e.g., BTC ↔ EUR/USD) with pre-normalization and feed-forward networks.
"""

import torch
import torch.nn as nn


class CrossAssetTransformerLayer(nn.Module):
    """
    Cross-Asset Transformer Layer with Multi-Head Self-Attention.

    Learns cross-asset correlations using multi-head attention mechanism.
    Follows pre-normalization architecture (LayerNorm before attention/FFN)
    for better training stability.

    Key Features:
    - Multi-head self-attention (8 heads default)
    - Pre-normalization architecture (LayerNorm before sub-layers)
    - Feed-forward network with 4x expansion
    - Residual connections around attention & FFN
    - Dropout for regularization

    Architecture:
        Input
          → LayerNorm → Multi-Head Attention → Dropout → + Residual
          → LayerNorm → FFN (d_model → 4*d_model → d_model) → Dropout → + Residual
        Output

    FFN Structure:
        - Linear(d_model, ffn_expansion * d_model)
        - GELU activation
        - Dropout
        - Linear(ffn_expansion * d_model, d_model)

    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_heads: Number of attention heads
        ffn_expansion: FFN expansion factor (default 4x)
        dropout: Dropout probability for residual connections
        attention_dropout: Dropout probability for attention weights
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        """
        Initialize Cross-Asset Transformer Layer.

        Args:
            d_model: Channel dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            ffn_expansion: FFN expansion ratio (default 4x)
            dropout: Dropout probability for residual connections
            attention_dropout: Dropout probability within attention mechanism
        """
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion

        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,  # Expect [batch, seq, features]
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_expansion * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_expansion * d_model, d_model),
        )

        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through transformer layer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attn_mask: Attention mask [seq_len, seq_len] or [batch*num_heads, seq_len, seq_len]
                      Used for causal masking or other attention patterns
            key_padding_mask: Key padding mask [batch_size, seq_len]
                             True indicates positions to ignore
            return_attention: If True, return (output, attention_weights)

        Returns:
            Output tensor [batch_size, seq_len, d_model]
            or tuple (output, attention_weights) if return_attention=True
        """
        # 1. Pre-normalization + Multi-head attention + residual
        residual = x
        x_norm = self.norm1(x)

        # Multi-head self-attention (query, key, value all from same source)
        attn_output, attn_weights = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,  # Average across heads for interpretability
        )

        # Dropout + residual connection
        x = residual + self.dropout(attn_output)

        # 2. Pre-normalization + FFN + residual
        residual = x
        x_norm = self.norm2(x)

        # Feed-forward network
        ffn_output = self.ffn(x_norm)

        # Dropout + residual connection
        x = residual + self.dropout(ffn_output)

        if return_attention:
            return x, attn_weights
        return x

    def get_attention_weights(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Get attention weights without computing full forward pass.

        Useful for visualization and analysis of learned cross-asset correlations.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask

        Returns:
            Attention weights [batch_size, seq_len, seq_len]
        """
        with torch.no_grad():
            x_norm = self.norm1(x)
            _, attn_weights = self.attention(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
        return attn_weights

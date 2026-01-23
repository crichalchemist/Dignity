"""Performance optimizations for attention layers and longer look-backs."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class OptimizedMultiHeadAttention(nn.Module):
    """Memory-optimized multi-head attention for longer sequences."""

    def __init__(
            self,
            input_dim: int,
            attention_dim: int,
            n_heads: int = 4,
            max_seq_length: int = 2048,
            use_chunking: bool = True,
            chunk_size: int = 512
    ):
        """Initialize optimized multi-head attention.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        attention_dim : int
            Attention hidden dimension.
        n_heads : int
            Number of attention heads.
        max_seq_length : int
            Maximum sequence length to handle efficiently.
        use_chunking : bool
            Whether to use chunking for very long sequences.
        chunk_size : int
            Size of chunks for processing long sequences.
        """
        super().__init__()
        assert input_dim % n_heads == 0, "input_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads
        self.attention_dim = attention_dim
        self.max_seq_length = max_seq_length
        self.use_chunking = use_chunking and max_seq_length > chunk_size
        self.chunk_size = chunk_size

        # Optimized projection layers
        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)

        # Output projection
        self.output_proj = nn.Linear(attention_dim, input_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

        logger.info(
            f"Initialized optimized attention: {n_heads} heads, chunk_size={chunk_size if use_chunking else 'N/A'}")

    def _compute_attention_chunked(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention with chunking for memory efficiency."""
        batch_size, seq_len, _ = query.shape
        chunk_size = min(self.chunk_size, seq_len)

        # Process in chunks to save memory
        outputs = []
        attention_weights = []

        for i in range(0, seq_len, chunk_size):
            end_idx = min(i + chunk_size, seq_len)

            # Extract chunks
            q_chunk = query[:, i:end_idx, :]  # [B, chunk_len, D]
            k_chunk = key  # [B, seq_len, D]
            v_chunk = value  # [B, seq_len, D]

            # Compute attention for this chunk
            scores = torch.bmm(q_chunk, k_chunk.transpose(-2, -1)) / (self.attention_dim ** 0.5)

            if attention_mask is not None:
                attention_mask_chunk = attention_mask[:, i:end_idx, :]
                scores = scores.masked_fill(attention_mask_chunk == 0, float('-inf'))

            weights_chunk = F.softmax(scores, dim=-1)
            output_chunk = torch.bmm(weights_chunk, v_chunk)

            outputs.append(output_chunk)
            attention_weights.append(weights_chunk)

        # Concatenate chunks
        output = torch.cat(outputs, dim=1)
        weights = torch.cat(attention_weights, dim=1)

        return output, weights

    def _compute_attention_standard(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard attention computation for shorter sequences."""
        # Compute attention scores
        scores = torch.bmm(query, key.transpose(-2, -1)) / (self.attention_dim ** 0.5)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Apply softmax
        weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.bmm(weights, value)

        return output, weights

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with memory optimization.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence [batch_size, seq_len, input_dim].
        attention_mask : torch.Tensor, optional
            Attention mask [batch_size, seq_len, seq_len].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output context vector [batch_size, input_dim] and attention weights [batch_size, seq_len].
            Note: For compatibility with TemporalAttention, returns a single context vector per batch.
        """
        batch_size, seq_len, input_dim = x.shape

        # Project to query, key, value
        query = self.query_proj(x)  # [B, L, attention_dim]
        key = self.key_proj(x)  # [B, L, attention_dim]
        value = self.value_proj(x)  # [B, L, attention_dim]

        # Choose computation method based on sequence length
        if self.use_chunking and seq_len > self.chunk_size:
            output, weights = self._compute_attention_chunked(query, key, value, attention_mask)
        else:
            output, weights = self._compute_attention_standard(query, key, value, attention_mask)

        # Project output back to input dimension
        output = self.output_proj(output)  # [B, L, input_dim]

        # Apply residual connection and layer norm
        output = self.layer_norm(x + self.dropout(output))  # [B, L, input_dim]

        # CRITICAL FIX: Aggregate sequence to single context vector
        # Use attention weights to get weighted average across sequence
        # weights shape: [B, L, L], output shape: [B, L, input_dim]
        # First compute sequence aggregation: torch.bmm(weights.mean(dim=1), output)
        avg_weights_per_time = weights.mean(dim=1)  # [B, L] - average across heads
        context = torch.bmm(avg_weights_per_time.unsqueeze(1), output).squeeze(1)  # [B, input_dim]

        # Average attention weights across sequence for logging
        avg_weights = avg_weights_per_time  # [B, seq_len]

        return context, avg_weights


class MultiHeadTemporalAttention(nn.Module):
    """Multi‑head additive attention (optional).

    Mirrors the API of ``TemporalAttention`` but splits the representation
    into ``n_heads`` sub‑spaces and concatenates the resulting contexts.
    """

    def __init__(self, input_dim: int, attention_dim: int, n_heads: int = 4):
        super().__init__()
        assert input_dim % n_heads == 0, "input_dim must be divisible by n_heads"
        self.n_heads = n_heads
        head_dim = input_dim // n_heads
        self.head_dim = head_dim
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, n_heads)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        B, T, D = x.shape
        h = torch.tanh(self.proj(x))  # [B, T, attention_dim]
        # Produce a score per head.
        scores = self.score(h)  # [B, T, n_heads]
        scores = scores.permute(0, 2, 1)  # [B, n_heads, T]
        weights = F.softmax(scores, dim=2)  # per‑head softmax over time

        # Split x into heads and compute weighted sum
        x_heads = x.view(B, T, self.n_heads, self.head_dim)  # [B, T, n_heads, head_dim]
        x_heads = x_heads.permute(0, 2, 1, 3)  # [B, n_heads, T, head_dim]

        # Compute context for each head
        contexts = []
        for h in range(self.n_heads):
            head_weights = weights[:, h:h + 1, :]  # [B, 1, T]
            head_context = torch.bmm(head_weights, x_heads[:, h, :, :])  # [B, 1, head_dim]
            contexts.append(head_context)

        # Concatenate heads back to [B, D]
        context = torch.cat(contexts, dim=2).squeeze(1)  # [B, n_heads * head_dim] -> [B, D]
        context = context.reshape(B, D)

        # Return combined weights (average across heads for logging)
        avg_weights = weights.mean(dim=1)  # [B, T]
        return context, avg_weights


class SlidingWindowAttention(nn.Module):
    """Sliding window attention for very long sequences."""

    def __init__(
            self,
            input_dim: int,
            attention_dim: int,
            window_size: int = 512,
            step_size: int = 256
    ):
        """Initialize sliding window attention.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        attention_dim : int
            Attention hidden dimension.
        window_size : int
            Size of attention window.
        step_size : int
            Step size for sliding window.
        """
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size

        self.query_proj = nn.Linear(input_dim, attention_dim)
        self.key_proj = nn.Linear(input_dim, attention_dim)
        self.value_proj = nn.Linear(input_dim, attention_dim)
        self.output_proj = nn.Linear(attention_dim, input_dim)

        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
            self,
            x: torch.Tensor,
            global_context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with sliding window attention.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence [batch_size, seq_len, input_dim].
        global_context : torch.Tensor, optional
            Global context to attend to across windows.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output context vector [batch_size, input_dim] and attention weights [batch_size, seq_len].
        """
        batch_size, seq_len, input_dim = x.shape
        outputs = []
        attention_weights = []

        # Process sequence in sliding windows
        for start_idx in range(0, seq_len, self.step_size):
            end_idx = min(start_idx + self.window_size, seq_len)

            # Extract window
            window = x[:, start_idx:end_idx, :]  # [B, window_len, D]
            end_idx - start_idx

            # Apply attention within window
            q = self.query_proj(window)
            k = self.key_proj(window)
            v = self.value_proj(window)

            # Self-attention within window
            scores = torch.bmm(q, k.transpose(-2, -1)) / (self.attention_dim ** 0.5)
            weights = F.softmax(scores, dim=-1)

            if global_context is not None:
                # Cross-attention with global context
                global_q = self.query_proj(global_context)
                global_k = self.key_proj(window)
                global_v = self.value_proj(window)

                global_scores = torch.bmm(global_q, global_k.transpose(-2, -1)) / (self.attention_dim ** 0.5)
                global_weights = F.softmax(global_scores, dim=-1)

                # Combine local and global attention
                local_output = torch.bmm(weights, v)
                global_output = torch.bmm(global_weights, global_v)

                # Weighted combination
                window_output = 0.7 * local_output + 0.3 * global_output
                attention_weights.append(0.7 * weights + 0.3 * global_weights)
            else:
                window_output = torch.bmm(weights, v)
                attention_weights.append(weights)

            outputs.append(window_output)

        # Concatenate window outputs
        output = torch.cat(outputs, dim=1)
        weights = torch.cat(attention_weights, dim=1)

        # Project and apply residual connection
        output = self.output_proj(output)  # [B, seq_len, input_dim]
        output = self.layer_norm(x + self.dropout(output))  # [B, seq_len, input_dim]

        # CRITICAL FIX: Aggregate sequence to single context vector for compatibility
        avg_weights = weights.mean(dim=1)  # [B, seq_len]
        context = torch.bmm(avg_weights.unsqueeze(1), output).squeeze(1)  # [B, input_dim]

        return context, avg_weights


class AdaptiveAttention(nn.Module):
    """Adaptive attention that adjusts computation based on sequence complexity."""

    def __init__(
            self,
            input_dim: int,
            attention_dim: int,
            complexity_threshold: float = 0.5
    ):
        """Initialize adaptive attention.

        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        attention_dim : int
            Attention hidden dimension.
        complexity_threshold : float
            Threshold for determining if sequence is complex.
        """
        super().__init__()
        self.complexity_threshold = complexity_threshold

        # Standard attention
        self.standard_attention = OptimizedMultiHeadAttention(
            input_dim, attention_dim, n_heads=4, use_chunking=False
        )

        # Simplified attention for complex sequences
        self.simple_attention = TemporalAttention(input_dim, attention_dim)

        # Complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(input_dim * 2, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with adaptive computation.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence [batch_size, seq_len, input_dim].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output context vector [batch_size, input_dim] and attention weights [batch_size, seq_len].
        """
        batch_size, seq_len, input_dim = x.shape

        # Estimate sequence complexity
        # Use mean and std of the sequence to estimate complexity
        seq_mean = torch.mean(x, dim=1, keepdim=True)  # [B, 1, D]
        seq_std = torch.std(x, dim=1, keepdim=True)  # [B, 1, D]

        complexity_features = torch.cat([seq_mean, seq_std], dim=-1)
        complexity_score = self.complexity_estimator(complexity_features).squeeze(-1)  # [B, 1]

        # Choose attention mechanism based on complexity
        is_complex = complexity_score > self.complexity_threshold

        if is_complex.any():
            # Use simple attention for complex sequences
            simple_output, simple_weights = self.simple_attention(x)  # [B, input_dim], [B, seq_len]

            # Use standard attention for simple sequences
            standard_output, standard_weights = self.standard_attention(x)  # [B, input_dim], [B, seq_len]

            # CRITICAL FIX: Properly handle tensor dimensions for adaptive combination
            # Ensure complexity weight broadcasting works correctly
            complexity_weight = complexity_score.squeeze(1)  # [B,]

            # Use weights to create properly shaped weighting for each sequence element
            weight_matrix = complexity_weight.unsqueeze(1).expand(-1, seq_len)  # [B, seq_len]

            # Combine outputs using element-wise weights
            output = weight_matrix.unsqueeze(-1) * simple_output.unsqueeze(1).expand(-1, seq_len, -1) + \
                     (1 - weight_matrix).unsqueeze(-1) * standard_output.unsqueeze(1).expand(-1, seq_len, -1)

            # Aggregate sequence to single context vector
            avg_weights = weight_matrix  # [B, seq_len]
            torch.bmm(avg_weights.unsqueeze(1), output).squeeze(1)  # [B, input_dim]

            # Combine attention weights
            weights = weight_matrix * simple_weights + (1 - weight_matrix) * standard_weights
        else:
            # All sequences are simple, use standard attention
            output, weights = self.standard_attention(x)

        return output, weights


class TemporalAttention(nn.Module):
    """Single-head additive attention (simplified version)."""

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        h = torch.tanh(self.proj(x))
        scores = self.score(h).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, D]
        return context, weights


def create_optimized_attention(
        input_dim: int,
        attention_dim: int,
        n_heads: int = 4,
        max_seq_length: int = 1024,
        use_adaptive: bool = True
) -> nn.Module:
    """Factory function to create optimized attention layer.

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    attention_dim : int
        Attention hidden dimension.
    n_heads : int
        Number of attention heads.
    max_seq_length : int
        Maximum expected sequence length.
    use_adaptive : bool
        Whether to use adaptive attention.

    Returns
    -------
    nn.Module
        Optimized attention module.
    """
    if use_adaptive and max_seq_length > 512:
        logger.info("Using adaptive attention for long sequences")
        return AdaptiveAttention(input_dim, attention_dim)
    elif max_seq_length > 1024:
        logger.info("Using sliding window attention for very long sequences")
        return SlidingWindowAttention(input_dim, attention_dim, window_size=512, step_size=256)
    else:
        logger.info("Using standard optimized attention")
        return OptimizedMultiHeadAttention(
            input_dim, attention_dim, n_heads=n_heads,
            use_chunking=max_seq_length > 512, chunk_size=256
        )

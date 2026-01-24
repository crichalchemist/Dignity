"""Block-Biased S6 (B2S6) recurrent unit for long-range sequence processing.

Implementation based on "Block-Biased Mamba for Long-Range Sequence Processing"
(Yu & Erichson, 2025). Combines selective state space models with block partitioning
for efficient long-range dependency modeling.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class B2S6Unit(nn.Module):
    """
    Block-Biased Selective State Space (B2S6) Unit.

    Implements selective state space dynamics with block partitioning and
    input-dependent parameters for long-range sequence modeling.

    Key Features:
    - Block partitioning: Splits channels into independent processing blocks
    - Selective SSM: Input-dependent A, B, C matrices
    - Channel-specific bias: Learnable per-channel long-range memory
    - Efficient computation: Parallel scan for training, recurrent for inference

    Math:
        x_k = A_k * x_{k-1} + B_k * u_k  (state update)
        y_k = C_k * x_k                   (output)
        A_k = exp(delta_k * A_diag)       (discretization)
        delta_k = softplus(W_delta^T * u_k + b_delta)  (input-dependent)
    """

    def __init__(
        self,
        d_model: int = 512,
        state_dim: int = 64,
        num_blocks: int = 4,
        conv_kernel: int = 4,
        dt_rank: str = "auto",
    ):
        """
        Initialize B2S6 unit.

        Args:
            d_model: Channel dimension (must be divisible by num_blocks)
            state_dim: SSM state dimension (n)
            num_blocks: Number of blocks for channel partitioning
            conv_kernel: Kernel size for causal convolution
            dt_rank: Rank for delta projection ("auto" uses d_model // 16)
        """
        super().__init__()

        if d_model % num_blocks != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_blocks ({num_blocks})"
            )

        self.d_model = d_model
        self.state_dim = state_dim
        self.num_blocks = num_blocks
        self.conv_kernel = conv_kernel
        self.block_size = d_model // num_blocks

        # Determine delta rank
        if dt_rank == "auto":
            self.dt_rank = max(16, d_model // 16)
        else:
            self.dt_rank = int(dt_rank)

        # Causal convolution for local context (applied per block)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_model,  # Depthwise convolution
        )

        # SSM parameters (per block)
        # A matrix: Diagonal, negative real values for stability
        self.A_log = nn.Parameter(
            torch.log(
                torch.randn(num_blocks, state_dim).abs() + 1e-4
            )
        )  # [num_blocks, state_dim]

        # Input projection to SSM state (B matrix) - generates input-dependent B
        self.B_weight_gen = nn.Linear(
            self.block_size, state_dim, bias=False
        )  # Shared across blocks

        # Output projection from SSM state (C matrix) - generates input-dependent C
        self.C_weight_gen = nn.Linear(
            self.block_size, state_dim, bias=False
        )  # Shared across blocks

        # Delta (timestep) projection - input-dependent
        self.delta_proj = nn.Sequential(
            nn.Linear(self.block_size, self.dt_rank, bias=True),
            nn.Linear(self.dt_rank, 1, bias=True),
        )

        # Channel-specific bias for long-range memory
        self.channel_bias = nn.Parameter(torch.zeros(d_model))

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters with appropriate scaling."""
        # A matrix: Initialize to be negative (stable dynamics)
        with torch.no_grad():
            self.A_log.uniform_(-math.log(self.state_dim), -math.log(2.0))

        # Delta projection: Small initialization for stable learning
        nn.init.xavier_uniform_(self.delta_proj[0].weight)
        nn.init.constant_(self.delta_proj[0].bias, 0.0)
        nn.init.constant_(self.delta_proj[1].weight, 0.01)
        nn.init.constant_(self.delta_proj[1].bias, 0.01)

        # B and C weight generators
        nn.init.xavier_uniform_(self.B_weight_gen.weight)
        nn.init.xavier_uniform_(self.C_weight_gen.weight)

    def forward(self, x: torch.Tensor, use_parallel_scan: bool = True) -> torch.Tensor:
        """
        Forward pass through B2S6 unit.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            use_parallel_scan: Whether to use parallel scan (training) or
                             recurrent computation (inference)

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"

        # Apply causal convolution for local context
        # [B, T, D] -> [B, D, T]
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Remove extra padding
        x_conv = x_conv.transpose(1, 2)  # [B, T, D]

        # Split into blocks: [B, T, D] -> [B, T, num_blocks, block_size]
        x_blocks = x_conv.view(batch_size, seq_len, self.num_blocks, self.block_size)

        # Process each block independently
        outputs = []
        for block_idx in range(self.num_blocks):
            x_block = x_blocks[:, :, block_idx, :]  # [B, T, block_size]
            y_block = self._process_block(x_block, block_idx, use_parallel_scan)
            outputs.append(y_block)

        # Concatenate block outputs: [B, T, num_blocks, block_size] -> [B, T, D]
        y = torch.stack(outputs, dim=2)  # [B, T, num_blocks, block_size]
        y = y.view(batch_size, seq_len, d_model)

        # Add channel-specific bias
        y = y + self.channel_bias.view(1, 1, d_model)

        return y

    def _process_block(
        self, x: torch.Tensor, block_idx: int, use_parallel_scan: bool
    ) -> torch.Tensor:
        """
        Process a single block through selective SSM.

        Args:
            x: Block input [batch_size, seq_len, block_size]
            block_idx: Index of the current block
            use_parallel_scan: Whether to use parallel scan

        Returns:
            Block output [batch_size, seq_len, block_size]
        """
        batch_size, seq_len, block_size = x.shape

        # Compute input-dependent delta (sampling interval)
        # [B, T, block_size] -> [B, T, 1]
        delta = self.delta_proj(x)  # [B, T, 1]
        delta = F.softplus(delta)  # Ensure positive

        # Get A matrix for this block (diagonal)
        A_diag = -torch.exp(self.A_log[block_idx])  # [state_dim], negative for stability

        # Discretize A matrix: A_k = exp(delta_k * A_diag)
        # [B, T, 1] * [state_dim] -> [B, T, state_dim]
        A_discrete = torch.exp(delta * A_diag.view(1, 1, -1))  # [B, T, state_dim]

        # Compute input-dependent B matrix (selective SSM)
        # B_k = B_weight_gen(u_k) where u_k is input at timestep k
        # [B, T, block_size] -> [B, T, state_dim]
        B = self.B_weight_gen(x)  # [B, T, state_dim]

        # Compute input-dependent C matrix (selective SSM)
        # C_k = C_weight_gen(u_k) where u_k is input at timestep k
        # [B, T, block_size] -> [B, T, state_dim]
        C = self.C_weight_gen(x)  # [B, T, state_dim]

        # State space computation
        if use_parallel_scan:
            y = self._parallel_scan(x, A_discrete, B, C, delta)
        else:
            y = self._recurrent_scan(x, A_discrete, B, C)

        return y

    def _parallel_scan(
        self,
        u: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parallel scan for efficient training.

        Computes SSM recurrence in parallel using associative scan.

        Args:
            u: Input [batch_size, seq_len, block_size]
            A: Discretized A matrix [batch_size, seq_len, state_dim]
            B: B matrix [batch_size, seq_len, state_dim]
            C: C matrix [batch_size, seq_len, state_dim]
            delta: Timestep [batch_size, seq_len, 1]

        Returns:
            Output [batch_size, seq_len, block_size]
        """
        batch_size, seq_len, state_dim = A.shape
        block_size = u.shape[-1]

        # Compute discretized input matrix: B_tilde_k = delta_k * B_k
        # [B, T, 1] * [B, T, state_dim] -> [B, T, state_dim]
        B_discrete = delta * B  # [B, T, state_dim]

        # Initialize states
        states = torch.zeros(
            batch_size, seq_len, state_dim, device=u.device, dtype=u.dtype
        )

        # SSM recurrence: x_k = A_k * x_{k-1} + B_k * u_k
        # Simple sequential scan (can be optimized with parallel prefix sum)
        h = torch.zeros(batch_size, state_dim, device=u.device, dtype=u.dtype)
        for t in range(seq_len):
            # State update with input-dependent B_k
            # u[:, t]: [B, block_size], need to expand to [B, state_dim]
            u_expanded = u[:, t].unsqueeze(-1)  # [B, block_size, 1]
            # B_discrete[:, t]: [B, state_dim]
            # Broadcast multiplication: [B, state_dim] with input contribution
            h = A[:, t] * h + B_discrete[:, t] * u[:, t].sum(dim=-1, keepdim=True)
            states[:, t] = h

        # Compute outputs: y_k = C_k * x_k (input-dependent C)
        # [B, T, state_dim] * [B, T, state_dim] -> [B, T, state_dim]
        # Then project to block_size using C as projection weights
        # y = sum over state_dim weighted by C
        y = torch.einsum("bts,bts->bt", states, C).unsqueeze(-1)  # [B, T, 1]
        # Expand to block_size
        y = y.expand(-1, -1, block_size)  # [B, T, block_size]

        return y

    def _recurrent_scan(
        self, u: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """
        Recurrent scan for efficient inference.

        Computes SSM recurrence sequentially (lower memory, higher latency).

        Args:
            u: Input [batch_size, seq_len, block_size]
            A: Discretized A matrix [batch_size, seq_len, state_dim]
            B: B matrix [batch_size, seq_len, state_dim]
            C: C matrix [batch_size, seq_len, state_dim]

        Returns:
            Output [batch_size, seq_len, block_size]
        """
        batch_size, seq_len, block_size = u.shape
        state_dim = A.shape[-1]

        # Initialize hidden state
        h = torch.zeros(batch_size, state_dim, device=u.device, dtype=u.dtype)

        outputs = []
        for t in range(seq_len):
            # State update: h_k = A_k * h_{k-1} + B_k * u_k
            # B[:, t] is input-dependent B_k for timestep t
            # Element-wise: [B, state_dim] * scalar from [B, block_size]
            u_scalar = u[:, t].sum(dim=-1, keepdim=True)  # [B, 1]
            h = A[:, t] * h + B[:, t] * u_scalar  # [B, state_dim]

            # Output: y_k = C_k * h_k (input-dependent C_k)
            # C[:, t] is input-dependent C_k for timestep t
            # [B, state_dim] * [B, state_dim] -> [B]
            y_t = (h * C[:, t]).sum(dim=-1, keepdim=True)  # [B, 1]
            y_t = y_t.expand(-1, block_size)  # [B, block_size]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # [B, T, block_size]
        return y


class B2S6Layer(nn.Module):
    """
    B2S6 Layer with residual connections and layer normalization.

    Wraps B2S6Unit with modern architectural components:
    - Pre-normalization (LayerNorm before B2S6)
    - Residual connection (x + B2S6(LayerNorm(x)))
    - Optional 1D convolution for local modeling (separate from B2S6Unit's conv)
    - Dropout for regularization

    Flow:
        1. LayerNorm(x)
        2. Optional: Causal Conv1D (kernel=conv_kernel, padding left only)
        3. B2S6Unit
        4. Dropout
        5. Residual: output = x + dropout(b2s6(conv(layernorm(x))))
    """

    def __init__(
        self,
        d_model: int = 512,
        state_dim: int = 64,
        num_blocks: int = 4,
        use_conv: bool = True,
        conv_kernel: int = 4,
        dropout: float = 0.1,
        dt_rank: str = "auto",
    ):
        """
        Initialize B2S6 layer.

        Args:
            d_model: Channel dimension
            state_dim: SSM state dimension
            num_blocks: Number of blocks for partitioning
            use_conv: Whether to use optional causal convolution for local patterns
            conv_kernel: Convolution kernel size (for optional conv layer)
            dropout: Dropout probability
            dt_rank: Rank for delta projection
        """
        super().__init__()

        self.d_model = d_model
        self.use_conv = use_conv
        self.conv_kernel = conv_kernel

        # Layer normalization (pre-norm)
        self.norm = nn.LayerNorm(d_model)

        # Optional causal 1D convolution for local pattern modeling
        if use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=conv_kernel,
                padding=0,  # Manual causal padding
                groups=d_model,  # Depthwise convolution
            )
        else:
            self.conv1d = None

        # B2S6 unit (always has its own internal conv)
        self.b2s6 = B2S6Unit(
            d_model=d_model,
            state_dim=state_dim,
            num_blocks=num_blocks,
            conv_kernel=4,  # B2S6Unit always uses its internal conv
            dt_rank=dt_rank,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_parallel_scan: bool = True) -> torch.Tensor:
        """
        Forward pass through B2S6 layer.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            use_parallel_scan: Whether to use parallel scan

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Save residual
        residual = x

        # 1. Pre-normalization
        x = self.norm(x)

        # 2. Optional causal convolution for local patterns
        if self.use_conv:
            # Apply causal padding (pad left only, no future info)
            # [B, T, D] -> [B, D, T]
            x_conv = x.transpose(1, 2)
            # Pad left with (kernel_size - 1) zeros
            x_conv = F.pad(x_conv, (self.conv_kernel - 1, 0), mode='constant', value=0)
            x_conv = self.conv1d(x_conv)
            # [B, D, T] -> [B, T, D]
            x = x_conv.transpose(1, 2)

        # 3. B2S6 unit
        x = self.b2s6(x, use_parallel_scan=use_parallel_scan)

        # 4. Dropout
        x = self.dropout(x)

        # 5. Residual connection
        x = residual + x

        return x


class B2S6Backbone(nn.Module):
    """
    Multi-layer B2S6 backbone for temporal modeling.

    Stacks multiple B2S6 layers with optional gradient checkpointing
    for memory-efficient long-range sequence processing.

    Features:
    - Configurable depth: 6-24 layers
    - Gradient checkpointing: Memory efficiency for long sequences
    - Final layer normalization: Stabilize output before cross-attention
    - Optional input projection: Linear(input_features, d_model)

    Flow:
        1. Optional: Linear projection from input_features to d_model
        2. Stack of B2S6Layer modules (num_layers)
        3. Final LayerNorm
        4. Optional: Gradient checkpointing during forward pass
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 12,
        state_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
        input_features: Optional[int] = None,
        dt_rank: str = "auto",
    ):
        """
        Initialize B2S6 backbone.

        Args:
            d_model: Channel dimension
            num_layers: Number of B2S6 layers (configurable: 6-24 layers)
            state_dim: SSM state dimension
            num_blocks: Number of blocks per layer
            dropout: Dropout probability
            use_gradient_checkpointing: Whether to use gradient checkpointing for memory efficiency
            input_features: If set, adds input projection from input_features to d_model
            dt_rank: Rank for delta projection
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_features = input_features

        # Optional input projection layer
        if input_features is not None and input_features != d_model:
            self.input_projection = nn.Linear(input_features, d_model)
        else:
            self.input_projection = None

        # Stack of B2S6 layers
        self.layers = nn.ModuleList(
            [
                B2S6Layer(
                    d_model=d_model,
                    state_dim=state_dim,
                    num_blocks=num_blocks,
                    dropout=dropout,
                    dt_rank=dt_rank,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, use_parallel_scan: bool = True) -> torch.Tensor:
        """
        Forward pass through B2S6 backbone.

        Args:
            x: Input tensor [batch_size, seq_len, input_features or d_model]
            use_parallel_scan: Whether to use parallel scan

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Optional input projection
        if self.input_projection is not None:
            x = self.input_projection(x)

        # Process through all layers
        for layer in self.layers:
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint(
                    layer, x, use_parallel_scan, use_reentrant=False
                )
            else:
                x = layer(x, use_parallel_scan=use_parallel_scan)

        # Final normalization
        x = self.final_norm(x)

        return x

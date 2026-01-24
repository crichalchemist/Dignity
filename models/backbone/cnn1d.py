"""1D-CNN for local pattern extraction in transaction sequences."""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D Convolutional Network for extracting local temporal patterns.

    Processes sequences with multiple convolutional layers to capture
    short-term dependencies and local features.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
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
        layers.append(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Additional layers: hidden_size -> hidden_size
        for _ in range(num_layers - 1):
            layers.append(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
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


class CNN1DFeatureExtractor(nn.Module):
    """
    Multi-scale causal CNN feature extractor for temporal pattern extraction.

    Extracts local temporal patterns from raw price/volume data using parallel
    convolution pathways at different scales (kernel sizes) before B2S6 processing.
    Uses causal convolutions to prevent information leakage from future timesteps.

    Architecture:
        - Parallel causal convolutions with multiple kernel sizes
        - BatchNorm + GELU activation for stable training
        - Channel-wise concatenation of multi-scale features
        - Final projection to d_model dimension
        - Optional residual connection when input_features == d_model
    """

    def __init__(
        self,
        input_features: int = 13,  # OHLCV + 8 technical indicators
        d_model: int = 512,
        kernel_sizes: list[int] = None,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ):
        """
        Initialize CNN1D feature extractor.

        Args:
            input_features: Number of input features (default: 13 for OHLCV + indicators)
            d_model: Output dimension (model hidden size)
            kernel_sizes: List of kernel sizes for multi-scale convolutions (default: [3, 5, 7])
            hidden_channels: Number of channels per convolution pathway
            dropout: Dropout probability
        """
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.input_features = input_features
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        self.hidden_channels = hidden_channels

        # Multi-scale convolutional pathways
        self.conv_pathways = nn.ModuleList()

        for kernel_size in kernel_sizes:
            # Each pathway: Conv1D -> BatchNorm -> GELU -> Dropout
            pathway = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_features,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size - 1,  # Left padding for causal conv
                    bias=False,  # BatchNorm handles bias
                ),
                nn.BatchNorm1d(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.conv_pathways.append(pathway)

        # Total channels after concatenation
        total_channels = len(kernel_sizes) * hidden_channels

        # Projection to d_model
        self.projection = nn.Linear(total_channels, d_model)

        # Residual connection (only if dimensions match)
        self.use_residual = input_features == d_model
        if self.use_residual:
            self.residual_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale feature extractor.

        Args:
            x: Input tensor [batch_size, seq_len, input_features]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Save for residual connection
        residual = x if self.use_residual else None

        # Transpose to Conv1D format: [B, C, L]
        x = x.transpose(1, 2)  # [B, input_features, seq_len]

        # Apply parallel convolution pathways
        pathway_outputs = []
        for i, pathway in enumerate(self.conv_pathways):
            # Apply convolution pathway
            out = pathway(x)  # [B, hidden_channels, seq_len + padding]

            # Truncate to original sequence length (causal: remove right padding)
            out = out[:, :, :seq_len]  # [B, hidden_channels, seq_len]

            pathway_outputs.append(out)

        # Concatenate along channel dimension
        x = torch.cat(pathway_outputs, dim=1)  # [B, total_channels, seq_len]

        # Transpose back to [B, seq_len, channels]
        x = x.transpose(1, 2)  # [B, seq_len, total_channels]

        # Project to d_model
        x = self.projection(x)  # [B, seq_len, d_model]

        # Add residual connection if dimensions match
        if self.use_residual:
            x = x + residual
            x = self.residual_norm(x)

        return x

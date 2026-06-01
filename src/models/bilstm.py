"""
Bidirectional LSTM for Time-Series Anomaly Detection.

Architecture uses a reconstruction-based approach (autoencoder style):
the model learns to reconstruct normal time-series patterns, and
high reconstruction error indicates an anomaly.

Designed for RTX 3050 (4GB VRAM):
- ~15MB parameters
- Batch size 16 with AMP → ~500MB peak VRAM
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AnomalyBiLSTM(nn.Module):
    """
    Bidirectional LSTM Autoencoder for anomaly detection.

    The encoder compresses the input sequence into a latent representation,
    and the decoder reconstructs the original sequence. Anomalies are
    detected when reconstruction error exceeds a learned threshold.

    Architecture:
        Input  → (batch, seq_len, input_size)
        Encoder BiLSTM → hidden_size * 2 (bidirectional)
        Bottleneck → FC compress
        Decoder LSTM → hidden_size
        Output → (batch, seq_len, input_size)

    Args:
        input_size: Number of input features per timestep (default: 1).
        hidden_size: LSTM hidden state dimension (default: 64).
        num_layers: Number of stacked LSTM layers (default: 2).
        dropout: Dropout probability between LSTM layers (default: 0.3).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── Encoder (Bidirectional) ───────────────────────────
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Bottleneck ────────────────────────────────────────
        # Compress bidirectional output (hidden_size * 2) to hidden_size
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Decoder (Unidirectional) ──────────────────────────
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── Output projection ────────────────────────────────
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode → bottleneck → decode → reconstruct.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Reconstructed tensor of same shape as input.
        """
        # Encode
        encoder_out, _ = self.encoder(x)
        # encoder_out: (batch, seq_len, hidden_size * 2)

        # Bottleneck compression
        compressed = self.bottleneck(encoder_out)
        # compressed: (batch, seq_len, hidden_size)

        # Decode
        decoder_out, _ = self.decoder(compressed)
        # decoder_out: (batch, seq_len, hidden_size)

        # Project to original feature space
        reconstructed = self.output_layer(decoder_out)
        # reconstructed: (batch, seq_len, input_size)

        return reconstructed

    def get_reconstruction_error(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-sample reconstruction error (MSE).

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Tensor of shape (batch,) with mean squared error per sample.
        """
        reconstructed = self.forward(x)
        # Per-sample MSE: mean over (seq_len, input_size)
        error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    input_size: int = 1,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    device: str | torch.device = "auto",
) -> AnomalyBiLSTM:
    """
    Factory function to build and move the model to the appropriate device.

    Args:
        input_size: Number of input features.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability.
        device: Target device ('auto', 'cuda', 'cpu').

    Returns:
        Initialized AnomalyBiLSTM on the target device.
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = AnomalyBiLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model = model.to(device)

    param_count = AnomalyBiLSTM.count_parameters(model)
    param_mb = param_count * 4 / (1024 ** 2)  # float32 → 4 bytes
    print(f"[Apex] BiLSTM initialized on {device}")
    print(f"[Apex] Parameters: {param_count:,} ({param_mb:.2f} MB)")

    return model

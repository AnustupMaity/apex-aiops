"""
Real-Time Anomaly Inference Engine.

Loads the trained BiLSTM model and provides real-time anomaly scoring
for incoming telemetry data. Designed for low-latency inference on
RTX 3050 (~50MB VRAM during inference).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.config.settings import get_settings
from src.models.bilstm import AnomalyBiLSTM


@dataclass
class AnomalyScore:
    """Result of anomaly scoring for a single window."""

    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    anomaly_score: float  # normalized 0-1 score
    confidence: float     # how far above/below threshold (0-1)


class AnomalyDetector:
    """
    Real-time anomaly detector using the trained BiLSTM.

    Loads the model checkpoint and provides methods for scoring
    individual windows or batches of time-series data.

    Args:
        model_path: Path to the trained model checkpoint.
        threshold: Anomaly threshold (if None, loaded from checkpoint).
        device: Device for inference ('auto', 'cuda', 'cpu').
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        threshold: Optional[float] = None,
        device: str = "auto",
    ) -> None:
        settings = get_settings()

        if model_path is None:
            model_path = settings.models_dir / "best_bilstm.pt"

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load checkpoint
        self.checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=True
        )
        config = self.checkpoint.get("config", {})

        # Build model from checkpoint config
        self.model = AnomalyBiLSTM(
            input_size=config.get("input_size", 1),
            hidden_size=config.get("hidden_size", 64),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.3),
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Anomaly threshold: prioritize argument, then checkpoint, then settings fallback
        ckpt_threshold = self.checkpoint.get("threshold")
        self.threshold = threshold or ckpt_threshold or settings.anomaly_threshold
        self.seq_len = config.get("seq_len", 60)

        # AMP for GPU inference
        self.use_amp = self.device.type == "cuda"

        print(
            f"[Apex] AnomalyDetector loaded on {self.device} "
            f"(threshold: {self.threshold:.6f})"
        )

    @torch.no_grad()
    def score_window(self, window: np.ndarray) -> AnomalyScore:
        """
        Score a single time-series window for anomaly.

        Args:
            window: Numpy array of shape (seq_len,) or (seq_len, 1).

        Returns:
            AnomalyScore with reconstruction error, threshold, and flags.
        """
        # Reshape to (1, seq_len, 1) for batch processing
        if window.ndim == 1:
            window = window.reshape(-1, 1)
        tensor = (
            torch.tensor(window, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        # Compute reconstruction error
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                error = self.model.get_reconstruction_error(tensor)
        else:
            error = self.model.get_reconstruction_error(tensor)

        error_val = float(error.item())

        # Compute normalized anomaly score (0-1, clamped)
        if self.threshold > 0:
            anomaly_score = min(error_val / self.threshold, 1.0)
        else:
            anomaly_score = 1.0 if error_val > 0 else 0.0

        is_anomaly = error_val > self.threshold

        # Confidence: how far from threshold (0 = at threshold, 1 = very far)
        if is_anomaly:
            confidence = min(
                (error_val - self.threshold) / max(self.threshold, 1e-8), 1.0
            )
        else:
            confidence = min(
                (self.threshold - error_val) / max(self.threshold, 1e-8), 1.0
            )

        return AnomalyScore(
            reconstruction_error=error_val,
            threshold=self.threshold,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            confidence=confidence,
        )

    @torch.no_grad()
    def score_batch(self, windows: np.ndarray) -> list[AnomalyScore]:
        """
        Score a batch of time-series windows.

        Args:
            windows: Numpy array of shape (batch, seq_len) or (batch, seq_len, 1).

        Returns:
            List of AnomalyScore objects.
        """
        if windows.ndim == 2:
            windows = windows.reshape(windows.shape[0], -1, 1)

        tensor = torch.tensor(windows, dtype=torch.float32).to(self.device)

        if self.use_amp:
            with torch.amp.autocast("cuda"):
                errors = self.model.get_reconstruction_error(tensor)
        else:
            errors = self.model.get_reconstruction_error(tensor)

        errors_np = errors.cpu().numpy()
        results = []

        for error_val in errors_np:
            error_val = float(error_val)
            if self.threshold > 0:
                anomaly_score = min(error_val / self.threshold, 1.0)
            else:
                anomaly_score = 1.0 if error_val > 0 else 0.0

            is_anomaly = error_val > self.threshold

            if is_anomaly:
                confidence = min(
                    (error_val - self.threshold) / max(self.threshold, 1e-8),
                    1.0,
                )
            else:
                confidence = min(
                    (self.threshold - error_val) / max(self.threshold, 1e-8),
                    1.0,
                )

            results.append(
                AnomalyScore(
                    reconstruction_error=error_val,
                    threshold=self.threshold,
                    is_anomaly=is_anomaly,
                    anomaly_score=anomaly_score,
                    confidence=confidence,
                )
            )

        return results

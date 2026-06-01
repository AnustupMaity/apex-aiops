"""
Telemetry Preprocessor.

Feature engineering and normalization for BiLSTM input.
Converts raw TelemetrySnapshot sequences into model-ready tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.telemetry.collector import TelemetrySnapshot


@dataclass
class NormalizationStats:
    """Stores normalization parameters computed from training data."""

    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0


class TelemetryPreprocessor:
    """
    Feature engineering pipeline for telemetry data.

    Computes derived features (rate-of-change, z-scores, ratios)
    and normalizes using statistics from training data.

    Args:
        norm_stats: Optional normalization statistics from training.
    """

    def __init__(
        self,
        norm_stats: Optional[NormalizationStats] = None,
    ) -> None:
        self.norm_stats = norm_stats or NormalizationStats()
        self._prev_snapshot: Optional[TelemetrySnapshot] = None

    def process_window(
        self, snapshots: list[TelemetrySnapshot]
    ) -> np.ndarray:
        """
        Process a window of telemetry snapshots into a model-ready array.

        Args:
            snapshots: List of TelemetrySnapshot objects (ordered by time).

        Returns:
            Numpy array of shape (len(snapshots), 1) — the primary
            metric (mean_exec_time_ms) normalized for BiLSTM input.
        """
        values = np.array(
            [s.mean_exec_time_ms for s in snapshots], dtype=np.float32
        )

        # Apply min-max normalization
        range_val = self.norm_stats.max_val - self.norm_stats.min_val
        if range_val > 1e-8:
            values = (values - self.norm_stats.min_val) / range_val
        else:
            values = values - self.norm_stats.min_val

        return values.reshape(-1, 1)

    def compute_derived_features(
        self, snapshots: list[TelemetrySnapshot]
    ) -> dict[str, float]:
        """
        Compute derived features from a window of snapshots.

        Returns a dictionary of engineered features useful for
        anomaly context (passed to PydanticAI, not the BiLSTM).
        """
        if len(snapshots) < 2:
            return {}

        exec_times = [s.mean_exec_time_ms for s in snapshots]
        cache_ratios = [s.cache_hit_ratio for s in snapshots]
        seq_scans = [s.seq_scan_count for s in snapshots]

        # Rate of change (last vs first)
        exec_roc = (
            (exec_times[-1] - exec_times[0]) / max(abs(exec_times[0]), 1e-8)
        )

        # Z-score of latest value
        exec_mean = np.mean(exec_times)
        exec_std = np.std(exec_times)
        exec_zscore = (
            (exec_times[-1] - exec_mean) / max(exec_std, 1e-8)
        )

        # Sequential scan ratio
        total_seq = snapshots[-1].seq_scan_count
        total_idx = snapshots[-1].idx_scan_count
        total_scans = total_seq + total_idx
        seq_ratio = total_seq / max(total_scans, 1)

        return {
            "exec_time_roc": float(exec_roc),
            "exec_time_zscore": float(exec_zscore),
            "exec_time_mean": float(exec_mean),
            "exec_time_std": float(exec_std),
            "cache_hit_ratio_latest": float(cache_ratios[-1]),
            "cache_hit_ratio_mean": float(np.mean(cache_ratios)),
            "seq_scan_ratio": float(seq_ratio),
            "active_connections": int(snapshots[-1].active_connections),
        }

    def update_norm_stats(
        self, values: np.ndarray
    ) -> NormalizationStats:
        """
        Update normalization statistics from new training data.

        Args:
            values: Array of metric values from training.

        Returns:
            Updated NormalizationStats.
        """
        self.norm_stats = NormalizationStats(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min_val=float(np.min(values)),
            max_val=float(np.max(values)),
        )
        return self.norm_stats

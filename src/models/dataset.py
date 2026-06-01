"""
NAB (Numenta Anomaly Benchmark) Dataset Loader.

Loads CSV time-series files from the NAB dataset, applies sliding window
extraction, normalization, and provides PyTorch Dataset/DataLoader objects
for training the BiLSTM anomaly detector.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config.settings import get_settings


class NABDataset(Dataset):
    """
    PyTorch Dataset for NAB time-series anomaly benchmark.

    Extracts sliding windows from univariate time-series CSV files.
    Each sample is a window of `seq_len` consecutive timesteps.

    The dataset uses a reconstruction-based approach: both input
    and target are the same window (autoencoder training).

    Args:
        data_dir: Path to NAB CSV directory.
        seq_len: Length of each sliding window.
        stride: Step size between consecutive windows.
        split: Data split ('train', 'val', 'test').
        train_ratio: Fraction of data used for training.
        val_ratio: Fraction of data used for validation.
        normalize: Whether to apply min-max normalization.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        seq_len: int = 60,
        stride: int = 1,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        settings = get_settings()
        self.data_dir = Path(data_dir or settings.data_dir / "nab")
        self.seq_len = seq_len
        self.stride = stride
        self.split = split
        self.normalize = normalize

        # Load and process all CSV files
        self.windows: list[np.ndarray] = []
        self.labels: list[int] = []
        self._global_min: float = float("inf")
        self._global_max: float = float("-inf")

        # Load anomaly labels if available
        self.anomaly_windows = self._load_anomaly_labels()

        # Process all CSV files in the data directory
        all_values = []
        csv_files = sorted(self.data_dir.rglob("*.csv"))

        if not csv_files:
            # If no data exists, create synthetic data for development
            print("[Apex] No NAB CSV files found. Generating synthetic data...")
            self._generate_synthetic_data()
            return

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if "value" not in df.columns:
                continue

            values = df["value"].values.astype(np.float32)
            all_values.append(values)

        if not all_values:
            print("[Apex] No valid CSV files found. Generating synthetic data...")
            self._generate_synthetic_data()
            return

        # Compute global normalization statistics
        all_concat = np.concatenate(all_values)
        self._global_min = float(np.min(all_concat))
        self._global_max = float(np.max(all_concat))

        # Extract windows from each series
        for values in all_values:
            n = len(values)

            # Temporal split (no data leakage)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            if split == "train":
                segment = values[:train_end]
            elif split == "val":
                segment = values[train_end:val_end]
            else:  # test
                segment = values[val_end:]

            # Normalize if requested
            if self.normalize and (self._global_max - self._global_min) > 1e-8:
                segment = (segment - self._global_min) / (
                    self._global_max - self._global_min
                )

            # Extract sliding windows
            for i in range(0, len(segment) - seq_len, stride):
                window = segment[i : i + seq_len]
                self.windows.append(window.reshape(-1, 1))  # (seq_len, 1)

        print(
            f"[Apex] NABDataset ({split}): {len(self.windows)} windows "
            f"from {len(csv_files)} files"
        )
        
        # Pre-allocate as a single massive PyTorch tensor for infinitely fast data loading
        self.windows_tensor = torch.tensor(np.array(self.windows), dtype=torch.float32)

    def _load_anomaly_labels(self) -> dict:
        """Load anomaly window labels from combined_windows.json if available."""
        labels_file = self.data_dir / "labels" / "combined_windows.json"
        if labels_file.exists():
            with open(labels_file) as f:
                return json.load(f)
        return {}

    def _generate_synthetic_data(self) -> None:
        """
        Generate synthetic time-series data for development/testing.

        Creates normal sinusoidal patterns with injected anomalies
        (spikes, level shifts) to simulate NAB-like data.
        """
        np.random.seed(42)
        n_series = 5
        series_length = 5000

        for i in range(n_series):
            t = np.linspace(0, 50 * np.pi, series_length)

            # Base signal: sinusoidal + trend + noise
            signal = (
                np.sin(t * (1 + i * 0.1))  # varying frequency
                + 0.5 * np.sin(t * 0.3)     # low-frequency component
                + np.random.normal(0, 0.1, series_length)  # noise
            )

            # Inject anomalies (5% of data)
            n_anomalies = int(series_length * 0.05)
            anomaly_indices = np.random.choice(
                series_length, n_anomalies, replace=False
            )
            signal[anomaly_indices] += np.random.normal(0, 3.0, n_anomalies)

            # Normalize
            signal = signal.astype(np.float32)
            sig_min, sig_max = signal.min(), signal.max()
            if (sig_max - sig_min) > 1e-8:
                signal = (signal - sig_min) / (sig_max - sig_min)

            # Split temporally
            train_end = int(series_length * 0.7)
            val_end = int(series_length * 0.85)

            if self.split == "train":
                segment = signal[:train_end]
            elif self.split == "val":
                segment = signal[train_end:val_end]
            else:
                segment = signal[val_end:]

            # Extract windows
            for j in range(0, len(segment) - self.seq_len, self.stride):
                window = segment[j : j + self.seq_len]
                self.windows.append(window.reshape(-1, 1))

        print(
            f"[Apex] NABDataset ({self.split}): {len(self.windows)} "
            f"synthetic windows generated"
        )

    def __len__(self) -> int:
        return len(self.windows_tensor)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (input, target) pair — both are the same window
        for reconstruction-based anomaly detection.
        """
        window = self.windows_tensor[idx]
        return window, window  # autoencoder: input == target

    @property
    def normalization_stats(self) -> dict[str, float]:
        """Return normalization statistics for inference-time use."""
        return {
            "min": self._global_min,
            "max": self._global_max,
        }


def create_dataloaders(
    data_dir: Optional[Path] = None,
    seq_len: int = 60,
    stride: int = 1,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders for the NAB dataset.

    Args:
        data_dir: Path to NAB data directory.
        seq_len: Sliding window length.
        stride: Stride between windows.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of parallel data loading workers.

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders.
    """
    loaders = {}

    for split in ["train", "val", "test"]:
        dataset = NABDataset(
            data_dir=data_dir,
            seq_len=seq_len,
            stride=stride,
            split=split,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
            drop_last=(split == "train"),
        )

    return loaders

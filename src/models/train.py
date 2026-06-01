"""
Training Pipeline for the BiLSTM Anomaly Detector.

Optimized for RTX 3050 (4GB VRAM):
- Mixed Precision Training (AMP) with autocast + GradScaler
- Gradient Accumulation for effective batch size of 64
- Early Stopping on validation loss
- Rich progress bars for monitoring
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional
import subprocess

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.config.settings import get_settings
from src.models.bilstm import build_model
from src.models.dataset import create_dataloaders

console = Console()


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True

        return False


def train_model(
    data_dir: Optional[Path] = None,
    save_path: Optional[Path] = None,
    device: str = "auto",
) -> dict:
    """
    Train the BiLSTM anomaly detector.

    Args:
        data_dir: Path to NAB dataset directory.
        save_path: Path to save the best model checkpoint.
        device: Device to train on ('auto', 'cuda', 'cpu').

    Returns:
        Dictionary with training metrics and model path.
    """
    settings = get_settings()

    # ── Device Setup ──────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    use_amp = device.type == "cuda"
    console.print(f"\n[bold cyan]🚀 Project Apex — BiLSTM Training[/]")
    console.print(f"   Device: {device} | AMP: {use_amp}")

    # ── CUDA Performance Optimizations ────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Auto-tune convolution algorithms
        torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for matmuls
        torch.backends.cudnn.allow_tf32 = True  # Use TF32 for cuDNN
        console.print("   [green]⚡ cuDNN benchmark + TF32 enabled[/]")

    # Check for Ollama running (VRAM conflict warning)
    if device.type == "cuda":
        try:
            import httpx
            resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
            if resp.status_code == 200:
                console.print(
                    "[yellow]⚠ Ollama is running. Consider stopping it during "
                    "training to free VRAM.[/]"
                )
        except Exception:
            pass  # Ollama not running, good

    # ── Data ──────────────────────────────────────────────────
    console.print("\n[bold]📊 Loading datasets...[/]")
    loaders = create_dataloaders(
        data_dir=data_dir,
        seq_len=settings.bilstm_seq_len,
        stride=1,
        batch_size=settings.batch_size,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]

    console.print(
        f"   Train: {len(train_loader.dataset)} samples | "
        f"Val: {len(val_loader.dataset)} samples"
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_model(
        input_size=1,
        hidden_size=settings.bilstm_hidden_size,
        num_layers=settings.bilstm_num_layers,
        dropout=settings.bilstm_dropout,
        device=device,
    )

    # ── Optimizer & Loss ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    criterion = nn.MSELoss()

    # AMP scaler
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── Training State ────────────────────────────────────────
    early_stopping = EarlyStopping(patience=settings.early_stop_patience)
    accumulation_steps = settings.gradient_accumulation_steps

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

    # Save path
    if save_path is None:
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        save_path = settings.models_dir / "best_bilstm.pt"

    start_time = time.time()

    # ── Training Loop ─────────────────────────────────────────
    console.print(f"\n[bold]🏋️ Training for up to {settings.num_epochs} epochs...[/]")

    def get_gpu_util() -> float:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            return float(out.strip())
        except Exception:
            return 0.0

    # Push datasets to GPU once to bypass all CPU bottlenecks
    if device.type == "cuda":
        train_loader.dataset.windows_tensor = train_loader.dataset.windows_tensor.to(device)
        val_loader.dataset.windows_tensor = val_loader.dataset.windows_tensor.to(device)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task(
            "[cyan]Epochs", total=settings.num_epochs
        )
        
        # We know num_samples=252388, bs=512 -> ~493 batches
        # We'll dynamically set the total inside the epoch
        batch_task = progress.add_task(
            "[magenta]Batches (GPU: 0%)", total=100
        )

        for epoch in range(settings.num_epochs):
            # ── Train Phase ───────────────────────────────────
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0

            optimizer.zero_grad()

            # Pure GPU batching (Bypasses DataLoader completely)
            dataset_tensor = train_loader.dataset.windows_tensor
            num_samples = len(dataset_tensor)
            bs = settings.batch_size
            
            # Fast shuffle on GPU
            indices = torch.randperm(num_samples, device=device)
            total_batches = (num_samples + bs - 1) // bs
            progress.reset(batch_task, total=total_batches)

            for batch_idx, start_idx in enumerate(range(0, num_samples, bs)):
                # Update GPU usage every 50 batches to avoid subprocess overhead
                if batch_idx % 50 == 0:
                    gpu_usage = get_gpu_util()
                    progress.update(batch_task, description=f"[magenta]Batches (GPU: {gpu_usage:.0f}%)")
                    
                batch_indices = indices[start_idx:start_idx + bs]
                inputs = dataset_tensor[batch_indices]
                targets = inputs

                # Forward pass with AMP
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss = loss / accumulation_steps
                    loss.backward()

                # Gradient accumulation step
                if (batch_idx + 1) % accumulation_steps == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                epoch_train_loss += loss.item() * accumulation_steps
                num_batches += 1
                progress.advance(batch_task)

            avg_train_loss = epoch_train_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)

            # ── Validation Phase ──────────────────────────────
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0

            val_tensor = val_loader.dataset.windows_tensor
            val_samples = len(val_tensor)
            
            with torch.no_grad():
                for start_idx in range(0, val_samples, bs):
                    inputs = val_tensor[start_idx:start_idx + bs]
                    targets = inputs

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    epoch_val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = epoch_val_loss / max(num_val_batches, 1)
            val_losses.append(avg_val_loss)

            # Learning rate scheduling
            scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "config": {
                        "input_size": 1,
                        "hidden_size": settings.bilstm_hidden_size,
                        "num_layers": settings.bilstm_num_layers,
                        "dropout": settings.bilstm_dropout,
                        "seq_len": settings.bilstm_seq_len,
                    },
                }
                torch.save(checkpoint, save_path)

            # Update progress
            progress.update(
                epoch_task,
                advance=1,
                description=(
                    f"[cyan]Epoch {epoch+1}/{settings.num_epochs} | "
                    f"Train: {avg_train_loss:.6f} | "
                    f"Val: {avg_val_loss:.6f}"
                ),
            )

            # Early stopping check
            if early_stopping(avg_val_loss):
                console.print(
                    f"\n[yellow]⏹ Early stopping at epoch {epoch+1} "
                    f"(patience: {settings.early_stop_patience})[/]"
                )
                break

    # ── Training Summary ──────────────────────────────────────
    elapsed = time.time() - start_time

    # Compute threshold and detailed metrics
    metrics = _compute_threshold(model, train_loader, device, use_amp)
    threshold = metrics["threshold"]
    
    table = Table(title="Training Summary & Evaluation Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Epochs", str(len(train_losses)))
    table.add_row("Training Time", f"{elapsed:.1f}s")
    table.add_row("Best Val Loss", f"{best_val_loss:.8f}")
    table.add_row("Final Train Loss", f"{train_losses[-1]:.8f}")
    table.add_row("Mean Absolute Error (MAE)", f"{metrics['mae']:.8f}")
    table.add_row("Mean Squared Error (MSE)", f"{metrics['mean_mse']:.8f}")
    table.add_row("Max Reconstruction Error", f"{metrics['max_error']:.8f}")
    table.add_row("Model Saved To", str(save_path))
    console.print(table)

    # Compute threshold from training reconstruction errors
    console.print(f"[bold green]✅ Anomaly threshold (μ + 3σ): {threshold:.8f}[/]")

    # Re-save the best checkpoint with the newly calculated threshold included
    final_checkpoint = torch.load(save_path, weights_only=True)
    final_checkpoint["threshold"] = threshold
    torch.save(final_checkpoint, save_path)

    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "threshold": threshold,
        "metrics": metrics,
        "model_path": str(save_path),
        "elapsed_seconds": elapsed,
    }


def _compute_threshold(
    model: nn.Module,
    train_loader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """
    Compute anomaly threshold and detailed evaluation metrics.
    Uses pure GPU slicing to match training speeds.
    """
    model.eval()
    all_errors = []
    
    dataset_tensor = train_loader.dataset.windows_tensor
    num_samples = len(dataset_tensor)
    bs = 2048

    with torch.no_grad():
        for start_idx in range(0, num_samples, bs):
            inputs = dataset_tensor[start_idx:start_idx + bs]
            if use_amp:
                with torch.amp.autocast("cuda"):
                    errors = model.get_reconstruction_error(inputs)
            else:
                errors = model.get_reconstruction_error(inputs)
            all_errors.append(errors)

    all_errors = torch.cat(all_errors)
    mean_err = all_errors.mean().item()
    std_err = all_errors.std().item()
    max_err = all_errors.max().item()
    mae = all_errors.abs().mean().item()
    threshold = mean_err + 3 * std_err

    return {
        "threshold": threshold,
        "mean_mse": mean_err,
        "std_mse": std_err,
        "max_error": max_err,
        "mae": mae,
    }


if __name__ == "__main__":
    results = train_model()
    console.print(f"\n[bold]📈 Training complete![/]")

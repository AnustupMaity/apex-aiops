"""
Tests for the BiLSTM Anomaly Detector.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.bilstm import AnomalyBiLSTM, build_model


class TestBiLSTMArchitecture:
    """Test the BiLSTM model architecture."""

    def test_model_creation(self):
        """Model should initialize with default parameters."""
        model = AnomalyBiLSTM()
        assert model is not None
        assert model.hidden_size == 64
        assert model.num_layers == 2

    def test_model_custom_params(self):
        """Model should accept custom parameters."""
        model = AnomalyBiLSTM(
            input_size=3, hidden_size=32, num_layers=1, dropout=0.1
        )
        assert model.input_size == 3
        assert model.hidden_size == 32
        assert model.num_layers == 1

    def test_forward_pass_shape(self, device):
        """Forward pass should return correct output shape."""
        model = AnomalyBiLSTM(input_size=1, hidden_size=64).to(device)
        x = torch.randn(4, 60, 1).to(device)  # batch=4, seq=60, features=1

        output = model(x)
        assert output.shape == x.shape, (
            f"Expected {x.shape}, got {output.shape}"
        )

    def test_reconstruction_error(self, device):
        """Reconstruction error should return one value per sample."""
        model = AnomalyBiLSTM(input_size=1, hidden_size=64).to(device)
        x = torch.randn(8, 60, 1).to(device)

        errors = model.get_reconstruction_error(x)
        assert errors.shape == (8,)
        assert (errors >= 0).all(), "Errors should be non-negative"

    def test_parameter_count(self):
        """Parameter count should be reasonable for 4GB VRAM."""
        model = AnomalyBiLSTM()
        count = AnomalyBiLSTM.count_parameters(model)
        param_mb = count * 4 / (1024 ** 2)

        assert param_mb < 100, (
            f"Model is {param_mb:.1f}MB — too large for 4GB VRAM"
        )
        assert count > 0, "Model should have trainable parameters"

    def test_build_model_factory(self):
        """Build model factory should create and move to device."""
        model = build_model(device="cpu")
        assert model is not None
        assert next(model.parameters()).device == torch.device("cpu")

    def test_gradient_flow(self, device):
        """Gradients should flow through the model."""
        model = AnomalyBiLSTM(input_size=1, hidden_size=32).to(device)
        x = torch.randn(2, 30, 1, requires_grad=True).to(device)

        output = model(x)
        loss = output.mean()
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"No gradient for {name}"
                )


class TestBiLSTMInference:
    """Test inference-related functionality."""

    def test_eval_mode_no_grad(self, device):
        """Model in eval mode should work without gradients."""
        model = AnomalyBiLSTM().to(device)
        model.eval()

        with torch.no_grad():
            x = torch.randn(1, 60, 1).to(device)
            output = model(x)
            error = model.get_reconstruction_error(x)

        assert output.shape == (1, 60, 1)
        assert error.shape == (1,)

    def test_batch_consistency(self, device):
        """Single and batched inference should give same results."""
        model = AnomalyBiLSTM().to(device)
        model.eval()

        x = torch.randn(1, 60, 1).to(device)

        with torch.no_grad():
            single_error = model.get_reconstruction_error(x)

            # Same input in a batch
            batch = x.repeat(4, 1, 1)
            batch_errors = model.get_reconstruction_error(batch)

        # All batch errors should be equal to single error
        for i in range(4):
            assert torch.allclose(
                batch_errors[i], single_error[0], atol=1e-5
            ), f"Batch item {i} differs from single inference"

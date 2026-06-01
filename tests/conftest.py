"""
Shared pytest fixtures for Project Apex tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.config.settings import Settings


@pytest.fixture
def settings():
    """Provide a test settings instance."""
    return Settings(
        supabase_url="https://test.supabase.co",
        supabase_service_key="test-key",
        supabase_db_url="postgresql://test:test@localhost:5432/test",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5-coder:1.5b",
        langsmith_tracing=False,
    )


@pytest.fixture
def sample_window():
    """Provide a sample time-series window for testing."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 60)
    signal = np.sin(t) + np.random.normal(0, 0.1, 60)
    return signal.astype(np.float32)


@pytest.fixture
def sample_anomaly_window():
    """Provide a window with an injected anomaly."""
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 60)
    signal = np.sin(t) + np.random.normal(0, 0.1, 60)
    # Inject spike at position 30
    signal[30] += 5.0
    return signal.astype(np.float32)


@pytest.fixture
def sample_anomaly_data():
    """Provide sample anomaly data for graph testing."""
    return {
        "anomaly_score": 0.92,
        "affected_query": (
            "SELECT o.order_id, c.customer_name, r.restaurant_name "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "JOIN restaurants r ON o.restaurant_id = r.restaurant_id "
            "WHERE o.order_date > '2023-01-01' "
            "ORDER BY o.total_amount DESC"
        ),
        "baseline_exec_ms": 15.0,
        "current_exec_ms": 450.0,
        "source_metrics": {
            "mean_exec_time_ms": 450.0,
            "cache_hit_ratio": 0.45,
            "seq_scan_count": 15000,
        },
    }


@pytest.fixture
def device():
    """Get the appropriate test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

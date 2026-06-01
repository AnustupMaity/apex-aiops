"""
Tests for the LangGraph orchestration pipeline.
"""

from __future__ import annotations

import pytest

from src.orchestration.nodes import validate_node, route_node, verify_node
from src.orchestration.router import route_by_severity, should_retry


class TestRouter:
    """Test the severity-based router."""

    def test_low_routes_local(self):
        state = {"severity": "low"}
        assert route_by_severity(state) == "local"

    def test_medium_routes_local(self):
        state = {"severity": "medium"}
        assert route_by_severity(state) == "local"

    def test_high_routes_cloud(self):
        state = {"severity": "high"}
        assert route_by_severity(state) == "cloud"

    def test_critical_routes_cloud(self):
        state = {"severity": "critical"}
        assert route_by_severity(state) == "cloud"

    def test_unknown_severity_defaults_local(self):
        state = {"severity": "unknown"}
        assert route_by_severity(state) == "local"


class TestRetryLogic:
    """Test the retry/escalation logic."""

    def test_improved_returns_done(self):
        state = {"resolution": "improved", "retry_count": 1}
        assert should_retry(state) == "done"

    def test_regression_returns_done(self):
        state = {"resolution": "regression", "retry_count": 1}
        assert should_retry(state) == "done"

    def test_no_change_under_limit_retries(self):
        state = {"resolution": "no_change", "retry_count": 1}
        assert should_retry(state) == "retry"

    def test_no_change_at_limit_escalates(self):
        state = {"resolution": "no_change", "retry_count": 3}
        assert should_retry(state) == "escalate"

    def test_no_change_over_limit_escalates(self):
        state = {"resolution": "no_change", "retry_count": 5}
        assert should_retry(state) == "escalate"


class TestValidateNode:
    """Test the validate graph node."""

    def test_valid_input(self, sample_anomaly_data):
        state = {"anomaly_data": sample_anomaly_data}
        result = validate_node(state)

        assert result.get("error") == ""
        assert result.get("anomaly_report") is not None
        assert result.get("severity") in ("low", "medium", "high", "critical")
        assert result.get("incident_id") is not None

    def test_missing_query(self):
        state = {"anomaly_data": {"anomaly_score": 0.5}}
        result = validate_node(state)
        # Should handle gracefully (may succeed with empty query
        # or fail with validation error)
        assert isinstance(result, dict)

    def test_empty_data(self):
        state = {"anomaly_data": {}}
        result = validate_node(state)
        assert isinstance(result, dict)


class TestVerifyNode:
    """Test the verify graph node."""

    def test_improvement_detected(self):
        state = {
            "original_exec_ms": 100.0,
            "optimized_exec_ms": 30.0,
            "retry_count": 0,
        }
        result = verify_node(state)

        assert result["resolution"] == "improved"
        assert result["speedup_factor"] > 1.0
        assert result["retry_count"] == 1

    def test_no_change_detected(self):
        state = {
            "original_exec_ms": 100.0,
            "optimized_exec_ms": 95.0,
            "retry_count": 0,
        }
        result = verify_node(state)
        assert result["resolution"] == "no_change"

    def test_regression_detected(self):
        state = {
            "original_exec_ms": 100.0,
            "optimized_exec_ms": 200.0,
            "retry_count": 0,
        }
        result = verify_node(state)
        assert result["resolution"] == "regression"
        assert result["speedup_factor"] < 1.0

    def test_zero_exec_times(self):
        state = {
            "original_exec_ms": 0.0,
            "optimized_exec_ms": 0.0,
            "retry_count": 0,
        }
        result = verify_node(state)
        assert result["speedup_factor"] == 1.0

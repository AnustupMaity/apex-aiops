"""
Tests for Pydantic validation schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.validation.schemas import (
    AnomalyReport,
    IncidentRecord,
    OptimizationResult,
    QueryContext,
)
from src.validation.validator_agent import (
    compute_severity,
    extract_table_names,
    validate_anomaly,
)


class TestAnomalyReport:
    """Test AnomalyReport schema validation."""

    def test_valid_report(self):
        """Valid data should create an AnomalyReport."""
        report = AnomalyReport(
            severity="high",
            anomaly_score=0.95,
            affected_query="SELECT * FROM orders",
            table_names=["orders"],
            baseline_exec_ms=10.0,
            current_exec_ms=500.0,
            degradation_factor=50.0,
        )
        assert report.severity == "high"
        assert report.anomaly_score == 0.95
        assert report.incident_id is not None  # auto-generated

    def test_anomaly_score_bounds(self):
        """Anomaly score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            AnomalyReport(
                severity="low",
                anomaly_score=1.5,  # Out of bounds
                affected_query="SELECT 1",
                baseline_exec_ms=1.0,
                current_exec_ms=2.0,
                degradation_factor=2.0,
            )

    def test_severity_literal(self):
        """Severity must be one of the allowed values."""
        with pytest.raises(ValidationError):
            AnomalyReport(
                severity="extreme",  # Not a valid value
                anomaly_score=0.5,
                affected_query="SELECT 1",
                baseline_exec_ms=1.0,
                current_exec_ms=2.0,
                degradation_factor=2.0,
            )

    def test_auto_id_generation(self):
        """Each report should get a unique ID."""
        r1 = AnomalyReport(
            severity="low",
            anomaly_score=0.5,
            affected_query="SELECT 1",
            baseline_exec_ms=1.0,
            current_exec_ms=2.0,
            degradation_factor=2.0,
        )
        r2 = AnomalyReport(
            severity="low",
            anomaly_score=0.5,
            affected_query="SELECT 1",
            baseline_exec_ms=1.0,
            current_exec_ms=2.0,
            degradation_factor=2.0,
        )
        assert r1.anomaly_id != r2.anomaly_id


class TestOptimizationResult:
    """Test OptimizationResult schema."""

    def test_valid_result(self):
        """Valid optimization result should pass validation."""
        result = OptimizationResult(
            original_query="SELECT * FROM orders",
            optimized_query="SELECT order_id FROM orders WHERE status = 'active'",
            status="improved",
            speedup_factor=3.5,
        )
        assert result.status == "improved"
        assert result.speedup_factor == 3.5

    def test_status_literal(self):
        """Status must be improved, no_change, or regression."""
        with pytest.raises(ValidationError):
            OptimizationResult(
                original_query="SELECT 1",
                optimized_query="SELECT 1",
                status="unknown",
            )


class TestTableExtraction:
    """Test SQL table name extraction."""

    def test_simple_select(self):
        """Extract table from simple SELECT."""
        tables = extract_table_names("SELECT * FROM orders")
        assert "orders" in tables

    def test_join_query(self):
        """Extract tables from JOIN query."""
        sql = (
            "SELECT o.*, c.name FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id"
        )
        tables = extract_table_names(sql)
        assert "orders" in tables or "o" in tables
        assert "customers" in tables or "c" in tables

    def test_multiple_joins(self):
        """Extract tables from multiple JOINs."""
        sql = (
            "SELECT * FROM orders o "
            "JOIN customers c ON o.cid = c.id "
            "JOIN restaurants r ON o.rid = r.id"
        )
        tables = extract_table_names(sql)
        assert len(tables) >= 2  # Should find at least 2 tables

    def test_subquery(self):
        """Extract tables from subquery."""
        sql = (
            "SELECT * FROM orders WHERE customer_id IN "
            "(SELECT id FROM customers)"
        )
        tables = extract_table_names(sql)
        assert "orders" in tables
        assert "customers" in tables


class TestSeverityComputation:
    """Test severity level computation."""

    def test_low_severity(self):
        assert compute_severity(1.5) == "low"

    def test_medium_severity(self):
        assert compute_severity(3.0) == "medium"

    def test_high_severity(self):
        assert compute_severity(7.0) == "high"

    def test_critical_severity(self):
        assert compute_severity(15.0) == "critical"

    def test_boundary_values(self):
        assert compute_severity(2.0) == "medium"
        assert compute_severity(5.0) == "high"
        assert compute_severity(10.0) == "critical"


class TestValidateAnomaly:
    """Test the validate_anomaly function."""

    def test_valid_anomaly(self):
        """Should produce a valid AnomalyReport."""
        report = validate_anomaly(
            anomaly_score=0.9,
            affected_query="SELECT * FROM orders JOIN customers ON orders.cid = customers.id",
            baseline_exec_ms=10.0,
            current_exec_ms=100.0,
        )
        assert isinstance(report, AnomalyReport)
        assert report.severity == "high"  # 10x degradation
        assert report.degradation_factor == 10.0
        assert len(report.table_names) >= 1

"""
Tests for MCP tools (unit tests without database connection).
"""

from __future__ import annotations

import pytest

from src.mcp.tools import run_explain_analyze, get_table_stats, list_slow_queries


class TestExplainAnalyze:
    """Test explain_analyze tool behavior."""

    def test_invalid_connection_returns_error(self):
        """Should return error dict when database is unavailable."""
        result = run_explain_analyze(
            query="SELECT 1",
            db_url="postgresql://invalid:invalid@localhost:5432/nonexistent",
        )
        assert result["success"] is False
        assert "error" in result

    def test_result_structure(self):
        """Error result should have all expected keys."""
        result = run_explain_analyze(
            query="SELECT 1",
            db_url="postgresql://invalid:invalid@localhost:5432/nonexistent",
        )
        expected_keys = {
            "plan", "execution_time_ms", "planning_time_ms",
            "node_type", "total_cost", "actual_rows", "success",
        }
        assert expected_keys.issubset(result.keys())


class TestGetTableStats:
    """Test get_table_stats tool behavior."""

    def test_invalid_connection_returns_error(self):
        """Should return error when database is unavailable."""
        result = get_table_stats(
            db_url="postgresql://invalid:invalid@localhost:5432/nonexistent",
        )
        assert len(result) > 0
        assert "error" in result[0]


class TestListSlowQueries:
    """Test list_slow_queries tool behavior."""

    def test_invalid_connection_returns_error(self):
        """Should return error when database is unavailable."""
        result = list_slow_queries(
            db_url="postgresql://invalid:invalid@localhost:5432/nonexistent",
        )
        assert len(result) > 0
        assert "error" in result[0]

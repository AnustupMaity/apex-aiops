"""
PydanticAI Validator Agent.

Intercepts raw anomaly data from the BiLSTM, validates and structures
it into a typed AnomalyReport, extracts table names from SQL, and
computes severity based on degradation thresholds.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

import sqlparse

from src.validation.schemas import AnomalyReport


# ── Severity Thresholds ───────────────────────────────────────
SEVERITY_THRESHOLDS = {
    "low": (1.0, 2.0),      # 1x-2x degradation
    "medium": (2.0, 5.0),   # 2x-5x degradation
    "high": (5.0, 10.0),    # 5x-10x degradation
    "critical": (10.0, float("inf")),  # 10x+ degradation
}


def extract_table_names(sql: str) -> list[str]:
    """
    Extract table names from a SQL query using sqlparse.

    Handles:
    - Simple SELECT FROM
    - JOINs (INNER, LEFT, RIGHT, CROSS, FULL)
    - Subqueries
    - INSERT INTO, UPDATE, DELETE FROM

    Args:
        sql: The SQL query string.

    Returns:
        List of unique table names found in the query.
    """
    tables: set[str] = set()

    # Parse SQL
    try:
        parsed = sqlparse.parse(sql)
    except Exception:
        # Fallback to regex extraction
        return _extract_tables_regex(sql)

    for statement in parsed:
        tables.update(_extract_from_parsed(statement))

    # Also try regex as a supplement
    regex_tables = _extract_tables_regex(sql)
    tables.update(regex_tables)

    # Filter out SQL keywords and common non-table words
    sql_keywords = {
        "select", "from", "where", "and", "or", "not", "in",
        "between", "like", "is", "null", "true", "false",
        "order", "by", "group", "having", "limit", "offset",
        "insert", "into", "values", "update", "set", "delete",
        "create", "alter", "drop", "index", "table", "view",
        "join", "inner", "left", "right", "full", "outer",
        "cross", "on", "as", "case", "when", "then", "else",
        "end", "distinct", "all", "exists", "any", "union",
        "intersect", "except", "asc", "desc", "with",
    }

    filtered = [
        t for t in tables
        if t.lower() not in sql_keywords and len(t) > 1
    ]

    return list(set(filtered))


def _extract_from_parsed(statement) -> set[str]:
    """Extract table names from a parsed sqlparse statement."""
    tables = set()

    from_seen = False
    for token in statement.tokens:
        if token.ttype is sqlparse.tokens.Keyword and token.normalized in (
            "FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN",
            "FULL JOIN", "CROSS JOIN", "INTO", "UPDATE",
        ):
            from_seen = True
            continue

        if from_seen:
            if hasattr(token, "get_name") and token.get_name():
                tables.add(token.get_name())
            elif token.ttype is sqlparse.tokens.Name:
                tables.add(str(token))
            from_seen = False

    return tables


def _extract_tables_regex(sql: str) -> set[str]:
    """Regex-based fallback for table name extraction."""
    patterns = [
        r'\bFROM\s+(\w+)',
        r'\bJOIN\s+(\w+)',
        r'\bINTO\s+(\w+)',
        r'\bUPDATE\s+(\w+)',
    ]

    tables = set()
    for pattern in patterns:
        matches = re.findall(pattern, sql, re.IGNORECASE)
        tables.update(matches)

    return tables


def compute_severity(degradation_factor: float) -> str:
    """
    Compute severity level from degradation factor.

    Args:
        degradation_factor: Ratio of current to baseline execution time.

    Returns:
        Severity string: 'low', 'medium', 'high', or 'critical'.
    """
    for severity, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= degradation_factor < high:
            return severity
    return "critical"


def validate_anomaly(
    anomaly_score: float,
    affected_query: str,
    baseline_exec_ms: float,
    current_exec_ms: float,
    source_metrics: Optional[dict[str, Any]] = None,
    timestamp: Optional[datetime] = None,
) -> AnomalyReport:
    """
    Validate and structure raw anomaly data into an AnomalyReport.

    This is the main entry point for the validation layer. It:
    1. Extracts table names from the SQL query
    2. Computes degradation factor and severity
    3. Validates all fields with Pydantic
    4. Returns a fully typed AnomalyReport

    Args:
        anomaly_score: Normalized anomaly score from BiLSTM (0-1).
        affected_query: The SQL query causing issues.
        baseline_exec_ms: Normal execution time.
        current_exec_ms: Current (degraded) execution time.
        source_metrics: Raw telemetry snapshot.
        timestamp: When the anomaly was detected.

    Returns:
        Validated AnomalyReport instance.

    Raises:
        pydantic.ValidationError: If any field fails validation.
    """
    # Compute derived fields
    degradation_factor = current_exec_ms / max(baseline_exec_ms, 0.001)
    severity = compute_severity(degradation_factor)
    table_names = extract_table_names(affected_query)

    # Build and validate the report
    report = AnomalyReport(
        timestamp=timestamp or datetime.utcnow(),
        severity=severity,
        anomaly_score=anomaly_score,
        affected_query=affected_query,
        table_names=table_names,
        baseline_exec_ms=baseline_exec_ms,
        current_exec_ms=current_exec_ms,
        degradation_factor=degradation_factor,
        source_metrics=source_metrics or {},
    )

    return report

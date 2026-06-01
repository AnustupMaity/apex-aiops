"""
SQL Optimizer — Unified Query Rewrite Interface.

Wraps the LLM call (Ollama or Cloud) with pre/post processing:
- Pre: Formats schema context from pg_catalog introspection
- Post: Validates generated SQL syntax with sqlparse
"""

from __future__ import annotations

from typing import Any, Optional

import sqlparse

from src.config.settings import get_settings
from src.reasoning.ollama_agent import optimize_with_ollama
from src.reasoning.cloud_agent import optimize_with_cloud


def optimize_query(
    query: str,
    table_names: list[str],
    query_context: Optional[dict[str, Any]] = None,
    use_local: bool = True,
) -> dict[str, Any]:
    """
    Optimize a SQL query using either local or cloud LLM.

    This is the main entry point for query optimization, called
    by the LangGraph reason_node.

    Args:
        query: The slow SQL query to optimize.
        table_names: Tables referenced in the query.
        query_context: Optional database schema context.
        use_local: True for Ollama, False for cloud LLM.

    Returns:
        Dictionary with optimized_query, reasoning, and index_recommendations.
    """
    context = query_context or {}

    # Format schema context for the prompt
    schema_context = _format_schema_context(context)

    # Get baseline metrics from context
    current_exec_ms = context.get("current_exec_ms", 0.0)
    baseline_exec_ms = context.get("baseline_exec_ms", 0.0)

    # Call the appropriate agent
    if use_local:
        result = optimize_with_ollama(
            query=query,
            table_names=table_names,
            current_exec_ms=current_exec_ms,
            baseline_exec_ms=baseline_exec_ms,
            schema_context=schema_context,
        )
    else:
        result = optimize_with_cloud(
            query=query,
            table_names=table_names,
            current_exec_ms=current_exec_ms,
            baseline_exec_ms=baseline_exec_ms,
            schema_context=schema_context,
        )

    # Post-process: validate the optimized SQL
    optimized = result.get("optimized_query", query)
    validated_sql = _validate_sql(optimized, original=query)
    result["optimized_query"] = validated_sql

    # Post-process: validate index recommendations
    result["index_recommendations"] = _validate_indexes(
        result.get("index_recommendations", [])
    )

    return result


def _format_schema_context(context: dict[str, Any]) -> str:
    """Format database schema context into a readable string for the LLM."""
    parts = []

    # Table schemas
    table_schemas = context.get("table_schemas", {})
    if table_schemas:
        parts.append("## Table Schemas")
        for table_name, columns in table_schemas.items():
            parts.append(f"\n### {table_name}")
            for col in columns:
                col_name = col.get("column", "unknown")
                col_type = col.get("type", "unknown")
                nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
                parts.append(f"  - {col_name} {col_type} {nullable}")

    # Existing indexes
    indexes = context.get("existing_indexes", [])
    if indexes:
        parts.append("\n## Existing Indexes")
        for idx in indexes:
            parts.append(f"  - {idx}")

    # Table statistics
    stats = context.get("table_statistics", {})
    if stats:
        parts.append("\n## Table Statistics")
        for table_name, table_stats in stats.items():
            row_count = table_stats.get("row_count", "unknown")
            size = table_stats.get("size", "unknown")
            parts.append(f"  - {table_name}: {row_count} rows, {size}")

    return "\n".join(parts) if parts else "No schema context available."


def _validate_sql(sql: str, original: str) -> str:
    """
    Validate that the optimized SQL is syntactically valid.

    If validation fails, returns the original query.
    """
    try:
        parsed = sqlparse.parse(sql)
        if not parsed or not parsed[0].tokens:
            print("[Apex] SQL validation failed: empty parse result")
            return original

        # Check that it's a valid statement type
        stmt_type = parsed[0].get_type()
        if stmt_type not in ("SELECT", "INSERT", "UPDATE", "DELETE", None):
            print(f"[Apex] Unexpected statement type: {stmt_type}")

        # Format the SQL for consistency
        formatted = sqlparse.format(
            sql,
            reindent=True,
            keyword_case="upper",
        )

        return formatted.strip()

    except Exception as e:
        print(f"[Apex] SQL validation error: {e}")
        return original


def _validate_indexes(recommendations: list) -> list[str]:
    """
    Validate index recommendation strings.

    Ensures each recommendation is a valid CREATE INDEX statement.
    """
    validated = []
    for rec in recommendations:
        if not isinstance(rec, str):
            continue

        rec = rec.strip()
        if rec.upper().startswith("CREATE INDEX") or rec.upper().startswith(
            "CREATE UNIQUE INDEX"
        ):
            validated.append(rec)

    return validated

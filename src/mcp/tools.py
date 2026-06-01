"""
MCP Tools — PostgreSQL Database Operations.

Defines the read-only tools exposed through the MCP server:
1. explain_analyze — Execute EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
2. get_table_stats — Return pg_stat_user_tables data
3. list_slow_queries — Return top-N slowest queries

All tools are READ-ONLY by design. No data mutations are allowed.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import psycopg

from src.config.settings import get_settings


import uuid

def run_explain_analyze(
    query: str,
    db_url: Optional[str] = None,
    table_names: Optional[list[str]] = None,
    index_recommendations: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Execute EXPLAIN (FORMAT JSON) on a query using HypoPG.

    If index_recommendations are provided, creates hypothetical indexes
    using the hypopg extension before running the explain. This provides
    accurate Cost estimates from the query planner based on real table
    statistics, without the overhead or false negatives of a physical sandbox.
    """
    settings = get_settings()
    mcp_url = settings.mcp_db_url

    index_statements = index_recommendations or []
    # Note: We cannot use ANALYZE with hypothetical indexes.
    # We rely on the Estimated Total Cost.
    explain_sql = f"EXPLAIN (FORMAT JSON) {query}"
    result = None

    try:
        # Step 1: Execute EXPLAIN as the restricted MCP role using HypoPG
        with psycopg.connect(mcp_url) as mcp_conn:
            mcp_conn.autocommit = True
            with mcp_conn.cursor() as cur:
                # Reset any previous hypothetical indexes in this session
                try:
                    cur.execute("SELECT * FROM hypopg_reset();")
                except Exception as e:
                    print(f"[Apex HypoPG] Reset failed (is extension enabled?): {e}")

                # Apply index recommendations hypothetically
                for idx in index_statements:
                    try:
                        # hypopg_create_index takes the CREATE INDEX statement as a string
                        cur.execute("SELECT * FROM hypopg_create_index(%s)", (idx,))
                    except Exception as e:
                        print(f"[Apex HypoPG] Index creation failed for '{idx}': {e}")
                
                cur.execute(explain_sql)
                result = cur.fetchone()

    except Exception as e:
        return {
            "plan": {},
            "execution_time_ms": 0.0,
            "planning_time_ms": 0.0,
            "node_type": "Error",
            "total_cost": 0.0,
            "actual_rows": 0,
            "success": False,
            "error": str(e),
        }

    # Process the result
    if result and result[0]:
        plan = result[0]
        if isinstance(plan, list) and len(plan) > 0:
            top_plan = plan[0]
            
            # Since we don't use ANALYZE, we don't get 'Execution Time'.
            # We map 'Total Cost' to 'execution_time_ms' to keep the rest of the
            # system's pipeline mathematically consistent (speedup = old / new).
            total_cost = top_plan.get("Plan", {}).get("Total Cost", 0.0)
            
            # Planning Time is also not returned without ANALYZE usually, 
            # but we can try to fetch it if available.
            planning_time = top_plan.get("Planning Time", 0.0)
            plan_node = top_plan.get("Plan", {})
            node_type = plan_node.get("Node Type", "Unknown")
            actual_rows = plan_node.get("Plan Rows", 0)  # Estimated rows

            return {
                "plan": plan,
                "execution_time_ms": float(total_cost),  # Map Cost to Time for speedup
                "planning_time_ms": float(planning_time),
                "node_type": node_type,
                "total_cost": float(total_cost),
                "actual_rows": int(actual_rows),
                "success": True,
            }

    return {
        "plan": {},
        "execution_time_ms": 0.0,
        "planning_time_ms": 0.0,
        "node_type": "Unknown",
        "total_cost": 0.0,
        "actual_rows": 0,
        "success": False,
        "error": "Empty result from EXPLAIN",
    }


def get_table_stats(
    table_names: Optional[list[str]] = None,
    db_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Get statistics for database tables from pg_stat_user_tables.

    Args:
        table_names: Optional list of table names to filter.
        db_url: Optional database connection URL.

    Returns:
        List of dictionaries with table statistics.
    """
    settings = get_settings()
    db_url = db_url or settings.supabase_db_url

    query = """
        SELECT
            schemaname,
            relname AS table_name,
            seq_scan,
            seq_tup_read,
            idx_scan,
            idx_tup_fetch,
            n_tup_ins AS inserts,
            n_tup_upd AS updates,
            n_tup_del AS deletes,
            n_live_tup AS live_rows,
            n_dead_tup AS dead_rows,
            pg_size_pretty(pg_total_relation_size(relid)) AS total_size
        FROM pg_stat_user_tables
    """

    if table_names:
        placeholders = ", ".join([f"'{t}'" for t in table_names])
        query += f" WHERE relname IN ({placeholders})"

    query += " ORDER BY n_live_tup DESC"

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc.name for desc in cur.description]
                rows = cur.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        return [{"error": str(e)}]


def list_slow_queries(
    limit: int = 10,
    db_url: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Get the top-N slowest queries from pg_stat_statements.

    Args:
        limit: Maximum number of queries to return.
        db_url: Optional database connection URL.

    Returns:
        List of dictionaries with query performance data.
    """
    settings = get_settings()
    db_url = db_url or settings.supabase_db_url

    query = f"""
        SELECT
            queryid::text AS query_id,
            LEFT(query, 500) AS query_text,
            calls,
            ROUND((total_exec_time / GREATEST(calls, 1))::numeric, 3)
                AS avg_exec_time_ms,
            ROUND(total_exec_time::numeric, 3) AS total_exec_time_ms,
            rows,
            CASE WHEN (shared_blks_hit + shared_blks_read) > 0
                THEN ROUND(
                    (shared_blks_hit::decimal /
                     (shared_blks_hit + shared_blks_read))::numeric, 4
                )
                ELSE 1.0
            END AS cache_hit_ratio
        FROM pg_stat_statements
        WHERE calls > 0
        ORDER BY total_exec_time / GREATEST(calls, 1) DESC
        LIMIT {limit}
    """

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                columns = [desc.name for desc in cur.description]
                rows = cur.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        return [{"error": str(e)}]


def get_table_schema(
    table_name: str,
    db_url: Optional[str] = None,
) -> dict[str, Any]:
    """
    Get the schema definition for a specific table.

    Returns column definitions, indexes, and constraints.
    """
    settings = get_settings()
    db_url = db_url or settings.supabase_db_url

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Get columns
                cur.execute("""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_name = %s
                    AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
                columns = [
                    {
                        "column": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3],
                    }
                    for row in cur.fetchall()
                ]

                # Get indexes
                cur.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = %s
                    AND schemaname = 'public'
                """, (table_name,))
                indexes = [
                    {"name": row[0], "definition": row[1]}
                    for row in cur.fetchall()
                ]

        return {
            "table_name": table_name,
            "columns": columns,
            "indexes": indexes,
        }

    except Exception as e:
        return {"table_name": table_name, "error": str(e)}

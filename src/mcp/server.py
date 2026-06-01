"""
MCP Server — PostgreSQL Bridge.

Model Context Protocol server that exposes read-only database
tools for the reasoning agents. Uses stdio transport for local
operation with connection pooling to Supabase.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.config.settings import get_settings
from src.mcp.tools import (
    get_table_schema,
    get_table_stats,
    list_slow_queries,
    run_explain_analyze,
)

# Create MCP server instance
mcp_server = Server("apex-postgres")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """Register available MCP tools."""
    return [
        Tool(
            name="explain_analyze",
            description=(
                "Execute EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) on a SQL query. "
                "Returns the execution plan, timing, and cost metrics. "
                "The query is executed in a rolled-back transaction for safety."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to analyze",
                    },
                    "table_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tables to clone into the sandbox schema",
                    },
                    "index_recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Indexes to apply in the sandbox",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_table_stats",
            description=(
                "Get statistics for database tables from pg_stat_user_tables. "
                "Returns scan counts, row counts, and size information."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of table names to filter",
                    },
                },
            },
        ),
        Tool(
            name="list_slow_queries",
            description=(
                "Get the top-N slowest queries from pg_stat_statements. "
                "Returns query text, call count, avg execution time, "
                "and cache hit ratio."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of queries to return (default: 10)",
                        "default": 10,
                    },
                },
            },
        ),
        Tool(
            name="get_table_schema",
            description=(
                "Get the schema definition (columns, indexes, constraints) "
                "for a specific table."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The table name to inspect",
                    },
                },
                "required": ["table_name"],
            },
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool invocations."""

    if name == "explain_analyze":
        result = run_explain_analyze(
            query=arguments["query"],
            table_names=arguments.get("table_names"),
            index_recommendations=arguments.get("index_recommendations"),
        )
    elif name == "get_table_stats":
        result = get_table_stats(
            table_names=arguments.get("table_names"),
        )
    elif name == "list_slow_queries":
        result = list_slow_queries(
            limit=arguments.get("limit", 10),
        )
    elif name == "get_table_schema":
        result = get_table_schema(
            table_name=arguments["table_name"],
        )
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [
        TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )
    ]


async def run_server() -> None:
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )


if __name__ == "__main__":
    print("[Apex] Starting MCP PostgreSQL Bridge Server...")
    asyncio.run(run_server())

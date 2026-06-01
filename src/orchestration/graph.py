"""
LangGraph Compilation.

Builds and compiles the stateful orchestration graph with:
- Conditional routing (local vs cloud)
- Retry loops with escalation
- Supabase PostgreSQL checkpointer for persistence
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, StateGraph

from src.config.settings import get_settings
from src.orchestration.nodes import (
    execute_node,
    log_node,
    reason_node,
    route_node,
    validate_node,
    verify_node,
)
from src.orchestration.router import should_retry
from src.orchestration.state import ApexGraphState


def build_graph(use_checkpointer: bool = True) -> StateGraph:
    """
    Build and compile the LangGraph orchestration pipeline.

    The graph implements the following flow:
        START → validate → route → reason → execute → verify
            ↓ (improved/regression) → log → END
            ↓ (no_change + retries < 3) → reason (retry)
            ↓ (no_change + retries >= 3) → log (escalated) → END

    Args:
        use_checkpointer: Whether to attach a Supabase checkpointer.

    Returns:
        Compiled LangGraph StateGraph.
    """
    # ── Build the Graph ───────────────────────────────────────
    builder = StateGraph(ApexGraphState)

    # Add nodes
    builder.add_node("validate", validate_node)
    builder.add_node("route", route_node)
    builder.add_node("reason", reason_node)
    builder.add_node("execute", execute_node)
    builder.add_node("verify", verify_node)
    builder.add_node("log", log_node)

    # ── Define Edges ──────────────────────────────────────────

    # Entry point
    builder.set_entry_point("validate")

    # validate → route (or END if validation failed)
    builder.add_conditional_edges(
        "validate",
        lambda state: "route" if not state.get("error") else "log",
        {"route": "route", "log": "log"},
    )

    # route → reason
    builder.add_edge("route", "reason")

    # reason → execute
    builder.add_edge("reason", "execute")

    # execute → verify
    builder.add_edge("execute", "verify")

    # verify → conditional (done/retry/escalate)
    builder.add_conditional_edges(
        "verify",
        should_retry,
        {
            "done": "log",
            "retry": "reason",
            "escalate": "log",
        },
    )

    # log → END
    builder.add_edge("log", END)

    # ── Compile with Checkpointer ─────────────────────────────
    checkpointer = None

    if use_checkpointer:
        try:
            checkpointer = _create_checkpointer()
        except Exception as e:
            print(
                f"[Apex] Could not create checkpointer: {e}\n"
                "       Running without persistence."
            )

    graph = builder.compile(checkpointer=checkpointer)
    print("[Apex] Orchestration graph compiled successfully")

    return graph


def _create_checkpointer():
    """
    Create a PostgreSQL checkpointer connected to Supabase.

    Uses langgraph-checkpoint-postgres for durable state persistence.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool

    settings = get_settings()

    pool = ConnectionPool(
        conninfo=settings.supabase_db_url,
        max_size=5,
        min_size=1,
    )

    checkpointer = PostgresSaver(pool)

    # Create checkpointer tables if they don't exist
    try:
        checkpointer.setup()
    except Exception:
        pass  # Tables may already exist

    return checkpointer


def invoke_graph(
    anomaly_data: dict,
    graph=None,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Invoke the orchestration graph with anomaly data.

    Args:
        anomaly_data: Raw anomaly data from the BiLSTM.
        graph: Pre-compiled graph (if None, builds a new one).
        thread_id: Thread ID for checkpointing.

    Returns:
        Final state dictionary with resolution.
    """
    if graph is None:
        graph = build_graph(use_checkpointer=False)

    # Initial state
    initial_state = {
        "anomaly_data": anomaly_data,
        "anomaly_report": None,
        "route": "",
        "severity": "low",
        "query_context": {},
        "optimized_query": "",
        "optimization_reasoning": "",
        "index_recommendations": [],
        "explain_before": {},
        "explain_after": {},
        "original_exec_ms": 0.0,
        "optimized_exec_ms": 0.0,
        "speedup_factor": 1.0,
        "resolution": "pending",
        "messages": [],
        "retry_count": 0,
        "incident_id": "",
        "model_used": "",
        "error": "",
    }

    # Configuration with thread ID for checkpointing
    config = {
        "configurable": {
            "thread_id": thread_id or anomaly_data.get("query_id", "default"),
        },
    }

    # Invoke the graph
    result = graph.invoke(initial_state, config=config)

    return result

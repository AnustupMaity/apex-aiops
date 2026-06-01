"""
LangGraph State Definitions.

Defines the TypedDict state that flows through the LangGraph
orchestration pipeline, tracking the full incident lifecycle
from anomaly detection to resolution.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ApexState(dict):
    """
    State dictionary for the LangGraph orchestration pipeline.

    Keys:
        anomaly_data: Raw anomaly data from BiLSTM + telemetry.
        anomaly_report: Validated AnomalyReport (from PydanticAI).
        route: Routing decision ('local' for Ollama, 'cloud' for escalation).
        query_context: Database schema context for the reasoning agent.
        optimized_query: The rewritten SQL from the reasoning agent.
        optimization_reasoning: LLM explanation of changes.
        index_recommendations: Suggested CREATE INDEX statements.
        explain_before: EXPLAIN ANALYZE output for original query.
        explain_after: EXPLAIN ANALYZE output for optimized query.
        original_exec_ms: Original query execution time.
        optimized_exec_ms: Optimized query execution time.
        speedup_factor: Performance improvement ratio.
        resolution: Current resolution status.
        messages: Conversation history for the reasoning agent.
        retry_count: Number of optimization retry attempts.
        incident_id: Unique ID for this incident.
        model_used: Which model performed the optimization.
        error: Any error encountered during processing.
    """
    pass


# Type annotations for the state — used by LangGraph's StateGraph
from typing import TypedDict


class ApexGraphState(TypedDict):
    """Typed state definition for LangGraph StateGraph."""

    # ── Input ─────────────────────────────────────────────────
    anomaly_data: dict[str, Any]

    # ── Validation ────────────────────────────────────────────
    anomaly_report: Optional[dict[str, Any]]

    # ── Routing ───────────────────────────────────────────────
    route: Literal["local", "cloud", ""]
    severity: str

    # ── Reasoning ─────────────────────────────────────────────
    query_context: dict[str, Any]
    optimized_query: str
    optimization_reasoning: str
    index_recommendations: list[str]

    # ── Execution ─────────────────────────────────────────────
    explain_before: dict[str, Any]
    explain_after: dict[str, Any]
    original_exec_ms: float
    optimized_exec_ms: float
    speedup_factor: float

    # ── Resolution ────────────────────────────────────────────
    resolution: Literal[
        "pending", "improved", "no_change", "escalated", "failed"
    ]

    # ── Metadata ──────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: int
    incident_id: str
    model_used: str
    error: str

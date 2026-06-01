"""
Severity-Based Router for the LangGraph Pipeline.

Routes anomaly incidents to either the local Ollama SLM
or the cloud LLM based on degradation factor severity.
"""

from __future__ import annotations

from typing import Literal

from src.orchestration.state import ApexGraphState


# ── Routing Thresholds ────────────────────────────────────────
# Degradation factor ranges and corresponding routes
ROUTE_MAP: dict[str, Literal["local", "cloud"]] = {
    "low": "local",       # < 2x degradation → Ollama
    "medium": "local",    # 2x-5x degradation → Ollama
    "high": "cloud",      # 5x-10x degradation → Cloud LLM
    "critical": "cloud",  # 10x+ degradation → Cloud LLM
}


def route_by_severity(state: ApexGraphState) -> Literal["local", "cloud"]:
    """
    Determine routing based on anomaly severity.

    Args:
        state: Current graph state containing the anomaly report.

    Returns:
        'local' for Ollama processing, 'cloud' for escalation.
    """
    severity = state.get("severity", "low")
    return ROUTE_MAP.get(severity, "local")


def should_retry(state: ApexGraphState) -> Literal["retry", "escalate", "done"]:
    """
    Determine whether to retry optimization or escalate.

    Called after the verify node to decide next action:
    - 'done': Query was improved, proceed to logging
    - 'retry': Query not improved but retries remain
    - 'escalate': Query not improved and retries exhausted

    Args:
        state: Current graph state with resolution and retry count.

    Returns:
        Next action: 'retry', 'escalate', or 'done'.
    """
    resolution = state.get("resolution", "pending")
    retry_count = state.get("retry_count", 0)

    if resolution == "improved":
        return "done"

    if resolution == "regression":
        return "done"  # Don't retry regressions, just log them

    # No change — retry if under limit
    if retry_count < 3:
        return "retry"

    return "escalate"

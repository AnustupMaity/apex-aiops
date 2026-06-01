"""
LangSmith Observability Setup.

Configures tracing for the entire Project Apex pipeline,
including custom spans for non-LangChain functions and
metadata tagging for incident-level correlation.
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable, Optional

from src.config.settings import get_settings


def setup_tracing() -> bool:
    """
    Initialize LangSmith tracing from environment configuration.

    Returns True if tracing was successfully configured.
    """
    settings = get_settings()

    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        print("[Apex] LangSmith tracing disabled (no API key)")
        return False

    # Set environment variables for automatic tracing
    settings.setup_langsmith_env()

    print(
        f"[Apex] LangSmith tracing enabled "
        f"(project: {settings.langsmith_project})"
    )
    return True


def traceable_apex(
    name: Optional[str] = None,
    run_type: str = "chain",
    metadata: Optional[dict[str, Any]] = None,
) -> Callable:
    """
    Decorator that wraps a function with LangSmith tracing.

    Falls back to a no-op decorator if LangSmith is not configured.

    Args:
        name: Name for the trace span.
        run_type: Type of run ('chain', 'tool', 'llm', 'retriever').
        metadata: Additional metadata to attach to the trace.

    Usage:
        @traceable_apex(name="validate_anomaly", run_type="chain")
        def validate_anomaly(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Try to import langsmith traceable
        try:
            from langsmith import traceable

            traced_func = traceable(
                name=name or func.__name__,
                run_type=run_type,
                metadata=metadata or {},
            )(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                return traced_func(*args, **kwargs)

            return wrapper

        except ImportError:
            # LangSmith not installed, return function as-is
            return func

    return decorator


def create_run_config(
    incident_id: str,
    severity: str = "unknown",
    model_used: str = "unknown",
    tags: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Create a LangGraph/LangChain run configuration with
    incident-level metadata for trace correlation.

    Args:
        incident_id: Unique incident identifier.
        severity: Anomaly severity level.
        model_used: Which model performed the optimization.
        tags: Additional tags for filtering traces.

    Returns:
        Configuration dict compatible with LangGraph invoke().
    """
    return {
        "metadata": {
            "incident_id": incident_id,
            "severity": severity,
            "model_used": model_used,
            "project": "apex",
        },
        "tags": tags or [f"severity:{severity}", "apex"],
        "configurable": {
            "thread_id": incident_id,
        },
    }

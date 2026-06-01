"""
Ollama Agent — Local SQL Reasoning.

Integrates with Ollama running Qwen2.5-Coder 1.5B for local,
low-latency SQL query optimization. Ensures data privacy by
keeping all reasoning on-device.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.config.settings import get_settings
from src.reasoning.prompts import SQL_OPTIMIZER_SYSTEM_PROMPT, SQL_OPTIMIZER_USER_PROMPT


def create_ollama_agent() -> ChatOllama:
    """
    Create a ChatOllama instance configured for SQL optimization.

    Uses low temperature for deterministic SQL generation and
    JSON output mode for structured responses.
    """
    settings = get_settings()

    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_predict=2048,
        format="json",
    )

    return llm


def optimize_with_ollama(
    query: str,
    table_names: list[str],
    current_exec_ms: float = 0.0,
    baseline_exec_ms: float = 0.0,
    schema_context: str = "",
) -> dict[str, Any]:
    """
    Use Ollama (Qwen2.5-Coder 1.5B) to optimize a SQL query.

    Args:
        query: The slow SQL query to optimize.
        table_names: Tables referenced in the query.
        current_exec_ms: Current execution time in ms.
        baseline_exec_ms: Baseline execution time in ms.
        schema_context: Formatted database schema information.

    Returns:
        Dictionary with 'optimized_query', 'reasoning',
        and 'index_recommendations'.
    """
    llm = create_ollama_agent()

    # Compute degradation factor
    degradation_factor = (
        current_exec_ms / max(baseline_exec_ms, 0.001)
        if baseline_exec_ms > 0 else 1.0
    )

    # Format the user prompt
    user_prompt = SQL_OPTIMIZER_USER_PROMPT.format(
        query=query,
        table_names=", ".join(table_names),
        current_exec_ms=current_exec_ms,
        baseline_exec_ms=baseline_exec_ms,
        degradation_factor=degradation_factor,
        schema_context=schema_context,
    )

    messages = [
        SystemMessage(content=SQL_OPTIMIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = llm.invoke(messages)
        content = response.content

        # Parse JSON response
        result = json.loads(content)

        return {
            "optimized_query": result.get("optimized_query", query),
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "index_recommendations": result.get("index_recommendations", []),
            "model": "ollama:" + get_settings().ollama_model,
            "raw_response": content,
        }

    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract SQL from the response
        return {
            "optimized_query": query,
            "reasoning": f"Failed to parse JSON response: {content[:200]}",
            "index_recommendations": [],
            "model": "ollama:" + get_settings().ollama_model,
            "error": "JSON parse error",
        }
    except Exception as e:
        return {
            "optimized_query": query,
            "reasoning": f"Ollama error: {str(e)}",
            "index_recommendations": [],
            "model": "ollama:" + get_settings().ollama_model,
            "error": str(e),
        }

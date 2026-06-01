"""
Cloud Agent — Escalation LLM for Complex Queries.

Implements a highly resilient, free-tier fallback chain for reasoning:
Primary: Gemini 2.5 Flash
Fallback 1: Groq Llama 3.3 70B
Fallback 2: OpenRouter DeepSeek R1
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.config.settings import get_settings
from src.reasoning.prompts import SQL_OPTIMIZER_SYSTEM_PROMPT, SQL_OPTIMIZER_USER_PROMPT


def create_cloud_agent():
    """
    Create the cloud LLM agent with a robust fallback chain.
    """
    settings = get_settings()

    # Primary: Gemini 2.5 Flash
    primary_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.1,
        max_output_tokens=2048,
    )

    fallbacks = []

    # Fallback 1: Groq (Llama 3.3 70B)
    if settings.groq_api_key:
        groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0.1,
            max_tokens=2048,
        )
        fallbacks.append(groq_llm)

    # Fallback 2: OpenRouter (DeepSeek R1)
    if settings.openrouter_api_key:
        openrouter_llm = ChatOpenAI(
            model="deepseek/deepseek-r1:free",
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=2048,
        )
        fallbacks.append(openrouter_llm)

    # Chain them together
    if fallbacks:
        return primary_llm.with_fallbacks(fallbacks)
    
    return primary_llm


def optimize_with_cloud(
    query: str,
    table_names: list[str],
    current_exec_ms: float = 0.0,
    baseline_exec_ms: float = 0.0,
    schema_context: str = "",
) -> dict[str, Any]:
    """
    Use cloud LLM fallback chain to optimize a complex SQL query.
    """
    llm = create_cloud_agent()

    degradation_factor = (
        current_exec_ms / max(baseline_exec_ms, 0.001)
        if baseline_exec_ms > 0 else 1.0
    )

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

        # Try to extract JSON from the response
        json_content = content
        if "```json" in content:
            json_content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(json_content)

        return {
            "optimized_query": result.get("optimized_query", query),
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "index_recommendations": result.get("index_recommendations", []),
            "model": "cloud_fallback_chain",
            "raw_response": content,
        }

    except json.JSONDecodeError:
        return {
            "optimized_query": query,
            "reasoning": f"Failed to parse cloud LLM response",
            "index_recommendations": [],
            "model": "cloud_fallback_chain",
            "error": "JSON parse error",
        }
    except Exception as e:
        return {
            "optimized_query": query,
            "reasoning": f"Cloud LLM error: {str(e)}",
            "index_recommendations": [],
            "model": "cloud_fallback_chain",
            "error": str(e),
        }

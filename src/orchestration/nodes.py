"""
LangGraph Node Functions.

Each function represents a node in the orchestration graph:
1. validate_node   — PydanticAI validation
2. route_node      — Severity-based routing
3. reason_node     — LLM query optimization (Ollama or Cloud)
4. execute_node    — MCP EXPLAIN ANALYZE execution
5. verify_node     — Before/after comparison
6. log_node        — Persist incident to Supabase
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import get_settings
from src.orchestration.router import route_by_severity
from src.orchestration.state import ApexGraphState
from src.validation.schemas import AnomalyReport, IncidentRecord, OptimizationResult
from src.validation.validator_agent import validate_anomaly


def validate_node(state: ApexGraphState) -> dict[str, Any]:
    """
    Node 1: Validate raw anomaly data into a structured AnomalyReport.

    Takes raw anomaly_data from the BiLSTM and produces a validated,
    typed AnomalyReport using PydanticAI validation.
    """
    anomaly_data = state.get("anomaly_data", {})

    try:
        report = validate_anomaly(
            anomaly_score=anomaly_data.get("anomaly_score", 0.0),
            affected_query=anomaly_data.get("affected_query", ""),
            baseline_exec_ms=anomaly_data.get("baseline_exec_ms", 1.0),
            current_exec_ms=anomaly_data.get("current_exec_ms", 1.0),
            source_metrics=anomaly_data.get("source_metrics", {}),
        )

        return {
            "anomaly_report": report.model_dump(),
            "severity": report.severity,
            "incident_id": report.anomaly_id,
            "error": "",
        }

    except Exception as e:
        return {
            "anomaly_report": None,
            "severity": "low",
            "incident_id": str(uuid4()),
            "error": f"Validation failed: {str(e)}",
            "resolution": "failed",
        }


def route_node(state: ApexGraphState) -> dict[str, Any]:
    """
    Node 2: Route to local or cloud LLM based on severity.
    """
    route = route_by_severity(state)
    settings = get_settings()

    model_used = (
        settings.ollama_model if route == "local"
        else f"cloud:{settings.cloud_llm_provider}"
    )

    return {
        "route": route,
        "model_used": model_used,
    }


def reason_node(state: ApexGraphState) -> dict[str, Any]:
    """
    Node 3: Call the reasoning agent (Ollama or Cloud) to optimize the query.

    Uses the SQL optimizer to rewrite the query and generate
    index recommendations.
    """
    from src.reasoning.sql_optimizer import optimize_query

    anomaly_report = state.get("anomaly_report", {})
    route = state.get("route", "local")
    query_context = state.get("query_context", {})

    affected_query = anomaly_report.get("affected_query", "")
    table_names = anomaly_report.get("table_names", [])

    try:
        result = optimize_query(
            query=affected_query,
            table_names=table_names,
            query_context=query_context,
            use_local=(route == "local"),
        )

        return {
            "optimized_query": result.get("optimized_query", affected_query),
            "optimization_reasoning": result.get("reasoning", ""),
            "index_recommendations": result.get("index_recommendations", []),
            "messages": [
                HumanMessage(content=f"Optimize query: {affected_query}"),
            ],
        }

    except Exception as e:
        return {
            "optimized_query": affected_query,
            "optimization_reasoning": f"Optimization failed: {str(e)}",
            "index_recommendations": [],
            "error": str(e),
        }


def execute_node(state: ApexGraphState) -> dict[str, Any]:
    """
    Node 4: Execute EXPLAIN ANALYZE via MCP for both original and optimized queries.

    Runs the queries through the MCP PostgreSQL bridge to collect
    execution plans and timing data.
    """
    from src.mcp.tools import run_explain_analyze

    anomaly_report = state.get("anomaly_report", {})
    original_query = anomaly_report.get("affected_query", "")
    optimized_query = state.get("optimized_query", "")

    try:
        # Run EXPLAIN ANALYZE on original query
        original_result = run_explain_analyze(
            query=original_query,
            table_names=anomaly_report.get("table_names", [])
        )
        original_exec_ms = original_result.get("execution_time_ms", 0.0)

        # Run EXPLAIN ANALYZE on optimized query
        optimized_result = run_explain_analyze(
            query=optimized_query,
            table_names=anomaly_report.get("table_names", []),
            index_recommendations=state.get("index_recommendations", [])
        )
        optimized_exec_ms = optimized_result.get("execution_time_ms", 0.0)

        return {
            "explain_before": original_result,
            "explain_after": optimized_result,
            "original_exec_ms": original_exec_ms,
            "optimized_exec_ms": optimized_exec_ms,
        }

    except Exception as e:
        return {
            "explain_before": {},
            "explain_after": {},
            "original_exec_ms": 0.0,
            "optimized_exec_ms": 0.0,
            "error": f"Execution failed: {str(e)}",
        }


def verify_node(state: ApexGraphState) -> dict[str, Any]:
    """
    Node 5: Compare before/after execution plans and determine resolution.

    Computes speedup factor and sets the resolution status:
    - 'improved': Optimized query is faster
    - 'no_change': No significant improvement
    - 'regression': Optimized query is slower
    """
    original_ms = state.get("original_exec_ms", 0.0)
    optimized_ms = state.get("optimized_exec_ms", 0.0)
    retry_count = state.get("retry_count", 0)

    # Compute speedup
    if optimized_ms > 0 and original_ms > 0:
        speedup = original_ms / optimized_ms
    else:
        speedup = 1.0

    # Determine resolution
    if speedup > 1.1:  # At least 10% improvement
        resolution = "improved"
    elif speedup < 0.9:  # Regression (optimized is slower)
        resolution = "regression"
    else:
        resolution = "no_change"

    return {
        "speedup_factor": speedup,
        "resolution": resolution,
        "retry_count": retry_count + 1,
    }


def log_node(state: ApexGraphState) -> dict[str, Any]:
    """
    Node 6: Persist the incident record to Supabase.

    Creates an IncidentRecord from the final state and writes
    it to the apex_incidents table.
    """
    import psycopg

    settings = get_settings()

    try:
        anomaly_report = state.get("anomaly_report") or {}
        
        # Safely get values, handling the case where anomaly_report is a dict or a Pydantic model
        def get_val(key, default):
            if hasattr(anomaly_report, "get"):
                return anomaly_report.get(key, default)
            elif hasattr(anomaly_report, key):
                return getattr(anomaly_report, key, default)
            return default

        record = {
            "incident_id": state.get("incident_id", str(uuid4())),
            "severity": state.get("severity", "low"),
            "anomaly_score": get_val("anomaly_score", 0.0),
            "affected_query": get_val("affected_query", ""),
            "table_names": get_val("table_names", []),
            "baseline_exec_ms": get_val("baseline_exec_ms", 0.0),
            "current_exec_ms": get_val("current_exec_ms", 0.0),
            "degradation_factor": get_val("degradation_factor", 1.0),
            "optimized_query": state.get("optimized_query", ""),
            "original_plan": json.dumps(state.get("explain_before", {})),
            "optimized_plan": json.dumps(state.get("explain_after", {})),
            "speedup_factor": state.get("speedup_factor", 1.0),
            "index_recommendations": state.get("index_recommendations", []),
            "resolution": state.get("resolution", "pending"),
            "model_used": state.get("model_used", ""),
            "langgraph_thread_id": state.get("incident_id", ""),
            "resolved_at": datetime.utcnow().isoformat(),
        }

        with psycopg.connect(settings.supabase_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO apex_incidents (
                        incident_id, severity, anomaly_score, affected_query,
                        table_names, baseline_exec_ms, current_exec_ms,
                        degradation_factor, optimized_query, original_plan,
                        optimized_plan, speedup_factor, index_recommendations,
                        resolution, model_used, langgraph_thread_id, resolved_at
                    ) VALUES (
                        %(incident_id)s, %(severity)s, %(anomaly_score)s,
                        %(affected_query)s, %(table_names)s, %(baseline_exec_ms)s,
                        %(current_exec_ms)s, %(degradation_factor)s,
                        %(optimized_query)s, %(original_plan)s, %(optimized_plan)s,
                        %(speedup_factor)s, %(index_recommendations)s,
                        %(resolution)s, %(model_used)s, %(langgraph_thread_id)s,
                        %(resolved_at)s
                    )
                    ON CONFLICT (incident_id) DO UPDATE SET
                        resolution = EXCLUDED.resolution,
                        optimized_query = EXCLUDED.optimized_query,
                        speedup_factor = EXCLUDED.speedup_factor,
                        resolved_at = EXCLUDED.resolved_at
                """, record)
                conn.commit()

        print(
            f"[Apex] Incident {record['incident_id'][:8]}... "
            f"logged as '{record['resolution']}'"
        )

        return {"resolution": record["resolution"]}

    except Exception as e:
        print(f"[Apex] Failed to log incident: {e}")
        return {"error": f"Logging failed: {str(e)}"}

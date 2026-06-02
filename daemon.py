"""
Project Apex — Background Monitoring Daemon.

Main entry point that orchestrates the full AIOps pipeline:
1. Initializes settings, loads BiLSTM model
2. Starts async telemetry collector
3. On anomaly detection → creates LangGraph thread → invokes graph
4. Logs results to Supabase
5. Provides a FastAPI health endpoint

Usage:
    python daemon.py
    # Or with uvicorn:
    uvicorn daemon:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rich.console import Console

from src.config.settings import get_settings
from src.observability.tracer import setup_tracing

console = Console()

# ── Global State ──────────────────────────────────────────────
detector = None
collector = None
graph = None
daemon_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler — start/stop the daemon."""
    global detector, collector, graph, daemon_running

    console.print("\n[bold magenta]🚀 Project Apex — Daemon Starting[/]")
    settings = get_settings()

    # Step 1: Initialize LangSmith tracing
    setup_tracing()

    # Step 2: Load BiLSTM model
    try:
        from src.models.inference import AnomalyDetector
        model_path = settings.models_dir / "best_bilstm.pt"
        if model_path.exists():
            detector = AnomalyDetector(model_path=model_path)
            console.print("[green]✅ BiLSTM model loaded[/]")
        else:
            console.print(
                "[yellow]⚠ No trained model found at "
                f"{model_path}. Run training first.[/]"
            )
    except Exception as e:
        console.print(f"[yellow]⚠ Could not load BiLSTM: {e}[/]")

    # Step 3: Build LangGraph
    try:
        from src.orchestration.graph import build_graph
        graph = build_graph(use_checkpointer=False)
        console.print("[green]✅ LangGraph orchestration compiled[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not build graph: {e}[/]")

    # Step 4: Start telemetry collector
    try:
        from src.telemetry.collector import TelemetryCollector
        collector = TelemetryCollector()
        daemon_running = True

        # Start collector and monitoring loop in background
        asyncio.create_task(collector.start())
        asyncio.create_task(_monitoring_loop())
        asyncio.create_task(_active_learning_loop())
        console.print("[green]✅ Telemetry collector started[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not start collector: {e}[/]")

    console.print("\n[bold green]✅ Daemon ready![/]\n")

    yield  # App is running

    # Shutdown
    daemon_running = False
    if collector:
        await collector.stop()
    console.print("\n[bold yellow]⏹ Daemon stopped[/]")


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="Project Apex",
    description="Autonomous Database Performance Tuning Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running" if daemon_running else "stopped",
        "model_loaded": detector is not None,
        "graph_compiled": graph is not None,
        "collector_active": collector is not None and collector._running,
        "buffer_size": len(collector.buffer) if collector else 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get recent telemetry metrics."""
    if not collector:
        return JSONResponse(
            status_code=503,
            content={"error": "Collector not running"},
        )

    recent = list(collector.buffer)[-60:]  # Last 5 minutes
    return {
        "metrics": [
            {
                "timestamp": s.timestamp.isoformat(),
                "mean_exec_time_ms": s.mean_exec_time_ms,
                "cache_hit_ratio": s.cache_hit_ratio,
                "active_connections": s.active_connections,
                "seq_scan_count": s.seq_scan_count,
                "idx_scan_count": s.idx_scan_count,
            }
            for s in recent
        ],
        "count": len(recent),
    }


@app.get("/api/incidents")
async def get_incidents():
    """Get recent incidents from Supabase."""
    try:
        import psycopg
        settings = get_settings()

        with psycopg.connect(settings.supabase_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        incident_id, created_at, resolved_at, severity,
                        anomaly_score, affected_query, table_names,
                        baseline_exec_ms, current_exec_ms, degradation_factor,
                        optimized_query, speedup_factor, index_recommendations,
                        resolution, model_used
                    FROM apex_incidents
                    ORDER BY created_at DESC
                    LIMIT 50
                """)
                columns = [desc.name for desc in cur.description]
                rows = cur.fetchall()

        incidents = []
        for row in rows:
            incident = dict(zip(columns, row))
            # Serialize datetime objects
            for key in ["created_at", "resolved_at"]:
                if incident.get(key):
                    incident[key] = incident[key].isoformat()
            incidents.append(incident)

        return {"incidents": incidents, "count": len(incidents)}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/api/trigger")
async def trigger_anomaly(data: dict):
    """
    Manually trigger an anomaly for testing.

    Body:
        {
            "query": "SELECT * FROM orders ...",
            "baseline_exec_ms": 10.0,
            "current_exec_ms": 500.0
        }
    """
    if not graph:
        return JSONResponse(
            status_code=503,
            content={"error": "Graph not compiled"},
        )

    anomaly_data = {
        "anomaly_score": 0.95,
        "affected_query": data.get("query", "SELECT 1"),
        "baseline_exec_ms": data.get("baseline_exec_ms", 10.0),
        "current_exec_ms": data.get("current_exec_ms", 500.0),
        "source_metrics": {"is_simulated": True},
        "query_id": str(uuid4()),
    }

    try:
        from src.orchestration.graph import invoke_graph
        result = invoke_graph(anomaly_data, graph=graph)

        return {
            "incident_id": result.get("incident_id", ""),
            "resolution": result.get("resolution", "unknown"),
            "speedup_factor": result.get("speedup_factor", 1.0),
            "optimized_query": result.get("optimized_query", ""),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

@app.post("/api/retrain")
async def trigger_retraining():
    """Manually trigger the Active Learning pipeline."""
    try:
        from src.models.active_learning import run_active_learning
        # Run in a separate thread so we don't block the async loop
        asyncio.create_task(asyncio.to_thread(run_active_learning))
        return {"status": "Active learning pipeline triggered."}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )

@app.get("/api/settings")
async def get_app_settings():
    """Retrieve current application settings."""
    settings = get_settings()
    return settings.model_dump(exclude={"supabase_service_key", "gemini_api_key", "groq_api_key", "openrouter_api_key"})

@app.post("/api/settings")
async def update_app_settings(updates: dict):
    """Update environment variables and reload settings."""
    try:
        from src.config.settings import Settings
        Settings.update_env_file(updates)
        
        # Reload detector model if threshold changed (optional, but since threshold is used during scoring, reload is not strictly needed for threshold, but we do need the new settings)
        settings = get_settings()
        
        return {"status": "success", "settings": settings.model_dump(exclude={"supabase_service_key", "gemini_api_key", "groq_api_key", "openrouter_api_key"})}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )



# ── Monitoring Loop ───────────────────────────────────────────

async def _monitoring_loop():
    """
    Background loop that feeds telemetry windows to the BiLSTM
    and triggers the LangGraph pipeline on anomaly detection.
    """
    global detector, collector, graph, daemon_running

    settings = get_settings()

    while daemon_running:
        try:
            # Wait for enough data in the buffer
            if not collector or not detector:
                await asyncio.sleep(5)
                continue

            window = collector.get_latest_window(
                window_size=settings.bilstm_seq_len
            )

            if window is None:
                await asyncio.sleep(settings.telemetry_poll_interval_seconds)
                continue

            # Score the window
            score = detector.score_window(window)

            if score.is_anomaly:
                console.print(
                    f"[bold red]🚨 ANOMALY DETECTED[/] "
                    f"(score: {score.anomaly_score:.4f}, "
                    f"confidence: {score.confidence:.4f})"
                )

                # Get the slowest query as the affected query
                slow_queries = collector.get_slow_queries()
                if slow_queries:
                    slowest = slow_queries[0]
                    anomaly_data = {
                        "anomaly_score": score.anomaly_score,
                        "affected_query": slowest.query_text,
                        "baseline_exec_ms": slowest.avg_exec_time_ms / 2,
                        "current_exec_ms": slowest.avg_exec_time_ms,
                        "source_metrics": {
                            "reconstruction_error": score.reconstruction_error,
                            "threshold": score.threshold,
                        },
                        "query_id": slowest.query_id,
                    }

                    # Trigger the orchestration pipeline
                    if graph:
                        from src.orchestration.graph import invoke_graph
                        asyncio.create_task(
                            asyncio.to_thread(
                                invoke_graph, anomaly_data, graph
                            )
                        )

        except Exception as e:
            console.print(f"[red]Monitoring error: {e}[/]")

        await asyncio.sleep(settings.telemetry_poll_interval_seconds)

async def _active_learning_loop():
    """
    Background loop that runs the active learning pipeline periodically.
    Defaults to running every 7 days (604800 seconds).
    """
    global daemon_running
    from src.models.active_learning import run_active_learning

    # Wait an hour before the first check
    await asyncio.sleep(3600)
    
    while daemon_running:
        try:
            # Run in a separate thread to avoid blocking
            await asyncio.to_thread(run_active_learning)
        except Exception as e:
            console.print(f"[red]Active learning loop error: {e}[/]")
            
        # Sleep for a week
        await asyncio.sleep(7 * 24 * 3600)


# ── CLI Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "daemon:app",
        host=settings.daemon_host,
        port=settings.daemon_port,
        reload=False,
        log_level="info",
    )

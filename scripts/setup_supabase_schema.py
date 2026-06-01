"""
Supabase Schema Setup Script.

Creates all required database objects:
1. Food-delivery sandbox tables (from SQL Practice Dataset 2)
2. Project Apex internal tables (incidents, metrics)
3. pg_stat_statements extension and views
4. LangGraph checkpointer tables
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg
from rich.console import Console

from src.config.settings import get_settings

console = Console()


# ── SQL Statements ────────────────────────────────────────────

ENABLE_EXTENSIONS = """
-- Enable pg_stat_statements for query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
-- Enable hypopg for hypothetical indexing
CREATE EXTENSION IF NOT EXISTS hypopg;
"""

CREATE_SANDBOX_TABLES = """
-- ============================================================
-- Food Delivery Sandbox Schema (from SQL Practice Dataset 2)
-- ============================================================

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    city VARCHAR(100),
    signup_date DATE DEFAULT CURRENT_DATE,
    email VARCHAR(200),
    phone VARCHAR(20)
);

-- Restaurants table
CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id SERIAL PRIMARY KEY,
    restaurant_name VARCHAR(200) NOT NULL,
    city VARCHAR(100),
    cuisine_type VARCHAR(100),
    rating DECIMAL(2,1) DEFAULT 0.0,
    opening_hours VARCHAR(50)
);

-- Menu items table
CREATE TABLE IF NOT EXISTS menu_items (
    item_id SERIAL PRIMARY KEY,
    restaurant_id INTEGER REFERENCES restaurants(restaurant_id) ON DELETE CASCADE,
    item_name VARCHAR(200) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(100),
    is_available BOOLEAN DEFAULT TRUE
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id) ON DELETE CASCADE,
    restaurant_id INTEGER REFERENCES restaurants(restaurant_id) ON DELETE CASCADE,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10,2),
    delivery_time_minutes INTEGER,
    status VARCHAR(50) DEFAULT 'delivered'
);

-- Order details (bridge table)
CREATE TABLE IF NOT EXISTS order_details (
    detail_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id) ON DELETE CASCADE,
    item_id INTEGER REFERENCES menu_items(item_id) ON DELETE CASCADE,
    quantity INTEGER DEFAULT 1,
    subtotal DECIMAL(10,2)
);

-- Create indexes that would normally exist
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_restaurant ON orders(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_order_details_order ON order_details(order_id);
CREATE INDEX IF NOT EXISTS idx_menu_items_restaurant ON menu_items(restaurant_id);
"""

CREATE_APEX_TABLES = """
-- ============================================================
-- Project Apex Internal Tables
-- ============================================================

-- Anomaly incidents tracked by the system
CREATE TABLE IF NOT EXISTS apex_incidents (
    incident_id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    severity VARCHAR(20) NOT NULL,
    anomaly_score DECIMAL(8,6),
    affected_query TEXT,
    table_names TEXT[],
    baseline_exec_ms DECIMAL(12,3),
    current_exec_ms DECIMAL(12,3),
    degradation_factor DECIMAL(8,3),
    optimized_query TEXT,
    original_plan JSONB,
    optimized_plan JSONB,
    speedup_factor DECIMAL(8,3),
    index_recommendations TEXT[],
    resolution VARCHAR(30) DEFAULT 'pending',
    model_used VARCHAR(50),
    langgraph_thread_id VARCHAR(100),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Real-time telemetry metrics
CREATE TABLE IF NOT EXISTS apex_metrics (
    id SERIAL PRIMARY KEY,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_exec_time DECIMAL(15,3),
    total_calls BIGINT,
    rows_returned BIGINT,
    mean_exec_time_ms DECIMAL(12,3),
    shared_blks_hit BIGINT,
    shared_blks_read BIGINT,
    cache_hit_ratio DECIMAL(5,4),
    active_connections INTEGER,
    seq_scan_count BIGINT,
    idx_scan_count BIGINT
);

-- Indexes for Apex tables
CREATE INDEX IF NOT EXISTS idx_incidents_created ON apex_incidents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON apex_incidents(severity);
CREATE INDEX IF NOT EXISTS idx_incidents_resolution ON apex_incidents(resolution);
CREATE INDEX IF NOT EXISTS idx_metrics_recorded ON apex_metrics(recorded_at DESC);
"""

CREATE_PERF_VIEWS = """
-- ============================================================
-- Performance Monitoring Views
-- ============================================================

-- View: Slowest queries from pg_stat_statements
CREATE OR REPLACE VIEW apex_slow_queries AS
SELECT
    queryid,
    query,
    calls,
    total_exec_time / calls AS avg_exec_time_ms,
    total_exec_time,
    rows,
    shared_blks_hit,
    shared_blks_read,
    CASE
        WHEN shared_blks_hit + shared_blks_read > 0
        THEN shared_blks_hit::decimal / (shared_blks_hit + shared_blks_read)
        ELSE 1.0
    END AS cache_hit_ratio
FROM pg_stat_statements
WHERE calls > 0
ORDER BY avg_exec_time_ms DESC;

-- View: Table-level scan statistics
CREATE OR REPLACE VIEW apex_table_stats AS
SELECT
    schemaname,
    relname AS table_name,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_live_tup,
    n_dead_tup,
    last_vacuum,
    last_analyze
FROM pg_stat_user_tables
ORDER BY seq_scan DESC;
"""


def setup_schema() -> None:
    """Execute all schema setup steps."""
    settings = get_settings()

    console.print("[bold magenta]🔧 Project Apex — Supabase Schema Setup[/]\n")
    console.print(f"   Connecting to: {settings.supabase_db_url[:50]}...")

    try:
        with psycopg.connect(settings.supabase_db_url) as conn:
            with conn.cursor() as cur:
                # Step 1: Extensions
                console.print("\n[cyan]1/4[/] Enabling extensions...")
                try:
                    cur.execute(ENABLE_EXTENSIONS)
                    conn.commit()
                    console.print("   [green]✅ pg_stat_statements enabled[/]")
                except Exception as e:
                    conn.rollback()
                    console.print(
                        f"   [yellow]⚠ Could not enable pg_stat_statements: {e}[/]\n"
                        "   (May require superuser privileges)"
                    )

                # Step 2: Sandbox tables
                console.print("\n[cyan]2/4[/] Creating sandbox tables...")
                cur.execute(CREATE_SANDBOX_TABLES)
                conn.commit()
                console.print(
                    "   [green]✅ customers, restaurants, menu_items, "
                    "orders, order_details[/]"
                )

                # Step 3: Apex tables
                console.print("\n[cyan]3/4[/] Creating Apex internal tables...")
                cur.execute(CREATE_APEX_TABLES)
                conn.commit()
                console.print(
                    "   [green]✅ apex_incidents, apex_metrics[/]"
                )

                # Step 4: Performance views
                console.print("\n[cyan]4/4[/] Creating performance views...")
                try:
                    cur.execute(CREATE_PERF_VIEWS)
                    conn.commit()
                    console.print(
                        "   [green]✅ apex_slow_queries, apex_table_stats[/]"
                    )
                except Exception as e:
                    conn.rollback()
                    console.print(
                        f"   [yellow]⚠ Could not create views: {e}[/]\n"
                        "   (pg_stat_statements may not be available)"
                    )

        # Step 6: Create restricted RBAC role for MCP execution
        console.print("\n[bold cyan]🔒 Setting up restricted MCP role (RBAC)...[/]")
        try:
            with conn.cursor() as cur:
                # We catch errors here because creating roles can fail if it already exists
                # and PostgreSQL doesn't have CREATE ROLE IF NOT EXISTS
                cur.execute("""
                    DO $$
                    BEGIN
                      IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'apex_mcp_role') THEN
                        CREATE ROLE apex_mcp_role WITH LOGIN PASSWORD 'mcp_restricted_pass123';
                      END IF;
                    END
                    $$;
                """)
                # Grant usage on schema
                cur.execute("GRANT USAGE ON SCHEMA public TO apex_mcp_role;")
                # Grant read-only access to all tables
                cur.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO apex_mcp_role;")
                # Ensure future tables also get read-only access
                cur.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO apex_mcp_role;")
            conn.commit()
            console.print("[green]✅ RBAC role 'apex_mcp_role' configured successfully[/]")
        except Exception as role_err:
            console.print(f"[yellow]⚠ Could not create MCP role (may already exist or require superuser): {role_err}[/]")
            conn.rollback()

        console.print("\n[bold green]✅ Schema setup complete![/]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Connection failed: {e}[/]")
        console.print(
            "   Make sure your SUPABASE_DB_URL is set correctly in .env"
        )
        sys.exit(1)


if __name__ == "__main__":
    setup_schema()

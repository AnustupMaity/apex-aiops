"""
Async Telemetry Collector.

Streams database performance metrics from Supabase by polling
pg_stat_statements and pg_stat_user_tables at configurable intervals.
Maintains a rolling buffer fed into the BiLSTM inference engine.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from src.config.settings import get_settings


@dataclass
class TelemetrySnapshot:
    """A single telemetry reading from the database."""

    timestamp: datetime
    total_exec_time: float = 0.0
    total_calls: int = 0
    rows_returned: int = 0
    mean_exec_time_ms: float = 0.0
    shared_blks_hit: int = 0
    shared_blks_read: int = 0
    cache_hit_ratio: float = 1.0
    active_connections: int = 0
    seq_scan_count: int = 0
    idx_scan_count: int = 0

    def to_feature_vector(self) -> list[float]:
        """Convert to a normalized feature vector for the BiLSTM."""
        return [
            self.mean_exec_time_ms,
            self.cache_hit_ratio,
            float(self.total_calls),
            float(self.seq_scan_count),
            float(self.active_connections),
        ]


@dataclass
class SlowQuery:
    """A slow query captured from pg_stat_statements."""

    query_id: str
    query_text: str
    calls: int
    avg_exec_time_ms: float
    total_exec_time_ms: float
    rows: int
    cache_hit_ratio: float


class TelemetryCollector:
    """
    Asynchronous telemetry collector that polls Supabase metrics.

    Maintains a rolling buffer of TelemetrySnapshot objects and
    provides the latest window for BiLSTM inference.

    Args:
        poll_interval: Seconds between polls (default: 5).
        buffer_size: Maximum snapshots to retain (default: 300).
    """

    def __init__(
        self,
        poll_interval: Optional[int] = None,
        buffer_size: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.poll_interval = poll_interval or settings.telemetry_poll_interval_seconds
        self.buffer_size = buffer_size or settings.telemetry_buffer_size
        self.db_url = settings.supabase_db_url

        # Rolling buffer
        self.buffer: deque[TelemetrySnapshot] = deque(maxlen=self.buffer_size)
        self._running = False
        self._pool = None
        self._slow_queries: list[SlowQuery] = []
        
        # State for calculating live deltas
        self._prev_exec_time = 0.0
        self._prev_calls = 0

    async def start(self) -> None:
        """Start the async telemetry collection loop."""
        import asyncpg

        self._running = True
        self._pool = await asyncpg.create_pool(
            self.db_url, min_size=1, max_size=3
        )
        print(
            f"[Apex] TelemetryCollector started "
            f"(poll: {self.poll_interval}s, buffer: {self.buffer_size})"
        )

        while self._running:
            try:
                snapshot = await self._collect_snapshot()
                self.buffer.append(snapshot)

                # Periodically collect slow queries
                if len(self.buffer) % 12 == 0:  # Every ~60 seconds
                    await self._collect_slow_queries()

            except Exception as e:
                print(f"[Apex] Telemetry collection error: {e}")

            await asyncio.sleep(self.poll_interval)

    async def stop(self) -> None:
        """Stop the telemetry collection loop."""
        self._running = False
        if self._pool:
            await self._pool.close()
        print("[Apex] TelemetryCollector stopped")

    async def _collect_snapshot(self) -> TelemetrySnapshot:
        """Collect a single telemetry snapshot from the database."""
        async with self._pool.acquire() as conn:
            # Query pg_stat_statements aggregate metrics
            stats_row = await conn.fetchrow("""
                SELECT
                    COALESCE(SUM(total_exec_time), 0) AS total_exec_time,
                    COALESCE(SUM(calls), 0) AS total_calls,
                    COALESCE(SUM(rows), 0) AS rows_returned,
                    CASE WHEN SUM(calls) > 0
                        THEN SUM(total_exec_time) / SUM(calls)
                        ELSE 0
                    END AS mean_exec_time_ms,
                    COALESCE(SUM(shared_blks_hit), 0) AS shared_blks_hit,
                    COALESCE(SUM(shared_blks_read), 0) AS shared_blks_read
                FROM pg_stat_statements
            """)

            # Active connections
            conn_count = await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
            )

            # Table scan statistics
            scan_row = await conn.fetchrow("""
                SELECT
                    COALESCE(SUM(seq_scan), 0) AS seq_scan_count,
                    COALESCE(SUM(idx_scan), 0) AS idx_scan_count
                FROM pg_stat_user_tables
            """)

            # Calculate cache hit ratio
            blks_hit = stats_row["shared_blks_hit"] or 0
            blks_read = stats_row["shared_blks_read"] or 0
            total_blks = blks_hit + blks_read
            cache_ratio = blks_hit / total_blks if total_blks > 0 else 1.0

            total_exec_time = float(stats_row["total_exec_time"] or 0)
            total_calls = int(stats_row["total_calls"] or 0)
            
            # Calculate live mean exec time (delta over poll interval)
            delta_exec = total_exec_time - self._prev_exec_time
            delta_calls = total_calls - self._prev_calls
            
            if self._prev_calls == 0:
                # First run, use global average
                mean_exec_time_ms = float(stats_row["mean_exec_time_ms"] or 0)
            elif delta_calls > 0:
                # Active queries in this interval
                mean_exec_time_ms = delta_exec / delta_calls
            else:
                # No active queries, drop to 0 to show true live state
                mean_exec_time_ms = 0.0
                
            # Update state
            self._prev_exec_time = total_exec_time
            self._prev_calls = total_calls

            return TelemetrySnapshot(
                timestamp=datetime.utcnow(),
                total_exec_time=total_exec_time,
                total_calls=total_calls,
                rows_returned=int(stats_row["rows_returned"] or 0),
                mean_exec_time_ms=mean_exec_time_ms,
                shared_blks_hit=int(blks_hit),
                shared_blks_read=int(blks_read),
                cache_hit_ratio=cache_ratio,
                active_connections=int(conn_count or 0),
                seq_scan_count=int(scan_row["seq_scan_count"] or 0),
                idx_scan_count=int(scan_row["idx_scan_count"] or 0),
            )

    async def _collect_slow_queries(self) -> None:
        """Collect the top-10 slowest queries from pg_stat_statements."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    queryid::text AS query_id,
                    query,
                    calls,
                    total_exec_time / GREATEST(calls, 1) AS avg_exec_time_ms,
                    total_exec_time,
                    rows,
                    CASE WHEN (shared_blks_hit + shared_blks_read) > 0
                        THEN shared_blks_hit::decimal /
                             (shared_blks_hit + shared_blks_read)
                        ELSE 1.0
                    END AS cache_hit_ratio
                FROM pg_stat_statements
                WHERE calls > 0
                ORDER BY avg_exec_time_ms DESC
                LIMIT 10
            """)

            self._slow_queries = [
                SlowQuery(
                    query_id=row["query_id"],
                    query_text=row["query"],
                    calls=row["calls"],
                    avg_exec_time_ms=float(row["avg_exec_time_ms"]),
                    total_exec_time_ms=float(row["total_exec_time"]),
                    rows=row["rows"],
                    cache_hit_ratio=float(row["cache_hit_ratio"]),
                )
                for row in rows
            ]

    def get_latest_window(self, window_size: int = 60) -> Optional[np.ndarray]:
        """
        Get the latest window of metric values for BiLSTM inference.

        Returns a numpy array of shape (window_size, 1) containing
        the mean_exec_time_ms values, or None if insufficient data.
        """
        if len(self.buffer) < window_size:
            return None

        # Extract mean_exec_time_ms from the last N snapshots
        recent = list(self.buffer)[-window_size:]
        values = np.array(
            [s.mean_exec_time_ms for s in recent], dtype=np.float32
        )
        return values

    def get_slow_queries(self) -> list[SlowQuery]:
        """Return the cached list of slowest queries."""
        return self._slow_queries.copy()

    @property
    def latest_snapshot(self) -> Optional[TelemetrySnapshot]:
        """Get the most recent telemetry snapshot."""
        return self.buffer[-1] if self.buffer else None

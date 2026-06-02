"use client";

import React, { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import MetricsChart from "@/components/MetricsChart";

interface MetricPoint {
  timestamp: string;
  mean_exec_time_ms: number;
  cache_hit_ratio: number;
  active_connections: number;
  seq_scan_count: number;
  idx_scan_count: number;
}

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);

  useEffect(() => {
    // Fetch live metrics

    // Fetch live metrics
    const fetchMetrics = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/metrics");
        if (res.ok) {
          const data = await res.json();
          if (data.metrics && data.metrics.length > 0) {
            setMetrics(data.metrics);
          }
        }
      } catch (err) {
        // Silent fail if backend offline
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <h2>Telemetry Overview</h2>
          <p>Real-time database performance and resource utilization</p>
        </div>

        <div className="glass-card" style={{ marginBottom: "32px" }}>
          <MetricsChart data={metrics} title="Execution Time & Cache Hit Ratio" />
        </div>

        <div className="stats-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
          <div className="stat-card blue">
            <div className="stat-label">Active Connections</div>
            <div className="stat-value">
              {metrics[metrics.length - 1]?.active_connections || 0}
            </div>
            <div className="stat-change positive">Healthy load</div>
          </div>
          
          <div className="stat-card purple">
            <div className="stat-label">Index Scans / min</div>
            <div className="stat-value">
              {metrics[metrics.length - 1]?.idx_scan_count || 0}
            </div>
            <div className="stat-change positive">Efficient queries</div>
          </div>

          <div className="stat-card orange">
            <div className="stat-label">Seq Scans / min</div>
            <div className="stat-value">
              {metrics[metrics.length - 1]?.seq_scan_count || 0}
            </div>
            <div className="stat-change negative">Potential bottlenecks</div>
          </div>
        </div>
      </main>
    </div>
  );
}

"use client";

import React, { useState, useEffect } from "react";
import { BrainCircuit, ShieldCheck, Bot } from "lucide-react";
import Sidebar from "@/components/Sidebar";
import MetricsChart from "@/components/MetricsChart";
import IncidentCard from "@/components/IncidentCard";
import IncidentTimeline from "@/components/IncidentTimeline";
import QueryDiff from "@/components/QueryDiff";
import StatusBadge from "@/components/StatusBadge";
import type { Incident, MetricPoint } from "@/lib/types";

// Fake data removed in favor of strict live data

// ── Main Dashboard Page ───────────────────────────────────────
export default function DashboardPage() {
  const [metrics, setMetrics] = useState<MetricPoint[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [metricsRes, incidentsRes] = await Promise.all([
          fetch("http://localhost:8000/api/metrics"),
          fetch("http://localhost:8000/api/incidents"),
        ]);

        if (metricsRes.ok) {
          const metricsData = await metricsRes.json();
          if (metricsData.metrics?.length > 0) {
            setMetrics(metricsData.metrics);
            setIsLive(true);
          }
        }

        if (incidentsRes.ok) {
          const incidentsData = await incidentsRes.json();
          if (incidentsData.incidents?.length > 0) {
            setIncidents(incidentsData.incidents);
            setSelectedIncident((prev) => prev || incidentsData.incidents[0]);
          }
        }
      } catch {
        setIsLive(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Computed stats
  const totalIncidents = incidents.length;
  const resolvedCount = incidents.filter((i) => i.resolution === "improved").length;
  const avgSpeedup =
    incidents
      .filter((i) => i.speedup_factor && i.speedup_factor > 1)
      .reduce((sum, i) => sum + (i.speedup_factor || 0), 0) /
      Math.max(resolvedCount, 1);
  const latestMetric = metrics[metrics.length - 1];
  const activeAgentsCard = (
    <div style={{ marginTop: "24px" }}>
      <h3
        style={{
          fontSize: "16px",
          fontWeight: 600,
          marginBottom: "16px",
        }}
      >
        Active Orchestration Agents
      </h3>
      <div className="glass-card" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: "var(--bg-elevated)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--accent-blue)" }}>
            <BrainCircuit size={16} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: "13px", fontWeight: 600 }}>BiLSTM Autoencoder</div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>Anomaly Detection Layer</div>
          </div>
          <span className="badge low" style={{ fontSize: "10px" }}>Online</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: "var(--bg-elevated)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--accent-green)" }}>
            <ShieldCheck size={16} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: "13px", fontWeight: 600 }}>PydanticAI</div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>Schema Validation Layer</div>
          </div>
          <span className="badge low" style={{ fontSize: "10px" }}>Online</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: "var(--bg-elevated)", display: "flex", alignItems: "center", justifyContent: "center", color: "var(--accent-purple)" }}>
            <Bot size={16} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: "13px", fontWeight: 600 }}>Qwen2.5-Coder</div>
            <div style={{ fontSize: "11px", color: "var(--text-muted)" }}>Reasoning & Planning Layer</div>
          </div>
          <span className="badge low" style={{ fontSize: "10px" }}>Online</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <h2>Apex Autonomous Orchestrator</h2>
            <span
              className={`badge ${isLive ? "low" : "pending"}`}
              style={
                isLive
                  ? {
                      animation: "pulse-badge 2s infinite",
                    }
                  : {
                      background: "transparent",
                      border: "1px solid var(--border-subtle)",
                      color: "var(--text-secondary)",
                    }
              }
            >
              {isLive ? "System Online" : "Simulated Environment"}
            </span>
          </div>
          <p>
            Enterprise database telemetry, anomaly detection, and autonomous query optimization
          </p>
        </div>

        {/* Stat Cards */}
        <div className="stats-grid">
          <div className="stat-card green">
            <div className="stat-label">Mean Latency</div>
            <div className="stat-value">
              {latestMetric?.mean_exec_time_ms?.toFixed(1) || "—"}
              <span style={{ fontSize: "14px", fontWeight: 400 }}> ms</span>
            </div>
            <div className="stat-change positive">
              <span className="status-dot online" />
              Nominal Baseline
            </div>
          </div>

          <div className="stat-card blue">
            <div className="stat-label">Cache Hit Ratio</div>
            <div className="stat-value">
              {latestMetric
                ? (latestMetric.cache_hit_ratio * 100).toFixed(1)
                : "—"}
              <span style={{ fontSize: "14px", fontWeight: 400 }}>%</span>
            </div>
            <div className="stat-change positive">Optimal Efficiency</div>
          </div>

          <div className="stat-card purple">
            <div className="stat-label">Automated Resolutions</div>
            <div className="stat-value">
              {resolvedCount}
              <span style={{ fontSize: "14px", fontWeight: 400 }}>
                {" "}
                / {totalIncidents}
              </span>
            </div>
            <div className="stat-change positive">
              {((resolvedCount / Math.max(totalIncidents, 1)) * 100).toFixed(0)}%
              Resolution Rate
            </div>
          </div>

          <div className="stat-card orange">
            <div className="stat-label">Mean Latency Reduction</div>
            <div className="stat-value">
              {avgSpeedup > 0 ? avgSpeedup.toFixed(1) : "—"}
              <span style={{ fontSize: "14px", fontWeight: 400 }}>x</span>
            </div>
            <div className="stat-change positive">
              Across {resolvedCount} autonomous optimizations
            </div>
          </div>
        </div>

        {/* Metrics Chart */}
        <MetricsChart data={metrics} />

        {/* Two Column Layout: Incidents + Detail */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)",
            gap: "32px",
            minWidth: 0,
          }}
        >
          {/* Recent Incidents */}
          <div style={{ minWidth: 0, display: "flex", flexDirection: "column" }}>
            <h3
              style={{
                fontSize: "16px",
                fontWeight: 600,
                marginBottom: "16px",
              }}
            >
              Recent Anomaly Reports
            </h3>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "12px",
                flex: 1,
                minHeight: 0,
                maxHeight: "450px", /* Restored maxHeight to enforce scrolling */
                overflowY: "auto",
                paddingRight: "8px",
              }}
            >
              {incidents.length === 0 ? (
                <div
                  className="glass-card"
                  style={{
                    textAlign: "center",
                    padding: "32px",
                    color: "var(--text-muted)",
                    fontSize: "13px",
                  }}
                >
                  <div style={{ marginBottom: "8px", color: "var(--accent-green)", display: "flex", justifyContent: "center" }}>
                    <ShieldCheck size={24} />
                  </div>
                  <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>System Operating Normally</div>
                  <div style={{ fontSize: "11px", marginTop: "4px" }}>
                    No database anomalies detected. Waiting for telemetry...
                  </div>
                </div>
              ) : (
                incidents.map((incident) => (
                  <IncidentCard
                    key={incident.incident_id}
                    incident={incident}
                    onClick={setSelectedIncident}
                  />
                ))
              )}
            </div>

            {/* Moved Incident Detail to left side */}
            {selectedIncident && (
              <div style={{ marginTop: "24px" }}>
                <h3
                  style={{
                    fontSize: "16px",
                    fontWeight: 600,
                    marginBottom: "16px",
                  }}
                >
                  Incident Detail
                </h3>

                <div className="glass-card" style={{ minHeight: "220px", display: "flex", flexDirection: "column", justifyContent: "space-evenly" }}>
                  <div
                    style={{
                      display: "flex",
                      gap: "8px",
                      marginBottom: "8px",
                    }}
                  >
                    <StatusBadge severity={selectedIncident.severity} />
                    <StatusBadge severity={selectedIncident.resolution} />
                  </div>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      gap: "12px",
                      fontSize: "13px",
                    }}
                  >
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Model:</span>{" "}
                      <span style={{ color: "var(--accent-purple)" }}>
                        {selectedIncident.model_used || "—"}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Score:</span>{" "}
                      <span>{selectedIncident.anomaly_score.toFixed(3)}</span>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>Before:</span>{" "}
                      <span style={{ color: "var(--severity-critical)" }}>
                        {selectedIncident.current_exec_ms.toFixed(1)} ms
                      </span>
                    </div>
                    <div>
                      <span style={{ color: "var(--text-muted)" }}>After:</span>{" "}
                      <span style={{ color: "var(--accent-green)" }}>
                        {selectedIncident.speedup_factor
                          ? (
                              selectedIncident.current_exec_ms /
                              selectedIncident.speedup_factor
                            ).toFixed(1)
                          : "—"}{" "}
                        ms
                      </span>
                    </div>
                  </div>

                  {/* Index Recommendations */}
                  {selectedIncident.index_recommendations?.length > 0 && (
                    <div style={{ marginTop: "16px" }}>
                      <div
                        style={{
                          fontSize: "11px",
                          textTransform: "uppercase",
                          letterSpacing: "1px",
                          color: "var(--text-muted)",
                          marginBottom: "8px",
                          fontWeight: 600,
                        }}
                      >
                        Index Recommendations
                      </div>
                      {selectedIncident.index_recommendations.map((idx, i) => (
                        <div
                          key={i}
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "11px",
                            color: "var(--accent-green)",
                            background: "rgba(62, 207, 142, 0.08)",
                            padding: "6px 10px",
                            borderRadius: "6px",
                            marginBottom: "4px",
                          }}
                        >
                          {idx}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Pipeline and Agents */}
          <div style={{ minWidth: 0 }}>
            {selectedIncident ? (
              <>

                {/* Timeline */}
                <div className="glass-card" style={{ marginBottom: "16px" }}>
                  <h4
                    style={{
                      fontSize: "13px",
                      fontWeight: 600,
                      marginBottom: "12px",
                      textTransform: "uppercase",
                      letterSpacing: "1px",
                      color: "var(--text-muted)",
                    }}
                  >
                    Resolution Pipeline
                  </h4>
                  <IncidentTimeline
                    resolution={selectedIncident.resolution}
                    modelUsed={selectedIncident.model_used}
                    speedup={selectedIncident.speedup_factor}
                  />
                </div>

                {/* Active Agents Card in detail view */}
                {activeAgentsCard}
              </>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "24px", height: "100%" }}>
                <div
                  className="glass-card"
                  style={{
                    textAlign: "center",
                    padding: "48px",
                    color: "var(--text-muted)",
                  }}
                >
                  Select an incident to view details
                </div>

                {/* Active Agents Card moved to right side */}
                {activeAgentsCard}
              </div>
            )}
          </div>
        </div>

        {/* Full-width Query Diff below the grid */}
        {selectedIncident && (
          <div style={{ marginTop: "24px" }}>
            <h3
              style={{
                fontSize: "16px",
                fontWeight: 600,
                marginBottom: "16px",
              }}
            >
              SQL Optimization Details
            </h3>
            <QueryDiff
              original={selectedIncident.affected_query}
              optimized={selectedIncident.optimized_query}
            />
          </div>
        )}
      </main>
    </div>
  );
}

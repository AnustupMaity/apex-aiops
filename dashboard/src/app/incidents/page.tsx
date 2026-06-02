"use client";

import React, { useState, useEffect } from "react";
import Sidebar from "@/components/Sidebar";
import StatusBadge from "@/components/StatusBadge";
import QueryDiff from "@/components/QueryDiff";
import { ShieldCheck } from "lucide-react";
import type { Incident } from "@/lib/types";

// Demo data removed in favor of strict live data

export default function IncidentsPage() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selected, setSelected] = useState<Incident | null>(null);
  const [filter, setFilter] = useState<string>("all");

  useEffect(() => {
    const fetchIncidents = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/incidents");
        if (res.ok) {
          const data = await res.json();
          if (data.incidents?.length > 0) setIncidents(data.incidents);
        }
      } catch {}
    };
    fetchIncidents();
  }, []);

  const filtered =
    filter === "all"
      ? incidents
      : incidents.filter((i) => i.severity === filter || i.resolution === filter);

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <h2>Incident History</h2>
          <p>Complete record of detected anomalies and optimization results</p>
        </div>

        {/* Filter Buttons */}
        <div style={{ display: "flex", gap: "8px", marginBottom: "24px" }}>
          {["all", "critical", "high", "medium", "low", "improved", "pending"].map(
            (f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: "6px 14px",
                  borderRadius: "100px",
                  border:
                    filter === f
                      ? "1px solid var(--accent-green)"
                      : "1px solid var(--border-glass)",
                  background:
                    filter === f
                      ? "rgba(62, 207, 142, 0.12)"
                      : "var(--bg-glass)",
                  color:
                    filter === f
                      ? "var(--accent-green)"
                      : "var(--text-secondary)",
                  fontSize: "12px",
                  fontWeight: 600,
                  cursor: "pointer",
                  textTransform: "capitalize",
                  fontFamily: "var(--font-sans)",
                  transition: "all 0.2s ease",
                }}
                id={`filter-${f}`}
              >
                {f}
              </button>
            )
          )}
        </div>

        {/* Incidents Table */}
        <div className="glass-card" style={{ padding: 0, overflow: "hidden" }}>
          <table className="incident-table" id="incidents-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Severity</th>
                <th>Query</th>
                <th>Tables</th>
                <th>Degradation</th>
                <th>Speedup</th>
                <th>Resolution</th>
                <th>Model</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={8} style={{ textAlign: "center", padding: "48px", color: "var(--text-muted)", fontSize: "13px" }}>
                    <div style={{ marginBottom: "8px", color: "var(--accent-green)", display: "flex", justifyContent: "center" }}>
                      <ShieldCheck size={24} />
                    </div>
                    <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>No Incidents Found</div>
                    <div style={{ fontSize: "11px", marginTop: "4px" }}>
                      System is operating normally. Waiting for anomaly detections...
                    </div>
                  </td>
                </tr>
              ) : (
                filtered.map((incident) => (
                  <tr
                    key={incident.incident_id}
                    onClick={() => setSelected(incident)}
                    style={{ cursor: "pointer" }}
                  >
                    <td className="time-cell" suppressHydrationWarning>
                      {new Date(incident.created_at).toLocaleString([], {
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </td>
                    <td>
                      <StatusBadge severity={incident.severity} size="sm" />
                    </td>
                    <td className="query-cell">
                      {incident.affected_query.slice(0, 60)}...
                    </td>
                    <td style={{ fontSize: "12px" }}>
                      {incident.table_names?.join(", ") || "—"}
                    </td>
                    <td>
                      <span
                        style={{
                          color: "var(--severity-high)",
                          fontWeight: 700,
                        }}
                      >
                        {incident.degradation_factor?.toFixed(1)}x
                      </span>
                    </td>
                    <td className={`speedup-cell ${incident.speedup_factor && incident.speedup_factor > 1 ? "positive" : ""}`}>
                      {incident.speedup_factor
                        ? `${incident.speedup_factor.toFixed(1)}x`
                        : "—"}
                    </td>
                    <td>
                      <StatusBadge severity={incident.resolution} size="sm" />
                    </td>
                    <td
                      style={{
                        fontSize: "11px",
                        color: "var(--accent-purple)",
                        fontFamily: "var(--font-mono)",
                        display: "flex",
                        alignItems: "center",
                        gap: "6px"
                      }}
                    >
                      {incident.model_used?.startsWith("[TEST]") && (
                        <span title="Simulated Test Incident" style={{ fontSize: "14px" }}>🧪</span>
                      )}
                      {incident.model_used?.replace("[TEST] ", "").split(":").pop() || "—"}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Selected Incident SQL Diff */}
        {selected && (
          <div style={{ marginTop: "24px" }}>
            <h3
              style={{
                fontSize: "16px",
                fontWeight: 600,
                marginBottom: "16px",
              }}
            >
              Query Comparison —{" "}
              <span style={{ color: "var(--text-muted)", fontWeight: 400 }}>
                {selected.incident_id.slice(0, 8)}
              </span>
            </h3>
            <QueryDiff
              original={selected.affected_query}
              optimized={selected.optimized_query}
            />
          </div>
        )}
      </main>
    </div>
  );
}

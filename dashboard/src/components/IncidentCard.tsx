"use client";

import React from "react";
import StatusBadge from "./StatusBadge";
import type { Incident } from "@/lib/types";

interface IncidentCardProps {
  incident: Incident;
  onClick?: (incident: Incident) => void;
}

export default function IncidentCard({ incident, onClick }: IncidentCardProps) {
  const timeAgo = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  };

  return (
    <div
      className="glass-card"
      style={{ cursor: onClick ? "pointer" : "default", padding: "16px 20px" }}
      onClick={() => onClick?.(incident)}
      id={`incident-${incident.incident_id.slice(0, 8)}`}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: "12px",
        }}
      >
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {incident.model_used?.startsWith("[TEST]") && (
            <span
              title="Test Simulation"
              style={{
                background: "rgba(167, 139, 250, 0.15)",
                color: "var(--accent-purple)",
                padding: "2px 6px",
                borderRadius: "4px",
                fontSize: "10px",
                fontWeight: 700,
                border: "1px solid rgba(167, 139, 250, 0.3)"
              }}
            >
              🧪 TEST
            </span>
          )}
          <StatusBadge severity={incident.severity} size="sm" />
          <StatusBadge severity={incident.resolution} size="sm" />
        </div>
        <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
          {timeAgo(incident.created_at)}
        </span>
      </div>

      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "12px",
          color: "var(--accent-blue)",
          marginBottom: "12px",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
        }}
      >
        {incident.affected_query.slice(0, 80)}...
      </div>

      <div style={{ display: "flex", gap: "24px", fontSize: "12px" }}>
        <div>
          <span style={{ color: "var(--text-muted)" }}>Tables: </span>
          <span>{incident.table_names?.join(", ") || "—"}</span>
        </div>
        <div>
          <span style={{ color: "var(--text-muted)" }}>Degradation: </span>
          <span style={{ color: "var(--severity-high)", fontWeight: 700 }}>
            {incident.degradation_factor?.toFixed(1)}x
          </span>
        </div>
        {incident.speedup_factor && incident.speedup_factor > 1 && (
          <div>
            <span style={{ color: "var(--text-muted)" }}>Speedup: </span>
            <span style={{ color: "var(--accent-green)", fontWeight: 700 }}>
              {incident.speedup_factor.toFixed(1)}x
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

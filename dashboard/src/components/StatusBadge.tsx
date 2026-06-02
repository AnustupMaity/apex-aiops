"use client";

import React from "react";

interface StatusBadgeProps {
  severity: string;
  size?: "sm" | "md";
}

export default function StatusBadge({ severity, size = "md" }: StatusBadgeProps) {
  const label = severity.charAt(0).toUpperCase() + severity.slice(1);

  const dotColors: Record<string, string> = {
    low: "var(--severity-low)",
    medium: "var(--severity-medium)",
    high: "var(--severity-high)",
    critical: "var(--severity-critical)",
    improved: "var(--accent-green)",
    pending: "var(--text-secondary)",
    no_change: "var(--text-muted)",
    escalated: "var(--severity-high)",
    failed: "var(--severity-critical)",
  };

  return (
    <span className={`badge ${severity}`} style={size === "sm" ? { fontSize: "10px", padding: "2px 8px" } : {}}>
      <span
        style={{
          width: "6px",
          height: "6px",
          borderRadius: "50%",
          background: dotColors[severity] || "var(--text-muted)",
          display: "inline-block",
        }}
      />
      {label.replace("_", " ")}
    </span>
  );
}

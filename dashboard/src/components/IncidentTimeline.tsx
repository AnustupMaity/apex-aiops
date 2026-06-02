"use client";

import React from "react";

interface Step {
  title: string;
  detail: string;
  status: "completed" | "active" | "pending";
}

interface IncidentTimelineProps {
  resolution: string;
  modelUsed?: string | null;
  speedup?: number | null;
}

export default function IncidentTimeline({
  resolution,
  modelUsed,
  speedup,
}: IncidentTimelineProps) {
  const isResolved = resolution === "improved";
  const isFailed = resolution === "failed" || resolution === "regression";

  const steps: Step[] = [
    {
      title: "Detection",
      detail: "BiLSTM anomaly detector triggered",
      status: "completed",
    },
    {
      title: "Validation",
      detail: "PydanticAI structured the anomaly report",
      status: "completed",
    },
    {
      title: "Routing",
      detail: modelUsed
        ? `Routed to ${modelUsed}`
        : "Evaluating severity...",
      status: resolution === "pending" ? "active" : "completed",
    },
    {
      title: "Reasoning",
      detail: "SQL query rewritten & indexes recommended",
      status:
        resolution === "pending"
          ? "pending"
          : "completed",
    },
    {
      title: "Execution",
      detail: "EXPLAIN ANALYZE via MCP bridge",
      status:
        resolution === "pending"
          ? "pending"
          : "completed",
    },
    {
      title: "Verification",
      detail: speedup
        ? `${speedup.toFixed(1)}x speedup achieved`
        : "Comparing execution plans...",
      status: isResolved
        ? "completed"
        : isFailed
        ? "completed"
        : "pending",
    },
  ];

  return (
    <div className="timeline" id="incident-timeline">
      {steps.map((step, i) => (
        <div
          className={`timeline-step ${step.status === "active" ? "active" : ""}`}
          key={i}
          style={{
            opacity: step.status === "pending" ? 0.4 : 1,
          }}
        >
          <div className="step-title">{step.title}</div>
          <div className="step-detail">{step.detail}</div>
        </div>
      ))}
    </div>
  );
}

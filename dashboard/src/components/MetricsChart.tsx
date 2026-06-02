"use client";

import React from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface MetricsChartProps {
  data: Array<{
    timestamp: string;
    mean_exec_time_ms: number;
    cache_hit_ratio: number;
  }>;
  title?: string;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;

  return (
    <div
      style={{
        background: "var(--bg-elevated)",
        border: "1px solid var(--border-glass)",
        borderRadius: "var(--radius-sm)",
        padding: "12px 16px",
        backdropFilter: "blur(12px)",
        fontSize: "12px",
      }}
    >
      <p style={{ color: "var(--text-muted)", marginBottom: "8px" }}>
        {new Date(label).toLocaleTimeString()}
      </p>
      {payload.map((entry: any, i: number) => (
        <p key={i} style={{ color: entry.color, marginBottom: "2px" }}>
          {entry.name}: {typeof entry.value === "number" ? entry.value.toFixed(2) : entry.value}
          {entry.name.includes("Ratio") ? "" : " ms"}
        </p>
      ))}
    </div>
  );
};

export default function MetricsChart({ data, title = "Performance Metrics" }: MetricsChartProps) {
  const chartData = data.map((d) => ({
    ...d,
    time: new Date(d.timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    }),
  }));

  return (
    <div className="glass-card chart-container" id="metrics-chart">
      <div className="chart-header">
        <h3>{title}</h3>
        <span className="chart-badge">⚡ Live</span>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="execGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.1} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="cacheGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0070f3" stopOpacity={0.1} />
              <stop offset="95%" stopColor="#0070f3" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.04)"
            vertical={false}
          />
          <XAxis
            dataKey="time"
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            axisLine={{ stroke: "var(--border-subtle)" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={48}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="mean_exec_time_ms"
            name="Avg Exec Time"
            stroke="#10b981"
            strokeWidth={2}
            fill="url(#execGradient)"
            dot={false}
            activeDot={{ r: 4, fill: "#10b981", stroke: "#000", strokeWidth: 2 }}
          />
          <Area
            type="monotone"
            dataKey="cache_hit_ratio"
            name="Cache Hit Ratio"
            stroke="#0070f3"
            strokeWidth={2}
            fill="url(#cacheGradient)"
            dot={false}
            activeDot={{ r: 4, fill: "#0070f3", stroke: "#000", strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

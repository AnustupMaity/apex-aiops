"use client";

import React, { useState } from "react";
import Sidebar from "@/components/Sidebar";
import { Terminal, AlertTriangle, Play } from "lucide-react";

export default function TestPage() {
  const [query, setQuery] = useState("SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleTest = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/trigger", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          baseline_exec_ms: 10.0,
          current_exec_ms: 850.0,
        }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      setResult({ error: err.message });
    }
    setLoading(false);
  };

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <h2>Diagnostic Console</h2>
          <p>Manual testing interface for autonomous orchestration</p>
        </div>
        
        <div className="glass-card">
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "24px" }}>
            <div style={{ background: "var(--bg-elevated)", padding: "12px", borderRadius: "var(--radius-md)", color: "var(--severity-high)" }}>
              <AlertTriangle size={24} />
            </div>
            <div>
              <h3 style={{ fontSize: "16px", fontWeight: 600 }}>Execute Diagnostic Simulation</h3>
              <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
                Use this console to manually simulate a latency anomaly and trigger the orchestration pipeline.
              </p>
            </div>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <label style={{ display: "block", marginBottom: "8px", fontWeight: 500, color: "var(--text-secondary)" }}>
              Target SQL Query
            </label>
            <textarea 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              style={{
                width: "100%",
                height: "100px",
                backgroundColor: "var(--bg-primary)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-sm)",
                padding: "16px",
                fontFamily: "var(--font-mono)",
                fontSize: "13px",
                color: "var(--text-primary)",
                resize: "vertical"
              }}
            />
          </div>
          
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "24px", marginBottom: "24px" }}>
            <div>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: 500, color: "var(--text-secondary)" }}>
                Baseline Latency (ms)
              </label>
              <input 
                type="text" 
                value="10.0" 
                disabled 
                style={{
                  width: "100%",
                  backgroundColor: "var(--bg-primary)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  padding: "12px",
                  color: "var(--text-muted)",
                  fontFamily: "var(--font-mono)",
                }} 
              />
            </div>
            <div>
              <label style={{ display: "block", marginBottom: "8px", fontWeight: 500, color: "var(--text-secondary)" }}>
                Simulated Latency Spike (ms)
              </label>
              <input 
                type="text" 
                value="850.0" 
                disabled 
                style={{
                  width: "100%",
                  backgroundColor: "var(--bg-primary)",
                  border: "1px solid var(--border-subtle)",
                  borderRadius: "var(--radius-sm)",
                  padding: "12px",
                  color: "var(--text-primary)",
                  fontFamily: "var(--font-mono)",
                }} 
              />
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button 
              onClick={handleTest}
              disabled={loading}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "10px 20px",
                backgroundColor: "var(--text-primary)",
                color: "var(--bg-primary)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                fontWeight: 600,
                cursor: loading ? "wait" : "pointer",
                opacity: loading ? 0.7 : 1,
                transition: "opacity 0.2s"
              }}
            >
              <Play size={16} fill="currentColor" />
              {loading ? "Executing..." : "Execute Simulation"}
            </button>
          </div>

          {result && (
            <div style={{
              marginTop: "24px",
              padding: "16px",
              backgroundColor: "var(--bg-primary)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-sm)",
              maxHeight: "300px",
              overflowY: "auto",
              fontFamily: "var(--font-mono)",
              fontSize: "12px",
              color: "var(--text-primary)",
              whiteSpace: "pre-wrap"
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px", color: "var(--text-secondary)", fontWeight: 600 }}>
                <Terminal size={14} /> Execution Output
              </div>
              {JSON.stringify(result, null, 2)}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

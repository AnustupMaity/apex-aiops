"use client";

import React, { useState, useEffect } from "react";
import Sidebar from "@/components/Sidebar";

export default function SettingsPage() {
  const [dbStatus, setDbStatus] = useState("checking");
  const [daemonStatus, setDaemonStatus] = useState("checking");
  const [settings, setSettings] = useState({
    anomaly_threshold: 0.85,
    ollama_model: "qwen2.5-coder:1.5b",
    cloud_llm_provider: "gemini",
  });
  const [saving, setSaving] = useState(false);
  
  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/settings");
        if (res.ok) {
          const data = await res.json();
          setSettings({
            anomaly_threshold: data.anomaly_threshold,
            ollama_model: data.ollama_model,
            cloud_llm_provider: data.cloud_llm_provider,
          });
        }
      } catch (err) {
        console.error("Failed to fetch settings", err);
      }
    };
    fetchSettings();

    const checkStatus = async () => {
      try {
        const res = await fetch("http://localhost:8000/health");
        if (res.ok) {
          setDaemonStatus("online");
          const data = await res.json();
          setDbStatus(data.collector_active ? "online" : "offline");
        } else {
          setDaemonStatus("offline");
          setDbStatus("offline");
        }
      } catch (err) {
        setDaemonStatus("offline");
        setDbStatus("offline");
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      const res = await fetch("http://localhost:8000/api/settings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(settings),
      });
      if (res.ok) {
        alert("Settings saved successfully!");
      } else {
        alert("Failed to save settings.");
      }
    } catch (err) {
      alert("Error saving settings.");
    }
    setSaving(false);
  };

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <div className="page-header">
          <h2>System Configuration</h2>
          <p>Apex Orchestrator Environment Variables & Hyperparameters</p>
        </div>

        <div className="stats-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
          <div className="glass-card">
            <h3 style={{ marginBottom: "16px", fontSize: "16px", color: "var(--text-secondary)" }}>
              Engine Status
            </h3>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px", background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)" }}>
                <div>
                  <div style={{ fontWeight: 600 }}>Backend Daemon</div>
                  <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>http://localhost:8000</div>
                </div>
                <div className={`badge ${daemonStatus === "online" ? "low" : "critical"}`}>
                  <span className={`status-dot ${daemonStatus}`}></span>
                  {daemonStatus.toUpperCase()}
                </div>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px", background: "var(--bg-elevated)", borderRadius: "var(--radius-sm)" }}>
                <div>
                  <div style={{ fontWeight: 600 }}>Supabase Postgres</div>
                  <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>Telemetry Target</div>
                </div>
                <div className={`badge ${dbStatus === "online" ? "low" : "critical"}`}>
                  <span className={`status-dot ${dbStatus}`}></span>
                  {dbStatus.toUpperCase()}
                </div>
              </div>
            </div>
          </div>

          <div className="glass-card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
              <h3 style={{ fontSize: "16px", color: "var(--text-secondary)" }}>
                Model Hyperparameters
              </h3>
              <button 
                onClick={handleSave}
                disabled={saving}
                style={{ 
                  background: "var(--accent-blue)", 
                  color: "#fff", 
                  border: "none", 
                  padding: "6px 12px", 
                  borderRadius: "4px", 
                  cursor: saving ? "not-allowed" : "pointer",
                  fontSize: "12px",
                  fontWeight: 600,
                  opacity: saving ? 0.7 : 1
                }}
              >
                {saving ? "Saving..." : "Save Configuration"}
              </button>
            </div>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "8px" }}>
                <span style={{ color: "var(--text-secondary)" }}>Anomaly Threshold (MSE)</span>
                <input 
                  type="number" 
                  step="0.0001"
                  value={settings.anomaly_threshold}
                  onChange={(e) => setSettings({...settings, anomaly_threshold: parseFloat(e.target.value)})}
                  style={{ 
                    fontFamily: "var(--font-mono)", 
                    color: "var(--accent-purple)", 
                    fontWeight: 600, 
                    background: "transparent", 
                    border: "1px solid var(--border-subtle)", 
                    padding: "4px 8px", 
                    borderRadius: "4px",
                    width: "100px",
                    textAlign: "right"
                  }} 
                />
              </div>
              
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "8px" }}>
                <span style={{ color: "var(--text-secondary)" }}>Detector Model</span>
                <span style={{ fontFamily: "var(--font-mono)" }}>PyTorch BiLSTM Autoencoder</span>
              </div>
              
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "8px" }}>
                <span style={{ color: "var(--text-secondary)" }}>Reasoning Agent</span>
                <select
                  value={settings.ollama_model}
                  onChange={(e) => setSettings({...settings, ollama_model: e.target.value})}
                  style={{ 
                    fontFamily: "var(--font-mono)", 
                    background: "var(--bg-elevated)", 
                    color: "var(--text-primary)", 
                    border: "1px solid var(--border-subtle)", 
                    padding: "4px 8px", 
                    borderRadius: "4px",
                    outline: "none"
                  }}
                >
                  <option value="qwen2.5-coder:1.5b">Ollama (Qwen2.5-Coder:1.5b)</option>
                  <option value="llama3:8b">Ollama (Llama3:8b)</option>
                  <option value="mistral">Ollama (Mistral)</option>
                  <option value="deepseek-coder">Ollama (DeepSeek-Coder)</option>
                </select>
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: "8px" }}>
                <span style={{ color: "var(--text-secondary)" }}>Fallback Chain</span>
                <select
                  value={settings.cloud_llm_provider}
                  onChange={(e) => setSettings({...settings, cloud_llm_provider: e.target.value})}
                  style={{ 
                    fontFamily: "var(--font-mono)", 
                    background: "var(--bg-elevated)", 
                    color: "var(--text-primary)", 
                    border: "1px solid var(--border-subtle)", 
                    padding: "4px 8px", 
                    borderRadius: "4px",
                    outline: "none"
                  }}
                >
                  <option value="gemini">Gemini → Groq → DeepSeek</option>
                  <option value="groq">Groq → Gemini → DeepSeek</option>
                  <option value="deepseek">DeepSeek → Gemini → Groq</option>
                  <option value="openrouter">OpenRouter (Any) → Fallbacks</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

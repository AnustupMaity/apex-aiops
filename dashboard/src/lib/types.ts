export interface Incident {
  incident_id: string;
  created_at: string;
  resolved_at: string | null;
  severity: "low" | "medium" | "high" | "critical";
  anomaly_score: number;
  affected_query: string;
  table_names: string[];
  baseline_exec_ms: number;
  current_exec_ms: number;
  degradation_factor: number;
  optimized_query: string | null;
  original_plan: Record<string, unknown>;
  optimized_plan: Record<string, unknown>;
  speedup_factor: number | null;
  index_recommendations: string[];
  resolution: "pending" | "improved" | "no_change" | "escalated" | "failed";
  model_used: string | null;
  langgraph_thread_id: string | null;
}

export interface MetricPoint {
  timestamp: string;
  mean_exec_time_ms: number;
  cache_hit_ratio: number;
  active_connections: number;
  seq_scan_count: number;
  idx_scan_count: number;
}

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  graph_compiled: boolean;
  collector_active: boolean;
  buffer_size: number;
  timestamp: string;
}

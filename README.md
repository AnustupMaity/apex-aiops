# 🚀 Project Apex: Autonomous Database Engine

Project Apex is an advanced, AI-driven AIOps platform that provides autonomous database monitoring, anomaly detection, and automated SQL query optimization for PostgreSQL databases (specifically built for Supabase).

It leverages a local orchestration pipeline built on **LangGraph**, utilizing a multi-layered AI architecture to detect inefficiencies, validate schemas, reason about SQL performance, and mathematically prove speedups via a secure Model Context Protocol (MCP) bridge.

---

## 🏗️ Architecture Overview

The system is cleanly decoupled into three primary services:

1. **The Daemon (`daemon.py`)**: A background Python process that runs a continuous telemetry loop, hosts the FastAPI orchestration endpoints, and coordinates the LangGraph AI workflows.
2. **The MCP Bridge (`src/mcp`)**: A secure, read-only boundary that interfaces with your live Supabase PostgreSQL database. It safely executes `EXPLAIN` query plans and gathers statistical data without ever modifying production tables.
3. **The Next.js Dashboard (`dashboard/`)**: A sleek, real-time React UI providing a visual interface for the autonomous system. It monitors metrics, visualizes detected anomalies, and traces the AI's resolution pipeline.

---

## 🧠 AI Layers (The Orchestration Agents)

Project Apex utilizes three distinct intelligence layers to handle anomalies safely:

### 1. 🔬 Detection Layer (BiLSTM Autoencoder)
- **Role**: Continuous telemetry analysis.
- **Tech**: PyTorch, Bi-directional LSTM neural network.
- **Function**: Monitors incoming SQL execution metrics. When a query's execution time deviates from historical patterns, the autoencoder's reconstruction error spikes above a threshold, triggering an Anomaly Report.

### 2. 🛡️ Validation Layer (PydanticAI)
- **Role**: Schema enforcement and data structuring.
- **Tech**: Pydantic, FastAPI.
- **Function**: Intercepts the raw anomaly signal and validates its shape. It ensures that the required context (original query, baseline execution time, severity) is perfectly formatted before handing it to the reasoning agent.

### 3. 🤖 Reasoning & Planning Layer (Qwen2.5-Coder:1.5b via Ollama)
- **Role**: Autonomous SQL optimization.
- **Tech**: Ollama, LangChain, Qwen2.5.
- **Function**: The brain of the operation. It ingests the inefficient SQL query and intelligently rewrites it (e.g., converting correlated subqueries to `EXISTS` clauses, introducing `JOIN`s, or recommending indexes).

### 4. ⚙️ Execution & Verification (MCP Bridge)
- **Role**: Mathematical proof of speedup.
- **Tech**: PostgreSQL, psycopg.
- **Function**: The MCP bridge runs a safe `EXPLAIN (FORMAT JSON)` on both the original query and the LLM's optimized query against your live database. It compares the `Total Cost` of both query plans. If the new query proves to be mathematically faster, it marks the incident as **Improved** and calculates the exact speedup factor (e.g., 85.0x).

---

## 🛠️ Features & Use Cases

- **Autonomous Query Optimization**: Drop a slow SQL query in, and Apex will return a mathematically verified, optimized version without human intervention.
- **Safe Hypothetical Indexing**: Recommends database indexes and uses PostgreSQL's HypoPG extension to prove their effectiveness before you actually create them.
- **Real-Time Telemetry Dashboard**: Watch your database health, cache hit ratios, and mean latency reduction in real-time.
- **Incident Tracing**: Native LangSmith integration allows you to trace the exact thoughts, logic, and output of the LLM pipeline for every anomaly.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Ollama (running locally with the `qwen2.5-coder:1.5b` model pulled)
- A Supabase Project

### 1. Database Configuration
Copy the `.env` template and add your credentials:
```bash
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_DB_URL=postgresql://postgres.<tenant>:password@<pooler>:5432/postgres
MCP_DB_URL=postgresql://apex_mcp_role.<tenant>:password@<pooler>:5432/postgres
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key
```

### 2. Start the Daemon (Backend)
The daemon runs the AI orchestration, telemetry loop, and FastAPI server.
```bash
# From the project root:
py daemon.py
```
*Note: The daemon runs on port `8000`.*

### 3. Start the Dashboard (Frontend)
The frontend is a Next.js application. 
```bash
cd dashboard
npm install
npm run dev
```
*Note: The frontend runs on port `3000`.*

---

## 🛑 How to Stop & Restart the Frontend
If you need to stop the frontend dashboard (Next.js) and restart it:

1. **To Stop**: Go to the terminal window where you ran `npm run dev` and press `CTRL + C` on your keyboard. It will ask `Terminate batch job (Y/N)?`. Type `Y` and press Enter.
2. **To Run Again**: Make sure you are in the `dashboard` directory and run:
```bash
npm run dev
```

---

## 🔌 API Endpoints (Daemon)

The FastAPI daemon exposes the following critical endpoints:

- `GET /health` : Returns system health and model loading status.
- `GET /api/metrics` : Returns the latest telemetry baseline metrics.
- `GET /api/incidents` : Returns a list of the most recent anomaly reports and their optimization statuses.
- `POST /api/trigger` : Manually trigger an anomaly for testing purposes.
  - **Payload**:
    ```json
    {
      "query": "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders)",
      "baseline_exec_ms": 5.0,
      "current_exec_ms": 500.0
    }
    ```

---

## 🔒 Security & MCP
The Model Context Protocol (MCP) ensures that the LLM **cannot** execute arbitrary SQL or drop tables. All interaction is strictly read-only (`EXPLAIN`) and executed under a restricted PostgreSQL role (`apex_mcp_role`), providing enterprise-grade security for autonomous agents operating against production databases.

---

Developed by Anustup Maity.

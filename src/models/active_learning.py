"""
Active Learning Loop for Project Apex.

Queries the `apex_incidents` table for verified, resolved anomalies,
synthesizes new time-series telemetry representing these real-world
patterns, saves them as a CSV, and triggers a BiLSTM retraining cycle.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import psycopg
from pathlib import Path

from src.config.settings import get_settings
from src.models.train import train_model

def run_active_learning() -> None:
    """Execute the active learning retraining loop."""
    settings = get_settings()
    
    print("[Apex Active Learning] Starting weekly retraining pipeline...")
    
    # 1. Fetch resolved incidents
    incidents = []
    try:
        with psycopg.connect(settings.supabase_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT baseline_exec_ms, current_exec_ms
                    FROM apex_incidents
                    WHERE resolution = 'improved'
                      AND baseline_exec_ms > 0
                      AND current_exec_ms > 0
                """)
                incidents = cur.fetchall()
    except Exception as e:
        print(f"[Apex Active Learning] Failed to fetch incidents: {e}")
        return

    if not incidents:
        print("[Apex Active Learning] No resolved incidents found to train on. Skipping.")
        return
        
    print(f"[Apex Active Learning] Found {len(incidents)} resolved incidents for fine-tuning.")

    # 2. Synthesize new telemetry data based on real incidents
    series_length = 5000
    synthetic_signal = np.full(series_length, 10.0)  # Default 10ms baseline
    
    if incidents:
        avg_baseline = float(np.mean([i[0] for i in incidents]))
        synthetic_signal = np.full(series_length, avg_baseline)
        
        # Inject the real anomaly spikes
        n_anomalies = len(incidents) * 5  # amplify signal
        anomaly_indices = np.random.choice(
            series_length, n_anomalies, replace=True
        )
        
        for idx in anomaly_indices:
            # Pick a random incident's spike
            incident = incidents[np.random.randint(len(incidents))]
            spike_val = float(incident[1])
            # Inject spike with some noise
            synthetic_signal[idx] = spike_val + np.random.normal(0, spike_val * 0.1)

    # Add general noise
    synthetic_signal += np.random.normal(0, avg_baseline * 0.05, series_length)

    # 3. Save as CSV in the NAB data directory
    data_dir = settings.data_dir / "nab"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = data_dir / "synthetic_live_anomalies.csv"
    
    df = pd.DataFrame({"value": synthetic_signal})
    df.to_csv(csv_path, index=False)
    
    print(f"[Apex Active Learning] Saved synthetic dataset to {csv_path}")

    # 4. Trigger Retraining
    print("[Apex Active Learning] Triggering BiLSTM training...")
    try:
        results = train_model()
        print(f"[Apex Active Learning] Retraining complete. New best val loss: {results['best_val_loss']:.4f}")
    except Exception as e:
        print(f"[Apex Active Learning] Training failed: {e}")

if __name__ == "__main__":
    run_active_learning()

import requests

payload = {
    "query": "SELECT id, name FROM test_customers WHERE status = 'VIP' AND id IN (SELECT customer_id FROM test_orders WHERE total_amount > 1950);",
    "baseline_exec_ms": 10.0,
    "current_exec_ms": 1300.0
}

res = requests.post("http://localhost:8000/api/trigger", json=payload)
print(f"Trigger response: {res.status_code}")
print(res.json())

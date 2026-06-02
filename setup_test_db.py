import psycopg
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")
if not DB_URL:
    print("No DB_URL found")
    exit(1)

print("Connecting to Supabase...")
with psycopg.connect(DB_URL) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        print("Creating tables...")
        # Create test_customers and test_orders
        cur.execute("""
            DROP TABLE IF EXISTS test_orders;
            DROP TABLE IF EXISTS test_customers;
            
            CREATE TABLE test_customers (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT
            );
            
            CREATE TABLE test_orders (
                id SERIAL PRIMARY KEY,
                customer_id INTEGER REFERENCES test_customers(id),
                total_amount NUMERIC,
                order_date DATE
            );
        """)
        
        print("Inserting test_customers...")
        # Insert 10,000 customers
        cur.execute("""
            INSERT INTO test_customers (name, status)
            SELECT 'Customer ' || i, 
                   CASE WHEN i % 5 = 0 THEN 'VIP' ELSE 'REGULAR' END
            FROM generate_series(1, 10000) i;
        """)
        
        print("Inserting test_orders...")
        # Insert 50,000 orders
        cur.execute("""
            INSERT INTO test_orders (customer_id, total_amount, order_date)
            SELECT (random() * 9999 + 1)::INT,
                   (random() * 2000)::NUMERIC,
                   '2023-01-01'::DATE + (random() * 365)::INT
            FROM generate_series(1, 50000) i;
        """)
        
        # NOTE: Intentionally not creating indexes on customer_id or total_amount 
        # so that the query plan has a high cost and index recommendations can be applied!
        
        print("Granting access to mcp_role...")
        # Make sure the MCP role can read these tables
        try:
            cur.execute("GRANT SELECT ON test_customers, test_orders TO apex_mcp_role;")
        except Exception as e:
            print(f"Warning granting roles: {e}")
            
print("Database setup complete. Triggering anomaly...")

payload = {
    "query": "SELECT id, name FROM test_customers WHERE id IN (SELECT customer_id FROM test_orders WHERE total_amount > 1900);",
    "baseline_exec_ms": 10.0,
    "current_exec_ms": 1250.0
}

res = requests.post("http://localhost:8000/api/trigger", json=payload)
print(f"Trigger response: {res.status_code}")
print(res.json())


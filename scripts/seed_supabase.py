"""
Supabase Data Seeder.

Loads the SQL Practice Dataset 2 CSV files into the Supabase sandbox
tables, and creates intentionally suboptimal queries for testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psycopg
from rich.console import Console

from src.config.settings import get_settings

console = Console()


def _generate_synthetic_food_data() -> dict[str, pd.DataFrame]:
    """
    Generate synthetic food-delivery data if CSV files aren't available.

    Creates realistic data for all 5 tables with proper foreign key
    relationships and realistic distributions.
    """
    np.random.seed(42)
    console.print("[yellow]   Generating synthetic food-delivery data...[/]")

    # ── Customers ─────────────────────────────────────────────
    n_customers = 500
    cities = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
        "San Antonio", "San Diego", "Dallas", "San Jose", "Austin",
    ]
    customers = pd.DataFrame({
        "customer_name": [f"Customer_{i}" for i in range(1, n_customers + 1)],
        "city": np.random.choice(cities, n_customers),
        "signup_date": pd.date_range("2022-01-01", periods=n_customers, freq="4h"),
        "email": [f"customer{i}@email.com" for i in range(1, n_customers + 1)],
        "phone": [f"+1-555-{np.random.randint(1000,9999)}" for _ in range(n_customers)],
    })

    # ── Restaurants ───────────────────────────────────────────
    n_restaurants = 100
    cuisines = [
        "Italian", "Chinese", "Mexican", "Indian", "Japanese",
        "Thai", "American", "French", "Korean", "Mediterranean",
    ]
    restaurants = pd.DataFrame({
        "restaurant_name": [f"Restaurant_{i}" for i in range(1, n_restaurants + 1)],
        "city": np.random.choice(cities, n_restaurants),
        "cuisine_type": np.random.choice(cuisines, n_restaurants),
        "rating": np.round(np.random.uniform(2.5, 5.0, n_restaurants), 1),
        "opening_hours": ["08:00-22:00"] * n_restaurants,
    })

    # ── Menu Items ────────────────────────────────────────────
    items_per_restaurant = 15
    n_items = n_restaurants * items_per_restaurant
    categories = ["Appetizer", "Main Course", "Dessert", "Beverage", "Side Dish"]
    menu_items = pd.DataFrame({
        "restaurant_id": np.repeat(range(1, n_restaurants + 1), items_per_restaurant),
        "item_name": [f"Item_{i}" for i in range(1, n_items + 1)],
        "price": np.round(np.random.uniform(5.0, 45.0, n_items), 2),
        "category": np.random.choice(categories, n_items),
        "is_available": np.random.choice([True, False], n_items, p=[0.9, 0.1]),
    })

    # ── Orders ────────────────────────────────────────────────
    n_orders = 5000
    orders = pd.DataFrame({
        "customer_id": np.random.randint(1, n_customers + 1, n_orders),
        "restaurant_id": np.random.randint(1, n_restaurants + 1, n_orders),
        "order_date": pd.date_range(
            "2023-01-01", periods=n_orders, freq="30min"
        ),
        "total_amount": np.round(np.random.uniform(10.0, 150.0, n_orders), 2),
        "delivery_time_minutes": np.random.randint(15, 90, n_orders),
        "status": np.random.choice(
            ["delivered", "cancelled", "pending"],
            n_orders,
            p=[0.85, 0.10, 0.05],
        ),
    })

    # ── Order Details ─────────────────────────────────────────
    details = []
    for order_id in range(1, n_orders + 1):
        rest_id = orders.iloc[order_id - 1]["restaurant_id"]
        rest_items = menu_items[
            menu_items["restaurant_id"] == rest_id
        ].index.tolist()

        if rest_items:
            n_detail_items = np.random.randint(1, 5)
            selected_items = np.random.choice(
                rest_items, min(n_detail_items, len(rest_items)), replace=False
            )
            for item_idx in selected_items:
                qty = np.random.randint(1, 4)
                price = menu_items.iloc[item_idx]["price"]
                details.append({
                    "order_id": order_id,
                    "item_id": int(item_idx + 1),  # 1-indexed
                    "quantity": qty,
                    "subtotal": round(float(price) * qty, 2),
                })

    order_details = pd.DataFrame(details)

    return {
        "customers": customers,
        "restaurants": restaurants,
        "menu_items": menu_items,
        "orders": orders,
        "order_details": order_details,
    }


def seed_database() -> None:
    """Seed the Supabase database with food-delivery data."""
    settings = get_settings()

    console.print("[bold magenta]🌱 Project Apex — Database Seeder[/]\n")

    # Try loading from CSV files first
    sql_dir = settings.data_dir / "sql_practice"
    csv_files = list(sql_dir.rglob("*.csv")) if sql_dir.exists() else []

    if csv_files:
        console.print(f"   Found {len(csv_files)} CSV files in {sql_dir}")
        # Load from CSVs — map filenames to tables
        tables_data = {}
        table_mapping = {
            "customers": "customers",
            "restaurants": "restaurants",
            "menu_items": "menu_items",
            "menu": "menu_items",
            "orders": "orders",
            "order_details": "order_details",
        }
        for csv_file in csv_files:
            stem = csv_file.stem.lower()
            for key, table_name in table_mapping.items():
                if key in stem:
                    tables_data[table_name] = pd.read_csv(csv_file)
                    console.print(f"   Loaded {table_name} from {csv_file.name}")
                    break
    else:
        tables_data = _generate_synthetic_food_data()

    # Insert into Supabase
    try:
        with psycopg.connect(settings.supabase_db_url) as conn:
            with conn.cursor() as cur:
                # Insert order matters (foreign keys)
                insert_order = [
                    "customers", "restaurants", "menu_items",
                    "orders", "order_details",
                ]

                for table_name in insert_order:
                    if table_name not in tables_data:
                        console.print(
                            f"   [yellow]⚠ Skipping {table_name} "
                            "(not found in data)[/]"
                        )
                        continue

                    df = tables_data[table_name]

                    # Clear existing data
                    cur.execute(
                        f"DELETE FROM {table_name}"  # noqa: S608
                    )

                    # Bulk insert using COPY
                    columns = list(df.columns)
                    col_str = ", ".join(columns)

                    # Use executemany for compatibility
                    placeholders = ", ".join(["%s"] * len(columns))
                    insert_sql = (
                        f"INSERT INTO {table_name} ({col_str}) "  # noqa: S608
                        f"VALUES ({placeholders})"
                    )

                    rows = [
                        tuple(
                            None if pd.isna(v) else v
                            for v in row
                        )
                        for row in df[columns].values.tolist()
                    ]

                    # Batch insert
                    batch_size = 500
                    for i in range(0, len(rows), batch_size):
                        batch = rows[i:i + batch_size]
                        cur.executemany(insert_sql, batch)

                    conn.commit()
                    console.print(
                        f"   [green]✅ {table_name}: "
                        f"{len(df)} rows inserted[/]"
                    )

                # Reset sequences
                for table_name in insert_order:
                    if table_name in tables_data:
                        try:
                            cur.execute(f"""
                                SELECT setval(
                                    pg_get_serial_sequence('{table_name}',
                                    (SELECT column_name FROM information_schema.columns
                                     WHERE table_name = '{table_name}'
                                     AND column_default LIKE 'nextval%'
                                     LIMIT 1)),
                                    (SELECT COALESCE(MAX(
                                        (SELECT column_name FROM information_schema.columns
                                         WHERE table_name = '{table_name}'
                                         AND column_default LIKE 'nextval%'
                                         LIMIT 1)::integer
                                    ), 1) FROM {table_name})
                                )
                            """)
                        except Exception:
                            conn.rollback()

        console.print("\n[bold green]✅ Database seeded successfully![/]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Seeding failed: {e}[/]")
        sys.exit(1)


if __name__ == "__main__":
    seed_database()

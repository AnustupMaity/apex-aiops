"""
Dataset Downloader for Project Apex (KaggleHub Version).

Downloads the three required datasets using the modern kagglehub package:
1. NAB (Numenta Anomaly Benchmark) — time-series telemetry
2. Spider (Text-to-SQL) — SQL understanding training data
3. SQL Practice Dataset 2 — food-delivery simulation environment
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import kagglehub
from dotenv import load_dotenv
from rich.console import Console

console = Console()
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def safe_copy_tree(src: str, dst: Path) -> None:
    """Safely copy the downloaded dataset cache to our local data folder."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = dst / item
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def download_nab() -> None:
    """Download the Numenta Anomaly Benchmark dataset."""
    target_dir = DATA_DIR / "nab"
    console.print("\n[bold cyan]📥 Downloading NAB dataset...[/]")

    try:
        path = kagglehub.dataset_download("boltzmannbrain/nab")
        safe_copy_tree(path, target_dir)
        console.print("[green]✅ NAB dataset downloaded successfully[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ Kaggle API failed: {e}[/]")

    csv_files = list(target_dir.rglob("*.csv"))
    console.print(f"   Found {len(csv_files)} CSV files")


def download_spider() -> None:
    """Download the Spider Text-to-SQL dataset."""
    target_dir = DATA_DIR / "spider"
    console.print("\n[bold cyan]📥 Downloading Spider dataset...[/]")

    try:
        path = kagglehub.dataset_download("jeromeblanchet/spider-dataset")
        safe_copy_tree(path, target_dir)
        console.print("[green]✅ Spider dataset downloaded successfully[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ Kaggle API failed: {e}[/]")

    for expected in ["train.json", "dev.json", "tables.json"]:
        found = list(target_dir.rglob(expected))
        status = "✅" if found else "❌"
        console.print(f"   {status} {expected}: {'found' if found else 'missing'}")


def download_sql_practice() -> None:
    """Download the SQL Practice Dataset 2 (Medium)."""
    target_dir = DATA_DIR / "sql_practice"
    console.print("\n[bold cyan]📥 Downloading SQL Practice Dataset 2...[/]")

    try:
        path = kagglehub.dataset_download("bhanupratapbiswas/sql-practice-dataset-2-medium-queries")
        safe_copy_tree(path, target_dir)
        console.print("[green]✅ SQL Practice Dataset 2 downloaded successfully[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ Kaggle API failed: {e}[/]")

    csv_files = list(target_dir.rglob("*.csv"))
    console.print(f"   Found {len(csv_files)} CSV files")


def verify_datasets() -> dict[str, bool]:
    """Verify all datasets are downloaded and accessible."""
    console.print("\n[bold]🔍 Verifying datasets...[/]")
    status = {}

    nab_csvs = list((DATA_DIR / "nab").rglob("*.csv"))
    status["nab"] = len(nab_csvs) > 0
    console.print(f"   NAB: {'✅' if status['nab'] else '❌'} ({len(nab_csvs)} CSV files)")

    spider_jsons = list((DATA_DIR / "spider").rglob("*.json"))
    status["spider"] = len(spider_jsons) > 0
    console.print(f"   Spider: {'✅' if status['spider'] else '❌'} ({len(spider_jsons)} JSON files)")

    sql_csvs = list((DATA_DIR / "sql_practice").rglob("*.csv"))
    status["sql_practice"] = len(sql_csvs) > 0
    console.print(f"   SQL Practice: {'✅' if status['sql_practice'] else '❌'} ({len(sql_csvs)} CSV files)")

    return status


if __name__ == "__main__":
    console.print("[bold magenta]🗂 Project Apex — Dataset Downloader[/]")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("KAGGLE_API_TOKEN") and not os.environ.get("KAGGLE_USERNAME"):
        console.print("\n[bold yellow]⚠ Missing Kaggle Credentials![/]\nEnsure KAGGLE_API_TOKEN is set in .env")
        sys.exit(1)

    download_nab()
    download_spider()
    download_sql_practice()

    status = verify_datasets()
    if all(status.values()):
        console.print("\n[bold green]✅ All datasets ready![/]")
    else:
        console.print("\n[bold red]❌ Missing datasets detected.[/]")
        sys.exit(1)


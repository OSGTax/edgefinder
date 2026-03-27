"""Initialize the EdgeFinder database.

Creates all tables from ORM models. Safe to run multiple times.
"""

import sys
sys.path.insert(0, ".")

from edgefinder.db.engine import Base, get_engine
from edgefinder.db import models  # noqa: F401 — registers all models

from rich.console import Console

console = Console()


def main():
    console.print("[bold]EdgeFinder v2 — Database Setup[/bold]\n")

    engine = get_engine()
    console.print(f"Database: {engine.url}")

    Base.metadata.create_all(engine)

    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    console.print(f"\n[green]{len(tables)} tables created:[/green]")
    for table in sorted(tables):
        cols = [c["name"] for c in inspector.get_columns(table)]
        console.print(f"  {table}: {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")

    console.print("\n[bold green]Database ready.[/bold green]")
    engine.dispose()


if __name__ == "__main__":
    main()

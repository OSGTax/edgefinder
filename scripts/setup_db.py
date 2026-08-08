"""Initialize the EdgeFinder database.

Creates all tables from ORM models. Safe to run multiple times.
"""

import sys
sys.path.insert(0, ".")

from edgefinder.db.engine import Base, get_engine
from edgefinder.db import models  # noqa: F401 — registers all models
from agent import models as agent_models  # noqa: F401 — registers desk_* tables

from rich.console import Console

console = Console()


def main():
    console.print("[bold]EdgeFinder v2 — Database Setup[/bold]\n")

    engine = get_engine()
    console.print(f"Database: {engine.url}")

    Base.metadata.create_all(engine)

    # Postgres extras create_all can't express (idempotent): the RLS toggles
    # and the edgefinder_db_size() RPC that agent.backup's size check calls
    # on the rest lane. Statement-by-statement, same as render_start.py.
    if engine.dialect.name == "postgresql":
        from sqlalchemy import text
        for ddl in agent_models.DESK_TABLE_DDL:
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
            except Exception as exc:  # noqa: BLE001 — idempotent best-effort
                console.print(f"[yellow]DDL skipped:[/yellow] {exc}")

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

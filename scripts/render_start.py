"""Startup script for Render deployment."""

import logging
import os
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Create all tables if they don't exist, and add any missing columns."""
    from edgefinder.db.engine import Base, get_engine
    from edgefinder.db import models  # noqa: F401 — registers ORM models
    from sqlalchemy import text, inspect

    engine = get_engine()
    Base.metadata.create_all(engine)

    # Add missing columns that create_all won't add to existing tables
    migrations = [
        ("strategy_accounts", "realized_pnl", "FLOAT DEFAULT 0.0"),
        ("tickers", "scan_batch", "INTEGER"),
    ]
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, col_type in migrations:
            existing = [c["name"] for c in inspector.get_columns(table)]
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                logger.info("Added column %s.%s", table, column)

    logger.info("Database tables verified/created")
    engine.dispose()


def main():
    init_database()

    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    main()

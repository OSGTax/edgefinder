"""Startup script for Render deployment."""

import logging
import os
import sys

sys.path.insert(0, ".")

from edgefinder.core.logging_config import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Create all tables if they don't exist, and add any missing columns."""
    from edgefinder.db.engine import Base, get_engine
    from edgefinder.db import models  # noqa: F401 — registers ORM models
    from sqlalchemy import text

    # Fail fast if DATABASE_URL is missing on Render — SQLite uses ephemeral
    # filesystem and all data (trades, accounts, research) is lost on redeploy
    db_url = os.getenv("DATABASE_URL", "")
    if os.getenv("RENDER") and not db_url:
        logger.error(
            "FATAL: DATABASE_URL is not set on Render. "
            "Without PostgreSQL, all data will be lost on every deploy.\n"
            "Set DATABASE_URL in Render Environment to your Supabase pooler URL:\n"
            "  postgresql://postgres.[ref]:[pass]@aws-0-[region].pooler.supabase.com:6543/postgres"
        )
        sys.exit(1)

    engine = get_engine()
    # Skip create_all on Render — tables already exist from prior deploys.
    # create_all hangs on schema reflection with Supabase pooler, causing
    # Render's port detection timeout. Only run on fresh databases.
    if not os.getenv("RENDER"):
        logger.info("Running create_all...")
        Base.metadata.create_all(engine)
        logger.info("create_all complete")
    else:
        logger.info("Render detected — skipping create_all (tables exist)")

    # Add missing columns that create_all won't add to existing tables
    logger.info("Running column migrations...")
    migrations = [
        ("strategy_accounts", "realized_pnl", "FLOAT DEFAULT 0.0"),
        ("tickers", "scan_batch", "INTEGER"),
        # Extended fundamentals columns (Phase 1 data expansion)
        ("fundamentals", "price_to_earnings", "FLOAT"),
        ("fundamentals", "price_to_book", "FLOAT"),
        ("fundamentals", "return_on_equity", "FLOAT"),
        ("fundamentals", "return_on_assets", "FLOAT"),
        ("fundamentals", "dividend_yield", "FLOAT"),
        ("fundamentals", "free_cash_flow", "FLOAT"),
        ("fundamentals", "quick_ratio", "FLOAT"),
        ("fundamentals", "short_shares", "INTEGER"),
        ("fundamentals", "days_to_cover", "FLOAT"),
        ("fundamentals", "dividend_amount", "FLOAT"),
        ("fundamentals", "ex_dividend_date", "VARCHAR(20)"),
        ("fundamentals", "news_sentiment", "VARCHAR(20)"),
        # Technical indicators from Massive API
        ("fundamentals", "rsi_14", "FLOAT"),
        ("fundamentals", "ema_21", "FLOAT"),
        ("fundamentals", "sma_50", "FLOAT"),
        ("fundamentals", "macd_value", "FLOAT"),
        ("fundamentals", "macd_signal", "FLOAT"),
        ("fundamentals", "macd_histogram", "FLOAT"),
        ("trades", "entry_reasoning", "TEXT"),
        ("trades", "exit_reasoning", "TEXT"),
        ("trades", "indicators_at_entry", "TEXT"),
        ("trades", "indicators_at_exit", "TEXT"),
        ("trades", "fundamentals_at_entry", "TEXT"),
        ("trades", "market_context_at_entry", "TEXT"),
        ("trades", "pdt_flag", "BOOLEAN DEFAULT 0"),
        ("trades", "hold_duration_hours", "FLOAT"),
    ]
    with engine.begin() as conn:
        for table, column, col_type in migrations:
            # Use IF NOT EXISTS to avoid inspection (which hangs on Supabase pooler)
            try:
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
                ))
            except Exception:
                # SQLite doesn't support IF NOT EXISTS — fall back to inspect
                try:
                    from sqlalchemy import inspect
                    inspector = inspect(engine)
                    existing = [c["name"] for c in inspector.get_columns(table)]
                    if column not in existing:
                        conn.execute(text(
                            f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                        ))
                        logger.info("Added column %s.%s", table, column)
                except Exception:
                    pass  # Column likely already exists

    logger.info("Column migrations complete")

    # New tables that create_all (skipped on Render) won't create. Idempotent
    # CREATE TABLE IF NOT EXISTS, guarded like the column migrations above.
    logger.info("Ensuring new tables exist...")
    table_ddls = [
        """CREATE TABLE IF NOT EXISTS system_heartbeat (
            id SERIAL PRIMARY KEY,
            component VARCHAR(50) UNIQUE NOT NULL,
            last_run_at TIMESTAMP NOT NULL,
            ok BOOLEAN DEFAULT TRUE,
            detail JSON
        )""",
        """CREATE TABLE IF NOT EXISTS validation_runs (
            id SERIAL PRIMARY KEY,
            strategy_name VARCHAR(50) NOT NULL,
            run_at TIMESTAMP DEFAULT NOW(),
            git_sha VARCHAR(40),
            universe VARCHAR(50),
            config JSON,
            oos JSON,
            criteria JSON,
            holdout JSON,
            verdict VARCHAR(10) NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_validation_runs_strat_ts
            ON validation_runs (strategy_name, run_at)""",
    ]
    with engine.begin() as conn:
        for ddl in table_ddls:
            try:
                conn.execute(text(ddl))
            except Exception:
                logger.exception("Table DDL failed (may already exist)")
    logger.info("New-table check complete")

    engine.dispose()
    logger.info("Database init done")


def main():
    # Run DB migrations synchronously — they must complete before uvicorn
    # starts so the ORM columns exist when the first request hits.
    # create_all is skipped on Render (hangs on Supabase pooler), so
    # this only runs the fast ALTER TABLE migrations.
    init_database()

    # One-time account reset — set EDGEFINDER_RESET_ACCOUNTS=1 on Render,
    # deploy, then remove the env var. Wipes old trades and resets to $10k.
    if os.getenv("EDGEFINDER_RESET_ACCOUNTS") == "1":
        logger.info("EDGEFINDER_RESET_ACCOUNTS=1 — running account reset")
        from scripts.reset_accounts import reset
        reset()

    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting uvicorn on port %d...", port)
    uvicorn.run(
        "dashboard.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    main()

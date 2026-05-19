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
    from sqlalchemy import text, inspect

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
    logger.info("Running create_all...")
    Base.metadata.create_all(engine)
    logger.info("create_all complete")

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
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, column, col_type in migrations:
            existing = [c["name"] for c in inspector.get_columns(table)]
            if column not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                logger.info("Added column %s.%s", table, column)

    logger.info("Column migrations complete")
    engine.dispose()
    logger.info("Database init done")


def main():
    # Run DB init in background so uvicorn can bind the port immediately.
    # Render kills deploys that don't bind a port within ~60s.
    # The tables already exist from prior deploys — this just adds new columns.
    import threading

    def _bg_init():
        try:
            init_database()
        except Exception:
            logger.exception("Background database init failed — app will still serve")

    thread = threading.Thread(target=_bg_init, daemon=True, name="db-init")
    thread.start()

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

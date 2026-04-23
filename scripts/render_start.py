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
    Base.metadata.create_all(engine)

    # Add missing columns that create_all won't add to existing tables
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


def provision_agent_config() -> None:
    """Write .claude/agent-config.json from the AGENT_CONFIG_JSON env var.

    Required so `is_agent_enabled()` returns True on Render. If the env var
    is missing or empty we leave the file alone — the gate defaults to
    disabled, which is the correct fail-safe when running untrusted code.
    """
    from pathlib import Path

    payload = os.getenv("AGENT_CONFIG_JSON", "").strip()
    if not payload:
        logger.info(
            "AGENT_CONFIG_JSON is not set — in-process agents will default "
            "to disabled. Set the env var to a JSON body matching "
            ".claude/agent-config.example.json to enable them."
        )
        return

    target = Path(".claude/agent-config.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload)
    logger.info("Wrote %s from AGENT_CONFIG_JSON (%d bytes)", target, len(payload))


def main():
    init_database()
    provision_agent_config()

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

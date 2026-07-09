"""Startup script for Render deployment.

Post-cutover (2026-06-22): the only tables the runtime touches are the desk_*
set defined in ``agent.models.DESK_TABLE_DDL``. This script used to also
CREATE the retired trading-workbench tables (strategy_accounts,
promoted_strategies, agent_decisions, validation_runs, dividend_credits,
system_heartbeat, …) and ALTER old columns onto them — on every deploy —
silently undoing what ``scripts/cutover.py`` intentionally dropped. That
whole block is gone; only the greenfield desk schema is ensured here.

If the deploy still needs to recreate the desk tables (fresh DB, or after a
manual wipe), ``DESK_TABLE_DDL`` covers it with ``CREATE TABLE IF NOT
EXISTS``. Never add a CREATE / ALTER referring to a table not in the
``desk_*`` set — the cutover is durable; this script must not resurrect it.
"""

import logging
import os
import sys

sys.path.insert(0, ".")

from edgefinder.core.logging_config import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Ensure the desk_* tables exist. Idempotent.

    Two paths, same result:
    - Dev / SQLite: SQLAlchemy ``create_all`` — it's dialect-aware and handles
      the desk_* ORM models cleanly (the raw DESK_TABLE_DDL is Postgres-only:
      SERIAL, NOW()).
    - Render / Supabase: ``create_all`` hangs on the pooler's schema
      reflection so we skip it and run the raw ``DESK_TABLE_DDL`` instead —
      idempotent CREATE TABLE IF NOT EXISTS statements the pooler runs
      without reflection.
    """
    from edgefinder.db.engine import Base, get_engine
    from agent import models as desk_models  # noqa: F401 — registers desk_* ORM
    from sqlalchemy import text

    # Fail fast if DATABASE_URL is missing on Render — SQLite uses ephemeral
    # filesystem and all data (trades, positions, decisions) is lost on redeploy
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
    if os.getenv("RENDER"):
        logger.info("Render detected — running raw desk_* DDL (pooler-safe)")
        with engine.begin() as conn:
            for ddl in desk_models.DESK_TABLE_DDL:
                try:
                    conn.execute(text(ddl))
                except Exception:
                    logger.exception("desk DDL failed (may already exist)")
    else:
        logger.info("Running create_all for desk_* (dev/SQLite path)")
        Base.metadata.create_all(engine)
    engine.dispose()
    logger.info("Database init done")


def main():
    # Run DB init synchronously so tables exist before uvicorn starts.
    init_database()

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

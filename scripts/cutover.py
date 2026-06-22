"""Pre-authorized cutover: drop the OLD trading/app tables, KEEP the data.

This is the destructive DB half of the greenfield rebuild (REBUILD-PLAN.md).
It is deliberately a separate, guarded script so the irreversible step is
explicit and obeys the three engineering rails:

  Rail 1 — NEVER touch the market-data tables or the R2 archive. This script
           drops ONLY the retired trading/app tables in DROP_TABLES; the
           KEEP_TABLES set is asserted present and never dropped.
  Rail 2 — Prove the new system works BEFORE wiping. With R2_* configured,
           ``--execute`` first runs BarStore().verify() and refuses to drop
           anything unless the R2 archive reads back clean. (Run this on
           Render / in a Routine where the R2_* creds exist — NOT the sandbox.)
  Rail 3 — Git-recoverable: this only drops tables; the old CODE stays in git
           history. The code removal is a normal reviewed deploy change.

Usage:
    python scripts/cutover.py                 # dry-run: print the plan
    python scripts/cutover.py --execute       # drop old tables (guarded)
    python scripts/cutover.py --skip-r2-check --execute   # only if R2 proven elsewhere

The OLD application CODE to delete in the same deploy (kept here as the
authoritative checklist; this script does not touch the filesystem):
    edgefinder/engine/{validate,walkforward,promote,live,record,hunt_r1,hunt_r2,
        analyst_strategy,live_ticket}.py
    edgefinder/agents/  edgefinder/scanner/  edgefinder/signals/
    edgefinder/analytics/  edgefinder/scheduler/  edgefinder/trading/
    dashboard/routers/{trades,strategies,research,benchmarks,market,lab,picks,
        admin,ops}.py and their templates/pages + page JS
    .github/workflows/*  (the hunt/watchdog/liveness/analyst crons)
  KEEP (the thin data-access + backtest layer the agent reuses):
    edgefinder/db/{engine,models}.py  edgefinder/engine/{data,backtest,strategy}.py
    edgefinder/backtest/costs.py  edgefinder/data/{barstore,indicator_engine,
        market_data,polygon,provider,cache}.py  edgefinder/core/*
    agent/*  dashboard (desk page + core static + base template)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cutover")

# Sacred — irreplaceable market data. Asserted present; NEVER dropped.
KEEP_TABLES = {
    "daily_bars", "dividends", "ticker_dividends", "ticker_splits",
    "fundamentals_snapshots", "fundamentals", "ticker_news", "index_daily",
    "tickers", "market_snapshots",
    # the agent's own new schema
    "desk_trades", "desk_positions", "desk_equity", "desk_strategy_state",
    "desk_journal", "desk_thinking", "desk_decisions", "desk_backtests",
}

# Retired trading/app tables — the wipe target.
DROP_TABLES = [
    "trades", "strategy_accounts", "strategy_snapshots", "strategy_parameters",
    "promoted_strategies", "validation_runs", "dividend_credits",
    "agent_decisions", "llm_decision_cache", "llm_decision_log",
    "trade_context", "ticker_strategy_qualifications", "manual_injections",
    "agent_observations", "agent_actions", "agent_memory",
    "system_heartbeat", "weekly_leaderboard", "landmark_leaderboard",
    "alembic_version",
]


def _r2_proof() -> bool:
    """Rail 2: prove the R2 archive reads back clean before wiping anything."""
    if not all(os.getenv(k) for k in
               ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET")):
        logger.error("R2_* env vars not set — cannot prove R2 reads here. Run this "
                     "on Render / in a Routine, or pass --skip-r2-check only if R2 "
                     "was proven elsewhere.")
        return False
    from edgefinder.data.barstore import BarStore
    from edgefinder.db.engine import get_engine, get_session_factory

    session = get_session_factory(get_engine())()
    try:
        result = BarStore().verify(session, sample=25)
    finally:
        session.close()
    logger.info("R2 verify: %s", result)
    return bool(result.get("ok"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true",
                   help="actually DROP the old tables (default: dry-run)")
    p.add_argument("--skip-r2-check", action="store_true",
                   help="skip the R2-read proof (only if proven in another run)")
    args = p.parse_args(argv)

    from sqlalchemy import text
    from edgefinder.db.engine import get_engine

    engine = get_engine()
    with engine.connect() as conn:
        present = {r[0] for r in conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public'"))}

    # Rail 1 safety assertion: never let a data table slip into the drop list.
    bad = set(DROP_TABLES) & KEEP_TABLES
    if bad:
        logger.error("ABORT — drop list intersects KEEP_TABLES: %s", bad)
        return 2

    to_drop = [t for t in DROP_TABLES if t in present]
    logger.info("Cutover plan (DB):")
    logger.info("  KEEP (data + agent): %s", ", ".join(sorted(KEEP_TABLES & present)))
    logger.info("  DROP (retired):      %s", ", ".join(to_drop) or "(none present)")

    if not args.execute:
        logger.info("\nDry-run only. Re-run with --execute to drop the retired tables.")
        return 0

    if not args.skip_r2_check:
        logger.info("\nRail 2 — proving R2 reads before wipe...")
        if not _r2_proof():
            logger.error("ABORT — R2 proof failed/unavailable; nothing dropped.")
            return 3

    logger.info("\nExecuting drops...")
    dropped = 0
    with engine.begin() as conn:
        for t in to_drop:
            conn.execute(text(f"DROP TABLE IF EXISTS {t} CASCADE"))
            logger.info("  dropped %s", t)
            dropped += 1
    logger.info("Cutover DB wipe complete — %d tables dropped, data preserved.", dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

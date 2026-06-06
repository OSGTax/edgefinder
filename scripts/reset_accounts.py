"""One-time account reset: wipe old trades, reset strategy accounts to fresh $10k.

Run directly:
    python scripts/reset_accounts.py

Or set EDGEFINDER_RESET_ACCOUNTS=1 env var on Render for a one-deploy reset.
The script deletes all trades before 2026-05-20 and resets all strategy
accounts to $10,000 starting capital with zero P&L.
"""

import logging
import os
import sys

sys.path.insert(0, ".")

from edgefinder.core.logging_config import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2026-06-07: bumped from 2026-05-20 to clear the 8 pre-v5.11 positions
# (sized before cap enforcement deployed; also the legacy-unverifiable
# chain rows). Override per-run with --cutoff.
CUTOFF_DATE = "2026-06-08"


def reset(cutoff: str = CUTOFF_DATE):
    from sqlalchemy import text
    from edgefinder.db.engine import get_engine

    engine = get_engine()
    logger.info("Starting account reset — cutoff date: %s", cutoff)

    with engine.begin() as conn:
        # 1. Delete trade_context rows FIRST (FK references trades.trade_id)
        result = conn.execute(text("""
            DELETE FROM trade_context
            WHERE trade_id IN (
                SELECT trade_id FROM trades
                WHERE entry_time < :cutoff OR entry_time IS NULL OR status = 'CANCELLED'
            )
        """), {"cutoff": cutoff})
        logger.info("Deleted %d trade context rows", result.rowcount)

        # 2. Delete all trades before cutoff + cancelled trades
        result = conn.execute(text(
            "DELETE FROM trades WHERE entry_time < :cutoff OR entry_time IS NULL OR status = 'CANCELLED'"
        ), {"cutoff": cutoff})
        logger.info("Deleted %d old/cancelled trades", result.rowcount)

        # 4. Reset all strategy accounts to fresh $10k
        result = conn.execute(text("""
            UPDATE strategy_accounts SET
                starting_capital = 10000.0,
                cash_balance = 10000.0,
                open_positions_value = 0.0,
                total_equity = 10000.0,
                peak_equity = 10000.0,
                drawdown_pct = 0.0,
                realized_pnl = 0.0,
                is_paused = false
        """))
        logger.info("Reset %d strategy accounts to $10,000", result.rowcount)

        # 5. Delete old equity curve snapshots before cutoff
        result = conn.execute(text(
            "DELETE FROM strategy_snapshots WHERE timestamp < :cutoff"
        ), {"cutoff": cutoff})
        logger.info("Deleted %d old equity snapshots", result.rowcount)

        # 6. Delete old agent observations and actions
        try:
            result = conn.execute(text(
                "DELETE FROM agent_observations WHERE timestamp < :cutoff"
            ), {"cutoff": cutoff})
            logger.info("Deleted %d old agent observations", result.rowcount)

            result = conn.execute(text(
                "DELETE FROM agent_actions WHERE timestamp < :cutoff"
            ), {"cutoff": cutoff})
            logger.info("Deleted %d old agent actions", result.rowcount)
        except Exception:
            logger.debug("No agent tables to clean (may not exist)")

    engine.dispose()
    logger.info("Account reset complete — all strategies at $10,000, no trade history before %s", cutoff)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Reset strategy accounts to a clean $10k")
    ap.add_argument("--cutoff", default=CUTOFF_DATE,
                    help="delete trades/snapshots before this date (YYYY-MM-DD)")
    args = ap.parse_args()
    reset(args.cutoff)

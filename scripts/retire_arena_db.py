"""Retire the old arena from the production database (owner approved
"just delete, no archive" 2026-06-10).

Removes every row belonging to the retired old-arena strategies and the
two feature tables whose features no longer exist:

  - trades + trade_context + their now-orphaned market_snapshots
    (coward / gambler / degenerate)
  - strategy_accounts (the three arena accounts + the four dead
    alpha/bravo/charlie/echo test shells that never traded)
  - strategy_snapshots, strategy_parameters (arena names only)
  - ticker_strategy_qualifications (whole table — per-strategy
    qualification retired with the scanner's qualify step)
  - manual_injections (whole table — inject feature removed)

The v2 lane (equal_weight, dual_momentum_v2, dividend_credits,
validation_runs, promoted_strategies) is untouched; a hard guard
aborts if any v2 name ever appears in the delete list.

Default is a DRY RUN that prints counts. ``--execute`` deletes, in one
transaction. Raw SQL on purpose: the ORM models for these tables are
being deleted from the codebase in the same initiative.
"""

from __future__ import annotations

import argparse
import os
import sys

from sqlalchemy import create_engine, text

ARENA_STRATEGIES = ("coward", "gambler", "degenerate")
DEAD_TEST_ACCOUNTS = ("alpha", "bravo", "charlie", "echo")
V2_STRATEGIES = ("equal_weight", "dual_momentum_v2")

ACCOUNT_NAMES = ARENA_STRATEGIES + DEAD_TEST_ACCOUNTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="actually delete (default: dry-run counts only)")
    args = parser.parse_args()

    for name in ACCOUNT_NAMES:
        if name in V2_STRATEGIES:
            print(f"ABORT: v2 strategy {name!r} in delete list")
            return 1

    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ABORT: DATABASE_URL not set")
        return 1
    engine = create_engine(url)

    arena = {"names": list(ARENA_STRATEGIES)}
    accounts = {"names": list(ACCOUNT_NAMES)}
    steps = [
        ("trade_context (arena trades)",
         "DELETE FROM trade_context WHERE trade_id IN "
         "(SELECT trade_id FROM trades WHERE strategy_name = ANY(:names))",
         arena),
        ("trades (arena)",
         "DELETE FROM trades WHERE strategy_name = ANY(:names)", arena),
        # exclude arena trades from the reference scan so the dry-run
        # count matches what --execute will remove after step 2
        ("market_snapshots (orphaned)",
         "DELETE FROM market_snapshots WHERE id NOT IN "
         "(SELECT market_snapshot_id FROM trades "
         " WHERE market_snapshot_id IS NOT NULL "
         " AND NOT (strategy_name = ANY(:names)))", arena),
        ("strategy_snapshots (arena)",
         "DELETE FROM strategy_snapshots WHERE strategy_name = ANY(:names)",
         arena),
        ("strategy_parameters (arena)",
         "DELETE FROM strategy_parameters WHERE strategy_name = ANY(:names)",
         arena),
        ("strategy_accounts (arena + dead test shells)",
         "DELETE FROM strategy_accounts WHERE strategy_name = ANY(:names)",
         accounts),
        ("ticker_strategy_qualifications (ALL — feature retired)",
         "DELETE FROM ticker_strategy_qualifications", {}),
        ("manual_injections (ALL — feature retired)",
         "DELETE FROM manual_injections", {}),
    ]

    with engine.begin() as conn:
        total = 0
        for label, delete_sql, params in steps:
            count_sql = delete_sql.replace("DELETE FROM", "SELECT count(*) FROM", 1)
            n = conn.execute(text(count_sql), params).scalar() or 0
            total += n
            print(f"{'DELETE' if args.execute else 'would delete':>12}  {n:>6}  {label}")
            if args.execute:
                conn.execute(text(delete_sql), params)
        keep = conn.execute(
            text("SELECT strategy_name FROM strategy_accounts ORDER BY 1")
        ).scalars().all()
        if args.execute:
            print(f"\nDeleted {total} rows. Remaining accounts: {keep}")
        else:
            print(f"\nDRY RUN — {total} rows would be deleted. "
                  f"Accounts currently present: {keep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

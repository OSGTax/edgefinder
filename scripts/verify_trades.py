#!/usr/bin/env python3
"""
EdgeFinder Trade Integrity Verifier
=====================================
Walks the hash chain on the arena trade log and verifies that
no records have been inserted, edited, deleted, or reordered.

Usage: python scripts/verify_trades.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import ArenaTradeLog, get_session, init_db
from modules.utils import verify_hash_chain, to_eastern


def main():
    print("=" * 60)
    print("  EDGEFINDER — Trade Integrity Verification")
    print("=" * 60)
    print()

    # Initialize database connection
    init_db()

    session = get_session()
    try:
        records = session.query(ArenaTradeLog).filter(
            ArenaTradeLog.integrity_hash != None  # noqa: E711
        ).order_by(ArenaTradeLog.sequence_num.asc()).all()

        if not records:
            print("  No trades with integrity hashes found.")
            print("  (Trades recorded before the hash chain was added won't have hashes.)")
            print()
            return

        trades = [
            {
                "trade_id": r.trade_id,
                "ticker": r.ticker,
                "action": r.action,
                "execution_price": r.execution_price,
                "shares": r.shares,
                "execution_timestamp": r.execution_timestamp,
                "strategy_name": r.strategy_name,
                "integrity_hash": r.integrity_hash,
                "sequence_num": r.sequence_num,
            }
            for r in records
        ]

        result = verify_hash_chain(trades)

        if result["valid"]:
            print(f"  ✓ {result['message']}")
        else:
            print(f"  ✗ INTEGRITY VIOLATION: {result['message']}")

        print()
        print(f"  Trades checked: {result['total_trades']}")
        if records:
            first = records[0]
            last = records[-1]
            print(f"  First trade:    #{first.sequence_num} — "
                  f"{first.ticker} @ {to_eastern(first.execution_timestamp)} ET")
            print(f"  Last trade:     #{last.sequence_num} — "
                  f"{last.ticker} @ {to_eastern(last.execution_timestamp)} ET")
        print()

        if not result["valid"]:
            sys.exit(1)

    finally:
        session.close()


if __name__ == "__main__":
    main()

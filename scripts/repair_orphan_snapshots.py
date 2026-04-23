"""Backfill `trades.market_snapshot_id` for rows that have NULL.

A trade can be persisted with `market_snapshot_id=NULL` if the Polygon
snapshot capture raises at trade-open time (by design — we'd rather keep
the trade than lose it). This script walks those rows and associates
each one with the nearest-in-time `market_snapshots` row.

Idempotent. Safe to run repeatedly.

Usage:
    python scripts/repair_orphan_snapshots.py             # dry-run
    python scripts/repair_orphan_snapshots.py --apply     # write changes
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import timedelta

sys.path.insert(0, ".")

from sqlalchemy import func

from edgefinder.core.logging_config import configure_logging
from edgefinder.db.engine import get_engine, get_session_factory
from edgefinder.db.models import MarketSnapshotRecord, TradeRecord

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LAG = timedelta(hours=6)  # don't associate a trade with a snapshot > 6h away


def _nearest_snapshot(session, when):
    """Return the MarketSnapshotRecord whose timestamp is closest to `when`.

    None if no snapshot exists within MAX_LAG.
    """
    before = (
        session.query(MarketSnapshotRecord)
        .filter(MarketSnapshotRecord.timestamp <= when)
        .order_by(MarketSnapshotRecord.timestamp.desc())
        .first()
    )
    after = (
        session.query(MarketSnapshotRecord)
        .filter(MarketSnapshotRecord.timestamp > when)
        .order_by(MarketSnapshotRecord.timestamp.asc())
        .first()
    )
    candidates = [c for c in (before, after) if c is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda s: abs(s.timestamp - when))
    if abs(best.timestamp - when) > MAX_LAG:
        return None
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually write changes")
    args = parser.parse_args()

    engine = get_engine()
    session_factory = get_session_factory(engine)
    session = session_factory()
    try:
        orphan_total = (
            session.query(func.count(TradeRecord.id))
            .filter(TradeRecord.market_snapshot_id.is_(None))
            .scalar()
        ) or 0
        if orphan_total == 0:
            logger.info("No orphan trades found. Nothing to do.")
            return 0

        logger.info("Found %d trades with NULL market_snapshot_id", orphan_total)

        orphans = (
            session.query(TradeRecord)
            .filter(TradeRecord.market_snapshot_id.is_(None))
            .order_by(TradeRecord.entry_time.asc())
            .all()
        )
        repaired = 0
        for trade in orphans:
            snap = _nearest_snapshot(session, trade.entry_time)
            if snap is None:
                continue
            logger.info(
                "Trade %s @ %s → snapshot #%d @ %s (lag %s)",
                trade.trade_id, trade.entry_time, snap.id, snap.timestamp,
                abs(snap.timestamp - trade.entry_time),
            )
            if args.apply:
                trade.market_snapshot_id = snap.id
            repaired += 1

        if args.apply:
            session.commit()
            logger.info("Committed %d repairs (out of %d orphans)", repaired, orphan_total)
        else:
            logger.info(
                "Dry run. Would repair %d of %d orphans. Re-run with --apply "
                "to persist changes.",
                repaired, orphan_total,
            )
        return 0
    except Exception:
        session.rollback()
        logger.exception("repair_orphan_snapshots failed")
        return 1
    finally:
        session.close()
        engine.dispose()


if __name__ == "__main__":
    sys.exit(main())

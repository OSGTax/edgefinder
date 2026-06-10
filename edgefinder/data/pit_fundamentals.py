"""Point-in-time fundamentals: snapshot writer + as-of reader.

Why this exists: the ``fundamentals`` table holds ONE row per symbol,
overwritten on every scan — so a backtest that reads it applies TODAY's
ratios to every historical decision date (pure look-ahead). The honest
Lynch/GARP lane needs "what was known on date X", which means dated
snapshots. This module:

- ``snapshot_fundamentals(session, as_of)`` — copies the current
  fundamentals table into ``fundamentals_snapshots`` keyed by ``as_of``
  (idempotent per date). Called after every nightly scan, so PIT history
  accumulates forward from the day this shipped.
- ``PITFundamentals(session)`` — the as-of reader the engine consumes:
  ``.asof(symbol, date)`` returns the latest snapshot at or before ``date``
  (None if none — honest: before coverage begins, a strategy sees nothing).

Historical backfill (pre-deployment dates) would need Polygon financials
keyed by FILING date; evaluated and deferred — see HANDOFF.

CLI:
    python -m edgefinder.data.pit_fundamentals snapshot   # snapshot now
    python -m edgefinder.data.pit_fundamentals coverage   # what exists
"""

from __future__ import annotations

import bisect
import logging
from datetime import date, datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from edgefinder.core.models import TickerFundamentals
from edgefinder.db.models import Fundamental, FundamentalsSnapshot

logger = logging.getLogger(__name__)


def _row_to_dict(row: Fundamental) -> dict:
    """The TickerFundamentals-shaped dict for a fundamentals row."""
    fields = set(TickerFundamentals.model_fields)
    out = {}
    for col in row.__table__.columns:
        if col.name in fields:
            v = getattr(row, col.name)
            if isinstance(v, datetime):
                v = v.isoformat()
            out[col.name] = v
    return out


def snapshot_fundamentals(session: Session, as_of: date | None = None) -> int:
    """Copy the current fundamentals table into the PIT store for ``as_of``.

    Idempotent per date: an existing snapshot for ``as_of`` is left alone
    (the first write of a day wins — re-running a scan does not silently
    rewrite history).
    """
    as_of = as_of or datetime.now(timezone.utc).date()
    existing = (session.query(func.count(FundamentalsSnapshot.id))
                .filter(FundamentalsSnapshot.as_of == as_of).scalar())
    if existing:
        logger.info("PIT snapshot for %s already exists (%d rows) — skipping",
                    as_of, existing)
        return 0

    n = 0
    for row in session.query(Fundamental).all():
        session.add(FundamentalsSnapshot(
            symbol=row.symbol, as_of=as_of, data=_row_to_dict(row)))
        n += 1
    session.commit()
    logger.info("PIT snapshot %s: %d symbols", as_of, n)
    return n


class PITFundamentals:
    """As-of fundamentals lookup for the engine (loads lazily per symbol).

    Satisfies the engine's PIT protocol: ``asof(symbol, as_of)`` returns the
    latest TickerFundamentals snapshot dated <= as_of, or None before
    coverage begins. Snapshot dates are cached per symbol; data rows are
    fetched on demand and memoized.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
        self._dates: dict[str, list[date]] = {}
        self._cache: dict[tuple[str, date], TickerFundamentals | None] = {}

    def _symbol_dates(self, symbol: str) -> list[date]:
        if symbol not in self._dates:
            rows = (self._session.query(FundamentalsSnapshot.as_of)
                    .filter(FundamentalsSnapshot.symbol == symbol)
                    .order_by(FundamentalsSnapshot.as_of).all())
            self._dates[symbol] = [r[0] for r in rows]
        return self._dates[symbol]

    def asof(self, symbol: str, as_of: date) -> TickerFundamentals | None:
        dates = self._symbol_dates(symbol)
        i = bisect.bisect_right(dates, as_of) - 1
        if i < 0:
            return None
        snap_date = dates[i]
        key = (symbol, snap_date)
        if key not in self._cache:
            row = (self._session.query(FundamentalsSnapshot)
                   .filter(FundamentalsSnapshot.symbol == symbol,
                           FundamentalsSnapshot.as_of == snap_date).one())
            try:
                self._cache[key] = TickerFundamentals(**row.data)
            except Exception:
                logger.exception("bad PIT snapshot %s@%s", symbol, snap_date)
                self._cache[key] = None
        return self._cache[key]


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from edgefinder.db.engine import get_engine, get_session_factory

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["snapshot", "coverage"])
    args = p.parse_args(argv)

    session = get_session_factory(get_engine())()
    try:
        if args.command == "snapshot":
            n = snapshot_fundamentals(session)
            print(f"snapshot rows written: {n}")
        else:
            rows = (session.query(
                        FundamentalsSnapshot.as_of,
                        func.count(FundamentalsSnapshot.id))
                    .group_by(FundamentalsSnapshot.as_of)
                    .order_by(FundamentalsSnapshot.as_of).all())
            print(json.dumps({str(d): n for d, n in rows}, indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

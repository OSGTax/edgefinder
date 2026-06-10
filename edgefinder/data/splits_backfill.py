"""Market-wide stock-split backfill into ticker_splits.

The fidelity audit (2026-06-10) found daily_bars' stock lane is RAW
as-traded: ~256 splits since 2021 produce fake ±60-99% one-day moves
(NVDA's 10:1 reads -89.9%; GE's 1:8 reverse reads +677%). ticker_splits had
correct ratios but covered only the 745 old-scan symbols. This backfill
pulls EVERY split from Polygon reference data (one paginated market-wide
stream, no per-symbol calls) so the load-time adjustment in
engine/data.load_bars covers the whole universe.

Non-integral ratios are stored as reduced integer fractions (the columns
are Integer); anything that cannot be expressed exactly is skipped loudly.

CLI:
    python -m edgefinder.data.splits_backfill run [--since 2019-01-01]
    python -m edgefinder.data.splits_backfill coverage
"""

from __future__ import annotations

import logging
from fractions import Fraction

from sqlalchemy import func
from sqlalchemy.orm import Session

from edgefinder.db.models import TickerSplit

logger = logging.getLogger(__name__)


def _as_int_ratio(split_from, split_to) -> tuple[int, int] | None:
    """(from, to) as reduced integers (INT32-safe), or None if inexpressible.

    A handful of OTC adjustment events carry absurd exact fractions (e.g.
    250000000000000:259290138161029); those are approximated to <1e-9 error
    via limit_denominator — far below price-data precision.
    """
    try:
        f = Fraction(str(split_to)) / Fraction(str(split_from))
    except (ValueError, ZeroDivisionError):
        return None
    if f <= 0:
        return None
    if f.numerator > 1_000_000 or f.denominator > 1_000_000:
        f = f.limit_denominator(100_000)
        if f <= 0 or f.numerator > 2_000_000_000:
            return None
    return f.denominator, f.numerator   # from, to


def backfill(session: Session, client, since: str = "2019-01-01") -> dict:
    existing = {(r.symbol, str(r.execution_date))
                for r in session.query(TickerSplit.symbol,
                                       TickerSplit.execution_date)}
    written = skipped = seen = 0
    for s in client.list_splits(execution_date_gte=since, limit=1000):
        seen += 1
        sym = getattr(s, "ticker", None)
        ex = getattr(s, "execution_date", None)
        ratio = _as_int_ratio(getattr(s, "split_from", None),
                              getattr(s, "split_to", None))
        if not sym or not ex or ratio is None:
            skipped += 1
            if ratio is None and sym:
                logger.warning("skipping inexpressible split %s %s %s:%s",
                               sym, ex, getattr(s, "split_from", "?"),
                               getattr(s, "split_to", "?"))
            continue
        if (sym, str(ex)) in existing:
            continue
        session.add(TickerSplit(symbol=sym, execution_date=str(ex),
                                split_from=ratio[0], split_to=ratio[1]))
        existing.add((sym, str(ex)))
        written += 1
        if written % 500 == 0:
            session.commit()
            logger.info("splits: %d seen, %d written", seen, written)
    session.commit()
    return {"seen": seen, "written": written, "skipped": skipped}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from edgefinder.data.polygon import PolygonDataProvider
    from edgefinder.db.engine import get_engine, get_session_factory

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "coverage"])
    p.add_argument("--since", default="2019-01-01")
    args = p.parse_args(argv)

    session = get_session_factory(get_engine())()
    try:
        if args.command == "coverage":
            n = session.query(func.count(TickerSplit.id)).scalar()
            syms = session.query(
                func.count(func.distinct(TickerSplit.symbol))).scalar()
            print(json.dumps({"rows": n, "symbols": syms}, indent=2))
            return
        client = PolygonDataProvider()._client
        print(json.dumps(backfill(session, client, since=args.since), indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

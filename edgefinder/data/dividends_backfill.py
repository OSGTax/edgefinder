"""Backfill cash-dividend history from Polygon into the dividends table.

One paginated reference call per symbol; idempotent (skip-existing per
(symbol, ex_date)). Raw bars stay dividend-UNadjusted everywhere — these rows
feed the load-time total-return transform (engine/data.adjust_for_dividends).

CLI:
    python -m edgefinder.data.dividends_backfill run [--symbols A,B]
    python -m edgefinder.data.dividends_backfill coverage
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session

from edgefinder.db.models import DailyBar, DividendRecord

logger = logging.getLogger(__name__)

_FETCH_WORKERS = 6


def _to_date(s) -> date | None:
    if not s:
        return None
    try:
        return s if isinstance(s, date) else date.fromisoformat(str(s))
    except ValueError:
        return None


def backfill(session: Session, client, symbols: list[str]) -> dict:
    def fetch(symbol: str):
        try:
            out = []
            for d in client.list_dividends(ticker=symbol, limit=1000):
                ex = _to_date(getattr(d, "ex_dividend_date", None))
                amt = getattr(d, "cash_amount", None)
                if ex is not None and amt and amt > 0:
                    out.append((ex, float(amt)))
            return symbol, out
        except Exception as e:
            logger.warning("dividends fetch failed for %s: %s", symbol, e)
            return symbol, []

    written = with_data = fetched = 0
    chunk = 50
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
            results = list(pool.map(fetch, batch))
        for symbol, divs in results:
            fetched += 1
            if not divs:
                continue
            with_data += 1
            existing = {r[0] for r in session.query(DividendRecord.ex_date)
                        .filter(DividendRecord.symbol == symbol).all()}
            seen: set = set()
            for ex, amt in divs:
                if ex in existing or ex in seen:
                    continue
                seen.add(ex)
                session.add(DividendRecord(symbol=symbol, ex_date=ex,
                                           cash_amount=amt))
                written += 1
        session.commit()
        if (i // chunk) % 10 == 0:
            logger.info("dividends: %d/%d symbols, %d with data, %d rows",
                        fetched, len(symbols), with_data, written)
    return {"symbols": len(symbols), "with_data": with_data,
            "rows_written": written}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from edgefinder.data.polygon import PolygonDataProvider
    from edgefinder.db.engine import get_engine, get_session_factory

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "coverage"])
    p.add_argument("--symbols", default=None)
    args = p.parse_args(argv)

    session = get_session_factory(get_engine())()
    try:
        if args.command == "coverage":
            total = session.query(func.count(DividendRecord.id)).scalar()
            syms = session.query(
                func.count(func.distinct(DividendRecord.symbol))).scalar()
            rng = session.query(func.min(DividendRecord.ex_date),
                                func.max(DividendRecord.ex_date)).one()
            print(json.dumps({"rows": total, "symbols": syms,
                              "from": str(rng[0]), "to": str(rng[1])}, indent=2))
            return
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",")]
        else:
            symbols = sorted(r[0] for r in
                             session.query(DailyBar.symbol).distinct().all())
        client = PolygonDataProvider()._client
        print(json.dumps(backfill(session, client, symbols), indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

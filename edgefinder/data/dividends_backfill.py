"""Backfill cash-dividend history from Polygon into the dividends table.

One paginated reference call per symbol; idempotent (skip-existing per
(symbol, ex_date)). Raw bars stay dividend-UNadjusted everywhere — these rows
feed the load-time total-return transform (engine/data.adjust_for_dividends).

CLI:
    python -m edgefinder.data.dividends_backfill run [--symbols A,B]
    python -m edgefinder.data.dividends_backfill coverage
    python -m edgefinder.data.dividends_backfill add --symbol X \
        --ex-date YYYY-MM-DD --amount N.NN   # patch a gap upstream data lacks
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


def add_record(session: Session, symbol: str, ex_date: date, amount: float) -> dict:
    """Insert one dividend row, e.g. a record the upstream vendor is missing
    (TLT 2015-08 is absent from Polygon entirely). Idempotent: an existing
    (symbol, ex_date) row is reported, never overwritten — corrections to a
    wrong amount should be deliberate SQL, not a silent CLI side effect.
    """
    existing = (session.query(DividendRecord)
                .filter(DividendRecord.symbol == symbol,
                        DividendRecord.ex_date == ex_date).one_or_none())
    if existing is not None:
        return {"status": "exists", "symbol": symbol, "ex_date": str(ex_date),
                "cash_amount": existing.cash_amount}
    if amount <= 0:
        raise ValueError(f"amount must be positive, got {amount}")
    session.add(DividendRecord(symbol=symbol, ex_date=ex_date, cash_amount=amount))
    session.commit()
    return {"status": "added", "symbol": symbol, "ex_date": str(ex_date),
            "cash_amount": amount}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from edgefinder.db.engine import get_engine, get_session_factory

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "coverage", "add"])
    p.add_argument("--symbols", default=None)
    p.add_argument("--symbol", default=None, help="add: ticker symbol")
    p.add_argument("--ex-date", default=None, help="add: ex-dividend date YYYY-MM-DD")
    p.add_argument("--amount", type=float, default=None,
                   help="add: cash amount per share")
    args = p.parse_args(argv)

    session = get_session_factory(get_engine())()
    try:
        if args.command == "add":
            if not (args.symbol and args.ex_date and args.amount is not None):
                p.error("add requires --symbol, --ex-date and --amount")
            print(json.dumps(add_record(
                session, args.symbol.strip().upper(),
                date.fromisoformat(args.ex_date), args.amount), indent=2))
            return
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
        from edgefinder.data.polygon import PolygonDataProvider

        client = PolygonDataProvider()._client
        print(json.dumps(backfill(session, client, symbols), indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

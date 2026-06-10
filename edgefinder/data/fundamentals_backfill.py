"""Historical PIT fundamentals backfill from Polygon SEC filings.

Fills ``fundamentals_snapshots`` with what was publicly knowable on each
filing date, computed from the raw quarterly statements (income/balance) that
Polygon serves back to ~2009 — including delisted names (verified: SIVB has
52 filings, 2010-2022). This converts the PIT store from forward-only to
~5 years of honestly backtestable fundamental history (bounded by our stock
BARS, which start 2021-06; we backfill snapshots from 2019 so trailing-TTM
context exists at the window's start).

Honesty rules:
- The as-of date is the FILING date (when the market learned it), never the
  period end. ~25% of filings lack filing_date; those get a CONSERVATIVE
  imputed date (period_end + 45d for quarters, + 75d for fiscal years — the
  SEC deadlines), which can only make information arrive later than reality,
  never earlier.
- Only price-INDEPENDENT facts are stored (EPS, growth, leverage, returns on
  capital). Price-dependent ratios (P/E, PEG) must be computed at decision
  time from AssetView.price — storing them would bake a price into history.
- TTM metrics need 4 trailing quarters; YoY growth needs 8. Filings without
  enough trailing context simply leave those fields None (no extrapolation).

CLI:
    python -m edgefinder.data.fundamentals_backfill run [--symbols A,B] [--since 2019-01-01]
    python -m edgefinder.data.fundamentals_backfill coverage
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from edgefinder.db.models import DailyBar, FundamentalsSnapshot

logger = logging.getLogger(__name__)

SINCE_DEFAULT = date(2019, 1, 1)
_QUARTER_LAG = timedelta(days=45)   # SEC 10-Q deadline (conservative for all filers)
_ANNUAL_LAG = timedelta(days=75)    # SEC 10-K deadline
_FETCH_WORKERS = 6


def _val(obj, *path) -> float | None:
    """Walk attributes, unwrap the final DataPoint's .value; None anywhere -> None."""
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    v = getattr(obj, "value", None)
    return float(v) if v is not None else None


def _to_date(s) -> date | None:
    if s is None:
        return None
    return s if isinstance(s, date) else date.fromisoformat(str(s))


def effective_date(filing) -> date | None:
    """Filing date if present, else conservative SEC-deadline imputation."""
    fd = _to_date(getattr(filing, "filing_date", None))
    if fd is not None:
        return fd
    end = _to_date(getattr(filing, "end_date", None))
    if end is None:
        return None
    lag = _ANNUAL_LAG if getattr(filing, "fiscal_period", "") == "FY" else _QUARTER_LAG
    return end + lag


def _ttm(values: list[float | None]) -> float | None:
    """Sum of exactly 4 trailing quarterly values; None unless all 4 known."""
    if len(values) < 4 or any(v is None for v in values[-4:]):
        return None
    return sum(values[-4:])


def _growth(now: float | None, prior: float | None) -> float | None:
    if now is None or prior is None or prior == 0:
        return None
    # use |prior| so a loss->profit swing reads as positive growth
    return round((now - prior) / abs(prior), 4)


def build_snapshots(symbol: str, filings: list, since: date) -> list[tuple[date, dict]]:
    """Pure function: filings -> [(as_of, TickerFundamentals-shaped dict)].

    Quarterly filings only (fiscal_period Q1-Q4), ordered by period end; each
    snapshot uses ONLY that filing and earlier ones (no future data by
    construction).
    """
    quarters = []
    for f in filings:
        if getattr(f, "fiscal_period", "") not in ("Q1", "Q2", "Q3", "Q4"):
            continue
        end = _to_date(getattr(f, "end_date", None))
        eff = effective_date(f)
        if end is None or eff is None:
            continue
        quarters.append({
            "end": end, "as_of": eff,
            "net_income": _val(f, "financials", "income_statement", "net_income_loss"),
            "revenues": _val(f, "financials", "income_statement", "revenues"),
            "eps": _val(f, "financials", "income_statement", "basic_earnings_per_share"),
            "liabilities": _val(f, "financials", "balance_sheet", "liabilities"),
            "equity": _val(f, "financials", "balance_sheet", "equity"),
            "assets": _val(f, "financials", "balance_sheet", "assets"),
            "current_assets": _val(f, "financials", "balance_sheet", "current_assets"),
            "current_liabilities": _val(f, "financials", "balance_sheet", "current_liabilities"),
        })
    quarters.sort(key=lambda q: q["end"])
    # dedup amended filings: keep the FIRST filing per period end (what the
    # market learned first); later amendments would rewrite history
    seen_ends: set = set()
    deduped = []
    for q in quarters:
        if q["end"] in seen_ends:
            continue
        seen_ends.add(q["end"])
        deduped.append(q)
    quarters = deduped

    # a delinquent filer can file SEVERAL catch-up quarters the same day
    # (seen live: ADMP filed three 10-Qs on 2021-11-22); on that day the
    # market learns all of them, and the latest period end dominates —
    # keep one snapshot per as_of date, from the most recent period
    by_asof: dict[date, int] = {}
    for i, q in enumerate(quarters):
        prev = by_asof.get(q["as_of"])
        if prev is None or quarters[prev]["end"] < q["end"]:
            by_asof[q["as_of"]] = i

    out: list[tuple[date, dict]] = []
    for i, q in enumerate(quarters):
        if q["as_of"] < since or by_asof[q["as_of"]] != i:
            continue
        window = quarters[max(0, i - 7): i + 1]
        ni_ttm = _ttm([w["net_income"] for w in window])
        rev_ttm = _ttm([w["revenues"] for w in window])
        eps_ttm = _ttm([w["eps"] for w in window])
        ni_ttm_prior = _ttm([w["net_income"] for w in window[:-4]]) if len(window) >= 8 else None
        rev_ttm_prior = _ttm([w["revenues"] for w in window[:-4]]) if len(window) >= 8 else None

        equity, assets = q["equity"], q["assets"]
        data = {
            "symbol": symbol,
            "earnings_per_share": round(eps_ttm, 4) if eps_ttm is not None else None,
            "earnings_growth": _growth(ni_ttm, ni_ttm_prior),
            "revenue_growth": _growth(rev_ttm, rev_ttm_prior),
            "debt_to_equity": (round(q["liabilities"] / equity, 4)
                               if q["liabilities"] is not None and equity else None),
            "return_on_equity": (round(ni_ttm / equity, 4)
                                 if ni_ttm is not None and equity else None),
            "return_on_assets": (round(ni_ttm / assets, 4)
                                 if ni_ttm is not None and assets else None),
            "current_ratio": (round(q["current_assets"] / q["current_liabilities"], 4)
                              if q["current_assets"] is not None
                              and q["current_liabilities"] else None),
        }
        out.append((q["as_of"], data))
    return out


def backfill(session: Session, client, symbols: list[str],
             since: date = SINCE_DEFAULT) -> dict:
    """Fetch filings and write snapshots (skip-existing; idempotent)."""
    def fetch(symbol: str):
        try:
            return symbol, list(client.vx.list_stock_financials(
                ticker=symbol, timeframe="quarterly", limit=100,
                sort="filing_date", order="asc"))
        except Exception as e:
            logger.warning("filings fetch failed for %s: %s", symbol, e)
            return symbol, []

    written = symbols_with_data = fetched = 0
    chunk = 50
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        with ThreadPoolExecutor(max_workers=_FETCH_WORKERS) as pool:
            results = list(pool.map(fetch, batch))
        for symbol, filings in results:
            fetched += 1
            if not filings:
                continue
            snaps = build_snapshots(symbol, filings, since)
            if not snaps:
                continue
            symbols_with_data += 1
            existing = {r[0] for r in session.query(FundamentalsSnapshot.as_of)
                        .filter(FundamentalsSnapshot.symbol == symbol).all()}
            for as_of, data in snaps:
                if as_of in existing:
                    continue
                session.add(FundamentalsSnapshot(
                    symbol=symbol, as_of=as_of, data=data))
                written += 1
        session.commit()
        if (i // chunk) % 10 == 0:
            logger.info("backfill: %d/%d symbols fetched, %d with data, %d rows",
                        fetched, len(symbols), symbols_with_data, written)
    return {"symbols": len(symbols), "with_data": symbols_with_data,
            "rows_written": written}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    from edgefinder.data.polygon import PolygonDataProvider
    from edgefinder.db.engine import get_engine, get_session_factory

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run", "coverage"])
    p.add_argument("--symbols", default=None,
                   help="comma-separated subset (default: every daily_bars symbol)")
    p.add_argument("--since", default=str(SINCE_DEFAULT))
    args = p.parse_args(argv)

    session = get_session_factory(get_engine())()
    try:
        if args.command == "coverage":
            total = session.query(func.count(FundamentalsSnapshot.id)).scalar()
            syms = session.query(
                func.count(func.distinct(FundamentalsSnapshot.symbol))).scalar()
            rng = session.query(func.min(FundamentalsSnapshot.as_of),
                                func.max(FundamentalsSnapshot.as_of)).one()
            print(json.dumps({"rows": total, "symbols": syms,
                              "from": str(rng[0]), "to": str(rng[1])}, indent=2))
            return
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",")]
        else:
            symbols = sorted(r[0] for r in
                             session.query(DailyBar.symbol).distinct().all())
        client = PolygonDataProvider()._client
        result = backfill(session, client, symbols,
                          since=date.fromisoformat(args.since))
        print(json.dumps(result, indent=2))
    finally:
        session.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

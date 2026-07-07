"""Keep the market-data asset fresh — the data-ingest tool for the Routine era.

The old in-process scheduler + GitHub Actions crons that grew ``daily_bars``
were retired in the v6.0 cutover. This tool replaces them with one idempotent
command the data-refresh Routine runs daily (before, or alongside, the trading
cycle):

  1. BARS — for each trading day missing since the latest ``daily_bars`` date,
     pull the whole US market in ONE grouped-daily call, keep that day's top-N
     by dollar volume (+ benchmark ETFs + the agent's open positions so a held
     name never loses bars), and upsert raw OHLCV. Survivorship-free by keeping
     a per-day top-N rather than "names alive today".
  2. R2 — when the R2_* creds are present, MERGE the new rows into the grow-only
     R2 archive and prune the DB's old rows back to the retention window.
  3. (optional) corporate actions / news — dividends + splits keep total-return
     adjustment honest; ``--with-news`` refreshes headlines for the picks panel.

All DB writes reuse the vetted ``scripts.backfill_daily_bars.upsert_daily_bars``
(Postgres/SQLite on-conflict upsert). Idempotent: re-running a day overwrites,
never duplicates.

CLI:
  python -m agent.refresh                      # bars (+ R2 if configured)
  python -m agent.refresh --max-days 10 --top 1000
  python -m agent.refresh --with-corporate-actions --with-news
  python -m agent.refresh --dry-run
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta

from config.settings import settings

logger = logging.getLogger(__name__)


def _rest_client():
    try:
        from massive import RESTClient
    except ImportError:  # pragma: no cover — legacy SDK name
        from polygon import RESTClient
    return RESTClient(settings.polygon_api_key)


def aggs_to_rows(aggs, day: date, *, top_n: int, keep: set[str],
                 min_price: float, max_price: float) -> list[dict]:
    """Turn one day's grouped aggs into daily_bars rows: top-N by dollar volume,
    plus every symbol in ``keep`` (benchmarks + held names), price-band filtered.

    Pure (no I/O) so it is unit-testable with stub bar objects.
    """
    scored: list[tuple[float, dict]] = []
    forced: list[dict] = []
    for bar in aggs:
        sym = getattr(bar, "ticker", None)
        c = getattr(bar, "close", None)
        v = getattr(bar, "volume", None)
        o = getattr(bar, "open", None)
        if not sym or c is None or v is None or o is None:
            continue
        row = {
            "symbol": sym, "date": day,
            "open": float(o), "high": float(getattr(bar, "high", c) or c),
            "low": float(getattr(bar, "low", c) or c), "close": float(c),
            "volume": float(v), "transactions": getattr(bar, "transactions", None),
            "source": "grouped_daily",
        }
        if sym in keep:
            forced.append(row)
            continue
        # universe filter: skip non-common-stock symbol shapes + price outliers
        if any(ch in sym for ch in (".", "/", "=")):
            continue
        if c < min_price or c > max_price:
            continue
        scored.append((float(c) * float(v), row))
    scored.sort(key=lambda t: t[0], reverse=True)
    rows = [r for _, r in scored[:top_n]]
    rows.extend(forced)
    # de-dup (a kept name could also rank top-N)
    seen, out = set(), []
    for r in rows:
        if r["symbol"] in seen:
            continue
        seen.add(r["symbol"])
        out.append(r)
    return out


def _missing_trading_days(engine, max_days: int) -> list[date]:
    """Weekdays from the day after the latest daily_bars date through yesterday,
    capped at ``max_days`` (ingested oldest first). Bars are research/backtest
    inputs only — fills price off live quotes — so we simply never ingest
    today's still-forming bar.
    """
    from sqlalchemy import func, select
    from edgefinder.db.models import DailyBar

    try:
        from zoneinfo import ZoneInfo
        today = datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:  # pragma: no cover
        today = date.today()

    with engine.connect() as conn:
        latest = conn.execute(select(func.max(DailyBar.date))).scalar()
    start = (latest + timedelta(days=1)) if latest else (today - timedelta(days=max_days))
    end = today - timedelta(days=1)
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri (holidays yield empty aggs and are skipped)
            days.append(d)
        d += timedelta(days=1)
    return days[-max_days:]


def _protected_symbols(engine) -> set[str]:
    """Benchmarks + the agent's open positions — never dropped from the top-N."""
    from edgefinder.data.barstore import DB_PROTECTED_ETFS
    from agent.models import ACCOUNT, DeskPosition

    keep = {s.upper() for s in DB_PROTECTED_ETFS}
    keep |= {s.strip().upper() for s in settings.index_symbols if s.strip()}
    from sqlalchemy.orm import Session
    with Session(engine) as sess:
        for (sym,) in sess.query(DeskPosition.symbol).filter(
                DeskPosition.account == ACCOUNT).all():
            keep.add(str(sym).upper())
    return keep


def refresh(*, max_days: int = 7, top_n: int = 1000, min_price: float = 1.0,
            max_price: float = 100_000.0, with_corporate_actions: bool = False,
            with_news: bool = False, dry_run: bool = False) -> dict:
    """Run one data-refresh pass. Returns a summary dict."""
    import os

    from scripts.backfill_daily_bars import upsert_daily_bars
    from edgefinder.db.engine import get_engine

    engine = get_engine()
    client = _rest_client()
    keep = _protected_symbols(engine)
    days = _missing_trading_days(engine, max_days)
    summary: dict = {"days_targeted": [str(d) for d in days], "bars_upserted": 0,
                     "days_ingested": 0, "dry_run": dry_run}

    for d in days:
        try:
            aggs = list(client.get_grouped_daily_aggs(d.isoformat()))
        except Exception as exc:  # noqa: BLE001 — one bad day can't abort the run
            logger.warning("grouped aggs failed for %s: %s", d, exc)
            continue
        if not aggs:
            logger.info("  %s: no aggs (holiday/weekend) — skip", d)
            continue
        rows = aggs_to_rows(aggs, d, top_n=top_n, keep=keep,
                            min_price=min_price, max_price=max_price)
        if dry_run:
            logger.info("  %s: would upsert %d bars", d, len(rows))
        else:
            n = upsert_daily_bars(engine, rows)
            summary["bars_upserted"] += n
            logger.info("  %s: upserted %d bars", d, n)
        summary["days_ingested"] += 1

    if with_corporate_actions and not dry_run:
        summary["corporate_actions"] = _accumulate(["dividends", "splits"])
    if with_news and not dry_run:
        summary["news"] = _accumulate(["news"])

    # mirror to the grow-only R2 archive + prune the DB back to the window
    if not dry_run and all(os.getenv(k) for k in (
            "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET")):
        try:
            from edgefinder.data.barstore import (
                DB_RETENTION_DAYS, BarStore, db_protected_symbols)
            from edgefinder.db.engine import get_session_factory

            store = BarStore()
            sess = get_session_factory(engine)()
            try:
                summary["r2_sync"] = store.sync(sess)
                summary["db_prune"] = store.prune_db(
                    sess, keep_days=DB_RETENTION_DAYS,
                    protected=db_protected_symbols(sess) | keep)
            finally:
                sess.close()
        except Exception:
            logger.exception("R2 sync/prune failed")
            summary["r2_sync"] = "error"
    elif not dry_run:
        summary["r2_sync"] = "skipped (no R2_* env)"
    return summary


def _accumulate(kinds: list[str]) -> dict:
    """Run the kept DataAccumulator for dividends/splits/news over active names."""
    from edgefinder.data.accumulator import DataAccumulator
    from edgefinder.data.polygon import PolygonDataProvider
    from edgefinder.db.engine import get_engine, get_session_factory

    acc = DataAccumulator(PolygonDataProvider(), get_session_factory(get_engine()))
    out: dict = {}
    if "dividends" in kinds:
        out["dividends"] = acc.accumulate_dividends()
    if "splits" in kinds:
        out["splits"] = acc.accumulate_splits()
    if "news" in kinds:
        out["news"] = acc.accumulate_news()
    return out


def alpaca_bars_to_rows(bars, symbol: str) -> list[dict]:
    """Alpaca daily-bar objects → daily_bars rows (pure; unit-tested)."""
    rows = []
    for bar in bars:
        ts = getattr(bar, "timestamp", None)
        c = getattr(bar, "close", None)
        if ts is None or c is None:
            continue
        rows.append({
            "symbol": symbol, "date": ts.date() if hasattr(ts, "date") else ts,
            "open": float(getattr(bar, "open", c) or c),
            "high": float(getattr(bar, "high", c) or c),
            "low": float(getattr(bar, "low", c) or c),
            "close": float(c),
            "volume": float(getattr(bar, "volume", 0) or 0),
            "transactions": getattr(bar, "trade_count", None),
            "source": "alpaca_daily",
        })
    return rows


def refresh_alpaca(max_days: int = 30, symbols: list[str] | None = None) -> dict:
    """Top up ``daily_bars`` from Alpaca daily bars for the LIVE universe
    (streamed symbols + held positions). This replaced the Polygon full-market
    ingest when that subscription was disabled — the agent only needs fresh
    bars for the names it actually watches/holds; the deep archive is R2.

    Works on both store transports (pg + REST): per symbol, missing dates are
    deleted-then-inserted, which is an upsert for our purposes. Idempotent.
    """
    from datetime import timedelta as _td

    from agent import broker
    from agent.models import ACCOUNT
    from agent.store import get_store
    from agent.streamer import stream_symbols

    if not broker.enabled():
        return {"error": "no Alpaca keys — cannot top up bars"}
    store = get_store()
    if symbols is None:
        symbols = stream_symbols()
        for r in store.select("desk_positions", filters={"account": ACCOUNT}):
            s = str(r["symbol"]).upper()
            if s not in symbols:
                symbols.append(s)

    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    b = broker.Broker()
    yesterday = date.today() - _td(days=1)
    summary: dict = {"symbols": len(symbols), "bars_upserted": 0, "per_symbol": {}}
    for sym in symbols:
        latest_rows = store.select("daily_bars", filters={"symbol": sym},
                                   order=[("date", "desc")], limit=1)
        latest = latest_rows[0]["date"] if latest_rows else None
        if latest is not None and not isinstance(latest, date):
            latest = date.fromisoformat(str(latest)[:10])
        start = (latest + _td(days=1)) if latest else yesterday - _td(days=max_days)
        if start > yesterday:
            summary["per_symbol"][sym] = 0
            continue
        try:
            req = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Day,
                                   start=start, end=yesterday)
            res = b.data.get_stock_bars(req)
            bars = res.data.get(sym, []) if hasattr(res, "data") else []
            rows = alpaca_bars_to_rows(bars, sym)
            for r in rows:  # delete-then-insert = transport-agnostic upsert
                store.delete("daily_bars", {"symbol": sym, "date": r["date"]})
            if rows:
                store.insert("daily_bars", rows, returning=False)
            summary["per_symbol"][sym] = len(rows)
            summary["bars_upserted"] += len(rows)
        except Exception as exc:  # noqa: BLE001 — one symbol can't abort the run
            logger.warning("bar top-up failed for %s: %s", sym, exc)
            summary["per_symbol"][sym] = f"error: {exc}"
    return summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["alpaca", "polygon"], default="alpaca",
                   help="alpaca = live-universe top-up (default); polygon = legacy full-market")
    p.add_argument("--max-days", type=int, default=7)
    p.add_argument("--top", type=int, default=1000)
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--max-price", type=float, default=100_000.0)
    p.add_argument("--with-corporate-actions", action="store_true")
    p.add_argument("--with-news", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.source == "alpaca":
        out = refresh_alpaca(max_days=args.max_days)
    else:
        out = refresh(max_days=args.max_days, top_n=args.top, min_price=args.min_price,
                      max_price=args.max_price,
                      with_corporate_actions=args.with_corporate_actions,
                      with_news=args.with_news, dry_run=args.dry_run)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

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
  python -m agent.refresh                      # alpaca (default): live-universe
                                               #   bars + news + dividends/splits
  python -m agent.refresh --source polygon --max-days 10 --top 1000
  python -m agent.refresh --source polygon --with-corporate-actions --with-news
  python -m agent.refresh --source polygon --dry-run
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
            "volume": int(round(float(getattr(bar, "volume", 0) or 0))),
            "transactions": (int(round(float(tc)))
                             if (tc := getattr(bar, "trade_count", None)) is not None
                             else None),
            "source": "alpaca_daily",
        })
    return rows


def _watch_symbols(store) -> list[str]:
    """Streamed symbols + the agent's held EQUITY names (OCC options excluded
    — the data APIs reject option symbols; a held option's underlying is
    already in the streamed set)."""
    from agent.models import ACCOUNT
    from agent.occ import is_option
    from agent.streamer import stream_symbols

    symbols = stream_symbols()
    for r in store.select("desk_positions", filters={"account": ACCOUNT}):
        s = str(r["symbol"]).upper()
        if s not in symbols and not is_option(s):
            symbols.append(s)
    return symbols


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
    from agent.store import get_store

    if not broker.enabled():
        return {"error": "no Alpaca keys — cannot top up bars"}
    store = get_store()
    if symbols is None:
        symbols = _watch_symbols(store)

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


# ── news + corporate actions via Alpaca ─────────────────────
# The Polygon DataAccumulator path retired with the old scheduler (and can't
# run on the REST transport at all), which silently froze ticker_news,
# dividends and ticker_splits. These top-ups run on the default alpaca
# refresh so the hourly Routine keeps all three current with the same keys
# that already price the fills.


def _pub_utc(ts) -> str | None:
    """Alpaca datetime → the 'YYYY-MM-DDTHH:MM:SSZ' shape Polygon rows use,
    so the (symbol, title, published_utc) dedupe key stays consistent."""
    if ts is None:
        return None
    try:
        from datetime import timezone
        return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:  # noqa: BLE001
        return str(ts)[:30]


def alpaca_news_to_rows(items, watch: set[str]) -> list[dict]:
    """Alpaca news articles → ticker_news rows, one per WATCHED symbol
    (pure; unit-tested). Articles without a headline/timestamp are skipped."""
    rows, seen = [], set()
    for n in items:
        head = getattr(n, "headline", None)
        pub = _pub_utc(getattr(n, "created_at", None))
        if not head or not pub:
            continue
        for sym in (getattr(n, "symbols", None) or []):
            s = str(sym).upper()
            key = (s, head, pub)
            if s not in watch or key in seen:
                continue
            seen.add(key)
            rows.append({
                "symbol": s, "title": head,
                "author": (getattr(n, "author", None) or None),
                "published_utc": pub,
                "article_url": getattr(n, "url", None),
                "description": getattr(n, "summary", None) or None,
                "publisher_name": getattr(n, "source", None) or None,
            })
    return rows


def refresh_alpaca_news(symbols: list[str], *, days: int = 3, limit: int = 50) -> dict:
    """Fetch recent Alpaca news for the watch list and insert only rows not
    already stored (dedupe on the uq_ticker_news key). Idempotent."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz

    from agent import broker
    from agent.store import get_store

    if not broker.enabled():
        return {"error": "no Alpaca keys"}
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    creds = broker.resolve_creds()
    client = NewsClient(api_key=creds["key"], secret_key=creds["secret"])
    start = _dt.now(_tz.utc) - _td(days=days)
    res = client.get_news(NewsRequest(symbols=",".join(symbols), start=start,
                                      limit=limit))
    items = res.data.get("news", []) if hasattr(res, "data") else []
    rows = alpaca_news_to_rows(items, {s.upper() for s in symbols})

    store = get_store()
    existing = {(r["symbol"], r["title"], r["published_utc"])
                for r in store.select(
                    "ticker_news", columns="symbol,title,published_utc",
                    filters={"symbol": ("in", symbols),
                             "published_utc": ("gte", start.strftime("%Y-%m-%dT%H:%M:%SZ"))},
                    limit=10000)}
    fresh = [r for r in rows
             if (r["symbol"], r["title"], r["published_utc"]) not in existing]
    if fresh:
        store.insert("ticker_news", fresh, returning=False)
    return {"fetched": len(items), "inserted": len(fresh)}


def alpaca_corp_actions_to_rows(data: dict) -> tuple[list[dict], list[dict]]:
    """Alpaca corporate-actions payload → (dividends rows, ticker_splits rows)
    (pure; unit-tested). Split rates become an integer from/to ratio."""
    from fractions import Fraction

    divs: list[dict] = []
    for item in (data.get("cash_dividends") or []):
        sym = getattr(item, "symbol", None)
        ex = getattr(item, "ex_date", None)
        rate = getattr(item, "rate", None)
        if not sym or ex is None or rate is None or float(rate) <= 0:
            continue
        divs.append({"symbol": str(sym).upper(), "ex_date": ex.isoformat(),
                     "cash_amount": float(rate)})

    splits: list[dict] = []
    for key in ("forward_splits", "reverse_splits"):
        for item in (data.get(key) or []):
            sym = getattr(item, "symbol", None)
            ex = getattr(item, "ex_date", None)
            new = getattr(item, "new_rate", None)
            old = getattr(item, "old_rate", None)
            if not sym or ex is None or not new or not old:
                continue
            fr = Fraction(float(new) / float(old)).limit_denominator(1000)
            if fr.numerator <= 0 or fr.denominator <= 0 or fr.numerator == fr.denominator:
                continue
            splits.append({"symbol": str(sym).upper(),
                           "execution_date": ex.isoformat(),
                           "split_from": fr.denominator, "split_to": fr.numerator})
    return divs, splits


def refresh_alpaca_corporate_actions(symbols: list[str], *, days: int = 45) -> dict:
    """Top up ``dividends`` + ``ticker_splits`` from Alpaca corporate actions
    over the trailing window, inserting only unseen (symbol, date) rows —
    keeps total-return adjustment and split adjustment honest. Idempotent."""
    from datetime import timedelta as _td

    from agent import broker
    from agent.store import get_store

    if not broker.enabled():
        return {"error": "no Alpaca keys"}
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    from alpaca.data.requests import CorporateActionsRequest

    creds = broker.resolve_creds()
    client = CorporateActionsClient(api_key=creds["key"], secret_key=creds["secret"])
    start = date.today() - _td(days=days)
    res = client.get_corporate_actions(CorporateActionsRequest(
        symbols=list(symbols), start=start, end=date.today()))
    data = res.data if hasattr(res, "data") else {}
    div_rows, split_rows = alpaca_corp_actions_to_rows(data)

    store = get_store()
    fresh_div: list[dict] = []
    if div_rows:
        # Alpaca windows on process date, so returned ex_dates can precede
        # ``start`` — bound the dedupe read by the rows actually returned.
        min_ex = min(r["ex_date"] for r in div_rows)
        have_div = {(r["symbol"], str(r["ex_date"])[:10])
                    for r in store.select("dividends", columns="symbol,ex_date",
                                          filters={"symbol": ("in", symbols),
                                                   "ex_date": ("gte", min_ex)},
                                          limit=10000)}
        fresh_div = [r for r in div_rows
                     if (r["symbol"], r["ex_date"]) not in have_div]
    if fresh_div:
        store.insert("dividends", fresh_div, returning=False)

    have_split = {(r["symbol"], str(r["execution_date"])[:10])
                  for r in store.select("ticker_splits",
                                        columns="symbol,execution_date",
                                        filters={"symbol": ("in", symbols)},
                                        limit=10000)}
    fresh_split = [r for r in split_rows
                   if (r["symbol"], r["execution_date"]) not in have_split]
    if fresh_split:
        store.insert("ticker_splits", fresh_split, returning=False)
    return {"dividends_inserted": len(fresh_div),
            "splits_inserted": len(fresh_split)}


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
        if "error" not in out:
            from agent.store import get_store

            watch = _watch_symbols(get_store())
            for name, fn in (("news", refresh_alpaca_news),
                             ("corporate_actions", refresh_alpaca_corporate_actions)):
                try:
                    out[name] = fn(watch)
                except Exception as exc:  # noqa: BLE001 — feeds must not block bars
                    logger.warning("%s top-up failed: %s", name, exc)
                    out[name] = f"error: {exc}"
    else:
        out = refresh(max_days=args.max_days, top_n=args.top, min_price=args.min_price,
                      max_price=args.max_price,
                      with_corporate_actions=args.with_corporate_actions,
                      with_news=args.with_news, dry_run=args.dry_run)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

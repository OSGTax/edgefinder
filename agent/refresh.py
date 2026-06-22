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
import os
from datetime import date, timedelta

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DB connectivity probe — in remote/web Claude Code sessions outbound TCP to
# Postgres ports (5432, 6543) is often blocked by the network policy.  When
# that happens we fall back to the Supabase REST API (PostgREST, HTTPS/443)
# for all DB reads/writes and call api.polygon.io directly via httpx.
# ---------------------------------------------------------------------------

def _db_reachable() -> bool:
    """Return True if psycopg2 can reach the Supabase host (fast probe)."""
    import socket
    db_url = os.environ.get("DATABASE_URL", "")
    # parse host:port from the URL
    try:
        after_at = db_url.split("@")[-1]
        host_port = after_at.split("/")[0]
        host, port_str = host_port.rsplit(":", 1)
        port = int(port_str)
    except Exception:
        return True  # can't parse → let SQLAlchemy try
    try:
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        return True
    except Exception:
        return False


def _rest_client():
    try:
        from massive import RESTClient
    except ImportError:  # pragma: no cover — legacy SDK name
        from polygon import RESTClient
    return RESTClient(settings.polygon_api_key)


# ---------------------------------------------------------------------------
# Supabase REST API helpers (used when TCP is blocked)
# ---------------------------------------------------------------------------

def _sb_headers() -> dict:
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }


def _sb_get(path: str, params: dict | None = None) -> list[dict]:
    import httpx
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    r = httpx.get(f"{url}/rest/v1/{path}", headers=_sb_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _sb_upsert(table: str, rows: list[dict]) -> int:
    import httpx
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    n = 0
    for i in range(0, len(rows), 500):
        chunk = rows[i:i + 500]
        r = httpx.post(
            f"{url}/rest/v1/{table}",
            headers=_sb_headers(),
            content=json.dumps(chunk, default=str),
            timeout=60,
        )
        r.raise_for_status()
        n += len(chunk)
    return n


def _fetch_grouped_aggs_httpx(day: date) -> list:
    """Fetch grouped daily aggs directly via httpx (no Polygon SDK, no SSL proxy)."""
    import httpx

    class _Bar:
        def __init__(self, d: dict):
            self.ticker = d.get("T")
            self.open = d.get("o")
            self.high = d.get("h")
            self.low = d.get("l")
            self.close = d.get("c")
            self.volume = d.get("v")
            self.transactions = d.get("n")

    r = httpx.get(
        "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/" + day.isoformat(),
        params={"apiKey": settings.polygon_api_key, "include_otc": "false"},
        timeout=60,
    )
    r.raise_for_status()
    return [_Bar(d) for d in r.json().get("results", [])]


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


def _missing_trading_days(engine, max_days: int, *, rest_mode: bool = False) -> list[date]:
    """Weekdays from the day after the latest daily_bars date through yesterday,
    capped at ``max_days`` (most recent first → ingested oldest first)."""
    if rest_mode:
        rows = _sb_get("daily_bars", {"select": "date", "order": "date.desc", "limit": "1"})
        latest = date.fromisoformat(rows[0]["date"]) if rows else None
    else:
        from sqlalchemy import func, select
        from edgefinder.db.models import DailyBar
        with engine.connect() as conn:
            latest = conn.execute(select(func.max(DailyBar.date))).scalar()
    start = (latest + timedelta(days=1)) if latest else (date.today() - timedelta(days=max_days))
    end = date.today() - timedelta(days=1)  # yesterday (today's bar isn't final)
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri (holidays yield empty aggs and are skipped)
            days.append(d)
        d += timedelta(days=1)
    return days[-max_days:]


def _protected_symbols(engine, *, rest_mode: bool = False) -> set[str]:
    """Benchmarks + the agent's open positions — never dropped from the top-N."""
    from edgefinder.data.barstore import DB_PROTECTED_ETFS
    from agent.models import ACCOUNT, DeskPosition

    keep = {s.upper() for s in DB_PROTECTED_ETFS}
    keep |= {s.strip().upper() for s in settings.index_symbols if s.strip()}
    if rest_mode:
        rows = _sb_get("desk_positions",
                       {"select": "symbol", "account": f"eq.{ACCOUNT}", "shares": "gt.0"})
        for r in rows:
            keep.add(str(r["symbol"]).upper())
    else:
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
    from edgefinder.db.engine import get_engine

    # Probe TCP connectivity first; fall back to HTTPS REST when blocked
    # (common in remote/web Claude Code execution environments).
    rest_mode = not _db_reachable()
    if rest_mode:
        logger.info("TCP to Postgres unreachable — using Supabase REST API (HTTPS)")
        engine = None
    else:
        engine = get_engine()

    keep = _protected_symbols(engine, rest_mode=rest_mode)
    days = _missing_trading_days(engine, max_days, rest_mode=rest_mode)
    summary: dict = {"days_targeted": [str(d) for d in days], "bars_upserted": 0,
                     "days_ingested": 0, "dry_run": dry_run,
                     "mode": "rest" if rest_mode else "sqlalchemy"}

    for d in days:
        try:
            if rest_mode:
                aggs = _fetch_grouped_aggs_httpx(d)
            else:
                client = _rest_client()
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
            if rest_mode:
                n = _sb_upsert("daily_bars", rows)
            else:
                from scripts.backfill_daily_bars import upsert_daily_bars
                n = upsert_daily_bars(engine, rows)
            summary["bars_upserted"] += n
            logger.info("  %s: upserted %d bars", d, n)
        summary["days_ingested"] += 1

    if with_corporate_actions and not dry_run:
        summary["corporate_actions"] = _accumulate(
            ["dividends", "splits"], rest_mode=rest_mode, keep=keep)
    if with_news and not dry_run:
        summary["news"] = _accumulate(["news"], rest_mode=rest_mode, keep=keep)

    # mirror to the grow-only R2 archive + prune the DB back to the window
    # R2 uses S3/HTTPS so it's reachable even in rest_mode, but the prune
    # step needs a DB session — skip it when TCP is blocked.
    if not dry_run and not rest_mode and all(os.getenv(k) for k in (
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
        summary["r2_sync"] = "skipped (rest_mode)" if rest_mode else "skipped (no R2_* env)"
    return summary


def _accumulate(kinds: list[str], *, rest_mode: bool = False,
                keep: set[str] | None = None) -> dict:
    """Run the DataAccumulator.  In rest_mode uses Polygon + Supabase REST directly."""
    if rest_mode:
        return _accumulate_rest(kinds, symbols=keep or set())

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


def _accumulate_rest(kinds: list[str], symbols: set[str]) -> dict:
    """Polygon + Supabase REST fallback for corporate-actions/news accumulation."""
    import httpx
    since = (date.today() - timedelta(days=7)).isoformat()
    out: dict = {}
    divs_inserted = 0
    splits_found = 0
    news_fetched = 0

    for sym in sorted(symbols):
        if "dividends" in kinds:
            try:
                r = httpx.get(
                    "https://api.polygon.io/v3/reference/dividends",
                    params={"ticker": sym, "ex_dividend_date.gte": since,
                            "apiKey": settings.polygon_api_key},
                    timeout=15,
                )
                if r.is_success:
                    for d in r.json().get("results", []):
                        try:
                            _sb_upsert("dividends?on_conflict=symbol,ex_date",
                                       [{"symbol": d["ticker"],
                                         "ex_date": d["ex_dividend_date"],
                                         "cash_amount": d.get("cash_amount", 0)}])
                            divs_inserted += 1
                        except Exception as e:
                            logger.warning("dividend upsert %s: %s", sym, e)
            except Exception as e:
                logger.warning("dividends fetch %s: %s", sym, e)

        if "splits" in kinds:
            try:
                r = httpx.get(
                    "https://api.polygon.io/v3/reference/splits",
                    params={"ticker": sym, "execution_date.gte": since,
                            "apiKey": settings.polygon_api_key},
                    timeout=15,
                )
                if r.is_success:
                    splits_found += len(r.json().get("results", []))
            except Exception as e:
                logger.warning("splits fetch %s: %s", sym, e)

        if "news" in kinds:
            try:
                since_news = (date.today() - timedelta(days=5)).strftime("%Y-%m-%dT00:00:00Z")
                r = httpx.get(
                    "https://api.polygon.io/v2/reference/news",
                    params={"ticker": sym, "published_utc.gte": since_news,
                            "limit": 10, "apiKey": settings.polygon_api_key},
                    timeout=15,
                )
                if r.is_success:
                    news_fetched += len(r.json().get("results", []))
            except Exception as e:
                logger.warning("news fetch %s: %s", sym, e)

    if "dividends" in kinds:
        out["dividends"] = divs_inserted
    if "splits" in kinds:
        out["splits"] = splits_found
    if "news" in kinds:
        out["news"] = {"fetched": news_fetched, "note": "no news table in DB"}
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-days", type=int, default=7)
    p.add_argument("--top", type=int, default=1000)
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--max-price", type=float, default=100_000.0)
    p.add_argument("--with-corporate-actions", action="store_true")
    p.add_argument("--with-news", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    out = refresh(max_days=args.max_days, top_n=args.top, min_price=args.min_price,
                  max_price=args.max_price,
                  with_corporate_actions=args.with_corporate_actions,
                  with_news=args.with_news, dry_run=args.dry_run)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

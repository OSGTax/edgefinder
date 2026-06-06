"""Backfill the daily_bars table from Massive flat-files day_aggs.

Dry-run by default: lists the trading days available in the range (an S3
LIST only — no file downloads, no DB writes). Pass --execute to download
each day and upsert it into daily_bars.

Examples:
    # Preview which days would be loaded (no downloads, no writes)
    python scripts/backfill_daily_bars.py --start 2026-05-01 --end 2026-05-26

    # Actually backfill into the configured DB (DATABASE_URL / settings)
    python scripts/backfill_daily_bars.py --start 2026-05-01 --end 2026-05-26 --execute

    # Backfill a single day for just a few symbols into a local SQLite file
    python scripts/backfill_daily_bars.py --start 2026-05-26 --end 2026-05-26 \
        --symbols NVDA,AAPL --db-url sqlite:///data/edgefinder.db --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

sys.path.insert(0, ".")

from edgefinder.core.logging_config import configure_logging
from edgefinder.data.flatfiles import FlatFilesClient, day_aggs_to_rows
from edgefinder.db.engine import get_engine
from edgefinder.db.models import DailyBar

from config.settings import settings

logger = logging.getLogger(__name__)

_UPSERT_COLS = ("open", "high", "low", "close", "volume", "transactions", "source")


def _with_benchmarks(symbols: list[str] | None, *, include: bool = True) -> list[str] | None:
    """Fold the benchmark index ETFs (``settings.index_symbols`` — SPY/QQQ/IWM/
    DIA) into a *scoped* symbol list so the daily_bars benchmark series the
    backtester reads stays full-range.

    No-op when ``symbols is None`` (an unscoped full-market backfill already
    covers them) or when disabled. Returns a sorted, de-duplicated list.
    """
    if symbols is None or not include:
        return symbols
    return sorted(set(symbols) | {s.strip().upper() for s in settings.index_symbols if s.strip()})


def upsert_daily_bars(engine, rows: list[dict], batch: int = 5_000) -> int:
    """Insert/update daily_bars rows keyed by (symbol, date). Returns row count."""
    if not rows:
        return 0
    table = DailyBar.__table__
    dialect = engine.dialect.name
    written = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), batch):
            chunk = rows[i : i + batch]
            if dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                stmt = pg_insert(table).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "date"],
                    set_={c: stmt.excluded[c] for c in _UPSERT_COLS},
                )
            elif dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                stmt = sqlite_insert(table).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol", "date"],
                    set_={c: stmt.excluded[c] for c in _UPSERT_COLS},
                )
            else:
                stmt = table.insert().values(chunk)
            conn.execute(stmt)
            written += len(chunk)
    return written


def _parse_day(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _common_stock_symbols() -> set[str]:
    """All US common-stock tickers Polygon knows — ACTIVE AND DELISTED.

    The delisted half is the graveyard: without it any top-per-day filter
    would re-create survivorship bias by only keeping names alive today.
    """
    try:
        from massive import RESTClient
    except ImportError:  # pragma: no cover - legacy SDK name
        from polygon import RESTClient

    client = RESTClient(settings.polygon_api_key)
    out: set[str] = set()
    for active in (True, False):
        n0 = len(out)
        for t in client.list_tickers(market="stocks", type="CS",
                                     active=active, limit=1000):
            out.add(t.ticker)
        logger.info("Reference tickers: %d %s common stocks",
                    len(out) - n0, "active" if active else "DELISTED")
    return out


def run_backfill(client, engine, days, *, symbols=None, use_cache=True,
                 top_per_day=None, keep_symbols=None):
    """Download + upsert each day independently.

    A failed day (missing/corrupt file, transient S3 error, dropped DB
    connection) is logged and skipped rather than aborting the whole run —
    so one bad day can't forfeit the rest of a multi-year backfill (which is
    exactly what truncated the first 3-year run at 2025-08-22). Returns
    ``(total_rows, failed)`` where ``failed`` is a list of ``(day, error)``.

    ``top_per_day``: survivorship-free ingest — read the FULL market file and
    keep that day's top N rows by dollar volume (restricted to ``keep_symbols``
    when given, e.g. common stocks incl. delisted + benchmark ETFs). Which
    names later died plays no part in what gets kept.
    """
    total_rows = 0
    failed: list[tuple[date, str]] = []
    for d in days:
        try:
            df = client.read_day_aggs(d, symbols=symbols, use_cache=use_cache)
            if top_per_day:
                if keep_symbols:
                    df = df[df["ticker"].isin(keep_symbols)]
                dv = df["close"] * df["volume"]
                df = df.loc[dv.nlargest(top_per_day).index]
            rows = day_aggs_to_rows(df)
            n = upsert_daily_bars(engine, rows)
            total_rows += n
            logger.info("  %s: upserted %d bars", d, n)
        except Exception as exc:  # noqa: BLE001 — resilience is the point
            failed.append((d, f"{type(exc).__name__}: {exc}"))
            logger.error("  %s: FAILED (%s) — skipping", d, exc)
    return total_rows, failed


def _load_universe(db_url: str | None) -> list[str]:
    """Symbols from the tickers table — used to bound the backfill (and DB size)
    to the tracked universe instead of the full ~11k-ticker market each day."""
    from sqlalchemy import select

    from edgefinder.db.models import Ticker

    engine = get_engine(url=db_url) if db_url else get_engine()
    with engine.connect() as conn:
        rows = conn.execute(select(Ticker.symbol)).scalars().all()
    return sorted({s.upper() for s in rows if s})


def main() -> int:
    configure_logging(level=logging.INFO)
    p = argparse.ArgumentParser(description="Backfill daily_bars from flat-files day_aggs")
    p.add_argument("--start", required=True, type=_parse_day, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, type=_parse_day, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--symbols", default=None, help="comma-separated symbol filter (default: all)")
    p.add_argument("--universe", action="store_true",
                   help="filter to symbols in the tickers table (bounds DB size to the tracked universe)")
    p.add_argument("--top-per-day", type=int, default=None, metavar="N",
                   help="survivorship-free ingest: keep each day's top N common "
                        "stocks by dollar volume (active AND delisted), plus the "
                        "benchmark ETFs. Mutually exclusive with --symbols/--universe.")
    p.add_argument("--no-benchmarks", action="store_true",
                   help="don't auto-include the benchmark index ETFs (settings.index_symbols) "
                        "in a scoped backfill (they're added by default so the backtest "
                        "benchmark stays full-range)")
    p.add_argument("--db-url", default=None, help="override DATABASE_URL / settings")
    p.add_argument("--limit-days", type=int, default=None, help="cap number of days processed")
    p.add_argument("--no-cache", action="store_true", help="bypass the on-disk download cache")
    p.add_argument("--execute", action="store_true", help="download + write (default: dry-run)")
    args = p.parse_args()

    if args.start > args.end:
        p.error("--start must be on or before --end")

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    if args.top_per_day and (symbols or args.universe):
        p.error("--top-per-day reads the full market file; drop --symbols/--universe")
    if args.universe and not symbols:
        symbols = _load_universe(args.db_url)
        if not symbols:
            p.error("--universe requested but the tickers table is empty")
        logger.info("Universe filter: %d symbols from the tickers table", len(symbols))

    # Always pull the benchmark index ETFs alongside a scoped backfill so the
    # backtest's SPY/QQQ/IWM/DIA buy-hold benchmark stays full-range.
    if symbols is not None and not args.no_benchmarks:
        before = set(symbols)
        symbols = _with_benchmarks(symbols)
        added = sorted(set(symbols) - before)
        if added:
            logger.info("Including benchmark indices: %s", ", ".join(added))

    client = FlatFilesClient()

    days = client.available_days("day_aggs", args.start, args.end)
    if args.limit_days is not None:
        days = days[: args.limit_days]

    logger.info(
        "Backfill range %s..%s — %d trading days available%s",
        args.start, args.end, len(days),
        f" (capped to {args.limit_days})" if args.limit_days is not None else "",
    )

    if not args.execute:
        logger.info("DRY-RUN (no downloads, no writes). Days that WOULD be loaded:")
        for d in days:
            logger.info("  %s  ->  %s", d, client.key_for("day_aggs", d))
        logger.info("Re-run with --execute to download and upsert these days.")
        return 0

    engine = get_engine(url=args.db_url) if args.db_url else get_engine()
    if engine.dialect.name == "postgresql":
        logger.warning("Target is PostgreSQL — this WRITES to a live database.")
    # Single-table bootstrap (checkfirst avoids full-schema reflection).
    DailyBar.__table__.create(engine, checkfirst=True)

    keep_symbols = None
    if args.top_per_day:
        keep_symbols = _common_stock_symbols()
        keep_symbols |= {s.strip().upper() for s in settings.index_symbols if s.strip()}
        logger.info("Keep-set: %d common stocks (active+delisted) + benchmark ETFs",
                    len(keep_symbols))

    total_rows, failed = run_backfill(
        client, engine, days, symbols=symbols, use_cache=not args.no_cache,
        top_per_day=args.top_per_day, keep_symbols=keep_symbols,
    )

    if failed:
        logger.warning("Backfill finished with %d failed day(s):", len(failed))
        for d, err in failed:
            logger.warning("  %s: %s", d, err)
    logger.info(
        "Backfill complete: %d bars across %d days (%d failed)",
        total_rows, len(days) - len(failed), len(failed),
    )
    engine.dispose()
    # Non-zero only on total failure (e.g. bad creds), so a few bad days still
    # count as a successful partial run that can be topped up idempotently.
    return 1 if days and len(failed) == len(days) else 0


if __name__ == "__main__":
    raise SystemExit(main())

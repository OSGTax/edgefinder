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

logger = logging.getLogger(__name__)

_UPSERT_COLS = ("open", "high", "low", "close", "volume", "transactions", "source")


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


def main() -> int:
    configure_logging(level=logging.INFO)
    p = argparse.ArgumentParser(description="Backfill daily_bars from flat-files day_aggs")
    p.add_argument("--start", required=True, type=_parse_day, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, type=_parse_day, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--symbols", default=None, help="comma-separated symbol filter (default: all)")
    p.add_argument("--db-url", default=None, help="override DATABASE_URL / settings")
    p.add_argument("--limit-days", type=int, default=None, help="cap number of days processed")
    p.add_argument("--no-cache", action="store_true", help="bypass the on-disk download cache")
    p.add_argument("--execute", action="store_true", help="download + write (default: dry-run)")
    args = p.parse_args()

    if args.start > args.end:
        p.error("--start must be on or before --end")

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
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

    total_rows = 0
    for d in days:
        df = client.read_day_aggs(d, symbols=symbols, use_cache=not args.no_cache)
        rows = day_aggs_to_rows(df)
        n = upsert_daily_bars(engine, rows)
        total_rows += n
        logger.info("  %s: upserted %d bars", d, n)

    logger.info("Backfill complete: %d bars across %d days", total_rows, len(days))
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

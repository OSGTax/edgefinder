"""One-shot Supabase slim: shed daily_bars breadth history that lives in R2.

WHY: daily_bars is 1.26 GB of a 1.36 GB database against Supabase's 500 MB
free-tier cap (2.7x over → statement timeouts, throttling). The permanent
data asset lives in the verified R2 Parquet mirror (Phase 4); the DB only
needs the operational hot set. Owner approved 2026-06-10 ("option B").

KEEP (everything else is dropped from the DB — never from R2):
- PROTECTED symbols, full history: the deep ETF menu + settings.index_symbols
  + every symbol in promoted_strategies + every symbol with an OPEN trade.
- For all other symbols: the per-day top-1000 by dollar volume over the
  trailing --keep-days calendar days (covers live indicator warmup, arena
  seeding, and the scanner; deep/breadth backtests read R2 via
  validate --bars-from r2).

HOW (atomic, no VACUUM FULL needed):
  build daily_bars_keep (LIKE ... INCLUDING ALL) → insert keep-set in
  month chunks (each statement small enough for a throttled DB) → swap
  names in one transaction → re-own the id sequence → drop the old table
  (which instantly reclaims the space).

SAFETY GATES:
- Aborts unless EVERY non-protected symbol is manifest-current in R2
  (rows + max_date match the DB exactly) — the mirror must provably hold
  what the DB is about to shed.
- Dry-run by default; --execute to act. Prints per-step counts.

CLI:
    python scripts/slim_daily_bars.py                # dry run (counts only)
    python scripts/slim_daily_bars.py --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

sys.path.insert(0, ".")

from sqlalchemy import text

from edgefinder.core.logging_config import configure_logging
from edgefinder.db.engine import get_engine

logger = logging.getLogger(__name__)

def protected_symbols(conn) -> set[str]:
    """Symbols whose FULL history stays in the DB (shared definition)."""
    from edgefinder.db.engine import get_session_factory, get_engine
    from edgefinder.data.barstore import db_protected_symbols

    session = get_session_factory(get_engine())()
    try:
        return db_protected_symbols(session)
    finally:
        session.close()


def assert_mirror_current(conn, protected: set[str]) -> None:
    """Abort unless every non-protected symbol is byte-current in R2."""
    from edgefinder.data.barstore import BarStore

    manifest = BarStore().read_manifest()
    rows = conn.execute(text(
        "SELECT symbol, count(*), max(date) FROM daily_bars GROUP BY symbol"))
    stale = []
    for sym, n, mx in rows:
        if sym in protected:
            continue
        m = manifest.get(sym) or {}
        # fingerprint = the DB state last merged into R2 (legacy entries
        # carry it implicitly as the parquet state)
        fp = (m.get("db_rows", m.get("rows")),
              m.get("db_max", m.get("max_date")))
        if fp != (n, str(mx)):
            stale.append(sym)
    if stale:
        raise SystemExit(
            f"ABORT: {len(stale)} non-protected symbols are not current in "
            f"the R2 mirror (e.g. {stale[:10]}). Run barstore sync until "
            "clean — slimming must never orphan data.")
    logger.info("Mirror gate passed: every non-protected symbol is R2-current")


def main() -> int:
    configure_logging(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep-days", type=int, default=365,
                   help="trailing calendar days of top-1000/day to keep")
    p.add_argument("--top-per-day", type=int, default=1000)
    p.add_argument("--execute", action="store_true",
                   help="actually rebuild (default: dry-run counts)")
    args = p.parse_args()

    engine = get_engine()
    cutoff = date.today() - timedelta(days=args.keep_days)

    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '600s'"))
        prot = protected_symbols(conn)
        logger.info("Protected symbols (full history): %d — %s",
                    len(prot), ", ".join(sorted(prot)))
        assert_mirror_current(conn, prot)

        total = conn.execute(text("SELECT count(*) FROM daily_bars")).scalar()
        prot_rows = conn.execute(
            text("SELECT count(*) FROM daily_bars WHERE symbol = ANY(:p)"),
            {"p": list(prot)}).scalar()
        window_kept = conn.execute(text("""
            SELECT count(*) FROM (
              SELECT row_number() OVER (PARTITION BY date
                         ORDER BY close * volume DESC, symbol) AS rn
              FROM daily_bars
              WHERE date >= :cutoff AND NOT (symbol = ANY(:p))
            ) t WHERE rn <= :k"""),
            {"cutoff": cutoff, "p": list(prot), "k": args.top_per_day}).scalar()
        keep = prot_rows + window_kept
        logger.info("daily_bars: %s rows total", f"{total:,}")
        logger.info("  keep: %s protected + %s trailing-window top-%d "
                    "= %s rows (%.0f%%)", f"{prot_rows:,}", f"{window_kept:,}",
                    args.top_per_day, f"{keep:,}", keep / total * 100)
        logger.info("  shed: %s rows (all present in R2)", f"{total - keep:,}")

    if not args.execute:
        logger.info("DRY-RUN complete. Re-run with --execute to rebuild.")
        return 0

    with engine.begin() as conn:
        conn.execute(text("SET statement_timeout = '600s'"))
        conn.execute(text("DROP TABLE IF EXISTS daily_bars_keep"))
        conn.execute(text(
            "CREATE TABLE daily_bars_keep (LIKE daily_bars INCLUDING ALL)"))
        n = conn.execute(text(
            "INSERT INTO daily_bars_keep SELECT * FROM daily_bars "
            "WHERE symbol = ANY(:p)"), {"p": list(prot)}).rowcount
        logger.info("inserted %s protected rows", f"{n:,}")

    # window keep-set in month chunks — each statement scans ~21 trading
    # days via the date index, far under the statement timeout
    chunk_start = cutoff
    today = date.today()
    while chunk_start <= today:
        nxt = (chunk_start.replace(day=1) + timedelta(days=40)).replace(day=1)
        with engine.begin() as conn:
            conn.execute(text("SET statement_timeout = '600s'"))
            n = conn.execute(text("""
                INSERT INTO daily_bars_keep
                SELECT db2.* FROM daily_bars db2 WHERE db2.id IN (
                  SELECT id FROM (
                    SELECT id, row_number() OVER (PARTITION BY date
                               ORDER BY close * volume DESC, symbol) AS rn
                    FROM daily_bars
                    WHERE date >= :a AND date < :b AND NOT (symbol = ANY(:p))
                  ) t WHERE rn <= :k)"""),
                {"a": chunk_start, "b": min(nxt, today + timedelta(days=1)),
                 "p": list(prot), "k": args.top_per_day}).rowcount
            logger.info("  %s..%s: +%s rows", chunk_start, nxt, f"{n:,}")
        chunk_start = nxt

    with engine.begin() as conn:
        conn.execute(text("SET statement_timeout = '600s'"))
        conn.execute(text("ALTER TABLE daily_bars RENAME TO daily_bars_old"))
        conn.execute(text("ALTER TABLE daily_bars_keep RENAME TO daily_bars"))
        # the id sequence is OWNED BY the old table's column — re-own it to
        # the new table or DROP ... CASCADE below would take it down
        conn.execute(text(
            "ALTER SEQUENCE daily_bars_id_seq OWNED BY daily_bars.id"))
    with engine.begin() as conn:
        conn.execute(text("SET statement_timeout = '600s'"))
        conn.execute(text("DROP TABLE daily_bars_old"))
    with engine.connect() as conn:
        kept = conn.execute(text("SELECT count(*) FROM daily_bars")).scalar()
        size = conn.execute(text(
            "SELECT pg_size_pretty(pg_total_relation_size('daily_bars'))")).scalar()
        logger.info("REBUILD COMPLETE: daily_bars now %s rows, %s",
                    f"{kept:,}", size)
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

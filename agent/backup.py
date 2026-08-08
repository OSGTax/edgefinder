"""Nightly R2 backup of everything we cannot reclaim (REBUILD-V4).

The owner's priority is explicit: trading history may reset, but the
KNOWLEDGE (claims, wiki, decisions, outcomes, journal) and the irreplaceable
market data (news history, PIT fundamentals, dividend/split history) must
never be lost. Supabase's free tier can pause; with this backup a pause
strands nothing.

Layout on R2 (same bucket as the parquet archive):
  backups/<YYYY-MM-DD>/<table>.jsonl.gz   — knowledge tables, dated (tiny,
                                            a history of states)
  backups/latest/<table>.jsonl.gz         — market tables, overwritten (one
                                            complete copy is sufficient —
                                            each dump supersedes the last)
  backups/<...>/manifest.json             — row counts + byte sizes per run

``daily_bars`` is deliberately absent from the nightly set: the R2 parquet
archive already mirrors it, deeper than the DB holds. ``--full`` (the one-time
cutover export, runbook step 2) adds it anyway, plus every remaining table,
so the pre-migration database exists somewhere immutable in its entirety.

Reads go through ``agent.store`` (keyset-paginated by id), so this runs on
both transports — including the Routine sandbox where only HTTPS/443 works.

CLI:
  python -m agent.backup run [--full]
  python -m agent.backup size-check
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os

logger = logging.getLogger(__name__)

# Knowledge layer — dated prefix, exported every night.
KNOWLEDGE_TABLES = [
    "desk_strategy_state", "desk_journal", "desk_thinking", "desk_decisions",
    "desk_backtests", "desk_changelog", "desk_options_snap", "desk_wiki",
    "desk_wiki_history", "desk_briefs", "desk_wakes", "desk_outcomes",
    "desk_claims", "desk_claim_events", "desk_commitments", "desk_proposals",
    "desk_orders", "desk_activities", "desk_portfolio_history",
    # Era-1 archive tables exist only after the cutover rename; missing
    # tables are skipped without error, so listing them early is free.
    "era1_trades", "era1_positions", "era1_equity",
]
# Irreplaceable market data — 'latest' prefix, overwritten nightly.
MARKET_TABLES = ["dividends", "ticker_splits", "ticker_news", "fundamentals_pit"]
# --full adds the whole pre-migration database (cutover runbook step 2).
FULL_EXTRA_TABLES = [
    "desk_trades", "desk_positions", "desk_equity", "desk_watch",
    "desk_dispatches", "daily_bars", "index_daily", "fundamentals_snapshots",
]

PAGE_SIZE = 1000  # PostgREST's hard page cap — the pg lane matches it
SIZE_WARN_BYTES = 400 * 1024 * 1024   # 400MB of the 500MB free-tier cap
SIZE_ALERT_BYTES = 450 * 1024 * 1024

__all__ = ["run", "size_check", "KNOWLEDGE_TABLES", "MARKET_TABLES"]


def _r2():
    """boto3 S3 client for R2, or None (with a reason) when unconfigured."""
    needed = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT",
              "R2_BUCKET")
    if not all(os.getenv(k) for k in needed):
        return None, None, "skipped (no R2_* env)"
    import boto3

    s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                      aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                      aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])
    return s3, os.environ["R2_BUCKET"], None


def _dump_table(store, table: str) -> tuple[bytes, int] | None:
    """Whole table → gzipped JSONL bytes + row count, keyset-paginated by id.
    None when the table does not exist on this database (pre-cutover era1_*,
    post-cleanup legacy tables) — the caller records it as skipped."""
    from agent.store import is_missing_table_error

    buf = io.BytesIO()
    rows_written = 0
    last_id = None
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        while True:
            filters = {"id": ("gt", last_id)} if last_id is not None else None
            try:
                page = store.select(table, filters=filters,
                                    order=[("id", "asc")], limit=PAGE_SIZE)
            except Exception as exc:
                if is_missing_table_error(exc):
                    return None
                raise
            for row in page:
                gz.write(json.dumps(row, default=str).encode() + b"\n")
            rows_written += len(page)
            if len(page) < PAGE_SIZE:
                break
            last_id = page[-1]["id"]
    return buf.getvalue(), rows_written


def run(*, full: bool = False, store=None) -> dict:
    """Export the backup set to R2. Best-effort per table — one failed table
    never aborts the rest; the manifest names every skip and failure."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from agent.store import get_store

    s3, bucket, skip_reason = _r2()
    if s3 is None:
        return {"status": skip_reason}
    store = store or get_store()
    today = datetime.now(ZoneInfo("America/New_York")).date().isoformat()

    plan: list[tuple[str, str]] = (
        [(t, f"backups/{today}") for t in KNOWLEDGE_TABLES]
        + [(t, "backups/latest") for t in MARKET_TABLES])
    if full:
        plan += [(t, f"backups/{today}") for t in FULL_EXTRA_TABLES
                 + MARKET_TABLES]

    manifest: dict = {"date": today, "full": full,
                      "ran_at": datetime.now(timezone.utc).isoformat(),
                      "tables": {}}
    seen: set[tuple[str, str]] = set()
    for table, prefix in plan:
        if (table, prefix) in seen:
            continue
        seen.add((table, prefix))
        key = f"{prefix}/{table}.jsonl.gz"
        try:
            dumped = _dump_table(store, table)
            if dumped is None:
                manifest["tables"][key] = {"status": "absent"}
                continue
            data, count = dumped
            s3.put_object(Bucket=bucket, Key=key, Body=data,
                          ContentType="application/gzip")
            manifest["tables"][key] = {"status": "ok", "rows": count,
                                       "bytes": len(data)}
        except Exception as exc:  # noqa: BLE001 — per-table isolation is the point
            logger.warning("backup: %s failed", table, exc_info=True)
            manifest["tables"][key] = {"status": "failed",
                                       "error": f"{type(exc).__name__}: {exc}"}
    mkey = (f"backups/{today}/manifest{'-full' if full else ''}.json")
    s3.put_object(Bucket=bucket, Key=mkey,
                  Body=json.dumps(manifest, indent=2).encode(),
                  ContentType="application/json")
    ok = sum(1 for v in manifest["tables"].values() if v["status"] == "ok")
    failed = [k for k, v in manifest["tables"].items()
              if v["status"] == "failed"]
    return {"status": "ok" if not failed else "partial", "exported": ok,
            "failed": failed, "manifest": mkey,
            "rows": sum(v.get("rows", 0) for v in manifest["tables"].values())}


def size_check(*, store=None) -> dict:
    """Database size vs the Supabase free-tier 500MB cap.

    pg lane: pg_database_size(current_database()) directly. rest lane: the
    ``edgefinder_db_size()`` RPC (created by the DDL in agent/models.py via
    render_start/setup_db — SECURITY DEFINER, returns bigint). A missing RPC
    reports unknown rather than failing the nightly."""
    from agent.store import get_store

    store = store or get_store()
    size = None
    how = None
    if getattr(store, "transport", "pg") == "pg":
        try:
            from sqlalchemy import text

            from edgefinder.db.engine import get_engine

            eng = get_engine()
            if eng.dialect.name == "postgresql":
                with eng.connect() as conn:
                    size = int(conn.execute(
                        text("SELECT pg_database_size(current_database())")
                    ).scalar_one())
                how = "pg_database_size"
            else:  # SQLite dev/test database — stat the file
                path = (str(eng.url.database) or "")
                if path and path != ":memory:" and os.path.exists(path):
                    size = os.path.getsize(path)
                how = "file size (sqlite)"
        except Exception as exc:  # noqa: BLE001
            return {"status": "unknown", "error": f"{type(exc).__name__}: {exc}"}
    else:
        try:
            from agent.rest import Rest

            size = int(Rest().rpc("edgefinder_db_size") or 0) or None
            how = "rpc edgefinder_db_size"
        except Exception as exc:  # noqa: BLE001
            return {"status": "unknown", "how": "rpc edgefinder_db_size",
                    "error": f"{type(exc).__name__}: {exc}",
                    "note": "create the RPC via scripts/setup_db.py (or the "
                            "DDL in agent/models.py) on the pg lane"}
    if size is None:
        return {"status": "unknown", "how": how}
    status = ("alert" if size >= SIZE_ALERT_BYTES
              else "warn" if size >= SIZE_WARN_BYTES else "ok")
    return {"status": status, "bytes": size,
            "mb": round(size / 1024 / 1024, 1),
            "cap_mb": 500, "how": how,
            "note": (None if status == "ok" else
                     "approaching the Supabase free-tier cap — prune "
                     "desk_thinking history or trim the daily_bars hot set")}


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="export the backup set to R2")
    r.add_argument("--full", action="store_true",
                   help="the one-time cutover export: every table, incl. "
                        "daily_bars and the pre-migration ledger")
    sub.add_parser("size-check", help="database size vs the 500MB free cap")
    args = p.parse_args(argv)

    if args.cmd == "run":
        out = run(full=args.full)
    else:
        out = size_check()
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

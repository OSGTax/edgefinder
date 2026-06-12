"""Backfill the R2 minute-bar store from Polygon REST aggregates.

Resumable, per-symbol, month-aligned. For each menu symbol it fetches
1-minute aggregates in chunks of WHOLE calendar months sized under the
~50k-bars-per-call REST cap, filters to regular trading hours, and merges
each month into ``minute/{SYMBOL}/{YYYY-MM}.parquet`` via the grow-only
MinuteStore. Months the manifest marks ``complete`` (their fetch window
covered the whole calendar month) are skipped on rerun, so a run that dies
continues where it stopped instead of starting over.

Chunk math (the 50k cap): Polygon returns EXTENDED-hours minutes (04:00–
20:00 ET, worst case 960 bars/day) even though we only keep RTH, and a
month has at most 23 trading days — so 2 months/call is the largest
month-aligned chunk that can never truncate (2 × 23 × 960 = 44,160
< 50,000). A guard still treats a maxed-out response as a loud failure.

Failure discipline (the lost-record lesson): a symbol that errors is
logged and the run CONTINUES to the next symbol, but the process exits
non-zero if any symbol ended incomplete — a half-finished backfill must
never look green.

Examples:
    # Plan only (no Polygon calls, no writes; works without R2 secrets)
    python scripts/backfill_minute_bars.py --menu intraday/menu.json --dry-run

    # Full pilot backfill (resumable — rerun continues where it died)
    python scripts/backfill_minute_bars.py --menu intraday/menu.json \
        --start 2021-06-01

    # Nightly append (merge-only, self-healing trailing window)
    python scripts/backfill_minute_bars.py --menu intraday/menu.json \
        --start $(date -u -d '5 days ago' +%F) --end $(date -u -d 'yesterday' +%F)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, ".")

import pandas as pd

from edgefinder.data.minutestore import (
    MinuteStore,
    filter_rth,
    month_bounds,
    month_of,
    months_between,
)

logger = logging.getLogger(__name__)

MAX_BARS_PER_CALL = 50_000   # Polygon REST aggs response cap
SESSION_MINUTES = 960        # worst case: extended session 04:00–20:00 ET
TRADING_DAYS_PER_MONTH = 23  # worst case
MONTHS_PER_CALL = 2          # largest month-aligned chunk under the cap


def est_worst_case_bars(months: int) -> int:
    """Upper bound on bars one call can return for ``months`` whole months."""
    return months * TRADING_DAYS_PER_MONTH * SESSION_MINUTES


assert est_worst_case_bars(MONTHS_PER_CALL) <= MAX_BARS_PER_CALL


def month_chunks(months: list[str],
                 per_call: int = MONTHS_PER_CALL) -> list[list[str]]:
    """Split an ordered month list into month-aligned fetch chunks."""
    return [months[i:i + per_call] for i in range(0, len(months), per_call)]


def plan_chunks(months_needed: list[str], manifest_months: dict,
                per_call: int = MONTHS_PER_CALL) -> list[list[str]]:
    """Chunks still to fetch: skip a chunk only when EVERY month in it is
    manifest-``complete`` (set only by a fetch that covered the whole
    calendar month — a partial nightly top-up never marks complete, so it
    can never hide a hole from a resumed backfill)."""
    done = {m for m, e in manifest_months.items() if e.get("complete")}
    return [c for c in month_chunks(months_needed, per_call)
            if not set(c) <= done]


def fetch_minute_df(client, symbol: str, start: date,
                    end: date) -> pd.DataFrame:
    """One REST aggs call -> DataFrame[ts(int64 s), o, h, l, c, v].

    Raises RuntimeError if the response hits the cap — a truncated chunk
    silently dropped would be a permanent hole, so it must fail loudly.
    """
    aggs = list(client.get_aggs(ticker=symbol, multiplier=1,
                                timespan="minute", from_=start.isoformat(),
                                to=end.isoformat(), limit=MAX_BARS_PER_CALL))
    if len(aggs) >= MAX_BARS_PER_CALL:
        raise RuntimeError(
            f"{symbol} {start}..{end}: response hit the {MAX_BARS_PER_CALL}"
            "-bar cap — likely truncated; shrink the chunk")
    if not aggs:
        return pd.DataFrame(columns=["ts", "open", "high", "low",
                                     "close", "volume"])
    return pd.DataFrame({
        "ts": [int(a.timestamp // 1000) for a in aggs],   # ms -> s, UTC
        "open": [float(a.open) for a in aggs],
        "high": [float(a.high) for a in aggs],
        "low": [float(a.low) for a in aggs],
        "close": [float(a.close) for a in aggs],
        "volume": [float(a.volume or 0) for a in aggs],
    })


def backfill_symbol(client, store: MinuteStore, symbol: str,
                    start: date, end: date, manifest: dict) -> dict:
    """Fetch + merge every non-complete chunk for one symbol."""
    months_needed = months_between(start, end)
    chunks = plan_chunks(months_needed, store.months(symbol, manifest))
    rows = months_synced = 0
    for chunk in chunks:
        f_start = max(month_bounds(chunk[0])[0], start)
        f_end = min(month_bounds(chunk[-1])[1], end)
        rth = filter_rth(fetch_minute_df(client, symbol, f_start, f_end))
        by_month = (dict(iter(rth.groupby(month_of(rth["ts"])))) if len(rth)
                    else {})
        for month in chunk:
            m_first, m_last = month_bounds(month)
            covered_full = f_start <= m_first and f_end >= m_last
            store.sync_symbol_month(
                by_month.get(month, rth.iloc[0:0]), symbol, month,
                complete=covered_full, manifest=manifest)
            months_synced += 1
        rows += int(len(rth))
        store.write_manifest(manifest)   # checkpoint per chunk (resumability)
    skipped = len(months_needed) - months_synced
    return {"rows": rows, "months": months_synced, "skipped": skipped}


def run(client, store: MinuteStore | None, symbols: list[str],
        start: date, end: date, dry_run: bool = False) -> dict:
    """Backfill every symbol; per-symbol failures are recorded, not fatal."""
    manifest = store.read_manifest() if store is not None else {}
    failed: list[str] = []
    total_rows = 0
    for i, symbol in enumerate(symbols, 1):
        symbol = symbol.strip().upper()
        months_needed = months_between(start, end)
        existing = (store.months(symbol, manifest) if store is not None
                    else {})
        chunks = plan_chunks(months_needed, existing)
        if dry_run:
            print(f"[{i}/{len(symbols)}] {symbol}: {len(months_needed)} "
                  f"months in range, {len(chunks)} chunk(s) to fetch "
                  f"({len(months_needed) - sum(len(c) for c in chunks)} "
                  "already complete)")
            continue
        try:
            out = backfill_symbol(client, store, symbol, start, end, manifest)
            total_rows += out["rows"]
            print(f"[{i}/{len(symbols)}] {symbol}: {out['rows']} RTH rows "
                  f"merged across {out['months']} month(s) "
                  f"({out['skipped']} already complete)")
        except Exception as e:                              # noqa: BLE001
            failed.append(symbol)
            logger.exception("backfill failed for %s", symbol)
            print(f"[{i}/{len(symbols)}] {symbol}: FAILED — {e}")
    if store is not None and not dry_run:
        store.write_manifest(manifest)
    return {"symbols": len(symbols), "rows": total_rows, "failed": failed,
            "dry_run": dry_run}


def _make_client():
    """Build the Polygon REST client (a seam tests monkeypatch)."""
    try:
        from massive import RESTClient
    except ImportError:                                     # pragma: no cover
        from polygon import RESTClient
    from config.settings import settings

    if not settings.polygon_api_key:
        raise SystemExit("EDGEFINDER_POLYGON_API_KEY is not set")
    return RESTClient(api_key=settings.polygon_api_key)


def _load_menu_symbols(path: str) -> list[str]:
    with open(path) as f:
        menu = json.load(f)
    return [s.strip().upper() for s in menu.get("symbols", []) if s.strip()]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--symbols", help="comma-separated symbol list")
    g.add_argument("--menu", help="path to intraday/menu.json")
    p.add_argument("--start", default="2021-06-01")
    p.add_argument("--end", default=None,
                   help="default: yesterday (UTC)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the per-symbol plan; no Polygon calls, "
                        "no writes")
    args = p.parse_args(argv)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")
                   if s.strip()]
    else:
        symbols = _load_menu_symbols(args.menu)
        if not symbols:
            print(f"{args.menu} has no symbols yet — run the menu resolver "
                  "first (MENU flag mode). Nothing to do; exiting green.")
            return

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = (datetime.strptime(args.end, "%Y-%m-%d").date()
           if args.end else date.today() - timedelta(days=1))
    if end < start:
        raise SystemExit(f"--end {end} precedes --start {start}")

    store: MinuteStore | None
    try:
        store = MinuteStore()
    except RuntimeError as e:
        if not args.dry_run:
            raise
        print(f"note: {e} — dry-run plan assumes an EMPTY store")
        store = None

    client = None if args.dry_run else _make_client()

    out = run(client, store, symbols, start, end, dry_run=args.dry_run)
    print(json.dumps(out, indent=2))
    if out["failed"]:
        print(f"INCOMPLETE: {len(out['failed'])} symbol(s) failed: "
              f"{', '.join(out['failed'])} — rerun to resume", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

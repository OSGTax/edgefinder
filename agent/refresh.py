"""Keep the market-data asset fresh — the data-ingest tool for the Routine era.

Alpaca is the sole data source (Polygon was retired). Two ingest modes, both
idempotent and safe to re-run:

  - ``alpaca`` (default, cheap/hourly) — top up ``daily_bars`` for the live
    universe (streamed seeds + held positions + last watchlist), snapshot
    option IV, and refresh Alpaca news headlines for those names.
  - ``alpaca-market`` (nightly) — enumerate Alpaca's whole tradable catalog
    (~13k names), rank by dollar volume, and keep fresh daily-bar history AND
    corporate actions (splits/dividends) for the top-N (+ benchmarks/held/
    watchlist) so discovery + indicators see the whole market, split-honest.

When the R2_* creds are present, fresh DB bars are merge-synced into the
grow-only R2 archive.

CLI:
  python -m agent.refresh                              # cheap live-universe top-up
  python -m agent.refresh --source alpaca-market --top 1000
  python -m agent.refresh --source alpaca-market --optionable --dry-run
"""

from __future__ import annotations

import json
import logging
import socket
from datetime import date, datetime, timedelta, timezone

from config.settings import settings

logger = logging.getLogger(__name__)

# Bound every outbound Alpaca call. The alpaca-py SDK builds its HTTP clients
# with no socket timeout, so a hung TLS handshake through the agent proxy blocks
# on the OS default (minutes) and takes the whole refresh down — the
# `_ssl.c:999: The handshake operation timed out` seen on the Routine lane. A
# short default socket timeout makes a stalled connection fail in seconds, where
# the per-symbol / per-pass `except` guards already catch it and the run
# continues instead of hanging.
NET_TIMEOUT_S = 20.0
# R2 parquet PUTs move multi-MB payloads over 443; a 20s socket timeout that
# works for Alpaca REST is aggressive for a healthy-but-slow parquet write.
# Widen it for the R2 sync path only, so a big archive push isn't spuriously
# killed by the same knob that keeps a hung TLS handshake bounded.
R2_NET_TIMEOUT_S = 90.0


def _bound_network(timeout: float = NET_TIMEOUT_S) -> None:
    """Set a process-wide socket timeout so no single Alpaca call can hang the
    refresh. Safe here: refresh is a CLI / Routine data tool, never imported
    into the dashboard's request path."""
    socket.setdefaulttimeout(timeout)


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
            # Alpaca returns trade_count as a float; the column is INTEGER —
            # the REST lane rejects an uncoerced float (caught live by the
            # agent's first routine run)
            "transactions": (int(tc) if (tc := getattr(bar, "trade_count", None))
                             is not None else None),
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
    from agent import broker
    from agent.models import ACCOUNT
    from agent.store import get_store
    from agent.streamer import stream_symbols

    if not broker.enabled():
        return {"error": "no Alpaca keys — cannot top up bars"}
    _bound_network()
    store = get_store()
    if symbols is None:
        from agent import occ

        symbols = stream_symbols()
        for r in store.select("desk_positions", filters={"account": ACCOUNT}):
            s = str(r["symbol"]).upper()
            if not occ.is_option(s) and s not in symbols:
                symbols.append(s)
        # watchlist candidates need fresh bars too — the agent researches them
        # before they're held/streamed
        try:
            dec = store.select("desk_decisions", filters={"account": ACCOUNT},
                               order=[("ts", "desc")], limit=1)
            for w in (dec[0].get("watchlist") or []) if dec else []:
                s = str(w.get("symbol") if isinstance(w, dict) else w).upper()
                if s and not occ.is_option(s) and s not in symbols:
                    symbols.append(s)
        except Exception:  # noqa: BLE001 — watchlist enrichments are best-effort
            pass

    b = broker.Broker()
    # Batched bar top-up: ONE multi-symbol Data-API call for the whole live
    # universe instead of one call per name (~15 round-trips → 1). Fewer
    # round-trips = far less exposure to a flaky handshake, and each per-symbol
    # write keeps its idempotent range-delete-then-insert + transient-error
    # retry via `_ingest_history_batched` (the same helper the nightly
    # full-market ingest uses). A short trailing window re-covers the 1-day gap
    # plus a weekend/holiday buffer; re-inserting a few identical days is
    # harmless (the delete-then-insert makes it an upsert).
    bar_stats = _ingest_history_batched(store, b, symbols,
                                        max_days=max(max_days, 5))
    summary: dict = {"symbols": len(symbols),
                     "bars_upserted": bar_stats.get("bars_upserted", 0),
                     "bar_batches": bar_stats.get("batches", 0),
                     "bar_errors": bar_stats.get("errors", 0)}

    # options IV data bank: one snapshot per underlying per day (first refresh
    # of the day wins; reruns are no-ops). Failures never abort the bar work.
    try:
        from agent import occ, options_data

        today = date.today()
        iv_names = [s for s in symbols if not occ.is_option(s)]
        # The IV data bank is one snapshot per underlying per DAY
        # (`persist_snapshot` dedupes at write time). Re-fetching a full option
        # chain every hour just to discover it's a no-op is the refresh's single
        # biggest cost (~2s/name) — so skip the FETCH outright for any name
        # already snapshotted today. Hourly runs after the first pay nothing.
        have_today = {str(r["symbol"]).upper() for r in store.select(
            "desk_options_snap", columns="symbol",
            filters={"symbol": ("in", sorted(iv_names)), "snap_date": today},
            limit=100000)} if iv_names else set()
        written = skipped = 0
        for sym in iv_names:
            if sym.upper() in have_today:
                skipped += 1
                continue
            # dte_max=30 is enough for the IV data bank — long-dated OPRA is
            # slower to fetch and rarely trades on the hourly agent's radar
            s = options_data.get_summary(sym, dte_max=30)
            if options_data.persist_snapshot(store, s, snap_date=today):
                written += 1
        summary["iv_snapshots"] = {"written": written, "skipped_have_today": skipped}
    except Exception as exc:  # noqa: BLE001
        logger.warning("IV snapshot pass failed: %s", exc)
        summary["iv_snapshots"] = f"error: {exc}"

    # news: Alpaca headlines for the equities we watch/hold (the Polygon news
    # source is gone). Best-effort; never aborts the bar work.
    try:
        from agent import occ

        equities = [s for s in symbols if not occ.is_option(s)]
        summary["news"] = _news_alpaca(store, equities)
    except Exception as exc:  # noqa: BLE001
        logger.warning("news pass failed: %s", exc)
        summary["news"] = f"error: {exc}"

    # corporate actions: dividends + splits into the existing tables (keeps
    # total-return backtests honest + feeds the catalysts panel). Best-effort.
    try:
        from agent import occ

        equities = [s for s in symbols if not occ.is_option(s)]
        summary["corp_actions"] = _corp_actions_alpaca(store, equities)
    except Exception as exc:  # noqa: BLE001
        logger.warning("corp actions pass failed: %s", exc)
        summary["corp_actions"] = f"error: {exc}"

    # R2 archive merge-sync: grow the 21-year parquet archive with the fresh
    # DB bars whenever this host can (needs direct Postgres + R2 creds — true
    # on Render/sandbox, skipped on the Routine's REST-only lane). The archive
    # is grow-only; the DB splice covers reads either way, so this is durable
    # archival, not a correctness dependency.
    summary["r2_sync"] = _r2_merge_sync(symbols)
    return summary


def _news_alpaca(store, symbols: list[str], *, limit: int = 15) -> dict:
    """Fetch Alpaca headlines PER symbol and upsert them into ticker_news.

    Fetched per symbol on purpose: a single batched call returns the newest
    articles ACROSS the requested names, so high-news-volume mega-caps starve
    quieter names of every slot (LLY/TSM got zero in a mixed batch). One call
    per symbol guarantees each name its own headlines. Deduped per symbol by
    (title, published_utc); transport-agnostic; best-effort per symbol."""
    from agent import broker

    if not broker.enabled() or not symbols:
        return {"fetched": 0, "inserted": 0}
    b = broker.Broker()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    fetched = inserted = 0
    for sym in {s.upper() for s in symbols}:
        try:
            arts = b.news([sym], limit=limit)
        except Exception as exc:  # noqa: BLE001 — one symbol can't abort the pass
            logger.debug("news fetch failed for %s: %s", sym, exc)
            continue
        fetched += len(arts)
        existing = store.select("ticker_news", columns="title,published_utc",
                                filters={"symbol": sym}, limit=500)
        seen = {(r.get("title"), str(r.get("published_utc"))) for r in existing}
        rows = []
        for a in arts:
            title, pub = a.get("title"), a.get("published_utc")
            if not title or sym not in (a.get("symbols") or []):
                continue
            if (title, str(pub)) in seen:
                continue
            seen.add((title, str(pub)))
            rows.append({"symbol": sym, "title": title, "author": a.get("author"),
                         "published_utc": pub, "article_url": a.get("url"),
                         "description": a.get("description"),
                         "publisher_name": a.get("publisher"), "created_at": now})
        if rows:
            store.insert("ticker_news", rows, returning=False)
            inserted += len(rows)
    return {"fetched": fetched, "inserted": inserted}


# Alpaca caps one corporate-announcements query at a 90-day window (see
# broker.corporate_announcements), so longer lookbacks are walked in chunks.
CA_WINDOW_DAYS = 90
# Below this many symbols the CA dedup pre-load is additionally bounded by
# symbol IN(): the hourly ~15-name pass must not scan the whole market's
# recent CA rows every hour, while the ~1000-name nightly keeps the pure
# date-window scan (an IN() that size gains nothing and bloats the query).
DEDUP_SYMBOL_IN_MAX = 50


def _insert_skip_dupes(store, table: str, rows: list[dict]) -> tuple[int, int]:
    """Insert ``rows`` as one fast batch; if the batch trips a unique
    constraint, fall back to per-row inserts that skip duplicate-key rows.

    Alpaca tiles corporate-actions windows on DECLARATION date, so a
    correction row can carry an ex-date OLDER than the dedup pre-load
    window — it escapes the dedup set, violates the table's unique
    constraint, and on the pure batch path that ONE row rolled back the
    WHOLE night's insert, every night. Returns ``(inserted, dupes_skipped)``;
    non-duplicate errors still raise."""
    from agent.store import is_duplicate_key_error

    if not rows:
        return 0, 0
    try:
        store.insert(table, rows, returning=False)
        return len(rows), 0
    except Exception as exc:  # noqa: BLE001 — inspect; re-raise if not a dupe
        if not is_duplicate_key_error(exc):
            raise
    inserted = skipped = 0
    for row in rows:
        try:
            store.insert(table, row, returning=False)
            inserted += 1
        except Exception as exc:  # noqa: BLE001
            if not is_duplicate_key_error(exc):
                raise
            skipped += 1
    return inserted, skipped


def _corp_actions_alpaca(store, symbols: list[str], *, back_days: int = 45,
                         fwd_days: int = 45) -> dict:
    """Ingest cash dividends + stock splits for ``symbols`` from Alpaca into the
    existing ``dividends`` / ``ticker_splits`` tables — the Polygon corporate-
    actions replacement. Deduped by (symbol, date) — pre-loaded for the fetch
    window, with a duplicate-skipping insert catching out-of-window
    corrections — so re-runs and overlapping windows stay idempotent.
    Keeping the ``dividends`` table fed is what keeps
    total-return backtests honest; the same rows drive the desk's upcoming-
    catalysts panel. Best-effort; ``back_days`` beyond Alpaca's 90-day CA cap
    (the nightly passes its bar window, ~400d) is walked in ≤90-day chunks —
    one failed window skips that window, never the pass."""
    from datetime import date as _date

    from agent import broker, occ

    wanted = {s.upper() for s in symbols if not occ.is_option(s)}
    if not broker.enabled() or not wanted:
        return {"dividends": 0, "splits": 0}
    today = _date.today()
    since = today - timedelta(days=back_days)
    until = today + timedelta(days=fwd_days)
    b = broker.Broker()
    cas: list[dict] = []
    windows = window_errors = 0
    lo = since
    while lo <= until:
        hi = min(lo + timedelta(days=CA_WINDOW_DAYS), until)
        windows += 1
        try:
            cas.extend(b.corporate_announcements(since=lo, until=hi))
        except Exception as exc:  # noqa: BLE001 — one window can't abort the pass
            logger.warning("corp actions window %s..%s failed: %s", lo, hi, exc)
            window_errors += 1
        lo = hi + timedelta(days=1)
    if window_errors and not cas:
        return {"error": f"all {windows} corporate-actions windows failed"}
    # Dedup against what's already stored, bounded to the fetch window (every
    # insertable row's ex-date lies inside it) — no giant IN() of the ~1000-name
    # nightly universe, and no unbounded whole-table scan. Small (hourly)
    # symbol sets ARE additionally bounded by symbol IN(), so the hourly pass
    # never scans the whole market's recent rows. Rows whose ex-date falls
    # OUTSIDE the window can still slip past this set (Alpaca tiles on
    # declaration date) — the duplicate-skipping insert below absorbs those.
    dedup_filters: dict = {}
    if len(wanted) <= DEDUP_SYMBOL_IN_MAX:
        dedup_filters["symbol"] = ("in", sorted(wanted))
    ex_div = {(r["symbol"], str(r["ex_date"])[:10]) for r in store.select(
        "dividends", columns="symbol,ex_date",
        filters={**dedup_filters, "ex_date": ("gte", since)}, limit=100000)}
    ex_split = {(r["symbol"], str(r["execution_date"])[:10]) for r in store.select(
        "ticker_splits", columns="symbol,execution_date",
        filters={**dedup_filters, "execution_date": ("gte", since.isoformat())},
        limit=100000)}
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    div_rows, split_rows, seen = [], [], set()
    for a in cas:
        sym, ex = a["symbol"], (a["ex_date"] or "")[:10]
        if sym not in wanted or not ex or (a["ca_type"], sym, ex) in seen:
            continue
        seen.add((a["ca_type"], sym, ex))
        try:  # Date columns want a real date object (works on pg AND REST)
            ex_d = _date.fromisoformat(ex)
        except ValueError:
            continue
        if a["ca_type"] == "dividend" and a["cash"] is not None and (sym, ex) not in ex_div:
            div_rows.append({"symbol": sym, "ex_date": ex_d, "cash_amount": a["cash"]})
        elif (a["ca_type"] == "split" and a["old_rate"] and a["new_rate"]
              and (sym, ex) not in ex_split):
            split_rows.append({"symbol": sym, "execution_date": ex_d,
                               "split_from": a["old_rate"], "split_to": a["new_rate"],
                               "created_at": now})
    div_ins, div_dupes = _insert_skip_dupes(store, "dividends", div_rows)
    split_ins, split_dupes = _insert_skip_dupes(store, "ticker_splits", split_rows)
    return {"dividends": div_ins, "splits": split_ins,
            "dupes_skipped": div_dupes + split_dupes,
            "windows": windows, "window_errors": window_errors}


def select_universe_symbols(day_rows, *, top_n: int, keep: set[str],
                            min_price: float, max_price: float) -> list[str]:
    """Rank ``{symbol, close, volume}`` rows by dollar volume and return the
    top-N symbols plus every name in ``keep`` (benchmarks + held + watchlist),
    de-duped and price-band/shape filtered. Pure — the ranking heart of the
    Alpaca full-market ingest, so it is unit-tested with plain dicts.

    This is the Polygon ``aggs_to_rows`` logic reframed for Alpaca: Polygon had
    one grouped-daily call for the whole market; Alpaca has no grouped endpoint,
    so we rank a batch of per-symbol daily bars instead — same dollar-volume
    top-N, same forced keep-set, same survivorship-free discipline.
    """
    kept = {s.upper() for s in keep}
    scored: list[tuple[float, str]] = []
    forced, seen_forced = [], set()
    for r in day_rows:
        sym = (r.get("symbol") or "").upper()
        c, v = r.get("close"), r.get("volume")
        if not sym or c is None or v is None:
            continue
        if sym in kept:
            if sym not in seen_forced:
                forced.append(sym)
                seen_forced.add(sym)
            continue
        if any(ch in sym for ch in (".", "/", "=")):
            continue
        if float(c) < min_price or float(c) > max_price:
            continue
        scored.append((float(c) * float(v), sym))
    scored.sort(key=lambda t: t[0], reverse=True)
    out, seen = [], set()
    for _, sym in scored[:top_n]:
        if sym not in seen:
            out.append(sym)
            seen.add(sym)
    for sym in forced:  # keep-set never falls off the edge of top-N
        if sym not in seen:
            out.append(sym)
            seen.add(sym)
    return out


def _keep_symbols_rest() -> set[str]:
    """Benchmarks + streamed seeds + held positions + last watchlist — the names
    that must never drop out of the universe, resolved transport-agnostically
    (works on the REST sandbox AND Render/pg)."""
    from agent import occ
    from agent.models import ACCOUNT
    from agent.store import get_store
    from agent.streamer import stream_symbols
    from edgefinder.data.barstore import DB_PROTECTED_ETFS

    keep = {s.upper() for s in DB_PROTECTED_ETFS}
    keep |= {s.strip().upper() for s in settings.index_symbols if s.strip()}
    keep |= {s.upper() for s in stream_symbols()}
    store = get_store()
    for r in store.select("desk_positions", filters={"account": ACCOUNT}):
        s = str(r["symbol"]).upper()
        if not occ.is_option(s):
            keep.add(s)
    try:
        dec = store.select("desk_decisions", filters={"account": ACCOUNT},
                           order=[("ts", "desc")], limit=1)
        for w in (dec[0].get("watchlist") or []) if dec else []:
            s = str(w.get("symbol") if isinstance(w, dict) else w).upper()
            if s and not occ.is_option(s):
                keep.add(s)
    except Exception:  # noqa: BLE001 — watchlist enrichment is best-effort
        pass
    return keep


def _alpaca_latest_daily(b, symbols: list[str], *, lookback_days: int = 7,
                         chunk: int = 200) -> list[dict]:
    """Latest daily ``{symbol, close, volume}`` for each symbol, via batched
    multi-symbol bars requests (network). Alpaca has no grouped-daily endpoint,
    so ranking the whole catalog means asking for a short trailing window in
    chunks and taking each name's most recent bar — robust to holidays."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start = date.today() - timedelta(days=lookback_days)
    out: list[dict] = []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            # `end=today` (midnight UTC) is INTENTIONAL here, unlike the
            # history ingest: ranking wants each name's last COMPLETED
            # session — the nightly runs post-midnight UTC so that's the
            # just-closed day, and an intraday manual run must not rank the
            # market on partial morning volume.
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                   start=start, end=date.today())
            res = b.data.get_stock_bars(req)
            data = res.data if hasattr(res, "data") else {}
        except Exception as exc:  # noqa: BLE001 — one bad chunk can't abort the rank
            logger.warning("rank batch @%d failed: %s", i, exc)
            continue
        for sym, bars in (data or {}).items():
            if not bars:
                continue
            last = bars[-1]
            c = getattr(last, "close", None)
            v = getattr(last, "volume", None)
            if c is None or v is None:
                continue
            out.append({"symbol": sym, "close": float(c), "volume": float(v)})
    return out


def _ingest_history_batched(store, b, symbols: list[str], *, max_days: int,
                            chunk: int = 100) -> dict:
    """Backfill ``daily_bars`` history for ``symbols`` using batched multi-symbol
    bars requests — one Data-API call per ~100 names instead of one per name.
    Per-symbol range-delete-then-insert keeps it an idempotent upsert on both
    transports."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start = date.today() - timedelta(days=max_days)
    start_iso = start.isoformat()
    summary = {"symbols": len(symbols), "bars_upserted": 0, "batches": 0,
               "errors": 0}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            # No `end`: the SDK turns a bare date into midnight UTC, which
            # silently EXCLUDES today's in-progress bar — that made the
            # "hourly top-up" a structural no-op all session. Defaulting to
            # now includes today's partial; the per-symbol delete-then-insert
            # upsert refines it each cycle and post-close runs finalize it.
            req = StockBarsRequest(symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                                   start=start)
            res = b.data.get_stock_bars(req)
            data = res.data if hasattr(res, "data") else {}
        except Exception as exc:  # noqa: BLE001 — one batch can't abort the run
            logger.warning("history batch @%d failed: %s", i, exc)
            summary["errors"] += 1
            continue
        summary["batches"] += 1
        for sym, bars in (data or {}).items():
            rows = alpaca_bars_to_rows(bars, sym)
            if not rows:
                continue
            # Isolate + retry each symbol's write: a single transient store/proxy
            # error (e.g. "connection reset by peer" over the REST lane) must not
            # abort a 500-name ingest. Re-runs are idempotent (delete-then-insert).
            if _write_bars_with_retry(store, sym, rows, start_iso):
                summary["bars_upserted"] += len(rows)
            else:
                summary["errors"] += 1
    return summary


def _write_bars_with_retry(store, sym: str, rows: list[dict], start_iso: str,
                           attempts: int = 3) -> bool:
    """Idempotent per-symbol upsert (range-delete + insert) with a short retry
    on transient store/network errors. Returns True on success."""
    import time

    for i in range(attempts):
        try:
            store.delete("daily_bars", {"symbol": sym, "date": ("gte", start_iso)})
            store.insert("daily_bars", rows, returning=False)
            return True
        except Exception as exc:  # noqa: BLE001 — transient network/proxy errors
            if i == attempts - 1:
                logger.warning("history write failed for %s: %s", sym, exc)
                return False
            time.sleep(0.5 * (i + 1))
    return False


def refresh_alpaca_market(*, top_n: int = 1000, max_days: int = 400,
                          min_price: float = 1.0, max_price: float = 100_000.0,
                          optionable_only: bool = False, lookback_days: int = 7,
                          dry_run: bool = False) -> dict:
    """Full-market ingest — the Alpaca replacement for the retired Polygon
    grouped-daily universe. Enumerate Alpaca's whole tradable catalog (~13k
    equities/ETFs), rank it by dollar volume, and keep fresh daily-bar history
    for the top-N (+ benchmarks/held/watchlist) so the agent's discovery step
    (`agent.market universe`) and its indicators/backtests see the WHOLE market,
    not a hand-picked seed list.

    ``top_n`` is a breadth dial, not a hard cap: anything outside it is still
    quote-and-fillable live, and any name the agent researches gets its history
    topped up on demand. ``optionable_only`` restricts the catalog to names with
    listed options. Idempotent; safe to re-run.
    """
    from agent import broker
    from agent.store import get_store

    if not broker.enabled():
        return {"error": "no Alpaca keys — cannot enumerate the market"}
    _bound_network()
    b = broker.Broker()
    catalog = [a["symbol"] for a in b.list_assets(optionable=optionable_only)]
    ranked = _alpaca_latest_daily(b, catalog, lookback_days=lookback_days)
    keep = _keep_symbols_rest()
    universe = select_universe_symbols(ranked, top_n=top_n, keep=keep,
                                       min_price=min_price, max_price=max_price)
    summary: dict = {"catalog": len(catalog), "ranked": len(ranked),
                     "universe": len(universe), "top_n": top_n,
                     "optionable_only": optionable_only, "dry_run": dry_run}
    if dry_run:
        summary["sample"] = universe[:25]
        return summary
    summary["ingest"] = _ingest_history_batched(get_store(), b, universe,
                                                 max_days=max_days)
    # Corporate actions for the SAME universe as the bar ingest, lookback
    # matching the bar window: ticker_splits/dividends were previously fed
    # only by the hourly ~15-name pass, so load-time split adjustment was a
    # no-op for ~99% of researched names — every split inside the fresh set's
    # research window read as a fake ±50-90% "move". Idempotent (deduped by
    # symbol+date); best-effort like the news/edgar passes.
    try:
        summary["corp_actions"] = _corp_actions_alpaca(get_store(), universe,
                                                       back_days=max_days)
    except Exception as exc:  # noqa: BLE001
        logger.warning("corp actions pass failed: %s", exc)
        summary["corp_actions"] = f"error: {exc}"
    # Grow the R2 archive for the WHOLE ingested universe — not just the ~15
    # live names the hourly path syncs. Charts read R2 for long ranges (1y+) on
    # non-protected symbols, so an ingest that freshens the DB but leaves R2
    # behind still renders a stale right edge at the default range. Change-
    # detected per symbol (unchanged names cost one tiny query), so this is
    # cheap when already current.
    summary["r2_sync"] = _r2_merge_sync(universe)
    # SEC EDGAR fundamentals for the same universe (isolated pass, like the
    # news/corp-actions passes on the hourly path): first run per symbol is a
    # full 2009+ backfill (companyfacts returns whole history in one call),
    # nightly reruns insert only new filings. ~1000 throttled calls ≈ a few
    # minutes at SEC's sanctioned rate; a failure here never aborts the bars.
    try:
        from agent import edgar

        summary["edgar"] = edgar.ingest(get_store(), symbols=universe)
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar fundamentals pass failed: %s", exc)
        summary["edgar"] = f"error: {exc}"
    return summary


def merge_bar_frames(r2_rows, db_rows):
    """GROW-ONLY union of an R2 parquet frame and fresh DB rows (pure).

    The DB wins on conflicting dates (corrections land there); dates the DB
    no longer holds are PRESERVED from R2 — a merge can never shrink the
    archive. Returns a date-sorted DataFrame with the archive schema."""
    import pandas as pd

    cols = ["date", "open", "high", "low", "close", "volume"]
    frames = []
    if r2_rows is not None and len(r2_rows):
        frames.append(pd.DataFrame(r2_rows)[cols])
    if db_rows:
        db = pd.DataFrame([{c: r[c] for c in cols} for r in db_rows])
        frames.append(db)
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = out["date"].astype(str).str[:10]
    for c in cols[1:]:
        out[c] = out[c].astype(float)
    # keep="last" → DB rows (appended last) win on duplicate dates
    out = out.drop_duplicates(subset="date", keep="last").sort_values("date")
    return out.reset_index(drop=True)


def _r2_merge_sync(symbols: list[str]):
    """Grow the R2 parquet archive from the DB — transport-agnostic.

    Bars are read through ``agent.store`` (REST or pg — works on the Routine
    sandbox AND Render) and parquet moves over S3/443, so this runs anywhere
    the R2 creds exist. Change-detected per symbol via the manifest's db_max
    fingerprint; unchanged symbols cost one tiny store query."""
    import io
    import json as _json
    import os
    import time

    if not all(os.getenv(k) for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
                                      "R2_ENDPOINT", "R2_BUCKET")):
        return "skipped (no R2_* env)"
    # R2 needs a wider socket timeout than Alpaca REST — parquet PUTs are
    # multi-MB and legitimately take longer than a JSON-shaped Alpaca call.
    _bound_network(R2_NET_TIMEOUT_S)
    try:  # setup only — a failure here really is fatal to the whole sync
        import boto3
        import pandas as pd

        from agent.store import get_store

        store = get_store()
        s3 = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT"],
                          aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                          aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"])
        bucket = os.environ["R2_BUCKET"]
    except Exception as exc:  # noqa: BLE001 — archival is best-effort
        logger.warning("R2 merge-sync setup failed: %s", exc)
        return f"skipped ({type(exc).__name__}: {exc})"

    try:
        manifest = _json.loads(s3.get_object(
            Bucket=bucket, Key="manifest.json")["Body"].read())
    except Exception:  # noqa: BLE001 — missing manifest = fresh entries
        manifest = {}

    def _flush_manifest() -> bool:
        try:
            s3.put_object(Bucket=bucket, Key="manifest.json",
                          Body=_json.dumps(manifest).encode())
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("R2 manifest flush failed: %s", exc)
            return False

    def _sync_one(sym: str):
        """Sync one symbol; returns "synced" | "unchanged". Raises on failure so
        the caller can retry — a symbol that keeps failing is counted, not fatal."""
        latest = store.select("daily_bars", filters={"symbol": sym},
                              order=[("date", "desc")], limit=1)
        db_max = str(latest[0]["date"])[:10] if latest else None
        entry = manifest.get(sym) or {}
        if not db_max or db_max <= str(entry.get("db_max") or
                                       entry.get("max_date") or ""):
            return "unchanged"
        db_rows = store.select("daily_bars", filters={"symbol": sym})
        try:
            body = s3.get_object(Bucket=bucket,
                                 Key=f"bars/{sym}.parquet")["Body"].read()
            r2_df = pd.read_parquet(io.BytesIO(body))
        except Exception:  # noqa: BLE001 — new symbol, no parquet yet
            r2_df = None
        merged = merge_bar_frames(r2_df, db_rows)
        buf = io.BytesIO()
        merged.to_parquet(buf, index=False)
        s3.put_object(Bucket=bucket, Key=f"bars/{sym}.parquet", Body=buf.getvalue())
        manifest[sym] = {"rows": int(len(merged)),
                         "max_date": str(merged["date"].iloc[-1]),
                         "db_rows": len(db_rows), "db_max": db_max}
        return "synced"

    # Per-symbol try/retry: a transient reset over 443 (seen live at scale as
    # "connection reset by peer") must skip/retry ONE name, never abort a
    # 2000-symbol archive sync. Flush the manifest periodically so a late failure
    # can't discard progress already written to parquet.
    synced = skipped = errors = dirty = 0
    for sym in symbols:
        for attempt in range(3):
            try:
                res = _sync_one(sym)
                if res == "synced":
                    synced += 1
                    dirty += 1
                    if dirty >= 200:
                        _flush_manifest()
                        dirty = 0
                else:
                    skipped += 1
                break
            except Exception as exc:  # noqa: BLE001 — transient network/proxy error
                if attempt == 2:
                    logger.warning("R2 sync failed for %s: %s", sym, exc)
                    errors += 1
                else:
                    time.sleep(0.5 * (attempt + 1))
    if dirty:
        _flush_manifest()
    return {"synced": synced, "unchanged": skipped, "errors": errors}


def main(argv: list[str] | None = None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["alpaca", "alpaca-market"], default="alpaca",
                   help="alpaca = live-universe top-up (default, cheap/hourly); "
                        "alpaca-market = full-market rank + top-N ingest (nightly)")
    p.add_argument("--max-days", type=int, default=7)
    p.add_argument("--top", type=int, default=1000)
    p.add_argument("--optionable", action="store_true",
                   help="alpaca-market: restrict the catalog to optionable names")
    p.add_argument("--min-price", type=float, default=1.0)
    p.add_argument("--max-price", type=float, default=100_000.0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.source == "alpaca-market":
        out = refresh_alpaca_market(top_n=args.top,
                                    max_days=max(args.max_days, 400),
                                    min_price=args.min_price, max_price=args.max_price,
                                    optionable_only=args.optionable,
                                    dry_run=args.dry_run)
    else:
        out = refresh_alpaca(max_days=args.max_days)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

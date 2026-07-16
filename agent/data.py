"""Thin, clean data-access for the agent — the one seam over the kept layer.

Reads delegate to the audited EdgeFinder data layer that the rebuild KEEPS:
``edgefinder.engine.data`` (bar loading, split/dividend adjustment),
``edgefinder.data.barstore`` (R2 archive), and
``edgefinder.data.indicator_engine`` (indicator snapshots).

Two transports (see ``agent.store``):
- **pg**   — SQLAlchemy ``Session`` against DATABASE_URL (Render/CI/Codespaces).
- **rest** — Supabase PostgREST over HTTPS for the web Routine, where the
  Postgres port is blocked. On this lane the market tables (splits, dividends,
  news, daily_bars, index_daily) are read via the Data API; **bars come from
  the R2 archive** (S3/443) when configured. The pure adjustment transforms
  (``adjust_for_splits`` / ``adjust_for_dividends``) are shared by both lanes,
  so the numbers don't depend on which transport served the rows.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd

from agent.store import transport


def r2_available() -> bool:
    return all(
        os.getenv(k)
        for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET")
    )


def _use_rest() -> bool:
    return transport() == "rest"


# ── pg-lane engine/session (unused on the rest lane) ────────

_session_factory = None


def session_factory():
    global _session_factory
    if _session_factory is None:
        from edgefinder.db.engine import get_engine, get_session_factory

        _session_factory = get_session_factory(get_engine())
    return _session_factory


# ── rest-lane reads of the kept market tables ───────────────


def _rest_splits(symbols: list[str]) -> dict[str, list[tuple]]:
    """{symbol: [(ex_date, factor=to/from), ...]} — mirrors engine.load_splits."""
    from agent.store import get_store

    if not symbols:
        return {}
    rows = get_store().select(
        "ticker_splits", columns="symbol,execution_date,split_from,split_to",
        filters={"symbol": ("in", symbols)}, limit=10000)
    out: dict[str, list[tuple]] = {}
    for r in rows:
        f, t = r.get("split_from"), r.get("split_to")
        if not f or not t or f <= 0 or t <= 0 or f == t:
            continue
        ex = date.fromisoformat(str(r["execution_date"])[:10])
        out.setdefault(r["symbol"], []).append((ex, t / f))
    for sym in out:
        out[sym].sort()
    return out


def _rest_dividends(symbols: list[str]) -> dict[str, list[tuple]]:
    """{symbol: [(ex_date, cash_amount), ...]} — mirrors engine.load_dividends."""
    from agent.store import get_store

    if not symbols:
        return {}
    rows = get_store().select(
        "dividends", columns="symbol,ex_date,cash_amount",
        filters={"symbol": ("in", symbols)},
        order=[("symbol", "asc"), ("ex_date", "asc")], limit=100000)
    out: dict[str, list[tuple]] = {}
    for r in rows:
        if r.get("cash_amount") is None:
            continue
        ex = date.fromisoformat(str(r["ex_date"])[:10])
        out.setdefault(r["symbol"], []).append((ex, float(r["cash_amount"])))
    return out


def _rest_bars(symbols: list[str], start: date | None, end: date | None
               ) -> dict[str, pd.DataFrame]:
    """Raw daily bars from daily_bars over REST (fallback when R2 is absent)."""
    from agent.store import get_store

    filters: dict = {"symbol": ("in", symbols)}
    if start is not None:
        filters["date"] = ("gte", start)
    rows = get_store().select(
        "daily_bars", columns="symbol,date,open,high,low,close,volume",
        filters=filters, order=[("symbol", "asc"), ("date", "asc")], limit=1000000)
    out: dict[str, list[dict]] = {}
    for r in rows:
        d = date.fromisoformat(str(r["date"])[:10])
        if end is not None and d > end:
            continue
        out.setdefault(r["symbol"], []).append({
            "date": d, "open": r["open"], "high": r["high"],
            "low": r["low"], "close": r["close"], "volume": r["volume"]})
    return {s: pd.DataFrame(v) for s, v in out.items() if v}


# ── bars (transport-aware) ──────────────────────────────────


def _splice_db_tail(r2_bars: dict, db_bars: dict) -> dict:
    """Union the deep R2 frames with the fresh DB frames, the DB winning on any
    shared or newer date. Pure.

    The R2 archive is grow-only but only as fresh as its last sync: a name the
    full-market ingest just wrote to the DB but hasn't R2-synced yet reads
    weeks-stale bars from R2 alone — which silently wrecks momentum/indicators.
    Splicing the DB's fresh tail on top (raw, pre-adjustment) gives R2's depth
    AND the DB's currency, so ``auto`` always sees the freshest close.
    """
    out = dict(r2_bars or {})
    for sym, dbf in (db_bars or {}).items():
        if dbf is None or not len(dbf):
            continue
        r2f = out.get(sym)
        if r2f is None or not len(r2f):
            out[sym] = dbf.sort_values("date").reset_index(drop=True)
            continue
        merged = pd.concat([r2f, dbf], ignore_index=True)
        out[sym] = (merged.drop_duplicates(subset="date", keep="last")
                    .sort_values("date").reset_index(drop=True))
    return out


def load_bars(
    symbols: list[str], *, start: date | None = None, end: date | None = None,
    div_adjust: bool = True, source: str = "auto",
) -> dict[str, pd.DataFrame]:
    """Split-adjusted (and optionally total-return) daily bars per symbol.

    ``source``: ``auto`` (R2 for depth, spliced with the DB's fresh tail),
    ``r2`` (archive only — the pure equivalence path), or ``db`` (hot set only).
    Returns ``{symbol: DataFrame[date, OHLCV(, close_raw)]}``.
    """
    from edgefinder.engine import data as eng

    use_r2 = source == "r2" or (source == "auto" and r2_available())

    if _use_rest():
        if use_r2:
            bars = eng.load_bars_from_store(symbols, start=start, end=end)
            if source == "auto":  # fresh DB tail beats a stale archive
                bars = _splice_db_tail(bars, _rest_bars(symbols, start, end))
        else:
            bars = _rest_bars(symbols, start, end)
        splits = _rest_splits(list(bars))
        bars = eng.adjust_for_splits(bars, splits)
        if div_adjust:
            divs = eng.adjust_dividends_for_splits(_rest_dividends(list(bars)), splits)
            bars = eng.adjust_for_dividends(bars, divs)
        return bars

    # pg lane
    if use_r2:
        bars = eng.load_bars_from_store(symbols, start=start, end=end)
        sess = session_factory()()
        try:
            if source == "auto":  # splice the fresh DB tail (raw, pre-adjust)
                db_raw = eng.load_bars(sess, symbols, start=start, end=end,
                                       split_adjust=False)
                bars = _splice_db_tail(bars, db_raw)
            splits = eng.load_splits(sess, list(bars))
            bars = eng.adjust_for_splits(bars, splits)
            if div_adjust:
                divs = eng.load_dividends(sess, list(bars))
                divs = eng.adjust_dividends_for_splits(divs, splits)
                bars = eng.adjust_for_dividends(bars, divs)
        finally:
            sess.close()
        return bars

    sess = session_factory()()
    try:
        bars = eng.load_bars(sess, symbols, start=start, end=end, split_adjust=True)
        if div_adjust and bars:
            splits = eng.load_splits(sess, list(bars))
            divs = eng.load_dividends(sess, list(bars))
            divs = eng.adjust_dividends_for_splits(divs, splits)
            bars = eng.adjust_for_dividends(bars, divs)
        return bars
    finally:
        sess.close()


def latest_indicators(symbols: list[str], *, as_of: date | None = None,
                      source: str = "auto") -> dict[str, dict]:
    """Latest close + computed indicators per symbol as of ``as_of`` (or today)."""
    from edgefinder.data.indicator_engine import compute_snapshot_series

    start = (as_of or date.today()) - timedelta(days=560)
    bars = load_bars(symbols, start=start, end=as_of, div_adjust=False, source=source)
    out: dict[str, dict] = {}
    for sym, df in bars.items():
        if df is None or not len(df):
            continue
        d = df.sort_values("date").reset_index(drop=True)
        snaps = compute_snapshot_series(d[["open", "high", "low", "close", "volume"]])
        if not snaps:
            continue
        snap = snaps[-1].to_dict()
        out[sym] = {
            "symbol": sym,
            "date": str(d["date"].iloc[-1]),
            "close": round(float(d["close"].iloc[-1]), 4),
            "bars": int(len(d)),
            "indicators": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in snap.items()},
            "ret_1m": _ret(d, 21),
            "ret_3m": _ret(d, 63),
            "ret_6m": _ret(d, 126),
            "ret_12m": _ret(d, 252),
        }
    return out


def _ret(df: pd.DataFrame, n: int) -> float | None:
    c = df["close"]
    if len(c) <= n or c.iloc[-n - 1] <= 0:
        return None
    return round(float(c.iloc[-1] / c.iloc[-n - 1] - 1.0), 4)


def history(symbol: str, *, days: int = 120, source: str = "auto") -> list[dict]:
    """Recent OHLCV rows (split+div adjusted) for one symbol, oldest→newest."""
    start = date.today() - timedelta(days=int(days * 1.6) + 10)
    bars = load_bars([symbol], start=start, div_adjust=True, source=source)
    df = bars.get(symbol)
    if df is None or not len(df):
        return []
    d = df.sort_values("date").reset_index(drop=True).tail(days)
    return [
        {"date": str(r.date), "open": round(float(r.open), 4),
         "high": round(float(r.high), 4), "low": round(float(r.low), 4),
         "close": round(float(r.close), 4), "volume": float(r.volume)}
        for r in d.itertuples()
    ]


def news(symbol: str, *, limit: int = 8) -> list[dict]:
    """Recent stored news headlines for a symbol (from ticker_news)."""
    if _use_rest():
        from agent.store import get_store

        rows = get_store().select(
            "ticker_news",
            columns="title,published_utc,publisher_name,article_url,description",
            filters={"symbol": symbol.upper()},
            order=[("published_utc", "desc")], limit=limit)
        return [
            {"title": r.get("title"), "published_utc": r.get("published_utc"),
             "publisher": r.get("publisher_name"), "url": r.get("article_url"),
             "description": (r.get("description") or "")[:400]}
            for r in rows
        ]

    from edgefinder.db.models import TickerNews

    sess = session_factory()()
    try:
        rows = (sess.query(TickerNews)
                .filter(TickerNews.symbol == symbol.upper())
                .order_by(TickerNews.published_utc.desc())
                .limit(limit).all())
        return [
            {"title": r.title, "published_utc": r.published_utc,
             "publisher": r.publisher_name, "url": r.article_url,
             "description": (r.description or "")[:400]}
            for r in rows
        ]
    finally:
        sess.close()


def universe(top_n: int = 200, *, as_of: date | None = None) -> list[str]:
    """The top-N most liquid symbols by recent dollar volume (PIT-safe-ish).

    pg lane uses the kept PIT ranking. rest lane ranks the latest stored
    session's dollar volume from daily_bars (the hot set already holds the
    live top-1000/day), which is fast over the Data API and good enough for
    the agent's hunting universe.
    """
    if _use_rest():
        return _rest_universe(top_n, as_of)

    from edgefinder.engine.data import resolve_universe

    sess = session_factory()()
    try:
        rank_start = (as_of or date.today()) - timedelta(days=190)
        return resolve_universe(sess, "top", [], top_n,
                                as_of=as_of, rank_start=rank_start)
    finally:
        sess.close()


def _rest_universe(top_n: int, as_of: date | None) -> list[str]:
    """Rank the hot set by dollar volume over a short TRAILING window, not a
    single latest day.

    A single-day rank collapses whenever coverage is uneven: the cheap hourly
    top-up writes ~10 held/watchlist names for the newest date, so "rank the
    latest day" saw only those 10 even after a full-market ingest wrote 1000+
    names for the prior session. Ranking each symbol's most-recent bar within a
    trailing window instead lets the broad ingest's names rank in regardless of
    which day the last thin top-up happened to land on.
    """
    from agent.store import get_store

    store = get_store()
    anchor = as_of or date.today()
    lo = (anchor - timedelta(days=10)).isoformat()
    hi = anchor.isoformat()
    rows = store.select("daily_bars", columns="symbol,close,volume,date",
                        filters={"date": ("gte", lo)}, limit=200000)
    latest_per_sym: dict[str, tuple[str, float, float]] = {}
    for r in rows:
        sym, c, v, d = r.get("symbol"), r.get("close"), r.get("volume"), str(r.get("date"))[:10]
        if not sym or c is None or v is None or d > hi:
            continue
        prev = latest_per_sym.get(sym)
        if prev is None or d > prev[0]:
            latest_per_sym[sym] = (d, float(c), float(v))
    scored = []
    for sym, (_d, c, v) in latest_per_sym.items():
        if any(ch in sym for ch in (".", "/", "=")):
            continue
        scored.append((c * v, sym))
    scored.sort(reverse=True)
    return [s for _, s in scored[:top_n]]


def spy_series_df(*, total_return: bool = False) -> pd.DataFrame:
    """Longest available SPY series for benchmarking (transport-aware).

    ``total_return=True`` dividend-adjusts the series through the SAME
    ``load_bars`` path the strategy bars take, so backtest/lab excess is
    TR-vs-TR. A price-only SPY against dividend-adjusted strategy bars hands
    EVERY strategy the benchmark's dividend yield as phantom "excess"
    (~+50pp compounded over 2006-2018) — the structural inflation fixed
    2026-07-16. Missing SPY dividend rows degrade gracefully: the adjustment
    is a no-op and TR ≈ PR (never a crash).

    The default stays price-only for the callers where that is the honest
    basis: the live book's vs-SPY (``agent.ledger``) books no dividend cash
    on either side, so its price-vs-price comparison is deliberate.
    """
    if total_return or _use_rest():
        bars = load_bars(["SPY"], div_adjust=total_return, source="auto")
        df = bars.get("SPY")
        if df is None or not len(df):
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        return df.sort_values("date").reset_index(drop=True)

    from edgefinder.engine.data import spy_series

    sess = session_factory()()
    try:
        return spy_series(sess)
    finally:
        sess.close()


def regime(*, as_of: date | None = None) -> dict:
    """A compact market-regime read from the index series (pure DB/R2 history).

    Honesty: ``bars_through`` is the actual date of the newest bar this read
    used — the indicators are only as current as that. ``bars_stale`` flags a
    gap > 5 calendar days so old closes are never mistaken for today's."""
    today = as_of or date.today()
    out: dict = {"as_of": str(today), "indices": {}}
    bars = latest_indicators(["SPY", "QQQ", "IWM"], as_of=as_of)
    bar_dates = [str(info.get("date"))[:10] for info in bars.values() if info.get("date")]
    if bar_dates:
        newest = max(bar_dates)
        out["bars_through"] = newest
        out["bars_stale"] = (today - date.fromisoformat(newest)).days > 5
    for sym, info in bars.items():
        ind = info.get("indicators", {})
        close = info["close"]
        sma50 = ind.get("ema_50")
        sma200 = ind.get("ema_200")
        out["indices"][sym] = {
            "close": close,
            "above_50": (close > sma50) if sma50 else None,
            "above_200": (close > sma200) if sma200 else None,
            "ret_1m": info.get("ret_1m"),
            "ret_3m": info.get("ret_3m"),
            "rsi": ind.get("rsi"),
        }
    spy = out["indices"].get("SPY", {})
    if spy.get("above_200") is True and spy.get("above_50") is True:
        out["tag"] = "risk_on"
    elif spy.get("above_200") is False:
        out["tag"] = "risk_off"
    else:
        out["tag"] = "neutral"
    return out


# ── market-data coverage (is the whole-market scan trustworthy?) ──
#
# Bar AGE alone cannot detect a dead nightly ingest: the hourly top-up keeps
# writing a handful of held/streamed names every session, so "latest bar =
# yesterday" stays true while the other ~2,000 symbols quietly rot (this is
# exactly how the 2026-07-08 outage went unnoticed). The honest unit of
# staleness is a *thin session*: a bar date newer than the last date with
# full-universe coverage. One thin session is normal (today's intraday
# partial); two means last night's ingest was missed; three or more means
# whole-market research is running on stale rankings.
#
# Thin sessions alone can't see TOTAL death, though — when every writer stops
# (all Routines dead), no new bar dates appear and the newest date IS the last
# full one, thin_sessions=0 forever. The calendar-age anchor below catches
# that: bars older than a long weekend can't be green no matter their shape.

# Bars on a date for it to count as a full nightly ingest. The nightly runs
# --top 1000 and lands ~1,020-1,035 rows (top-N + the forced keep set), so
# 900 leaves ~10% headroom: one failed 100-name fetch batch in an otherwise
# successful ingest must not flip the whole date to "thin".
FULL_COVERAGE_MIN = 900
COVERAGE_AMBER_AGE_DAYS = 4  # newest bar older than this → at most amber
COVERAGE_RED_AGE_DAYS = 6    # older than this → red (holiday weekend is 4)


def coverage_verdict(date_counts, *, full_min: int | None = None,
                     today: date | None = None) -> dict:
    """Classify market-data freshness from per-date bar counts (pure logic).

    ``date_counts``: iterable of ``(date-or-iso-str, count)`` in any order.
    Counts may be capped at ``full_min`` by the caller — only the threshold
    matters. Returns status green (0-1 thin sessions), amber (2), or red (3+,
    or no full-coverage date in the window); ``research_ok`` is False only on
    red, so one transient missed nightly degrades the display without benching
    the trader. Independently of shape, the newest bar's calendar age caps the
    verdict (amber past COVERAGE_AMBER_AGE_DAYS, red past
    COVERAGE_RED_AGE_DAYS) so a totally dead pipeline — no new dates at all —
    cannot read as fresh.
    """
    full_min = FULL_COVERAGE_MIN if full_min is None else full_min
    today = today or date.today()
    counts: dict[str, int] = {}
    for d, n in date_counts:
        key = str(d)[:10]
        counts[key] = max(counts.get(key, 0), int(n or 0))
    out: dict = {"full_min": full_min, "latest_date": None, "latest_count": 0,
                 "latest_age_days": None, "last_full_date": None,
                 "thin_sessions": 0, "status": "red", "research_ok": False}
    if not counts:
        return out
    latest = max(counts)
    out["latest_date"] = latest
    out["latest_count"] = counts[latest]
    try:
        out["latest_age_days"] = (today - date.fromisoformat(latest)).days
    except ValueError:
        pass
    full_dates = [d for d, n in counts.items() if n >= full_min]
    if full_dates:
        last_full = max(full_dates)
        out["last_full_date"] = last_full
        out["thin_sessions"] = sum(1 for d in counts if d > last_full)
    else:
        out["thin_sessions"] = len(counts)  # nothing full in the window
    thin = out["thin_sessions"]
    if full_dates and thin <= 1:
        out["status"] = "green"
    elif full_dates and thin == 2:
        out["status"] = "amber"
    else:
        out["status"] = "red"
    # Calendar anchor: silence is not health.
    age = out["latest_age_days"]
    if age is not None:
        if age > COVERAGE_RED_AGE_DAYS:
            out["status"] = "red"
        elif age > COVERAGE_AMBER_AGE_DAYS and out["status"] == "green":
            out["status"] = "amber"
    out["research_ok"] = out["status"] != "red"
    return out


def universe_coverage(*, window: int = 10,
                      full_min: int | None = None) -> dict:
    """Recent per-date coverage of ``daily_bars`` via the store (both lanes).

    The store has no aggregates on the REST lane, so counts are measured by
    fetching at most ``full_min`` symbol rows per date — enough to know
    full-vs-thin, never more. Walks dates newest-first and stops at the first
    full-coverage date (older history can't change the verdict).
    """
    from agent.store import get_store

    full_min = FULL_COVERAGE_MIN if full_min is None else full_min
    store = get_store()
    # Recent bar dates, anchored on the index ETFs (stream seeds + protected
    # keeps — always written by both the top-up and the nightly), plus the
    # true newest date in case even those are the rows missing. A thin date
    # none of the anchors traded would be invisible; the calendar-age anchor
    # in coverage_verdict bounds the damage of that blind spot.
    anchor_rows = store.select("daily_bars", columns="date",
                               filters={"symbol": ("in", ["SPY", "QQQ", "IWM"])},
                               order=[("date", "desc")], limit=window * 3)
    newest = store.select("daily_bars", columns="date",
                          order=[("date", "desc")], limit=1)
    dates = sorted({str(r["date"])[:10] for r in [*anchor_rows, *newest]},
                   reverse=True)[:window]
    date_counts: list[tuple[str, int]] = []
    for d in dates:
        rows = store.select("daily_bars", columns="symbol",
                            filters={"date": date.fromisoformat(d)},
                            limit=full_min)
        date_counts.append((d, len(rows)))
        if len(rows) >= full_min:
            break  # verdict can't change past the first full date
    return coverage_verdict(date_counts, full_min=full_min)

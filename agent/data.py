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

import logging
import os
from datetime import date, timedelta

import pandas as pd

from agent.store import transport

logger = logging.getLogger(__name__)


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


def merge_fresh_bars(bars: dict[str, pd.DataFrame],
                     fresh: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Append rows from ``fresh`` strictly newer than each frame in ``bars``.

    Pure (no I/O). Symbols present only in ``fresh`` are added whole; ``fresh``
    columns are aligned to the base frame so the merged shape never changes.
    """
    out = dict(bars)
    for sym, fdf in fresh.items():
        if fdf is None or not len(fdf):
            continue
        base = out.get(sym)
        if base is None or not len(base):
            out[sym] = fdf.reset_index(drop=True)
            continue
        newer = fdf[fdf["date"] > base["date"].max()]
        if len(newer):
            out[sym] = pd.concat([base, newer.reindex(columns=base.columns)],
                                 ignore_index=True)
    return out


def _db_raw_bars(symbols: list[str], start: date | None, end: date | None
                 ) -> dict[str, pd.DataFrame]:
    """Raw (split-UNadjusted) daily_bars from the active transport's DB."""
    if _use_rest():
        return _rest_bars(symbols, start, end)
    from edgefinder.engine import data as eng

    sess = session_factory()()
    try:
        return eng.load_bars(sess, symbols, start=start, end=end, split_adjust=False)
    finally:
        sess.close()


def _top_up_from_db(bars: dict[str, pd.DataFrame], symbols: list[str],
                    end: date | None) -> dict[str, pd.DataFrame]:
    """Overlay daily_bars rows newer than the R2 archive's frames.

    The R2 archive only grows when a sync job runs; the hourly Alpaca top-up
    lands in ``daily_bars``. Splicing the newer DB rows in BEFORE split/div
    adjustment keeps ``source="auto"`` reads (regime, indicators, backtests)
    as fresh as the DB without giving up R2 depth. Best-effort: a DB failure
    falls back to the pure R2 read (regime's ``bars_stale`` stays honest).
    """
    try:
        ends = [df["date"].max() for df in bars.values() if df is not None and len(df)]
        start = (min(ends) + timedelta(days=1)) if ends else None
        fresh = _db_raw_bars(symbols, start, end)
        return merge_fresh_bars(bars, fresh)
    except Exception:  # noqa: BLE001 — freshness top-up must never break reads
        logger.warning("DB freshness top-up failed; using R2 bars as-is",
                       exc_info=True)
        return bars


def load_bars(
    symbols: list[str], *, start: date | None = None, end: date | None = None,
    div_adjust: bool = True, source: str = "auto",
) -> dict[str, pd.DataFrame]:
    """Split-adjusted (and optionally total-return) daily bars per symbol.

    ``source``: ``auto`` (R2 when configured — topped up with any newer
    ``daily_bars`` rows so reads stay current between R2 syncs — else the
    active transport's DB), ``r2`` (pure archive), or ``db``.
    Returns ``{symbol: DataFrame[date, OHLCV(, close_raw)]}``.
    """
    from edgefinder.engine import data as eng

    use_r2 = source == "r2" or (source == "auto" and r2_available())

    if _use_rest():
        if use_r2:
            bars = eng.load_bars_from_store(symbols, start=start, end=end)
            if source == "auto":
                bars = _top_up_from_db(bars, symbols, end)
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
        if source == "auto":
            bars = _top_up_from_db(bars, symbols, end)
        sess = session_factory()()
        try:
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
    from agent.store import get_store

    store = get_store()
    latest = store.select("daily_bars", columns="date",
                          order=[("date", "desc")], limit=1)
    if not latest:
        return []
    day = str(latest[0]["date"])[:10]
    rows = store.select("daily_bars", columns="symbol,close,volume",
                        filters={"date": day}, limit=20000)
    scored = []
    for r in rows:
        sym, c, v = r.get("symbol"), r.get("close"), r.get("volume")
        if not sym or c is None or v is None:
            continue
        if any(ch in sym for ch in (".", "/", "=")):
            continue
        scored.append((float(c) * float(v), sym))
    scored.sort(reverse=True)
    return [s for _, s in scored[:top_n]]


def spy_series_df() -> pd.DataFrame:
    """Longest available SPY series for benchmarking (transport-aware)."""
    if _use_rest():
        bars = load_bars(["SPY"], div_adjust=False, source="auto")
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

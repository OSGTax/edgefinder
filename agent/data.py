"""Thin, clean data-access for the agent — the one seam over the kept layer.

Everything here delegates to the audited EdgeFinder data layer that the
rebuild KEEPS: ``edgefinder.engine.data`` (DB + R2 bar loading, split and
dividend adjustment), ``edgefinder.data.barstore`` (R2 archive), and
``edgefinder.data.indicator_engine`` (indicator snapshots). The agent never
touches raw SQL or S3 — it calls these typed helpers and gets back plain
dicts / DataFrames.

Bar source: ``auto`` reads the deep history from R2 when the R2_* env vars
are present (the 21-year archive), and falls back to the Postgres hot set
otherwise. The two paths are proven bit-equivalent (see CLAUDE.md), so a
backtest's numbers don't depend on which lane served the bars.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd

from edgefinder.db.engine import get_engine, get_session_factory


def r2_available() -> bool:
    return all(
        os.getenv(k)
        for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET")
    )


_session_factory = None


def session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = get_session_factory(get_engine())
    return _session_factory


def load_bars(
    symbols: list[str],
    *,
    start: date | None = None,
    end: date | None = None,
    div_adjust: bool = True,
    source: str = "auto",
) -> dict[str, pd.DataFrame]:
    """Split-adjusted (and optionally total-return) daily bars per symbol.

    ``source``: ``auto`` (R2 when configured, else DB), ``r2``, or ``db``.
    Returns ``{symbol: DataFrame[date, open, high, low, close, volume(, close_raw)]}``.
    """
    from edgefinder.engine import data as eng

    use_r2 = source == "r2" or (source == "auto" and r2_available())
    if use_r2:
        bars = eng.load_bars_from_store(symbols, start=start, end=end)
        # the store loader doesn't split-adjust (raw mirror) — apply it here
        # using DB split history so R2 and DB lanes agree.
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
    """Latest close + computed indicators per symbol as of ``as_of`` (or today).

    Pulls ~1.5y of bars so EMAs/RSI/MACD/BB are warm, then returns the last
    snapshot dict (the IndicatorSnapshot the strategy interface uses).
    """
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
    """The top-N most liquid symbols by trailing dollar volume (PIT-safe).

    Uses the DB ranking (the operational hot set already holds the live
    top-1000/day), so this is fast and needs no R2 full-market scan.
    """
    from edgefinder.engine.data import resolve_universe

    sess = session_factory()()
    try:
        rank_start = (as_of or date.today()) - timedelta(days=190)
        return resolve_universe(sess, "top", [], top_n,
                                as_of=as_of, rank_start=rank_start)
    finally:
        sess.close()


def regime(*, as_of: date | None = None) -> dict:
    """A compact market-regime read from the index series.

    SPY/QQQ/IWM levels + recent trend (last close vs its 50d and 200d SMA)
    and 1m/3m returns. No external calls — pure DB/R2 history.
    """
    out: dict = {"as_of": str(as_of or date.today()), "indices": {}}
    bars = latest_indicators(["SPY", "QQQ", "IWM"], as_of=as_of)
    for sym, info in bars.items():
        ind = info.get("indicators", {})
        close = info["close"]
        sma50 = ind.get("ema_50")  # snapshot exposes EMAs; use as trend proxy
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

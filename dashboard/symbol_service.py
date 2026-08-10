"""Bar access for the dashboard's symbol charts — the DB/R2 seam.

Two-tier storage (v5.35): the DB holds protected ETFs full-history plus a
trailing ~365d top-1000 window; the full market history lives in the R2
Parquet store. This service picks the right source per request, applies
split adjustment uniformly (the R2 loader returns raw frames), falls back
to the DB (flagged ``truncated``) when R2 is unavailable, and caches
frames in-process (R2 GETs are 100-300ms).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from config.settings import settings
from edgefinder.data.barstore import DB_PROTECTED_ETFS
from edgefinder.engine.data import (
    adjust_for_splits,
    load_bars,
    load_bars_from_store,
    load_splits,
)

from dashboard.ttl_cache import TTLCache

logger = logging.getLogger(__name__)

# how far back the DB reliably covers non-protected symbols (the slim
# keeps a trailing-365d window; leave margin for prune lag + weekends)
DB_WINDOW_DAYS = 350

_cache = TTLCache(maxsize=128, ttl_seconds=900)


def _protected(symbol: str) -> bool:
    prot = {s.upper() for s in DB_PROTECTED_ETFS}
    prot |= {s.strip().upper() for s in settings.index_symbols if s.strip()}
    return symbol.upper() in prot


def _last_expected_session(today: date) -> date:
    """The most recent date a daily bar could exist for (weekend-aware;
    holidays degrade to a no-op fetch, bounded by the frame cache TTL)."""
    if today.weekday() >= 5:  # Sat/Sun → the preceding Friday
        return today - timedelta(days=today.weekday() - 4)
    return today


def _live_tail(symbol: str, after: date) -> pd.DataFrame | None:
    """Daily bars AFTER ``after`` straight from Alpaca market data — the
    live top-up for symbols whose stored history has gone stale (the nightly
    only maintains the top-N universe + held names). Read-only: nothing is
    written to the market-data tables. None when data creds are absent, the
    call fails, or nothing new exists; split-adjusted to match the stored
    basis. Intraday this includes today's developing bar — the chart shows
    the candle forming, which is the point."""
    if not (settings.alpaca_api_key and settings.alpaca_api_secret):
        return None
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        try:
            feed = DataFeed(settings.alpaca_data_feed or "sip")
        except ValueError:
            feed = DataFeed.SIP
        client = StockHistoricalDataClient(settings.alpaca_api_key,
                                           settings.alpaca_api_secret)
        from datetime import datetime as _dt

        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=_dt.combine(after + timedelta(days=1), _dt.min.time()),
            adjustment=Adjustment.SPLIT, feed=feed)
        bars = (client.get_stock_bars(req).data or {}).get(symbol) or []
        rows = [{"date": b.timestamp.date(), "open": float(b.open),
                 "high": float(b.high), "low": float(b.low),
                 "close": float(b.close), "volume": float(b.volume)}
                for b in bars if b.timestamp.date() > after]
        return pd.DataFrame(rows) if rows else None
    except Exception:  # noqa: BLE001 — the top-up is additive, never a 500
        logger.warning("live bar top-up failed for %s", symbol, exc_info=True)
        return None


def get_bars(db, symbol: str, start: date | None) -> tuple[pd.DataFrame, dict]:
    """Split-adjusted daily bars for one symbol.

    Returns ``(frame, meta)`` where meta = {"source": "db"|"r2",
    "truncated": bool}. ``truncated`` means deep history was requested but
    only the DB window could be served (R2 unavailable). When the stored
    history ends before the last expected session the missing tail is
    fetched live from Alpaca and merged (``live_through`` in meta) — the
    stored layer is the nightly's job; the screen's job is to be current.
    """
    symbol = symbol.upper()
    key = (symbol, start.isoformat() if start else "max")
    hit = _cache.get(key)
    if hit is not None:
        return hit

    recent = start is not None and start >= date.today() - timedelta(days=DB_WINDOW_DAYS)
    meta = {"source": "db", "truncated": False}

    if _protected(symbol) or recent:
        frame = load_bars(db, [symbol], start=start).get(symbol)
    else:
        try:
            # raw store frame; split adjustment is the caller's job here
            frame = load_bars_from_store([symbol], start=start).get(symbol)
            if frame is not None:
                frame = adjust_for_splits(
                    {symbol: frame}, load_splits(db, [symbol]))[symbol]
            meta["source"] = "r2"
        except Exception:
            logger.exception("R2 read failed for %s — serving DB window", symbol)
            frame = load_bars(db, [symbol], start=start).get(symbol)
            meta = {"source": "db", "truncated": True}

    if frame is None:
        frame = pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"])

    # Live top-up (cached with the frame, so ≤1 Alpaca call per symbol per
    # cache TTL). Unknown symbols (empty frame) stay 404 at the router —
    # the searchable set is the stored universe.
    if len(frame):
        try:
            last = frame["date"].iloc[-1]
            last = last.date() if hasattr(last, "date") else last
            if last < _last_expected_session(date.today()):
                tail = _live_tail(symbol, after=max(
                    last, date.today() - timedelta(days=90)))
                if tail is not None and len(tail):
                    frame = pd.concat([frame, tail], ignore_index=True)
                    meta["live_through"] = str(tail["date"].iloc[-1])
                    meta["source"] = f"{meta['source']}+alpaca"
        except Exception:  # noqa: BLE001 — stored bars still serve
            logger.warning("bar top-up merge failed for %s", symbol,
                           exc_info=True)

    result = (frame, meta)
    _cache.set(key, result)
    return result


def clear_cache() -> None:
    _cache.clear()

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


def get_bars(db, symbol: str, start: date | None) -> tuple[pd.DataFrame, dict]:
    """Split-adjusted daily bars for one symbol.

    Returns ``(frame, meta)`` where meta = {"source": "db"|"r2",
    "truncated": bool}. ``truncated`` means deep history was requested but
    only the DB window could be served (R2 unavailable).
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

    result = (frame, meta)
    _cache.set(key, result)
    return result


def clear_cache() -> None:
    _cache.clear()

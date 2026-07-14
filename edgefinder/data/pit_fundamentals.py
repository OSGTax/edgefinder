"""Point-in-time fundamentals lookup — the engine's honest ``asof`` provider.

Restored from the retired Polygon-era module (purged in a83bc52) with the
same protocol the backtest engine duck-types
(``edgefinder/engine/backtest.py``: ``asof(symbol, date) ->
TickerFundamentals | None``), rebuilt on two changed foundations:

- Source rows come from ``fundamentals_pit`` (SEC EDGAR, one row per FILING,
  keyed by the ``filed`` date) instead of daily snapshots of a live vendor
  table. ``asof(sym, D)`` returns knowledge as of the last filing on or
  before D — restatements filed after D are invisible to D, by construction.
- Access goes through ``agent.store`` (pg or Supabase REST), not a bound
  SQLAlchemy session, so the same loader serves Render, CI/SQLite, and the
  web Routine sandbox.

Price-dependent fields (P/E, market cap, EV ratios) are intentionally None
on hydrated models — a filing cannot know a later price. Strategies compute
them at decision time from the ``_``-ingredients via ``agent.edgar
.price_ratios`` and the AssetView's own price.
"""

from __future__ import annotations

import bisect
import json
import logging
from datetime import date

from edgefinder.core.models import TickerFundamentals

logger = logging.getLogger(__name__)

_MODEL_FIELDS = set(TickerFundamentals.model_fields)


def _hydrate(symbol: str, data) -> TickerFundamentals | None:
    if not isinstance(data, dict):
        try:
            data = json.loads(data)
        except (TypeError, ValueError):
            return None
    clean = {k: v for k, v in data.items() if k in _MODEL_FIELDS}
    clean["symbol"] = symbol
    try:
        return TickerFundamentals(**clean)
    except Exception:  # noqa: BLE001 — a bad row must not kill a backtest
        logger.exception("bad fundamentals_pit row for %s", symbol)
        return None


class PITFundamentals:
    """As-of fundamentals lookup satisfying the engine's PIT protocol."""

    def __init__(self, store=None) -> None:
        if store is None:
            from agent.store import get_store

            store = get_store()
        self._store = store
        self._dates: dict[str, list[date]] = {}
        self._models: dict[tuple[str, date], TickerFundamentals | None] = {}
        self._raw: dict[tuple[str, date], dict] = {}

    def preload(self, symbols: list[str]) -> int:
        """Bulk-load all filings for ``symbols`` so asof() is in-memory.

        One chunked select instead of thousands of per-date round-trips —
        and the engine never touches the store mid-run."""
        n = 0
        todo = [s.upper() for s in dict.fromkeys(symbols)
                if s.upper() not in self._dates]
        for i in range(0, len(todo), 200):
            chunk = todo[i:i + 200]
            for s in chunk:
                self._dates[s] = []
            rows = self._store.select(
                "fundamentals_pit", columns="symbol,filed,data",
                filters={"symbol": ("in", chunk)},
                order=[("symbol", "asc"), ("filed", "asc")])
            for r in rows:
                sym = r["symbol"]
                filed = r["filed"]
                if isinstance(filed, str):
                    filed = date.fromisoformat(filed[:10])
                self._dates[sym].append(filed)
                data = r["data"] if isinstance(r["data"], dict) \
                    else json.loads(r["data"])
                self._raw[(sym, filed)] = data
                self._models[(sym, filed)] = _hydrate(sym, data)
                n += 1
        return n

    def _symbol_dates(self, symbol: str) -> list[date]:
        if symbol not in self._dates:
            self.preload([symbol])
        return self._dates[symbol]

    def asof(self, symbol: str, as_of: date) -> TickerFundamentals | None:
        """Latest filing on/before ``as_of``; None before coverage begins."""
        symbol = symbol.upper()
        dates = self._symbol_dates(symbol)
        idx = bisect.bisect_right(dates, as_of) - 1
        if idx < 0:
            return None
        return self._models.get((symbol, dates[idx]))

    def raw_asof(self, symbol: str, as_of: date) -> dict | None:
        """The raw PIT dict (incl. `_` ratio ingredients) for strategies that
        compute price ratios at decision time."""
        symbol = symbol.upper()
        dates = self._symbol_dates(symbol)
        idx = bisect.bisect_right(dates, as_of) - 1
        if idx < 0:
            return None
        return self._raw.get((symbol, dates[idx]))

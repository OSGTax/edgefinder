"""EdgeFinder v2 — CachedDataProvider wrapper.

Wraps any DataProvider with a local cache layer. Checks cache first,
delegates to underlying provider on miss, stores results.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import TickerFundamentals
from edgefinder.data.cache import DataCache


class CachedDataProvider:
    """Cache-first wrapper around any DataProvider."""

    def __init__(self, provider: DataProvider, cache: DataCache) -> None:
        self._provider = provider
        self._cache = cache

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame | None:
        cached = self._cache.get_bars(ticker, timeframe, start, end)
        if cached is not None and not cached.empty:
            return cached
        df = self._provider.get_bars(ticker, timeframe, start, end)
        if df is not None and not df.empty:
            self._cache.store_bars(ticker, timeframe, df)
        return df

    def get_latest_price(self, ticker: str) -> float | None:
        # Prices must always be fresh — never cached
        return self._provider.get_latest_price(ticker)

    def get_fundamentals(self, ticker: str, full_refresh: bool = False) -> TickerFundamentals | None:
        if not full_refresh:
            cached = self._cache.get_fundamentals(ticker)
            if cached is not None:
                return cached
        result = self._provider.get_fundamentals(ticker, full_refresh=full_refresh)
        if result is not None:
            self._cache.store_fundamentals(ticker, result)
        return result

    def get_ticker_universe(
        self, min_market_cap: int = 0, min_volume: int = 0
    ) -> list[str]:
        cached = self._cache.get_universe()
        if cached is not None:
            return cached
        result = self._provider.get_ticker_universe(min_market_cap, min_volume)
        if result:
            self._cache.store_universe(result)
        return result

    def is_market_open(self) -> bool:
        return self._provider.is_market_open()

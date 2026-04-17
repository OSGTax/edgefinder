"""EdgeFinder v2 — CachedDataProvider wrapper.

Wraps any DataProvider with a local cache layer. Checks cache first,
delegates to underlying provider on miss, stores results.
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd

from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import TickerFundamentals
from edgefinder.data.cache import DataCache

# In-memory price cache with TTL (avoids redundant API calls during position checks)
# Kept short so consecutive position-monitor cycles get fresh data — only meant
# to dedupe calls within a single check_positions sweep, not across sweeps.
_PRICE_CACHE: dict[str, tuple[float, float]] = {}  # ticker -> (price, timestamp)
_PRICE_CACHE_TTL = 30  # 30 seconds


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
        """Get latest price with 2-minute in-memory cache.

        Prevents redundant API calls when check_positions() loops over
        multiple tickers within seconds.
        """
        now = time.time()
        cached = _PRICE_CACHE.get(ticker)
        if cached and (now - cached[1]) < _PRICE_CACHE_TTL:
            return cached[0]
        price = self._provider.get_latest_price(ticker)
        if price is not None:
            _PRICE_CACHE[ticker] = (price, now)
        return price

    def get_fundamentals(self, ticker: str, full_refresh: bool = False) -> TickerFundamentals | None:
        if not full_refresh:
            cached = self._cache.get_fundamentals(ticker)
            if cached is not None:
                return cached
        result = self._provider.get_fundamentals(ticker, full_refresh=full_refresh)
        # Only cache if we got meaningful data (not empty/failed)
        if result is not None and result.company_name is not None:
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

    def get_top_dollar_volume_tickers(
        self,
        top_n: int = 1000,
        min_price: float = 5.0,
        max_price: float = 500.0,
    ) -> list[str]:
        """Delegate to underlying provider — never cached, always fresh."""
        return self._provider.get_top_dollar_volume_tickers(
            top_n=top_n, min_price=min_price, max_price=max_price,
        )

    def get_all_snapshots(self) -> dict[str, float]:
        # Prices must always be fresh — never cached
        return self._provider.get_all_snapshots()

    def is_market_open(self) -> bool:
        return self._provider.is_market_open()

    def get_news(self, ticker: str, limit: int = 10) -> list[dict]:
        return self._provider.get_news(ticker, limit)

    def get_dividends(self, ticker: str, limit: int = 20) -> list[dict]:
        return self._provider.get_dividends(ticker, limit)

    def get_splits(self, ticker: str, limit: int = 10) -> list[dict]:
        return self._provider.get_splits(ticker, limit)

    def get_market_holidays(self) -> list[dict]:
        return self._provider.get_market_holidays()

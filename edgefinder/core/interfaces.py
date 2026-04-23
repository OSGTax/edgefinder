"""EdgeFinder v2 — Protocol definitions for all pluggable components.

Using Protocol (structural subtyping) instead of ABC so implementations
don't need to inherit from a base class. Any object matching the shape works.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from edgefinder.core.models import TickerFundamentals

logger = logging.getLogger(__name__)


@runtime_checkable
class DataProvider(Protocol):
    """Contract for market data + fundamentals providers.

    Polygon.io is the primary implementation. DataHub wraps this with
    supplemental enrichment from additional sources.
    """

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV bars.

        Returns DataFrame with columns: open, high, low, close, volume.
        Index is DatetimeIndex named 'timestamp'.
        Returns None on failure.
        """
        ...

    def get_latest_price(self, ticker: str) -> float | None:
        """Get the most recent trade/quote price."""
        ...

    def get_fundamentals(self, ticker: str, full_refresh: bool = False) -> TickerFundamentals | None:
        """Get fundamental data. full_refresh=True fetches quarterly data too."""
        ...

    def get_ticker_universe(
        self, min_market_cap: int = 0, min_volume: int = 0
    ) -> list[str]:
        """Return a list of ticker symbols meeting criteria."""
        ...

    def get_all_snapshots(self) -> dict[str, float]:
        """Get latest prices for ALL tickers in one API call.

        Returns dict of ticker -> close price.
        """
        ...

    def is_market_open(self) -> bool:
        """Whether the US equity market is currently in regular trading hours."""
        ...


class DataHub:
    """Thin passthrough wrapper for a DataProvider.

    Historically supported registering SupplementalProviders that enriched
    fundamentals after the primary fetch; that hook was never exercised and
    has been removed. DataHub now just forwards every call to its primary.
    Kept as a shell so callers can get supplementary methods (get_news,
    get_dividends, etc.) that aren't on the DataProvider protocol proper.
    """

    def __init__(self, primary: DataProvider) -> None:
        self._primary = primary

    # ── DataProvider interface (delegates to primary) ────

    def get_bars(
        self, ticker: str, timeframe: str, start: date, end: date | None = None
    ) -> pd.DataFrame | None:
        return self._primary.get_bars(ticker, timeframe, start, end)

    def get_latest_price(self, ticker: str) -> float | None:
        return self._primary.get_latest_price(ticker)

    def get_ticker_universe(
        self, min_market_cap: int = 0, min_volume: int = 0
    ) -> list[str]:
        return self._primary.get_ticker_universe(min_market_cap, min_volume)

    def get_top_dollar_volume_tickers(
        self,
        top_n: int = 1000,
        min_price: float = 5.0,
        max_price: float = 500.0,
    ) -> list[str]:
        return self._primary.get_top_dollar_volume_tickers(
            top_n=top_n, min_price=min_price, max_price=max_price,
        )

    def get_all_snapshots(self) -> dict[str, float]:
        return self._primary.get_all_snapshots()

    def is_market_open(self) -> bool:
        return self._primary.is_market_open()

    def get_news(self, ticker: str, limit: int = 10) -> list[dict]:
        if hasattr(self._primary, "get_news"):
            return self._primary.get_news(ticker, limit)
        return []

    def get_dividends(self, ticker: str, limit: int = 20) -> list[dict]:
        if hasattr(self._primary, "get_dividends"):
            return self._primary.get_dividends(ticker, limit)
        return []

    def get_splits(self, ticker: str, limit: int = 10) -> list[dict]:
        if hasattr(self._primary, "get_splits"):
            return self._primary.get_splits(ticker, limit)
        return []

    def get_market_holidays(self) -> list[dict]:
        if hasattr(self._primary, "get_market_holidays"):
            return self._primary.get_market_holidays()
        return []

    def get_fundamentals(self, ticker: str, full_refresh: bool = False) -> TickerFundamentals | None:
        return self._primary.get_fundamentals(ticker, full_refresh=full_refresh)



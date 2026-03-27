"""EdgeFinder v2 — Protocol definitions for all pluggable components.

Using Protocol (structural subtyping) instead of ABC so implementations
don't need to inherit from a base class. Any object matching the shape works.
"""

from __future__ import annotations

from datetime import date
from typing import Awaitable, Callable, Protocol, runtime_checkable

import pandas as pd

from edgefinder.core.models import BarData, TickerFundamentals, TickerSentiment


@runtime_checkable
class DataProvider(Protocol):
    """Contract for all market data providers.

    Polygon.io is the sole implementation. The protocol exists so consuming
    code never imports Polygon directly, and so tests can use mock providers.
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

    def get_fundamentals(self, ticker: str) -> TickerFundamentals | None:
        """Get fundamental data (financials, sector, market cap, etc.)."""
        ...

    def get_ticker_universe(
        self, min_market_cap: int = 0, min_volume: int = 0
    ) -> list[str]:
        """Return a list of ticker symbols meeting criteria."""
        ...

    def is_market_open(self) -> bool:
        """Whether the US equity market is currently in regular trading hours."""
        ...


@runtime_checkable
class StreamProvider(Protocol):
    """Contract for real-time WebSocket streaming."""

    async def connect(self) -> None:
        """Establish WebSocket connection and authenticate."""
        ...

    async def subscribe(
        self,
        tickers: list[str],
        on_bar: Callable[[BarData], Awaitable[None]] | None = None,
        on_trade: Callable[[dict], Awaitable[None]] | None = None,
        on_quote: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        """Subscribe to real-time data for given tickers."""
        ...

    async def unsubscribe(self, tickers: list[str]) -> None:
        """Unsubscribe from tickers."""
        ...

    async def disconnect(self) -> None:
        """Cleanly close the WebSocket connection."""
        ...


@runtime_checkable
class SentimentProvider(Protocol):
    """Contract for sentiment data sources (Reddit, Twitter, News)."""

    @property
    def source_name(self) -> str:
        """Identifier for this source (e.g., 'reddit', 'twitter', 'news')."""
        ...

    def get_sentiment(self, ticker: str) -> TickerSentiment | None:
        """Get current sentiment for a single ticker."""
        ...

    def get_trending(self) -> list[TickerSentiment]:
        """Get tickers that are currently trending on this source."""
        ...

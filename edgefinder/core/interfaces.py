"""EdgeFinder v2 — Protocol definitions for all pluggable components.

Using Protocol (structural subtyping) instead of ABC so implementations
don't need to inherit from a base class. Any object matching the shape works.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Awaitable, Callable, Protocol, runtime_checkable

import pandas as pd

from edgefinder.core.models import BarData, TickerFundamentals

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


@runtime_checkable
class SupplementalProvider(Protocol):
    """Contract for supplemental data sources that enrich fundamentals.

    Supplements fill fields that the primary provider (Polygon) doesn't
    supply: earnings calendar, analyst ratings, insider activity, etc.
    Each supplement enriches a TickerFundamentals in-place, filling only
    None fields (never overwriting primary data).

    To add a new data source:
    1. Create edgefinder/data/<source>.py implementing this protocol
    2. Add API key setting to config/settings.py
    3. Register in DataHub via register_supplement()
    """

    @property
    def source_name(self) -> str:
        """Unique identifier for this source (e.g., 'finnhub', 'fmp')."""
        ...

    @property
    def available_fields(self) -> list[str]:
        """TickerFundamentals field names this provider can populate."""
        ...

    def enrich(self, fund: TickerFundamentals) -> None:
        """Enrich fundamentals in-place. Fill None fields only."""
        ...


class DataHub:
    """Central registry for all data providers.

    Wraps a primary DataProvider (Polygon) and optional SupplementalProviders
    (Finnhub, FMP, etc.). All existing code that takes a DataProvider works
    with DataHub since it exposes the same interface.

    Fundamentals flow:
    1. Primary provider fetches core data (financials, ratios, company info)
    2. Each supplement enriches with fields the primary doesn't provide
    3. Failed supplements are logged and skipped (graceful degradation)
    """

    def __init__(self, primary: DataProvider) -> None:
        self._primary = primary
        self._supplements: list[SupplementalProvider] = []

    def register_supplement(self, provider: SupplementalProvider) -> None:
        """Register a supplemental data provider."""
        self._supplements.append(provider)
        logger.info(
            "Registered supplement '%s' (fields: %s)",
            provider.source_name, ", ".join(provider.available_fields),
        )

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

    def get_fundamentals(self, ticker: str, full_refresh: bool = False) -> TickerFundamentals | None:
        """Get fundamentals from primary, then enrich via supplements."""
        fund = self._primary.get_fundamentals(ticker, full_refresh=full_refresh)
        if fund is None:
            return None

        sources: dict[str, str] = {}
        for supplement in self._supplements:
            try:
                supplement.enrich(fund)
                for field_name in supplement.available_fields:
                    if getattr(fund, field_name, None) is not None:
                        sources[field_name] = supplement.source_name
            except Exception:
                logger.warning(
                    "Supplement '%s' failed for %s — skipping",
                    supplement.source_name, ticker, exc_info=True,
                )

        if sources:
            fund.data_sources = sources

        return fund


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



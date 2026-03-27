"""EdgeFinder v2 — Polygon.io data provider implementation.

This is the sole market data source. Covers: bars, fundamentals,
ticker universe, latest prices, and market status.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
from polygon import RESTClient

from config.settings import settings
from edgefinder.core.models import TickerFundamentals

logger = logging.getLogger(__name__)


class PolygonDataProvider:
    """Polygon.io implementation of the DataProvider protocol."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.polygon_api_key
        if not key:
            raise ValueError(
                "Polygon API key required. Set EDGEFINDER_POLYGON_API_KEY in .env"
            )
        self._client = RESTClient(api_key=key)
        self._max_retries = settings.polygon_max_retries
        self._retry_delay = settings.polygon_retry_delay

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV bars from Polygon aggregates endpoint."""
        multiplier, timespan = self._parse_timeframe(timeframe)
        end_date = end or date.today()

        aggs = self._retry(
            lambda: list(
                self._client.get_aggs(
                    ticker=ticker,
                    multiplier=multiplier,
                    timespan=timespan,
                    from_=start.isoformat(),
                    to=end_date.isoformat(),
                    limit=50000,
                )
            ),
            context=f"get_bars({ticker})",
        )
        if not aggs:
            return None
        return self._aggs_to_dataframe(aggs)

    def get_latest_price(self, ticker: str) -> float | None:
        """Get latest price via snapshot endpoint."""
        snapshot = self._retry(
            lambda: self._client.get_snapshot_ticker("stocks", ticker),
            context=f"get_latest_price({ticker})",
        )
        if not snapshot:
            return None
        if snapshot.day and snapshot.day.close:
            return float(snapshot.day.close)
        if snapshot.prev_day and snapshot.prev_day.close:
            return float(snapshot.prev_day.close)
        return None

    def get_fundamentals(self, ticker: str) -> TickerFundamentals | None:
        """Get fundamentals via ticker details + financials endpoints."""
        # Ticker details (sector, industry, market cap)
        details = self._retry(
            lambda: self._client.get_ticker_details(ticker),
            context=f"get_fundamentals.details({ticker})",
        )

        # Financials (experimental vX endpoint)
        financials = self._retry(
            lambda: next(
                self._client.vx.list_stock_financials(
                    ticker=ticker, limit=1, sort="period_of_report_date", order="desc"
                ),
                None,
            ),
            context=f"get_fundamentals.financials({ticker})",
        )

        return self._build_fundamentals(ticker, details, financials)

    def get_ticker_universe(
        self, min_market_cap: int = 0, min_volume: int = 0
    ) -> list[str]:
        """Get active US common stock tickers from Polygon."""
        tickers: list[str] = []
        try:
            for t in self._client.list_tickers(
                market="stocks",
                type="CS",
                active=True,
                locale="us",
                limit=1000,
            ):
                tickers.append(t.ticker)
        except Exception as e:
            logger.error("Polygon get_ticker_universe failed: %s", e)
        logger.info("Fetched %d tickers from Polygon universe", len(tickers))
        return tickers

    def is_market_open(self) -> bool:
        """Check market status via Polygon."""
        status = self._retry(
            lambda: self._client.get_market_status(),
            context="is_market_open",
        )
        if not status:
            return False
        return getattr(status, "market", "") == "open"

    # ── Private helpers ──────────────────────────────

    def _retry(self, fn, context: str = "") -> Any:
        """Retry a function with exponential backoff."""
        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as e:
                if attempt == self._max_retries - 1:
                    logger.error("Polygon %s failed after %d retries: %s", context, self._max_retries, e)
                    return None
                delay = self._retry_delay * (2 ** attempt)
                logger.warning("Polygon %s attempt %d failed, retrying in %.1fs: %s", context, attempt + 1, delay, e)
                time.sleep(delay)
        return None

    @staticmethod
    def _parse_timeframe(timeframe: str) -> tuple[int, str]:
        mapping = {
            "1": (1, "minute"),
            "5": (5, "minute"),
            "15": (15, "minute"),
            "60": (1, "hour"),
            "day": (1, "day"),
        }
        return mapping.get(timeframe, (1, "day"))

    @staticmethod
    def _aggs_to_dataframe(aggs: list) -> pd.DataFrame:
        data = [
            {
                "timestamp": datetime.fromtimestamp(a.timestamp / 1000),
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume,
                "vwap": getattr(a, "vwap", None),
                "trade_count": getattr(a, "transactions", None),
            }
            for a in aggs
        ]
        df = pd.DataFrame(data)
        df = df.set_index("timestamp")
        return df

    def _build_fundamentals(
        self, ticker: str, details: Any, financials: Any
    ) -> TickerFundamentals:
        """Build TickerFundamentals from Polygon API responses.

        Defensive parsing — missing fields become None.
        Raw data stored for research tool deep dive.
        """
        fund = TickerFundamentals(symbol=ticker)

        # From ticker details
        if details:
            fund.company_name = getattr(details, "name", None)
            fund.sector = getattr(details, "sic_description", None)
            fund.market_cap = getattr(details, "market_cap", None)
            # Polygon puts industry info in description or SIC codes
            fund.industry = getattr(details, "sic_description", None)
            fund.raw_data = {}
            # Store the full details for research
            for attr in ("name", "market_cap", "sic_code", "sic_description",
                         "total_employees", "list_date", "description",
                         "homepage_url", "locale", "primary_exchange"):
                val = getattr(details, attr, None)
                if val is not None:
                    fund.raw_data[attr] = val

        # From financials (experimental vX endpoint)
        if financials and hasattr(financials, "financials"):
            fin = financials.financials
            raw_fin = {}

            # Income statement
            if hasattr(fin, "income_statement"):
                inc = fin.income_statement
                revenues = getattr(inc, "revenues", None)
                if revenues:
                    raw_fin["revenues"] = getattr(revenues, "value", None)
                net_income = getattr(inc, "net_income_loss", None)
                if net_income:
                    raw_fin["net_income"] = getattr(net_income, "value", None)

            # Balance sheet
            if hasattr(fin, "balance_sheet"):
                bs = fin.balance_sheet
                total_assets = getattr(bs, "assets", None)
                if total_assets:
                    raw_fin["total_assets"] = getattr(total_assets, "value", None)
                total_liabilities = getattr(bs, "liabilities", None)
                if total_liabilities:
                    raw_fin["total_liabilities"] = getattr(total_liabilities, "value", None)
                equity = getattr(bs, "equity", None)
                if equity:
                    raw_fin["equity"] = getattr(equity, "value", None)
                current_assets = getattr(bs, "current_assets", None)
                if current_assets:
                    raw_fin["current_assets"] = getattr(current_assets, "value", None)
                current_liabilities = getattr(bs, "current_liabilities", None)
                if current_liabilities:
                    raw_fin["current_liabilities"] = getattr(current_liabilities, "value", None)

                # Derive current_ratio
                ca = raw_fin.get("current_assets")
                cl = raw_fin.get("current_liabilities")
                if ca and cl and cl != 0:
                    fund.current_ratio = ca / cl

                # Derive debt_to_equity
                tl = raw_fin.get("total_liabilities")
                eq = raw_fin.get("equity")
                if tl and eq and eq != 0:
                    fund.debt_to_equity = tl / eq

            # Cash flow statement
            if hasattr(fin, "cash_flow_statement"):
                cf = fin.cash_flow_statement
                op_cf = getattr(cf, "net_cash_flow_from_operating_activities", None)
                if op_cf:
                    raw_fin["operating_cash_flow"] = getattr(op_cf, "value", None)

            if fund.raw_data is None:
                fund.raw_data = {}
            fund.raw_data["financials"] = raw_fin

        return fund

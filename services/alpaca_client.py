"""
Alpaca Market Data Client
=========================
Handles all bar/quote data from Alpaca's free paper trading API.
Supports historical bars, latest quotes, and account info.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Alpaca API base URLs
PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"


class AlpacaClient:
    """Client for Alpaca Markets API (paper trading + market data)."""

    def __init__(self, api_key: str, secret_key: str):
        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API key and secret key are required. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in config/secrets.env"
            )
        self.api_key = api_key
        self.secret_key = secret_key
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._rate_limit_remaining = 200
        self._rate_limit_reset = 0.0

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[dict] = None,
        max_retries: int = 3,
    ) -> dict | list | None:
        """Make an API request with retry logic and rate limiting."""
        for attempt in range(max_retries):
            try:
                # Respect rate limits
                if self._rate_limit_remaining < 5:
                    wait = max(0, self._rate_limit_reset - time.time()) + 0.5
                    logger.info(f"Rate limit approaching, waiting {wait:.1f}s")
                    time.sleep(wait)

                resp = requests.request(
                    method, url, headers=self._headers, params=params, timeout=30
                )

                # Track rate limits from headers
                self._rate_limit_remaining = int(
                    resp.headers.get("X-Ratelimit-Remaining", 200)
                )
                reset_ts = resp.headers.get("X-Ratelimit-Reset")
                if reset_ts:
                    self._rate_limit_reset = float(reset_ts)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limited, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                elif resp.status_code == 422:
                    logger.warning(f"Alpaca validation error: {resp.text}")
                    return None
                else:
                    logger.error(
                        f"Alpaca API error {resp.status_code}: {resp.text}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(2 ** (attempt + 1))
                    continue

            except requests.exceptions.Timeout:
                logger.warning(f"Alpaca request timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
            except requests.exceptions.ConnectionError:
                logger.warning(f"Alpaca connection error (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))

        logger.error(f"Alpaca request failed after {max_retries} attempts: {url}")
        return None

    def get_account(self) -> Optional[dict]:
        """Get paper trading account info (validates credentials)."""
        return self._request("GET", f"{PAPER_BASE_URL}/v2/account")

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical bars for a ticker.

        Args:
            ticker: Stock symbol (e.g., "AAPL")
            timeframe: Bar size — "1Min", "5Min", "15Min", "1Hour", "1Day"
            start: Start date as RFC3339 or YYYY-MM-DD
            end: End date as RFC3339 or YYYY-MM-DD
            limit: Max bars to return (max 10000)

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap, trade_count
            Index is DatetimeIndex (UTC). Returns None on failure.
        """
        params = {
            "timeframe": timeframe,
            "limit": min(limit, 10000),
            "adjustment": "split",  # Split-adjusted prices
            "feed": "iex",  # Free tier uses IEX
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        url = f"{DATA_BASE_URL}/v2/stocks/{ticker}/bars"
        all_bars = []
        next_page_token = None

        while True:
            if next_page_token:
                params["page_token"] = next_page_token

            data = self._request("GET", url, params=params)
            if data is None:
                return None

            bars = data.get("bars") or []
            all_bars.extend(bars)

            next_page_token = data.get("next_page_token")
            if not next_page_token or len(all_bars) >= limit:
                break

        if not all_bars:
            logger.warning(f"No bars returned for {ticker}")
            return None

        df = pd.DataFrame(all_bars)
        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t")
        df = df.rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "trade_count",
        })
        df = df[["open", "high", "low", "close", "volume", "vwap", "trade_count"]]
        df.index.name = "timestamp"

        logger.info(f"Fetched {len(df)} bars for {ticker} ({timeframe})")
        return df

    def get_latest_quote(self, ticker: str) -> Optional[dict]:
        """Get the latest quote (bid/ask) for a ticker."""
        url = f"{DATA_BASE_URL}/v2/stocks/{ticker}/quotes/latest"
        params = {"feed": "iex"}
        data = self._request("GET", url, params=params)
        if data and "quote" in data:
            q = data["quote"]
            return {
                "bid": q.get("bp", 0),
                "ask": q.get("ap", 0),
                "bid_size": q.get("bs", 0),
                "ask_size": q.get("as", 0),
                "timestamp": q.get("t"),
            }
        return None

    def get_latest_bar(self, ticker: str) -> Optional[dict]:
        """Get the latest bar for a ticker."""
        url = f"{DATA_BASE_URL}/v2/stocks/{ticker}/bars/latest"
        params = {"feed": "iex"}
        data = self._request("GET", url, params=params)
        if data and "bar" in data:
            b = data["bar"]
            return {
                "open": b.get("o"),
                "high": b.get("h"),
                "low": b.get("l"),
                "close": b.get("c"),
                "volume": b.get("v"),
                "vwap": b.get("vw"),
                "timestamp": b.get("t"),
            }
        return None

    def get_snapshot(self, ticker: str) -> Optional[dict]:
        """Get a full snapshot (latest trade, quote, bar, prev bar) for a ticker."""
        url = f"{DATA_BASE_URL}/v2/stocks/{ticker}/snapshot"
        params = {"feed": "iex"}
        return self._request("GET", url, params=params)

    def get_multi_bars(
        self,
        tickers: list[str],
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple tickers in one call.

        Returns:
            Dict mapping ticker -> DataFrame. Missing tickers are omitted.
        """
        params = {
            "symbols": ",".join(tickers),
            "timeframe": timeframe,
            "limit": min(limit, 10000),
            "adjustment": "split",
            "feed": "iex",
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        url = f"{DATA_BASE_URL}/v2/stocks/bars"
        data = self._request("GET", url, params=params)
        if data is None:
            return {}

        result = {}
        bars_by_ticker = data.get("bars") or {}
        for sym, bars in bars_by_ticker.items():
            if bars:
                df = pd.DataFrame(bars)
                df["t"] = pd.to_datetime(df["t"])
                df = df.set_index("t")
                df = df.rename(columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "trade_count",
                })
                df = df[["open", "high", "low", "close", "volume", "vwap", "trade_count"]]
                df.index.name = "timestamp"
                result[sym] = df

        logger.info(f"Fetched bars for {len(result)}/{len(tickers)} tickers")
        return result

    def get_tradeable_assets(
        self,
        exchanges: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Get all tradeable US equities from Alpaca's Assets API.

        Returns list of dicts with: symbol, name, exchange, status, tradable.
        Single API call, no pagination needed — returns ~8,000+ assets.
        """
        if exchanges is None:
            exchanges = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
        else:
            exchanges = set(e.upper() for e in exchanges)

        data = self._request(
            "GET",
            f"{PAPER_BASE_URL}/v2/assets",
            params={"asset_class": "us_equity", "status": "active"},
        )
        if not data:
            return []

        assets = [
            {
                "symbol": a["symbol"],
                "name": a.get("name", ""),
                "exchange": a.get("exchange", ""),
                "tradable": a.get("tradable", False),
            }
            for a in data
            if a.get("tradable") and a.get("exchange", "") in exchanges
        ]
        logger.info(f"Alpaca assets: {len(assets)} tradeable equities")
        return assets

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        data = self._request("GET", f"{DATA_BASE_URL}/v2/stocks/clock")
        # Fallback to paper trading endpoint
        if data is None:
            data = self._request("GET", f"{PAPER_BASE_URL}/v2/clock")
        if data:
            return data.get("is_open", False)
        return False

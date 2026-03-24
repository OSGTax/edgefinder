"""
Unified Data Service
====================
Single entry point for all market data. Handles:
  - Cache-first lookups (check local cache before hitting APIs)
  - Automatic fallback: Alpaca → yfinance (for bars)
  - Automatic fallback: FMP → yfinance (for fundamentals)
  - All results cached for future use

Usage:
    from services import DataService
    ds = DataService()
    bars = ds.get_bars("AAPL", timeframe="1Day", days_back=30)
    profile = ds.get_profile("AAPL")
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from services.alpaca_client import AlpacaClient
from services.cache import DataCache

logger = logging.getLogger(__name__)


class DataService:
    """
    Unified data interface with cache-first lookups.

    Data flow for bars:  Cache → Alpaca → yfinance (fallback)
    Data flow for fundamentals:  Cache → FMP → yfinance (fallback)
    """

    def __init__(
        self,
        alpaca_key: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
        cache_path: str = "data/cache.db",
    ):
        # Load secrets from env file
        secrets_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "secrets.env"
        )
        if os.path.exists(secrets_path):
            load_dotenv(secrets_path)

        # Initialize cache (always available)
        self.cache = DataCache(cache_path)

        # Initialize Alpaca (optional — bars fallback to yfinance)
        ak = alpaca_key or os.getenv("ALPACA_API_KEY", "")
        ask = alpaca_secret or os.getenv("ALPACA_SECRET_KEY", "")
        self.alpaca: Optional[AlpacaClient] = None
        if ak and ask:
            try:
                self.alpaca = AlpacaClient(ak, ask)
                logger.info("Alpaca client initialized")
            except ValueError as e:
                logger.warning(f"Alpaca not configured: {e}")

        # FMP removed — yfinance provides fundamentals (faster, no rate limits)
        self.fmp = None

        # Track which sources are available
        self._sources = {
            "alpaca": self.alpaca is not None,
            "yfinance": True,  # Fundamentals + bar fallback
            "cache": True,
        }
        logger.info(f"Data sources: {self._sources}")

    @property
    def available_sources(self) -> dict[str, bool]:
        """Which data sources are configured and available."""
        return self._sources.copy()

    def get_diagnostics(self) -> dict:
        """Return diagnostic info about all data sources."""
        return {
            "sources": self._sources.copy(),
            "fundamentals_source": "yfinance",
            "bars_source": "alpaca" if self.alpaca else "yfinance",
            "cache_stats": self.cache.get_stats() if self.cache else {},
        }

    # ── BAR DATA ────────────────────────────────────────────

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "1Day",
        days_back: int = 30,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical bars. Cache → Alpaca → yfinance.

        Args:
            ticker: Stock symbol
            timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
            days_back: How many days of history (used if start/end not provided)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            use_cache: Whether to check cache first

        Returns:
            DataFrame with OHLCV data, or None on failure.
        """
        # Calculate date range
        if not start:
            start_dt = datetime.utcnow() - timedelta(days=days_back)
            start = start_dt.strftime("%Y-%m-%d")

        # 1. Check cache
        if use_cache:
            start_dt = datetime.strptime(start, "%Y-%m-%d") if isinstance(start, str) else start
            end_dt = datetime.strptime(end, "%Y-%m-%d") if end and isinstance(end, str) else None
            cached = self.cache.get_bars(ticker, timeframe, start_dt, end_dt)
            if cached is not None and len(cached) > 0:
                logger.debug(f"Cache hit: {len(cached)} bars for {ticker}")
                return cached

        # 2. Try Alpaca
        if self.alpaca:
            df = self.alpaca.get_bars(ticker, timeframe=timeframe, start=start, end=end)
            if df is not None and not df.empty:
                self.cache.store_bars(ticker, timeframe, df)
                return df
            logger.debug(f"Alpaca miss for {ticker}, trying yfinance")

        # 3. Fallback to yfinance
        df = self._yfinance_bars(ticker, timeframe, start, end)
        if df is not None and not df.empty:
            self.cache.store_bars(ticker, timeframe, df)
            return df

        logger.warning(f"All sources failed for {ticker} bars")
        return None

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the most recent price for a ticker. Fast path.

        Tries: Alpaca quote → yfinance fast_info → cached bars (last close).
        """
        # 1. Alpaca latest quote
        if self.alpaca:
            quote = self.alpaca.get_latest_quote(ticker)
            if quote:
                # Use midpoint of bid/ask
                bid = quote.get("bid", 0) or 0
                ask = quote.get("ask", 0) or 0
                if bid > 0 and ask > 0:
                    return round((bid + ask) / 2, 4)

        # 2. yfinance fast_info
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            price = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice")
            if price and price > 0:
                return round(float(price), 4)
        except Exception as e:
            logger.debug(f"yfinance price lookup failed for {ticker}: {e}")

        # 3. Last cached close
        cached = self.cache.get_bars(ticker, "1Day")
        if cached is not None and not cached.empty:
            return round(float(cached.iloc[-1]["close"]), 4)

        return None

    def get_multi_bars(
        self,
        tickers: list[str],
        timeframe: str = "1Day",
        days_back: int = 30,
    ) -> dict[str, pd.DataFrame]:
        """
        Get bars for multiple tickers. Uses Alpaca batch API when available.

        Returns dict mapping ticker -> DataFrame.
        """
        start = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        result = {}
        uncached = []

        # Check cache first for each ticker
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        for ticker in tickers:
            cached = self.cache.get_bars(ticker, timeframe, start_dt)
            if cached is not None and len(cached) > 0:
                result[ticker] = cached
            else:
                uncached.append(ticker)

        if not uncached:
            return result

        # Batch fetch uncached from Alpaca
        if self.alpaca and uncached:
            batch = self.alpaca.get_multi_bars(
                uncached, timeframe=timeframe, start=start
            )
            for ticker, df in batch.items():
                if df is not None and not df.empty:
                    self.cache.store_bars(ticker, timeframe, df)
                    result[ticker] = df
                    uncached.remove(ticker)

        # Fallback remaining to yfinance one by one
        for ticker in uncached:
            df = self._yfinance_bars(ticker, timeframe, start)
            if df is not None and not df.empty:
                self.cache.store_bars(ticker, timeframe, df)
                result[ticker] = df

        logger.info(f"Got bars for {len(result)}/{len(tickers)} tickers")
        return result

    # ── FUNDAMENTAL DATA ────────────────────────────────────

    def get_profile(self, ticker: str, use_cache: bool = True) -> Optional[dict]:
        """
        Get company profile via yfinance.

        Returns dict with: symbol, companyName, sector, industry, mktCap, etc.
        """
        # 1. Cache
        if use_cache:
            cached = self.cache.get_fundamental(ticker, "profile")
            if cached:
                return cached

        # 2. yfinance
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            if info and info.get("marketCap"):
                data = {
                    "symbol": ticker,
                    "companyName": info.get("shortName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "mktCap": info.get("marketCap"),
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "volAvg": info.get("averageVolume"),
                }
                self.cache.store_fundamental(ticker, "profile", data)
                return data
        except Exception as e:
            logger.debug(f"yfinance profile failed for {ticker}: {e}")

        return None

    def get_fundamentals(self, ticker: str, use_cache: bool = True) -> Optional[dict]:
        """
        Get comprehensive fundamental data for scoring via yfinance.

        Single yfinance .info call (~0.2s) provides all fields needed by
        the scanner's Lynch + Burry scoring. No FMP dependency.

        Returns dict with all fields needed by scanner.score_stock().
        """
        # 1. Cache
        if use_cache:
            cached = self.cache.get_fundamental(ticker, "combined_fundamentals")
            if cached:
                return cached

        # 2. yfinance — single .info call gets everything
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info
            if not info or not info.get("marketCap"):
                return None

            # D/E: yfinance returns as percentage (e.g. 102.63 = 1.0263 ratio)
            de_raw = info.get("debtToEquity")
            debt_to_equity = de_raw / 100.0 if de_raw is not None else None

            # PEG: use direct value, or calculate from P/E and growth
            peg = info.get("pegRatio")
            if peg is None:
                trailing_pe = info.get("trailingPE")
                eg = info.get("earningsGrowth")
                if trailing_pe and eg and eg > 0:
                    peg = trailing_pe / (eg * 100)  # eg is decimal (0.18 = 18%)

            result = {
                "ticker": ticker,
                "company_name": info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "avg_volume": info.get("averageVolume"),
                "exchange": info.get("exchange"),
                # Lynch fields
                "peg_ratio": peg,
                "earnings_growth": info.get("earningsGrowth"),
                "debt_to_equity": debt_to_equity,
                "revenue_growth": info.get("revenueGrowth"),
                "institutional_pct": info.get("heldPercentInstitutions"),
                # Burry fields
                "fcf_yield": info.get("freeCashflow") / info.get("marketCap")
                    if info.get("freeCashflow") and info.get("marketCap")
                    else None,
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "current_ratio": info.get("currentRatio"),
                "short_interest": info.get("shortPercentOfFloat"),
                "price_to_tangible_book": info.get("priceToBook"),
            }

            self.cache.store_fundamental(ticker, "combined_fundamentals", result)
            return result

        except Exception as e:
            logger.debug(f"yfinance fundamentals failed for {ticker}: {e}")
            return None

    def get_earnings_calendar(
        self, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> Optional[list]:
        """Get upcoming earnings dates. Currently unsupported without FMP."""
        return None

    # ── MARKET STATUS ───────────────────────────────────────

    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open."""
        if self.alpaca:
            return self.alpaca.is_market_open()
        # Fallback: simple time check
        from datetime import timezone
        now = datetime.now(timezone.utc)
        # Rough check: M-F, 14:30-21:00 UTC (9:30-4:00 ET)
        if now.weekday() >= 5:
            return False
        return 14 * 60 + 30 <= now.hour * 60 + now.minute <= 21 * 60

    # ── CACHE MANAGEMENT ────────────────────────────────────

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self, ticker: Optional[str] = None) -> int:
        """Clear cache for a ticker, or all cache if ticker is None."""
        if ticker:
            return self.cache.clear_ticker(ticker)
        return self.cache.clear_all()

    def cleanup_cache(self) -> int:
        """Remove expired cache entries."""
        return self.cache.clear_stale()

    # ── YFINANCE FALLBACKS ──────────────────────────────────

    def _yfinance_bars(
        self,
        ticker: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch bars from yfinance as fallback."""
        try:
            import yfinance as yf

            # Map Alpaca timeframes to yfinance intervals
            interval_map = {
                "1Min": "1m",
                "5Min": "5m",
                "15Min": "15m",
                "1Hour": "1h",
                "1Day": "1d",
            }
            interval = interval_map.get(timeframe, "1d")

            # yfinance has limits on intraday history
            if interval in ("1m",) and not start:
                start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
            elif interval in ("5m", "15m") and not start:
                start = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")

            t = yf.Ticker(ticker)
            df = t.history(start=start, end=end, interval=interval, auto_adjust=True)

            if df is None or df.empty:
                return None

            # Normalize column names to match Alpaca format
            df.columns = [c.lower() for c in df.columns]
            df.index.name = "timestamp"

            # Keep only OHLCV columns
            cols = ["open", "high", "low", "close", "volume"]
            available = [c for c in cols if c in df.columns]
            df = df[available]

            # Add missing columns as None
            for c in ["vwap", "trade_count"]:
                if c not in df.columns:
                    df[c] = None

            logger.debug(f"yfinance: {len(df)} bars for {ticker} ({timeframe})")
            return df

        except Exception as e:
            logger.warning(f"yfinance bars failed for {ticker}: {e}")
            return None


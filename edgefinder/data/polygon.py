"""EdgeFinder v2 — Massive (formerly Polygon.io) data provider implementation.

Primary market data source using the Massive Starter plan ($30/mo, unlimited calls).
Covers: bars, fundamentals, pre-computed ratios, Benzinga earnings/analyst data,
short interest, dividends, splits, news with sentiment, related companies,
market status, and market holidays.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
from massive import RESTClient

from config.settings import settings
from edgefinder.core.models import TickerFundamentals

logger = logging.getLogger(__name__)


class PolygonDataProvider:
    """Massive (Polygon.io) implementation of the DataProvider protocol.

    Uses every relevant endpoint available on the Starter plan to maximize
    data coverage for strategy screening and trade decisions.
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.polygon_api_key
        if not key:
            raise ValueError(
                "Polygon API key required. Set EDGEFINDER_POLYGON_API_KEY in .env"
            )
        self._client = RESTClient(api_key=key, num_pools=4, retries=0)
        self._max_retries = settings.polygon_max_retries
        self._retry_delay = settings.polygon_retry_delay
        # Track which endpoints are not authorized on this plan
        self._disabled_endpoints: set[str] = set()
        # Plan access probe results (populated by probe_plan_access)
        self._plan_access: dict[str, bool] = {}

    def probe_plan_access(self) -> dict[str, bool]:
        """Test each endpoint once at startup to determine plan access.

        Returns dict of endpoint_name -> available (True/False).
        Pre-populates _disabled_endpoints so scanning doesn't waste retries.
        """
        test_ticker = "AAPL"
        probes = {
            "ratios": lambda: list(self._client.list_financials_ratios(ticker=test_ticker, limit=1)),
            "earnings": lambda: list(self._client.list_benzinga_earnings(ticker=test_ticker, limit=1)),
            "analyst": lambda: list(self._client.list_benzinga_consensus_ratings(ticker=test_ticker, limit=1)),
            "short_interest": lambda: list(self._client.list_short_interest(ticker=test_ticker, limit=1)),
            "financials_bs": lambda: list(self._client.list_financials_balance_sheets(tickers=test_ticker, limit=1)),
            "financials_is": lambda: list(self._client.list_financials_income_statements(tickers=test_ticker, limit=1)),
            "financials_cf": lambda: list(self._client.list_financials_cash_flow_statements(tickers=test_ticker, limit=1)),
            "financials_vx": lambda: list(self._client.vx.list_stock_financials(ticker=test_ticker, limit=1)),
            "technical_rsi": lambda: self._client.get_rsi(test_ticker, timespan="day", window=14, limit=1),
            "technical_ema": lambda: self._client.get_ema(test_ticker, timespan="day", window=21, limit=1),
            "technical_macd": lambda: self._client.get_macd(test_ticker, timespan="day", limit=1),
            "dividends": lambda: list(self._client.list_dividends(ticker=test_ticker, limit=1)),
            "splits": lambda: list(self._client.list_splits(ticker=test_ticker, limit=1)),
            "news": lambda: list(self._client.list_ticker_news(ticker=test_ticker, limit=1)),
            "related": lambda: self._client.get_related_companies(ticker=test_ticker),
            "ticker_events": lambda: self._client.get_ticker_events(test_ticker),
        }

        results = {}
        for name, fn in probes.items():
            try:
                fn()
                results[name] = True
                logger.info("  ✓ %s — available", name)
            except Exception as e:
                err = str(e)
                if "NOT_AUTHORIZED" in err or "not entitled" in err.lower():
                    results[name] = False
                    self._disabled_endpoints.add(name)
                    logger.warning("  ✗ %s — NOT on current plan", name)
                else:
                    # Other errors (network, etc.) — assume available
                    results[name] = True
                    logger.info("  ? %s — error but not plan-gated: %s", name, err[:80])

        self._plan_access = results
        available = sum(1 for v in results.values() if v)
        blocked = sum(1 for v in results.values() if not v)
        logger.info("Plan probe complete: %d available, %d blocked", available, blocked)
        return results

    # ── Technical Indicators (from Massive API) ──────

    def get_technical_rsi(self, ticker: str, timespan: str = "day", window: int = 14) -> float | None:
        """Get RSI from Massive's pre-computed technical indicators."""
        result = self._retry(
            lambda: self._client.get_rsi(ticker, timespan=timespan, window=window, limit=1),
            context=f"technical_rsi({ticker})",
        )
        if result and hasattr(result, "values") and result.values:
            return getattr(result.values[0], "value", None)
        return None

    def get_technical_ema(self, ticker: str, timespan: str = "day", window: int = 21) -> float | None:
        """Get EMA from Massive's pre-computed technical indicators."""
        result = self._retry(
            lambda: self._client.get_ema(ticker, timespan=timespan, window=window, limit=1),
            context=f"technical_ema({ticker})",
        )
        if result and hasattr(result, "values") and result.values:
            return getattr(result.values[0], "value", None)
        return None

    def get_technical_sma(self, ticker: str, timespan: str = "day", window: int = 50) -> float | None:
        """Get SMA from Massive's pre-computed technical indicators."""
        result = self._retry(
            lambda: self._client.get_sma(ticker, timespan=timespan, window=window, limit=1),
            context=f"technical_sma({ticker})",
        )
        if result and hasattr(result, "values") and result.values:
            return getattr(result.values[0], "value", None)
        return None

    def get_technical_macd(self, ticker: str, timespan: str = "day") -> dict | None:
        """Get MACD from Massive's pre-computed technical indicators."""
        result = self._retry(
            lambda: self._client.get_macd(ticker, timespan=timespan, limit=1),
            context=f"technical_macd({ticker})",
        )
        if result and hasattr(result, "values") and result.values:
            val = result.values[0]
            return {
                "value": getattr(val, "value", None),
                "signal": getattr(val, "signal", None),
                "histogram": getattr(val, "histogram", None),
            }
        return None

    # ── Core DataProvider Methods ────────────────────

    def get_bars(
        self,
        ticker: str,
        timeframe: str,
        start: date,
        end: date | None = None,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV bars from aggregates endpoint."""
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
        """Get comprehensive fundamentals from multiple Massive endpoints.

        Combines: ticker details, pre-computed ratios, raw financials (for YoY growth),
        Benzinga earnings calendar, analyst consensus, short interest, dividends,
        related companies, and news sentiment.
        """
        fund = TickerFundamentals(symbol=ticker)
        fund.raw_data = {}

        # 1. Ticker details (sector, market cap, company info)
        self._fill_ticker_details(fund)

        # 2. Pre-computed financial ratios (replaces manual calculation)
        self._fill_ratios(fund)

        # 3. Raw financials for YoY growth (ratios API doesn't include growth rates)
        self._fill_growth_metrics(fund)

        # 4. Benzinga earnings calendar
        self._fill_earnings(fund)

        # 5. Benzinga analyst consensus
        self._fill_analyst_consensus(fund)

        # 6. Short interest
        self._fill_short_interest(fund)

        # 7. Dividends
        self._fill_dividends(fund)

        # 8. Related companies
        self._fill_related_companies(fund)

        # 9. News with sentiment
        self._fill_news_sentiment(fund)

        return fund

    def get_ticker_universe(
        self, min_market_cap: int = 0, min_volume: int = 0
    ) -> list[str]:
        """Get active US common stock tickers."""
        tickers: list[str] = []
        try:
            for t in self._client.list_tickers(
                market="stocks",
                type="CS",
                active=True,
                limit=1000,
            ):
                tickers.append(t.ticker)
        except Exception as e:
            logger.error("get_ticker_universe failed: %s", e)
        logger.info("Fetched %d tickers from Massive universe", len(tickers))
        return tickers

    def is_market_open(self) -> bool:
        """Check market status."""
        status = self._retry(
            lambda: self._client.get_market_status(),
            context="is_market_open",
        )
        if not status:
            return False
        return getattr(status, "market", "") == "open"

    # ── Standalone Data Methods (for use outside get_fundamentals) ──

    def get_market_holidays(self) -> list[dict]:
        """Get upcoming market holidays and early closes."""
        holidays = self._retry(
            lambda: self._client.get_market_holidays(),
            context="get_market_holidays",
        )
        if not holidays:
            return []
        result = []
        for h in holidays:
            result.append({
                "date": getattr(h, "date", None),
                "name": getattr(h, "name", None),
                "status": getattr(h, "status", None),
                "exchange": getattr(h, "exchange", None),
                "open": getattr(h, "open", None),
                "close": getattr(h, "close", None),
            })
        return result

    def get_snapshot_enriched(self, ticker: str) -> dict | None:
        """Get full snapshot data (day OHLCV, prev day, bid/ask, change %)."""
        snapshot = self._retry(
            lambda: self._client.get_snapshot_ticker("stocks", ticker),
            context=f"get_snapshot_enriched({ticker})",
        )
        if not snapshot:
            return None
        result = {}
        if snapshot.day:
            for attr in ("open", "high", "low", "close", "volume", "vwap"):
                result[f"day_{attr}"] = getattr(snapshot.day, attr, None)
        if snapshot.prev_day:
            for attr in ("open", "high", "low", "close", "volume", "vwap"):
                result[f"prev_{attr}"] = getattr(snapshot.prev_day, attr, None)
        result["change"] = getattr(snapshot, "todaysChange", None)
        result["change_pct"] = getattr(snapshot, "todaysChangePerc", None)
        return result

    # ── Private: Fill TickerFundamentals from each endpoint ──

    def _fill_ticker_details(self, fund: TickerFundamentals) -> None:
        """Fill company info from ticker details endpoint."""
        details = self._retry(
            lambda: self._client.get_ticker_details(fund.symbol),
            context=f"details({fund.symbol})",
        )
        if not details:
            return

        fund.company_name = getattr(details, "name", None)
        fund.sector = getattr(details, "sic_description", None)
        fund.market_cap = getattr(details, "market_cap", None)
        # Use category for more granular industry if available
        fund.industry = getattr(details, "category", None) or getattr(details, "sic_description", None)

        # Store rich details in raw_data
        for attr in ("name", "market_cap", "sic_code", "sic_description",
                     "total_employees", "list_date", "description",
                     "homepage_url", "locale", "primary_exchange",
                     "cik", "share_class_shares_outstanding",
                     "weighted_shares_outstanding"):
            val = getattr(details, attr, None)
            if val is not None:
                fund.raw_data[attr] = val

    def _fill_ratios(self, fund: TickerFundamentals) -> None:
        """Fill pre-computed ratios from Massive's ratios endpoint.

        Replaces manual ratio calculation — more accurate, more ratios, less code.
        """
        ratios_list = self._retry(
            lambda: list(self._client.list_financials_ratios(
                ticker=fund.symbol, limit=1,
            )),
            context=f"ratios({fund.symbol})",
        )
        if not ratios_list:
            return

        r = ratios_list[0]

        # Map Massive ratio fields to TickerFundamentals fields
        fund.price = fund.price or getattr(r, "price", None)
        fund.market_cap = fund.market_cap or getattr(r, "market_cap", None)
        fund.price_to_earnings = getattr(r, "price_to_earnings", None)
        fund.price_to_book = getattr(r, "price_to_book", None)
        fund.price_to_sales = getattr(r, "price_to_sales", None)
        fund.price_to_free_cash_flow = getattr(r, "price_to_free_cash_flow", None)
        fund.ev_to_ebitda = getattr(r, "ev_to_ebitda", None)
        fund.ev_to_sales = getattr(r, "ev_to_sales", None)
        fund.dividend_yield = getattr(r, "dividend_yield", None)
        fund.return_on_assets = getattr(r, "return_on_assets", None)
        fund.return_on_equity = getattr(r, "return_on_equity", None)
        fund.debt_to_equity = getattr(r, "debt_to_equity", None)
        fund.current_ratio = getattr(r, "current", None)
        fund.quick_ratio = getattr(r, "quick", None)
        fund.enterprise_value = getattr(r, "enterprise_value", None)
        fund.free_cash_flow = getattr(r, "free_cash_flow", None)
        fund.earnings_per_share = getattr(r, "earnings_per_share", None)

        # Derive additional ratios
        if fund.price and fund.market_cap and fund.market_cap > 0:
            shares = fund.market_cap / fund.price
            if fund.free_cash_flow and shares > 0:
                fund.fcf_yield = fund.free_cash_flow / fund.market_cap

        # Derive price_to_tangible_book from price_to_book as proxy
        fund.price_to_tangible_book = fund.price_to_tangible_book or fund.price_to_book

        fund.raw_data["ratios"] = {
            attr: getattr(r, attr, None)
            for attr in r.__annotations__
            if getattr(r, attr, None) is not None
        }

    def _fill_growth_metrics(self, fund: TickerFundamentals) -> None:
        """Compute YoY growth and fallback ratios from raw financial statements.

        Always runs (even if ratios API succeeded) because:
        1. Ratios API doesn't include growth rates
        2. If ratios API is plan-gated, this provides fallback for D/E, current ratio, etc.

        Only fills fields that are still None (doesn't overwrite ratios API data).
        """
        financials_list = self._retry(
            lambda: list(
                self._client.vx.list_stock_financials(
                    ticker=fund.symbol, limit=2, timeframe="annual",
                    sort="period_of_report_date", order="desc",
                )
            ),
            context=f"financials({fund.symbol})",
        )
        if not financials_list:
            return

        curr = self._extract_financial_values(financials_list[0])
        if not curr:
            return

        # ── Fallback ratios (only if ratios API didn't populate them) ──
        mc = fund.market_cap

        # Current ratio
        ca = curr.get("current_assets")
        cl = curr.get("current_liabilities")
        if fund.current_ratio is None and ca and cl and cl != 0:
            fund.current_ratio = ca / cl

        # Debt to equity
        tl = curr.get("total_liabilities")
        eq = curr.get("equity")
        if fund.debt_to_equity is None and tl and eq and eq != 0:
            fund.debt_to_equity = tl / eq

        # FCF yield
        ocf = curr.get("operating_cash_flow")
        if fund.fcf_yield is None and ocf is not None and mc and mc > 0:
            fund.fcf_yield = ocf / mc

        # Price to tangible book
        ta = curr.get("total_assets")
        tl2 = curr.get("total_liabilities")
        if fund.price_to_tangible_book is None and ta is not None and tl2 is not None and mc:
            tangible_book = ta - tl2
            if tangible_book > 0:
                fund.price_to_tangible_book = mc / tangible_book

        # P/E ratio
        ni = curr.get("net_income")
        if fund.price_to_earnings is None and mc and ni and ni > 0:
            fund.price_to_earnings = mc / ni

        # ROE
        if fund.return_on_equity is None and ni and eq and eq != 0:
            fund.return_on_equity = ni / eq

        # ROA
        if fund.return_on_assets is None and ni and ta and ta != 0:
            fund.return_on_assets = ni / ta

        # ── YoY growth (always computed from 2-period comparison) ──
        prev = self._extract_financial_values(financials_list[1]) if len(financials_list) >= 2 else {}

        if prev:
            curr_ni = curr.get("net_income")
            prev_ni = prev.get("net_income")
            if curr_ni is not None and prev_ni is not None and prev_ni != 0:
                fund.earnings_growth = (curr_ni - prev_ni) / abs(prev_ni)

            curr_rev = curr.get("revenues")
            prev_rev = prev.get("revenues")
            if curr_rev is not None and prev_rev is not None and prev_rev != 0:
                fund.revenue_growth = (curr_rev - prev_rev) / abs(prev_rev)

        # PEG ratio
        if (fund.peg_ratio is None and mc and ni and ni > 0 and
                fund.earnings_growth is not None and fund.earnings_growth > 0):
            pe_ratio = mc / ni
            fund.peg_ratio = pe_ratio / (fund.earnings_growth * 100)

        fund.raw_data["financials"] = curr

    def _fill_earnings(self, fund: TickerFundamentals) -> None:
        """Fill earnings calendar from Benzinga earnings endpoint."""
        today = date.today()
        earnings_list = self._retry(
            lambda: list(self._client.list_benzinga_earnings(
                ticker=fund.symbol,
                date_gte=(today - timedelta(days=120)).isoformat(),
                date_lte=(today + timedelta(days=120)).isoformat(),
                limit=10,
            )),
            context=f"earnings({fund.symbol})",
        )
        if not earnings_list:
            return

        past = []
        future = []
        for e in earnings_list:
            d = getattr(e, "date", None)
            if not d:
                continue
            try:
                earnings_date = datetime.strptime(d, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                continue
            if earnings_date <= today:
                past.append((earnings_date, e))
            else:
                future.append((earnings_date, e))

        # Last earnings
        if past:
            past.sort(key=lambda x: x[0], reverse=True)
            last_date, last_earning = past[0]
            fund.last_earnings_date = last_date.isoformat()
            fund.eps_surprise_pct = getattr(last_earning, "eps_surprise_percent", None)
            fund.revenue_surprise_pct = getattr(last_earning, "revenue_surprise_percent", None)

        # Next earnings
        if future:
            future.sort(key=lambda x: x[0])
            next_date, next_earning = future[0]
            fund.estimated_next_earnings_date = next_date.isoformat()
            fund.estimated_eps = getattr(next_earning, "estimated_eps", None)
            days_until = (next_date - today).days
            fund.near_earnings = days_until <= settings.earnings_blackout_days

        fund.raw_data["earnings"] = [
            {attr: getattr(e, attr, None) for attr in e.__annotations__}
            for e in earnings_list[:5]
        ]

    def _fill_analyst_consensus(self, fund: TickerFundamentals) -> None:
        """Fill analyst consensus from Benzinga ratings endpoint."""
        consensus_list = self._retry(
            lambda: list(self._client.list_benzinga_consensus_ratings(
                ticker=fund.symbol, limit=1,
            )),
            context=f"analyst({fund.symbol})",
        )
        if not consensus_list:
            return

        c = consensus_list[0]
        buy = (getattr(c, "strong_buy_ratings", 0) or 0) + (getattr(c, "buy_ratings", 0) or 0)
        hold = getattr(c, "hold_ratings", 0) or 0
        sell = (getattr(c, "strong_sell_ratings", 0) or 0) + (getattr(c, "sell_ratings", 0) or 0)
        total = buy + hold + sell

        fund.analyst_buy_count = buy
        fund.analyst_hold_count = hold
        fund.analyst_sell_count = sell

        if total > 0:
            if buy / total > 0.5:
                fund.analyst_rating = "buy"
            elif sell / total > 0.3:
                fund.analyst_rating = "sell"
            else:
                fund.analyst_rating = "hold"

        fund.analyst_target_price = getattr(c, "consensus_price_target", None)

    def _fill_short_interest(self, fund: TickerFundamentals) -> None:
        """Fill short interest data."""
        si_list = self._retry(
            lambda: list(self._client.list_short_interest(
                ticker=fund.symbol, limit=1,
            )),
            context=f"short_interest({fund.symbol})",
        )
        if not si_list:
            return

        si = si_list[0]
        fund.short_shares = getattr(si, "short_interest", None)
        fund.days_to_cover = getattr(si, "days_to_cover", None)

        # Calculate short interest as percentage if we have shares outstanding
        shares_outstanding = fund.raw_data.get("share_class_shares_outstanding")
        if fund.short_shares and shares_outstanding and shares_outstanding > 0:
            fund.short_interest = fund.short_shares / shares_outstanding

    def _fill_dividends(self, fund: TickerFundamentals) -> None:
        """Fill most recent dividend data."""
        div_list = self._retry(
            lambda: list(self._client.list_dividends(
                ticker=fund.symbol, limit=1,
                sort="ex_dividend_date", order="desc",
            )),
            context=f"dividends({fund.symbol})",
        )
        if not div_list:
            return

        d = div_list[0]
        fund.dividend_amount = getattr(d, "cash_amount", None)
        fund.ex_dividend_date = getattr(d, "ex_dividend_date", None)

    def _fill_related_companies(self, fund: TickerFundamentals) -> None:
        """Fill related/competitor tickers."""
        related = self._retry(
            lambda: self._client.get_related_companies(ticker=fund.symbol),
            context=f"related({fund.symbol})",
        )
        if not related:
            return

        # The API returns a single object or list depending on version
        if isinstance(related, list):
            fund.related_tickers = [getattr(r, "ticker", None) for r in related if getattr(r, "ticker", None)]
        elif hasattr(related, "results"):
            fund.related_tickers = [getattr(r, "ticker", None) for r in related.results if getattr(r, "ticker", None)]
        elif hasattr(related, "ticker"):
            fund.related_tickers = [related.ticker]

    def _fill_news_sentiment(self, fund: TickerFundamentals) -> None:
        """Fill news sentiment from Polygon's ticker news with AI insights."""
        news_list = self._retry(
            lambda: list(self._client.list_ticker_news(
                ticker=fund.symbol, limit=10,
            )),
            context=f"news({fund.symbol})",
        )
        if not news_list:
            return

        fund.recent_news_count = len(news_list)

        # Aggregate sentiment from Polygon's built-in AI insights
        sentiments = []
        for article in news_list:
            insights = getattr(article, "insights", None)
            if not insights:
                continue
            for insight in insights:
                if getattr(insight, "ticker", None) == fund.symbol:
                    sent = getattr(insight, "sentiment", None)
                    if sent:
                        sentiments.append(sent.lower())

        if sentiments:
            pos = sum(1 for s in sentiments if s == "positive")
            neg = sum(1 for s in sentiments if s == "negative")
            if pos > neg:
                fund.news_sentiment = "positive"
            elif neg > pos:
                fund.news_sentiment = "negative"
            else:
                fund.news_sentiment = "neutral"

        # Store headlines in raw_data for research
        fund.raw_data["news"] = [
            {
                "title": getattr(a, "title", None),
                "published": getattr(a, "published_utc", None),
                "url": getattr(a, "article_url", None),
            }
            for a in news_list[:5]
        ]

    # ── Private helpers ──────────────────────────────

    def _retry(self, fn, context: str = "") -> Any:
        """Retry a function with exponential backoff.

        Auto-disables endpoints that return NOT_AUTHORIZED to avoid
        wasting retries on every ticker for plan-gated features.
        """
        # Check if this endpoint type is already known to be unauthorized
        endpoint_type = context.split("(")[0] if "(" in context else context
        if endpoint_type in self._disabled_endpoints:
            return None

        for attempt in range(self._max_retries):
            try:
                return fn()
            except Exception as e:
                err_str = str(e)
                # Detect plan-gated endpoints and disable them permanently
                if "NOT_AUTHORIZED" in err_str or "not entitled" in err_str.lower():
                    logger.warning(
                        "%s not available on current plan — disabling for this session", endpoint_type
                    )
                    self._disabled_endpoints.add(endpoint_type)
                    return None
                if attempt == self._max_retries - 1:
                    logger.error("%s failed after %d retries: %s", context, self._max_retries, e)
                    return None
                delay = self._retry_delay * (2 ** attempt)
                logger.warning("%s attempt %d failed, retrying in %.1fs: %s", context, attempt + 1, delay, e)
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

    @staticmethod
    def _extract_financial_values(financials_obj: Any) -> dict[str, float | None]:
        """Extract key numeric values from a Massive financials object."""
        result: dict[str, float | None] = {}
        if not financials_obj or not hasattr(financials_obj, "financials"):
            return result
        fin = financials_obj.financials

        if hasattr(fin, "income_statement"):
            inc = fin.income_statement
            rev = getattr(inc, "revenues", None)
            if rev:
                result["revenues"] = getattr(rev, "value", None)
            ni = getattr(inc, "net_income_loss", None)
            if ni:
                result["net_income"] = getattr(ni, "value", None)

        if hasattr(fin, "balance_sheet"):
            bs = fin.balance_sheet
            for key, attr in [
                ("total_assets", "assets"),
                ("total_liabilities", "liabilities"),
                ("equity", "equity"),
                ("current_assets", "current_assets"),
                ("current_liabilities", "current_liabilities"),
            ]:
                val = getattr(bs, attr, None)
                if val:
                    result[key] = getattr(val, "value", None)

        if hasattr(fin, "cash_flow_statement"):
            cf = fin.cash_flow_statement
            op_cf = getattr(cf, "net_cash_flow_from_operating_activities", None)
            if op_cf:
                result["operating_cash_flow"] = getattr(op_cf, "value", None)

        return result

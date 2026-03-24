"""
Financial Modeling Prep (FMP) Client
====================================
Handles fundamental data: financials, ratios, profiles, earnings calendars.
Free tier: 250 requests/day — cache aggressively.
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


class FMPClient:
    """Client for Financial Modeling Prep API."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "FMP API key is required. "
                "Set FMP_API_KEY in config/secrets.env"
            )
        self.api_key = api_key
        self._daily_requests = 0
        self._daily_limit = 250  # Free tier

    def _request(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        max_retries: int = 3,
    ) -> dict | list | None:
        """Make an FMP API request with retry logic."""
        if self._daily_requests >= self._daily_limit:
            logger.warning("FMP daily request limit reached (250). Using cache only.")
            return None

        url = f"{FMP_BASE_URL}/{endpoint}"
        req_params = {"apikey": self.api_key}
        if params:
            req_params.update(params)

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=req_params, timeout=30)
                self._daily_requests += 1

                if resp.status_code == 200:
                    data = resp.json()
                    # FMP returns error messages as dicts
                    if isinstance(data, dict) and "Error Message" in data:
                        logger.error(f"FMP error: {data['Error Message']}")
                        return None
                    return data
                elif resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"FMP rate limited, retrying in {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"FMP API error {resp.status_code}: {resp.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** (attempt + 1))
                    continue

            except requests.exceptions.RequestException as e:
                logger.warning(f"FMP request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))

        logger.error(f"FMP request failed after {max_retries} attempts: {endpoint}")
        return None

    def get_profile(self, ticker: str) -> Optional[dict]:
        """
        Get company profile: sector, industry, market cap, description, etc.

        Returns dict with keys: symbol, companyName, sector, industry,
        mktCap, price, volAvg, description, etc.
        """
        data = self._request(f"profile/{ticker}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None

    def get_key_metrics(self, ticker: str, period: str = "annual", limit: int = 5) -> Optional[list]:
        """
        Get key financial metrics: PE, PEG, ROE, FCF yield, etc.

        Args:
            ticker: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods to return
        """
        return self._request(
            f"key-metrics/{ticker}",
            params={"period": period, "limit": limit},
        )

    def get_ratios(self, ticker: str, period: str = "annual", limit: int = 5) -> Optional[list]:
        """
        Get financial ratios: P/E, P/B, debt-to-equity, current ratio, etc.
        """
        return self._request(
            f"ratios/{ticker}",
            params={"period": period, "limit": limit},
        )

    def get_income_statement(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> Optional[list]:
        """Get income statements for earnings growth calculations."""
        return self._request(
            f"income-statement/{ticker}",
            params={"period": period, "limit": limit},
        )

    def get_balance_sheet(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> Optional[list]:
        """Get balance sheets for asset/liability analysis."""
        return self._request(
            f"balance-sheet-statement/{ticker}",
            params={"period": period, "limit": limit},
        )

    def get_cash_flow(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> Optional[list]:
        """Get cash flow statements for FCF calculations."""
        return self._request(
            f"cash-flow-statement/{ticker}",
            params={"period": period, "limit": limit},
        )

    def get_enterprise_value(
        self, ticker: str, period: str = "annual", limit: int = 5
    ) -> Optional[list]:
        """Get enterprise value metrics (EV, EV/EBITDA, etc.)."""
        return self._request(
            f"enterprise-values/{ticker}",
            params={"period": period, "limit": limit},
        )

    def get_stock_screener(
        self,
        market_cap_min: Optional[int] = None,
        market_cap_max: Optional[int] = None,
        volume_min: Optional[int] = None,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        sector: Optional[str] = None,
        exchange: Optional[str] = None,
        limit: int = 1000,
    ) -> Optional[list]:
        """
        Screen stocks by fundamental criteria.
        Useful for building the initial ticker universe.
        """
        params = {"limit": limit}
        if market_cap_min:
            params["marketCapMoreThan"] = market_cap_min
        if market_cap_max:
            params["marketCapLowerThan"] = market_cap_max
        if volume_min:
            params["volumeMoreThan"] = volume_min
        if price_min:
            params["priceMoreThan"] = price_min
        if price_max:
            params["priceLowerThan"] = price_max
        if sector:
            params["sector"] = sector
        if exchange:
            params["exchange"] = exchange

        return self._request("stock-screener", params=params)

    def get_earnings_calendar(
        self, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> Optional[list]:
        """
        Get upcoming earnings dates.
        Dates as YYYY-MM-DD strings.
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._request("earning_calendar", params=params)

    def get_market_hours(self) -> Optional[dict]:
        """Get current market open/close status."""
        return self._request("is-the-market-open")

    @property
    def requests_remaining(self) -> int:
        """Estimated API requests remaining today."""
        return max(0, self._daily_limit - self._daily_requests)

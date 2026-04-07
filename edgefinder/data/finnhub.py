"""Finnhub supplemental data provider.

Enriches TickerFundamentals with data Polygon doesn't provide:
- Earnings calendar dates (last and estimated next)
- Analyst consensus rating and target price
- Insider transaction buy/sell ratio

Free tier: 60 API calls/minute, no cost.
Requires EDGEFINDER_FINNHUB_API_KEY in settings.

To enable: set finnhub_enabled=True and finnhub_api_key in config.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import httpx

from config.settings import settings
from edgefinder.core.models import TickerFundamentals

logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"


class FinnhubProvider:
    """Supplemental provider using Finnhub's free REST API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.finnhub_api_key
        if not self._api_key:
            raise ValueError("Finnhub API key required")
        self._client = httpx.Client(
            base_url=FINNHUB_BASE,
            timeout=10.0,
            params={"token": self._api_key},
        )

    @property
    def source_name(self) -> str:
        return "finnhub"

    @property
    def available_fields(self) -> list[str]:
        return [
            "last_earnings_date",
            "estimated_next_earnings_date",
            "near_earnings",
            "analyst_rating",
            "analyst_target_price",
            "insider_buy_ratio",
        ]

    def enrich(self, fund: TickerFundamentals) -> None:
        """Enrich fundamentals with Finnhub data. Fills None fields only."""
        self._enrich_earnings(fund)
        self._enrich_analyst(fund)
        self._enrich_insider(fund)

    def _enrich_earnings(self, fund: TickerFundamentals) -> None:
        """Fetch earnings calendar and set last/next earnings dates."""
        if fund.last_earnings_date is not None:
            return

        try:
            today = date.today()
            # Look back 120 days and forward 120 days
            resp = self._client.get("/calendar/earnings", params={
                "symbol": fund.symbol,
                "from": (today - timedelta(days=120)).isoformat(),
                "to": (today + timedelta(days=120)).isoformat(),
            })
            resp.raise_for_status()
            data = resp.json()

            earnings = data.get("earningsCalendar", [])
            if not earnings:
                return

            past = []
            future = []
            for e in earnings:
                d = e.get("date")
                if not d:
                    continue
                earnings_date = datetime.strptime(d, "%Y-%m-%d").date()
                if earnings_date <= today:
                    past.append(earnings_date)
                else:
                    future.append(earnings_date)

            if past:
                last = max(past)
                fund.last_earnings_date = last.isoformat()

            if future:
                nxt = min(future)
                fund.estimated_next_earnings_date = nxt.isoformat()
                days_until = (nxt - today).days
                fund.near_earnings = days_until <= settings.earnings_blackout_days

        except Exception:
            logger.debug("Finnhub earnings fetch failed for %s", fund.symbol)

    def _enrich_analyst(self, fund: TickerFundamentals) -> None:
        """Fetch analyst recommendation trends."""
        if fund.analyst_rating is not None:
            return

        try:
            resp = self._client.get("/stock/recommendation", params={
                "symbol": fund.symbol,
            })
            resp.raise_for_status()
            data = resp.json()

            if not data:
                return

            # Most recent recommendation period
            latest = data[0]
            buy = latest.get("buy", 0) + latest.get("strongBuy", 0)
            hold = latest.get("hold", 0)
            sell = latest.get("sell", 0) + latest.get("strongSell", 0)
            total = buy + hold + sell

            if total > 0:
                if buy / total > 0.5:
                    fund.analyst_rating = "buy"
                elif sell / total > 0.3:
                    fund.analyst_rating = "sell"
                else:
                    fund.analyst_rating = "hold"

            # Target price from price target endpoint
            pt_resp = self._client.get("/stock/price-target", params={
                "symbol": fund.symbol,
            })
            pt_resp.raise_for_status()
            pt_data = pt_resp.json()
            target = pt_data.get("targetMedian")
            if target and fund.analyst_target_price is None:
                fund.analyst_target_price = float(target)

        except Exception:
            logger.debug("Finnhub analyst fetch failed for %s", fund.symbol)

    def _enrich_insider(self, fund: TickerFundamentals) -> None:
        """Fetch insider transactions and compute buy ratio."""
        if fund.insider_buy_ratio is not None:
            return

        try:
            today = date.today()
            resp = self._client.get("/stock/insider-transactions", params={
                "symbol": fund.symbol,
            })
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                return

            # Count buys vs sells over last 90 days
            cutoff = today - timedelta(days=90)
            buys = 0
            sells = 0
            for txn in data:
                txn_date_str = txn.get("transactionDate", "")
                if not txn_date_str:
                    continue
                try:
                    txn_date = datetime.strptime(txn_date_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if txn_date < cutoff:
                    continue

                change = txn.get("change", 0)
                if change > 0:
                    buys += 1
                elif change < 0:
                    sells += 1

            total = buys + sells
            if total > 0:
                fund.insider_buy_ratio = round(buys / total, 3)

        except Exception:
            logger.debug("Finnhub insider fetch failed for %s", fund.symbol)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

"""EdgeFinder v2 — Research service.

Aggregates fundamentals, technicals, trade history, and
strategy qualification into a single per-ticker report. Read-only layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy.orm import Session

from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import TickerFundamentals
from edgefinder.db.models import Fundamental, ManualInjection, Ticker, TradeRecord
from edgefinder.signals.engine import compute_indicators
from edgefinder.strategies.base import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class TickerReport:
    """Complete research report for a single ticker."""

    symbol: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    price: float | None = None
    is_active: bool = False

    # Fundamental ratios
    fundamentals: dict = field(default_factory=dict)

    # Technical indicators (latest)
    indicators: dict = field(default_factory=dict)

    # Sentiment
    sentiment: dict = field(default_factory=dict)

    # Trade history summary
    trade_count: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    total_pnl: float = 0.0

    # Strategy qualification
    qualifying_strategies: list[str] = field(default_factory=list)


class ResearchService:
    """Aggregates all data sources for per-ticker research."""

    def __init__(
        self,
        provider: DataProvider,
        session: Session,
    ) -> None:
        self._provider = provider
        self._session = session

    def get_ticker_report(self, symbol: str) -> TickerReport:
        """One-call aggregation of everything known about a ticker."""
        report = TickerReport(symbol=symbol)

        # DB data
        ticker = self._session.query(Ticker).filter_by(symbol=symbol).first()
        if ticker:
            report.company_name = ticker.company_name
            report.sector = ticker.sector
            report.industry = ticker.industry
            report.market_cap = ticker.market_cap
            report.price = ticker.last_price
            report.is_active = ticker.is_active

            fund = self._session.query(Fundamental).filter_by(ticker_id=ticker.id).first()
            if fund:
                report.fundamentals = {}
                for attr in ("peg_ratio", "earnings_growth", "debt_to_equity",
                             "revenue_growth", "institutional_pct", "fcf_yield",
                             "price_to_tangible_book", "short_interest", "ev_to_ebitda",
                             "current_ratio", "price_to_earnings", "price_to_book",
                             "return_on_equity", "return_on_assets", "dividend_yield",
                             "free_cash_flow", "quick_ratio", "short_shares",
                             "days_to_cover", "dividend_amount", "ex_dividend_date",
                             "news_sentiment"):
                    val = getattr(fund, attr, None)
                    if val is not None:
                        report.fundamentals[attr] = val
                # Include news headlines from raw_data
                if fund.raw_data and "news" in fund.raw_data:
                    report.fundamentals["news_headlines"] = fund.raw_data["news"]

        # Live price + fundamentals from Polygon if not in DB
        if self._provider:
            live_price = self._provider.get_latest_price(symbol)
            if live_price:
                report.price = live_price

            # Fetch fundamentals live if not already in DB
            if not report.fundamentals:
                try:
                    fund_data = self._provider.get_fundamentals(symbol)
                    if fund_data:
                        report.company_name = report.company_name or fund_data.company_name
                        report.sector = report.sector or fund_data.sector
                        report.industry = report.industry or fund_data.industry
                        report.market_cap = report.market_cap or fund_data.market_cap
                        report.fundamentals = {
                            "peg_ratio": fund_data.peg_ratio,
                            "earnings_growth": fund_data.earnings_growth,
                            "debt_to_equity": fund_data.debt_to_equity,
                            "revenue_growth": fund_data.revenue_growth,
                            "institutional_pct": fund_data.institutional_pct,
                            "fcf_yield": fund_data.fcf_yield,
                            "price_to_tangible_book": fund_data.price_to_tangible_book,
                            "short_interest": fund_data.short_interest,
                            "ev_to_ebitda": fund_data.ev_to_ebitda,
                            "current_ratio": fund_data.current_ratio,
                        }
                except Exception:
                    logger.debug("Could not fetch live fundamentals for %s", symbol)

        # Technical indicators
        try:
            end = date.today()
            start = end - timedelta(days=365)
            bars = self._provider.get_bars(symbol, "day", start, end) if self._provider else None
            if bars is not None and not bars.empty:
                snapshot = compute_indicators(bars)
                if snapshot:
                    report.indicators = snapshot.to_dict()
        except Exception:
            logger.debug("Could not compute indicators for %s", symbol)

        # Trade history
        trades = self._session.query(TradeRecord).filter_by(symbol=symbol).all()
        report.trade_count = len(trades)
        report.open_trades = sum(1 for t in trades if t.status == "OPEN")
        report.closed_trades = sum(1 for t in trades if t.status == "CLOSED")
        report.total_pnl = sum(t.pnl_dollars or 0 for t in trades if t.status == "CLOSED")

        # Strategy qualification
        fund_model = self._build_fundamentals_model(report)
        for strategy in StrategyRegistry.get_instances():
            try:
                if strategy.qualifies_stock(fund_model):
                    report.qualifying_strategies.append(strategy.name)
            except Exception:
                pass

        return report

    def search_tickers(self, query: str, limit: int = 20) -> list[dict]:
        """Search tickers by symbol or company name."""
        results = (
            self._session.query(Ticker)
            .filter(
                (Ticker.symbol.ilike(f"%{query}%"))
                | (Ticker.company_name.ilike(f"%{query}%"))
            )
            .limit(limit)
            .all()
        )
        return [
            {
                "symbol": t.symbol,
                "company_name": t.company_name,
                "sector": t.sector,
                "is_active": t.is_active,
            }
            for t in results
        ]

    def get_active_tickers(self) -> list[dict]:
        """Get all research tickers: scanner-qualified + manually injected.

        Returns enriched data with fundamentals and strategy qualification.
        """
        results: dict[str, dict] = {}

        # Scanner-qualified tickers (from nightly scan)
        rows = (
            self._session.query(Ticker, Fundamental)
            .outerjoin(Fundamental, Ticker.id == Fundamental.ticker_id)
            .filter(Ticker.is_active == True)
            .order_by(Ticker.symbol)
            .all()
        )
        for ticker, fund in rows:
            entry = self._build_ticker_entry(ticker, fund, source="scanner")
            results[ticker.symbol] = entry

        # Manually injected tickers (non-expired)
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        injections = (
            self._session.query(ManualInjection)
            .filter(
                (ManualInjection.expires_at == None)
                | (ManualInjection.expires_at > now)
            )
            .all()
        )
        for inj in injections:
            if inj.symbol in results:
                results[inj.symbol]["source"] = "both"
                continue
            ticker = self._session.query(Ticker).filter_by(symbol=inj.symbol).first()
            fund = None
            if ticker:
                fund = self._session.query(Fundamental).filter_by(ticker_id=ticker.id).first()
            entry = self._build_ticker_entry(ticker, fund, source="manual", symbol=inj.symbol)
            results[inj.symbol] = entry

        return sorted(results.values(), key=lambda x: x.get("market_cap") or 0, reverse=True)

    def _build_ticker_entry(
        self, ticker: Ticker | None, fund: Fundamental | None,
        source: str = "scanner", symbol: str = "",
    ) -> dict:
        """Build an enriched ticker entry for the research table."""
        sym = ticker.symbol if ticker else symbol
        entry = {
            "symbol": sym,
            "company_name": ticker.company_name if ticker else None,
            "sector": ticker.sector if ticker else None,
            "market_cap": ticker.market_cap if ticker else None,
            "last_price": ticker.last_price if ticker else None,
            "source": source,
            # Fundamentals (all available fields)
            "earnings_growth": None, "revenue_growth": None,
            "peg_ratio": None, "fcf_yield": None,
            "current_ratio": None, "debt_to_equity": None,
            "price_to_tangible_book": None, "ev_to_ebitda": None,
            "price_to_earnings": None, "price_to_book": None,
            "return_on_equity": None, "return_on_assets": None,
            "dividend_yield": None, "free_cash_flow": None,
            "short_interest": None, "quick_ratio": None,
            "short_shares": None, "days_to_cover": None,
            "dividend_amount": None, "ex_dividend_date": None,
            "news_sentiment": None,
            # Strategies
            "qualifying_strategies": [],
        }

        if fund:
            for attr in ("earnings_growth", "revenue_growth", "peg_ratio",
                         "fcf_yield", "current_ratio", "debt_to_equity",
                         "price_to_tangible_book", "ev_to_ebitda",
                         "price_to_earnings", "price_to_book",
                         "return_on_equity", "return_on_assets",
                         "dividend_yield", "free_cash_flow",
                         "short_interest", "quick_ratio",
                         "short_shares", "days_to_cover",
                         "dividend_amount", "ex_dividend_date",
                         "news_sentiment"):
                val = getattr(fund, attr, None)
                if val is not None:
                    entry[attr] = val

        # Strategy qualification
        fund_model = TickerFundamentals(
            symbol=sym,
            company_name=entry["company_name"],
            sector=entry["sector"],
            market_cap=entry["market_cap"],
            price=entry["last_price"],
            earnings_growth=entry["earnings_growth"],
            revenue_growth=entry["revenue_growth"],
            peg_ratio=entry["peg_ratio"],
            fcf_yield=entry["fcf_yield"],
            current_ratio=entry["current_ratio"],
            debt_to_equity=entry["debt_to_equity"],
        )
        for strategy in StrategyRegistry.get_instances():
            try:
                if strategy.qualifies_stock(fund_model):
                    entry["qualifying_strategies"].append(strategy.name)
            except Exception:
                pass

        return entry

    @staticmethod
    def _build_fundamentals_model(report: TickerReport) -> TickerFundamentals:
        """Build TickerFundamentals from report data for strategy qualification."""
        return TickerFundamentals(
            symbol=report.symbol,
            company_name=report.company_name,
            sector=report.sector,
            market_cap=report.market_cap,
            price=report.price,
            **report.fundamentals,
        )

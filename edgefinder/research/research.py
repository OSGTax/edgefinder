"""EdgeFinder v2 — Research service (replaces watchlist).

Aggregates fundamentals, technicals, sentiment, trade history, and
strategy qualification into a single per-ticker report. Read-only layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy.orm import Session

from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import TickerFundamentals
from edgefinder.db.models import Fundamental, SentimentReading, Ticker, TradeRecord
from edgefinder.sentiment.aggregator import SentimentAggregator
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

    # Fundamental scores
    lynch_score: float | None = None
    lynch_category: str | None = None
    burry_score: float | None = None
    composite_score: float | None = None
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
        sentiment_agg: SentimentAggregator | None = None,
    ) -> None:
        self._provider = provider
        self._session = session
        self._sentiment = sentiment_agg or SentimentAggregator(session)

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
                report.lynch_score = fund.lynch_score
                report.lynch_category = fund.lynch_category
                report.burry_score = fund.burry_score
                report.composite_score = fund.composite_score
                report.fundamentals = {
                    "peg_ratio": fund.peg_ratio,
                    "earnings_growth": fund.earnings_growth,
                    "debt_to_equity": fund.debt_to_equity,
                    "revenue_growth": fund.revenue_growth,
                    "institutional_pct": fund.institutional_pct,
                    "fcf_yield": fund.fcf_yield,
                    "price_to_tangible_book": fund.price_to_tangible_book,
                    "short_interest": fund.short_interest,
                    "ev_to_ebitda": fund.ev_to_ebitda,
                    "current_ratio": fund.current_ratio,
                }

        # Live price
        live_price = self._provider.get_latest_price(symbol)
        if live_price:
            report.price = live_price

        # Technical indicators
        try:
            end = date.today()
            start = end - timedelta(days=365)
            bars = self._provider.get_bars(symbol, "day", start, end)
            if bars is not None and not bars.empty:
                snapshot = compute_indicators(bars)
                if snapshot:
                    report.indicators = snapshot.to_dict()
        except Exception:
            logger.debug("Could not compute indicators for %s", symbol)

        # Sentiment
        try:
            sent = self._sentiment.get_sentiment(symbol)
            report.sentiment = {
                "composite_score": sent.composite_score,
                "source_scores": sent.source_scores,
                "total_mentions": sent.total_mentions,
                "is_trending": sent.is_trending,
                "action": sent.action.value,
            }
        except Exception:
            logger.debug("Could not get sentiment for %s", symbol)

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
        """Get all active tickers with basic info."""
        tickers = (
            self._session.query(Ticker)
            .filter(Ticker.is_active == True)
            .order_by(Ticker.symbol)
            .all()
        )
        return [
            {
                "symbol": t.symbol,
                "company_name": t.company_name,
                "sector": t.sector,
                "market_cap": t.market_cap,
                "last_price": t.last_price,
            }
            for t in tickers
        ]

    @staticmethod
    def _build_fundamentals_model(report: TickerReport) -> TickerFundamentals:
        """Build TickerFundamentals from report data for strategy qualification."""
        return TickerFundamentals(
            symbol=report.symbol,
            company_name=report.company_name,
            sector=report.sector,
            market_cap=report.market_cap,
            price=report.price,
            lynch_score=report.lynch_score,
            burry_score=report.burry_score,
            composite_score=report.composite_score,
            **report.fundamentals,
        )

"""EdgeFinder v2 — Per-strategy scanner.

Each strategy runs its own complete scan with full data access:
fundamentals + daily technical indicators + relative strength.
All data is unified into a StockProfile that feeds into qualification,
scoring, and later signal generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import TickerFundamentals
from edgefinder.core.profile import StockProfile
from edgefinder.db.models import Fundamental, Ticker, TickerStrategyQualification
from edgefinder.scanner.scoring import compute_score, compute_universe_stats
from edgefinder.signals.engine import compute_indicators
from edgefinder.signals.relative_strength import (
    compute_relative_strength,
    get_sector_etf,
)
from edgefinder.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of scanning a single ticker for a strategy."""

    symbol: str
    profile: StockProfile
    qualified: bool = False
    score: float = 0.0


class StrategyScanner:
    """Runs a complete scan for a single strategy.

    Each strategy gets its own isolated scan pipeline:
    1. Fetch universe
    2. For each ticker: build full StockProfile (fundamentals + technicals + RS)
    3. Qualify using strategy's criteria
    4. Score qualifying stocks using strategy's scoring profile
    5. Persist qualifications + scores to DB
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        provider: DataProvider,
        session: Session,
    ) -> None:
        self._strategy = strategy
        self._provider = provider
        self._session = session

    def run(
        self,
        tickers: list[str] | None = None,
        batch_index: int | None = None,
    ) -> list[ScanResult]:
        """Execute a full scan for this strategy.

        Args:
            tickers: Specific tickers to scan (None = full universe).
            batch_index: Weekly batch index for deactivation scoping.
        """
        universe = tickers or self._provider.get_ticker_universe()
        logger.info(
            "[%s] Scanning %d tickers",
            self._strategy.name, len(universe),
        )

        # Fetch SPY bars once for relative strength (shared across all tickers)
        end = date.today()
        start = end - timedelta(days=60)
        spy_bars = self._provider.get_bars("SPY", "day", start, end)

        # Build profiles and qualify
        results: list[ScanResult] = []
        for ticker in universe:
            profile = self._build_profile(ticker, spy_bars, start, end)
            if profile.fundamentals is None:
                continue

            qualified = self._qualify(profile)
            results.append(ScanResult(
                symbol=ticker,
                profile=profile,
                qualified=qualified,
            ))

        # Score qualifying stocks
        qualifying = [r for r in results if r.qualified]
        if qualifying and self._strategy.scoring_profile:
            profiles = [r.profile for r in qualifying]
            stats = compute_universe_stats(profiles, self._strategy.scoring_profile.factors)
            for r in qualifying:
                r.score = compute_score(r.profile, self._strategy.scoring_profile, stats)

        elif qualifying:
            # No scoring profile — all qualifying stocks get score 100
            for r in qualifying:
                r.score = 100.0

        # Persist to DB
        self._persist(results, batch_index)

        scored_count = sum(1 for r in results if r.qualified)
        top_score = max((r.score for r in results if r.qualified), default=0)
        logger.info(
            "[%s] Scan complete: %d scanned, %d qualified, top score: %.1f",
            self._strategy.name, len(results), scored_count, top_score,
        )

        return results

    def _build_profile(
        self,
        ticker: str,
        spy_bars,
        start: date,
        end: date,
    ) -> StockProfile:
        """Build complete StockProfile: fundamentals + daily indicators + RS."""
        # Fundamentals (from Massive: ratios, earnings, analyst, short interest, etc.)
        fund = self._provider.get_fundamentals(ticker)

        # Daily bars for technical indicators
        bars = self._provider.get_bars(ticker, "day", start, end)
        indicators = None
        if bars is not None and not bars.empty:
            indicators = compute_indicators(bars)

        # Relative strength vs SPY
        rs_spy = None
        if bars is not None and spy_bars is not None:
            rs_spy = compute_relative_strength(bars, spy_bars)

        # Relative strength vs sector ETF
        rs_sector = None
        sector_etf = None
        if fund and fund.sector:
            sector_etf = get_sector_etf(fund.sector)
            if sector_etf and bars is not None:
                sector_bars = self._provider.get_bars(sector_etf, "day", start, end)
                if sector_bars is not None:
                    rs_sector = compute_relative_strength(bars, sector_bars)

        return StockProfile(
            symbol=ticker,
            fundamentals=fund,
            indicators=indicators,
            rs_vs_spy=rs_spy,
            rs_vs_sector=rs_sector,
            sector_etf=sector_etf,
        )

    def _qualify(self, profile: StockProfile) -> bool:
        """Check if this strategy wants this stock."""
        try:
            # Use qualifies_stock with fundamentals (backward compatible)
            if profile.fundamentals is None:
                return False
            return self._strategy.qualifies_stock(profile.fundamentals)
        except Exception:
            logger.exception(
                "[%s] qualifies_stock failed for %s",
                self._strategy.name, profile.symbol,
            )
            return False

    def _persist(self, results: list[ScanResult], batch_index: int | None) -> None:
        """Persist scan results to DB for this strategy."""
        scanned_symbols = set()

        for result in results:
            fund = result.profile.fundamentals
            if fund is None:
                continue
            scanned_symbols.add(result.symbol)

            # Upsert ticker
            ticker = (
                self._session.query(Ticker)
                .filter_by(symbol=result.symbol)
                .first()
            )
            is_active_for_any = result.qualified  # at minimum this strategy
            if ticker is None:
                ticker = Ticker(
                    symbol=result.symbol,
                    company_name=fund.company_name,
                    sector=fund.sector,
                    industry=fund.industry,
                    market_cap=fund.market_cap,
                    last_price=fund.price,
                    source="scanner",
                    is_active=is_active_for_any,
                )
                self._session.add(ticker)
                self._session.flush()
            else:
                ticker.company_name = fund.company_name or ticker.company_name
                ticker.sector = fund.sector or ticker.sector
                ticker.industry = fund.industry or ticker.industry
                ticker.market_cap = fund.market_cap or ticker.market_cap
                ticker.last_price = fund.price or ticker.last_price
                if is_active_for_any:
                    ticker.is_active = True
            ticker.scan_batch = batch_index

            # Upsert fundamental
            existing_fund = (
                self._session.query(Fundamental)
                .filter_by(ticker_id=ticker.id)
                .first()
            )
            fund_data = dict(
                ticker_id=ticker.id,
                symbol=result.symbol,
                peg_ratio=fund.peg_ratio,
                earnings_growth=fund.earnings_growth,
                debt_to_equity=fund.debt_to_equity,
                revenue_growth=fund.revenue_growth,
                institutional_pct=fund.institutional_pct,
                fcf_yield=fund.fcf_yield,
                price_to_tangible_book=fund.price_to_tangible_book,
                short_interest=fund.short_interest,
                ev_to_ebitda=fund.ev_to_ebitda,
                current_ratio=fund.current_ratio,
                raw_data=fund.raw_data,
                scan_date=datetime.now(timezone.utc),
            )
            if existing_fund is None:
                self._session.add(Fundamental(**fund_data))
            else:
                for key, val in fund_data.items():
                    if key != "ticker_id":
                        setattr(existing_fund, key, val)

            # Upsert strategy qualification with score
            existing_qual = (
                self._session.query(TickerStrategyQualification)
                .filter_by(ticker_id=ticker.id, strategy_name=self._strategy.name)
                .first()
            )
            now = datetime.now(timezone.utc)
            if existing_qual is None:
                self._session.add(TickerStrategyQualification(
                    ticker_id=ticker.id,
                    symbol=result.symbol,
                    strategy_name=self._strategy.name,
                    qualified=result.qualified,
                    score=result.score if result.qualified else None,
                    scan_date=now,
                ))
            else:
                existing_qual.qualified = result.qualified
                existing_qual.score = result.score if result.qualified else None
                existing_qual.scan_date = now

        self._session.commit()

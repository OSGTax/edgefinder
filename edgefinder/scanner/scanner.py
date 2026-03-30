"""EdgeFinder v2 — Fundamental scanner.

Nightly scan: fetches universe, pre-screens by market cap/price/sector,
fetches fundamentals from Polygon, checks strategy qualification, persists to DB.
Strategies handle their own qualification logic — no centralized scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.events import event_bus
from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import TickerFundamentals
from edgefinder.db.models import Fundamental, Ticker
from edgefinder.strategies.base import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class ScannedStock:
    """Result for a single scanned stock."""

    symbol: str
    fundamentals: TickerFundamentals
    qualifying_strategies: list[str] = field(default_factory=list)


class FundamentalScanner:
    """Nightly fundamental scanner.

    Injected with a DataProvider and DB session for testability.
    """

    def __init__(self, provider: DataProvider, session: Session) -> None:
        self._provider = provider
        self._session = session

    def run(self, tickers: list[str] | None = None) -> list[ScannedStock]:
        """Execute a full scan. Optionally pass specific tickers."""
        universe = tickers or self._get_universe()
        logger.info("Scanning %d tickers", len(universe))

        prescreened = self._pre_screen(universe)
        logger.info("%d tickers passed pre-screen", len(prescreened))

        results: list[ScannedStock] = []
        for fund in prescreened:
            qualifying = self._check_strategy_qualification(fund)
            results.append(ScannedStock(
                symbol=fund.symbol,
                fundamentals=fund,
                qualifying_strategies=qualifying,
            ))

        self._save_to_db(results)

        qualified_count = sum(1 for s in results if s.qualifying_strategies)
        logger.info(
            "Scan complete: %d scanned, %d pre-screened, %d qualified",
            len(universe), len(prescreened), qualified_count,
        )

        event_bus.publish("scan.completed", {
            "total_scanned": len(universe),
            "passed_prescreen": len(prescreened),
            "qualified": qualified_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return results

    # ── Pipeline steps ───────────────────────────────

    def _get_universe(self) -> list[str]:
        return self._provider.get_ticker_universe()

    def _pre_screen(self, tickers: list[str]) -> list[TickerFundamentals]:
        """Fetch fundamentals and filter by basic criteria."""
        passed: list[TickerFundamentals] = []
        for ticker in tickers:
            fund = self._provider.get_fundamentals(ticker)
            if fund is None:
                continue
            # Polygon fundamentals don't include price — fetch it separately
            if fund.price is None:
                fund.price = self._provider.get_latest_price(ticker)
            if not self._passes_prescreen(fund):
                logger.debug(
                    "Pre-screen rejected %s (price=%s, market_cap=%s, sector=%s)",
                    ticker, fund.price, fund.market_cap, fund.sector,
                )
                continue
            passed.append(fund)
        return passed

    def _passes_prescreen(self, fund: TickerFundamentals) -> bool:
        mc = fund.market_cap
        if mc is None or mc < settings.scanner_min_market_cap or mc > settings.scanner_max_market_cap:
            return False
        price = fund.price
        if price is None or price < settings.scanner_min_price or price > settings.scanner_max_price:
            return False
        if fund.sector and fund.sector in settings.scanner_excluded_sectors:
            return False
        return True

    # ── Strategy Qualification ───────────────────────

    @staticmethod
    def _check_strategy_qualification(fund: TickerFundamentals) -> list[str]:
        qualifying = []
        for strategy in StrategyRegistry.get_instances():
            try:
                if strategy.qualifies_stock(fund):
                    qualifying.append(strategy.name)
            except Exception:
                logger.exception(
                    "Strategy '%s' failed qualifies_stock for %s",
                    strategy.name, fund.symbol,
                )
        return qualifying

    # ── DB Persistence ───────────────────────────────

    def _save_to_db(self, results: list[ScannedStock]) -> None:
        """Upsert tickers and fundamentals to database."""
        scanned_symbols = set()

        for stock in results:
            fund = stock.fundamentals
            is_active = len(stock.qualifying_strategies) > 0
            scanned_symbols.add(stock.symbol)

            # Upsert ticker
            ticker = (
                self._session.query(Ticker)
                .filter_by(symbol=stock.symbol)
                .first()
            )
            if ticker is None:
                ticker = Ticker(
                    symbol=stock.symbol,
                    company_name=fund.company_name,
                    sector=fund.sector,
                    industry=fund.industry,
                    market_cap=fund.market_cap,
                    last_price=fund.price,
                    source="scanner",
                    is_active=is_active,
                )
                self._session.add(ticker)
                self._session.flush()
            else:
                ticker.company_name = fund.company_name
                ticker.sector = fund.sector
                ticker.industry = fund.industry
                ticker.market_cap = fund.market_cap
                ticker.last_price = fund.price
                ticker.is_active = is_active

            # Upsert fundamental
            existing_fund = (
                self._session.query(Fundamental)
                .filter_by(ticker_id=ticker.id)
                .first()
            )
            fund_data = dict(
                ticker_id=ticker.id,
                symbol=stock.symbol,
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

        # Deactivate tickers not in this scan that were previously active
        previously_active = (
            self._session.query(Ticker)
            .filter(Ticker.is_active == True, Ticker.source == "scanner")
            .all()
        )
        for ticker in previously_active:
            if ticker.symbol not in scanned_symbols:
                ticker.is_active = False

        self._session.commit()

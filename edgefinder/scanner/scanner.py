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
from edgefinder.db.models import Fundamental, Ticker, TickerStrategyQualification
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

    def run(
        self,
        tickers: list[str] | None = None,
        batch_index: int | None = None,
    ) -> list[ScannedStock]:
        """Execute a scan. Optionally pass specific tickers and batch index.

        Args:
            tickers: Specific tickers to scan (None = full universe).
            batch_index: Which weekly batch (0-4) this scan belongs to.
                When set, only deactivates tickers from this batch.
        """
        self._batch_index = batch_index
        universe = tickers or self._get_universe()
        logger.info("Scanning %d tickers (batch=%s)", len(universe), batch_index)

        fetched = self._fetch_fundamentals(universe)
        logger.info("%d tickers have fundamentals", len(fetched))

        results: list[ScannedStock] = []
        for fund in fetched:
            qualifying = self._check_strategy_qualification(fund)
            results.append(ScannedStock(
                symbol=fund.symbol,
                fundamentals=fund,
                qualifying_strategies=qualifying,
            ))

        self._save_to_db(results)

        qualified_count = sum(1 for s in results if s.qualifying_strategies)
        logger.info(
            "Scan complete: %d scanned, %d with fundamentals, %d qualified",
            len(universe), len(fetched), qualified_count,
        )

        event_bus.publish("scan.completed", {
            "total_scanned": len(universe),
            "with_fundamentals": len(fetched),
            "qualified": qualified_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return results

    # ── Pipeline steps ───────────────────────────────

    def _get_universe(self) -> list[str]:
        return self._provider.get_ticker_universe()

    def _fetch_fundamentals(self, tickers: list[str]) -> list[TickerFundamentals]:
        """Fetch fundamentals from Polygon for all tickers. No filtering."""
        results: list[TickerFundamentals] = []
        for ticker in tickers:
            fund = self._provider.get_fundamentals(ticker)
            if fund is None:
                continue
            if fund.price is None:
                fund.price = self._provider.get_latest_price(ticker)
            results.append(fund)
        return results

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
            ticker.scan_batch = self._batch_index

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

            # Upsert per-strategy qualifications
            all_strategy_names = StrategyRegistry.list_names()
            for strat_name in all_strategy_names:
                qualified = strat_name in stock.qualifying_strategies
                existing_qual = (
                    self._session.query(TickerStrategyQualification)
                    .filter_by(ticker_id=ticker.id, strategy_name=strat_name)
                    .first()
                )
                now = datetime.now(timezone.utc)
                if existing_qual is None:
                    self._session.add(TickerStrategyQualification(
                        ticker_id=ticker.id,
                        symbol=stock.symbol,
                        strategy_name=strat_name,
                        qualified=qualified,
                        scan_date=now,
                    ))
                else:
                    existing_qual.qualified = qualified
                    existing_qual.scan_date = now

        # Deactivate tickers not in this scan — scoped to this batch only
        query = self._session.query(Ticker).filter(
            Ticker.is_active == True, Ticker.source == "scanner"
        )
        if self._batch_index is not None:
            query = query.filter(Ticker.scan_batch == self._batch_index)
        previously_active = query.all()
        for ticker in previously_active:
            if ticker.symbol not in scanned_symbols:
                ticker.is_active = False

        self._session.commit()

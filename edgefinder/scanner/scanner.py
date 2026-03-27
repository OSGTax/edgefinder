"""EdgeFinder v2 — Fundamental scanner.

Nightly scan: fetches universe, pre-screens, scores Lynch + Burry,
checks strategy qualification, persists to DB.
All thresholds and weights come from config/settings.py.
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
class ScoredStock:
    """Result for a single scanned stock."""

    symbol: str
    fundamentals: TickerFundamentals
    lynch_score: float = 0.0
    lynch_category: str = "slow_grower"
    burry_score: float = 0.0
    composite_score: float = 0.0
    qualifying_strategies: list[str] = field(default_factory=list)


class FundamentalScanner:
    """Nightly fundamental scanner.

    Injected with a DataProvider and DB session for testability.
    """

    def __init__(self, provider: DataProvider, session: Session) -> None:
        self._provider = provider
        self._session = session

    def run(self, tickers: list[str] | None = None) -> list[ScoredStock]:
        """Execute a full scan. Optionally pass specific tickers."""
        universe = tickers or self._get_universe()
        logger.info("Scanning %d tickers", len(universe))

        prescreened = self._pre_screen(universe)
        logger.info("%d tickers passed pre-screen", len(prescreened))

        scored: list[ScoredStock] = []
        for fund in prescreened:
            lynch_score, lynch_cat = self._score_lynch(fund)
            burry_score = self._score_burry(fund)
            composite = self._compute_composite(lynch_score, burry_score)

            fund.lynch_score = lynch_score
            fund.lynch_category = lynch_cat
            fund.burry_score = burry_score
            fund.composite_score = composite

            qualifying = self._check_strategy_qualification(fund)

            scored.append(ScoredStock(
                symbol=fund.symbol,
                fundamentals=fund,
                lynch_score=lynch_score,
                lynch_category=lynch_cat,
                burry_score=burry_score,
                composite_score=composite,
                qualifying_strategies=qualifying,
            ))

        self._save_to_db(scored)

        qualified_count = sum(1 for s in scored if s.qualifying_strategies)
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

        return scored

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

    # ── Lynch Scoring ────────────────────────────────

    def _score_lynch(self, fund: TickerFundamentals) -> tuple[float, str]:
        """Score stock on Peter Lynch criteria (0-100)."""
        category = self._classify_lynch_category(fund)
        scores = {
            "peg": self._score_peg(fund.peg_ratio),
            "earnings_growth": self._score_earnings_growth(fund.earnings_growth),
            "debt_to_equity": self._score_debt_to_equity(fund.debt_to_equity),
            "revenue_growth": self._score_revenue_growth(fund.revenue_growth),
            "institutional": self._score_institutional(fund.institutional_pct),
            "category": self._score_category(category),
        }
        total = (
            scores["peg"] * settings.lynch_peg_weight
            + scores["earnings_growth"] * settings.lynch_earnings_growth_weight
            + scores["debt_to_equity"] * settings.lynch_debt_to_equity_weight
            + scores["revenue_growth"] * settings.lynch_revenue_growth_weight
            + scores["institutional"] * settings.lynch_institutional_weight
            + scores["category"] * settings.lynch_category_weight
        )
        return round(total, 2), category

    @staticmethod
    def _score_peg(peg: float | None) -> float:
        if peg is None or peg <= 0:
            return 0
        ideal = settings.lynch_peg_ideal
        mx = settings.lynch_peg_max
        if peg <= ideal:
            return 100
        if peg <= mx:
            return 100 - 50 * (peg - ideal) / (mx - ideal)
        return max(0, 50 * (1 - (peg - mx) / (3.0 - mx)))

    @staticmethod
    def _score_earnings_growth(eg: float | None) -> float:
        if eg is None or eg <= 0:
            return 0
        mn = settings.lynch_earnings_growth_min
        if eg >= 0.25:
            return 100
        if eg >= mn:
            return 50 + 50 * (eg - mn) / (0.25 - mn)
        return 50 * eg / mn

    @staticmethod
    def _score_debt_to_equity(de: float | None) -> float:
        if de is None:
            return 25
        pref = settings.lynch_debt_to_equity_preferred
        mx = settings.lynch_debt_to_equity_max
        if de <= pref:
            return 100
        if de <= mx:
            return 100 - 50 * (de - pref) / (mx - pref)
        return max(0, 50 * (1 - (de - mx) / (2.0 - mx)))

    @staticmethod
    def _score_revenue_growth(rg: float | None) -> float:
        if rg is None or rg <= 0:
            return 0
        mn = settings.lynch_revenue_growth_min
        if rg >= 0.20:
            return 100
        if rg >= mn:
            return 50 + 50 * (rg - mn) / (0.20 - mn)
        return 50 * rg / mn

    @staticmethod
    def _score_institutional(inst: float | None) -> float:
        if inst is None:
            return 30
        mn = settings.lynch_institutional_min
        mx = settings.lynch_institutional_max
        if mn <= inst <= mx:
            return 100
        if inst < mn:
            return 40 + 60 * inst / mn
        return max(20, 100 - 80 * (inst - mx) / (1.0 - mx))

    @staticmethod
    def _score_category(category: str) -> float:
        return {
            "fast_grower": 100,
            "turnaround": 85,
            "stalwart": 70,
            "asset_play": 65,
            "cyclical": 50,
            "slow_grower": 30,
        }.get(category, 30)

    @staticmethod
    def _classify_lynch_category(fund: TickerFundamentals) -> str:
        eg = fund.earnings_growth
        rg = fund.revenue_growth
        de = fund.debt_to_equity
        mc = fund.market_cap
        ptb = fund.price_to_tangible_book
        sector = fund.sector or ""

        if eg is not None and eg >= 0.20 and rg is not None and rg >= 0.15:
            return "fast_grower"
        if eg is not None and eg < 0 and de is not None and de < 1.5:
            return "turnaround"
        if eg is not None and 0.10 <= eg < 0.20 and mc is not None and mc >= 10_000_000_000:
            return "stalwart"
        if ptb is not None and ptb < 1.5:
            return "asset_play"
        if sector in ("Energy", "Materials", "Industrials"):
            return "cyclical"
        return "slow_grower"

    # ── Burry Scoring ────────────────────────────────

    def _score_burry(self, fund: TickerFundamentals) -> float:
        """Score stock on Michael Burry criteria (0-100)."""
        scores = {
            "fcf_yield": self._score_fcf_yield(fund.fcf_yield),
            "ptb": self._score_ptb(fund.price_to_tangible_book),
            "short_interest": self._score_short_interest(fund.short_interest),
            "ev_ebitda": self._score_ev_ebitda(fund.ev_to_ebitda),
            "current_ratio": self._score_current_ratio(fund.current_ratio),
        }
        total = (
            scores["fcf_yield"] * settings.burry_fcf_yield_weight
            + scores["ptb"] * settings.burry_price_to_tangible_book_weight
            + scores["short_interest"] * settings.burry_short_interest_weight
            + scores["ev_ebitda"] * settings.burry_ev_to_ebitda_weight
            + scores["current_ratio"] * settings.burry_current_ratio_weight
        )
        return round(total, 2)

    @staticmethod
    def _score_fcf_yield(fcf: float | None) -> float:
        if fcf is None or fcf <= 0:
            return 0
        strong = settings.burry_fcf_yield_strong
        acceptable = settings.burry_fcf_yield_acceptable
        if fcf >= strong:
            return 100
        if fcf >= acceptable:
            return 50 + 50 * (fcf - acceptable) / (strong - acceptable)
        return 50 * fcf / acceptable

    @staticmethod
    def _score_ptb(ptb: float | None) -> float:
        if ptb is None:
            return 0
        deep = settings.burry_price_to_tangible_book_deep
        value = settings.burry_price_to_tangible_book_value
        if ptb <= deep:
            return 100
        if ptb <= value:
            return 100 - 50 * (ptb - deep) / (value - deep)
        return max(0, 50 * (1 - (ptb - value) / (5.0 - value)))

    @staticmethod
    def _score_short_interest(si: float | None) -> float:
        if si is None:
            return 20
        contrarian = settings.burry_short_interest_contrarian
        if si >= contrarian:
            return 100
        if si >= 0.05:
            return 40 + 60 * (si - 0.05) / (contrarian - 0.05)
        return 20

    @staticmethod
    def _score_ev_ebitda(ev: float | None) -> float:
        if ev is None or ev <= 0:
            return 0
        mx = settings.burry_ev_to_ebitda_max
        if ev <= 4.0:
            return 100
        if ev <= mx:
            return 100 - 50 * (ev - 4.0) / (mx - 4.0)
        return max(0, 50 * (1 - (ev - mx) / (16.0 - mx)))

    @staticmethod
    def _score_current_ratio(cr: float | None) -> float:
        if cr is None:
            return 20
        mn = settings.burry_current_ratio_min
        if cr >= mn:
            return 100
        if cr >= 1.0:
            return 40 + 60 * (cr - 1.0) / (mn - 1.0)
        return 40 * cr

    # ── Composite ────────────────────────────────────

    @staticmethod
    def _compute_composite(lynch: float, burry: float) -> float:
        return round(
            settings.lynch_weight * lynch + settings.burry_weight * burry, 2
        )

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

    def _save_to_db(self, scored: list[ScoredStock]) -> None:
        """Upsert tickers and fundamentals to database."""
        scanned_symbols = set()

        for stock in scored:
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
                lynch_score=stock.lynch_score,
                lynch_category=stock.lynch_category,
                fcf_yield=fund.fcf_yield,
                price_to_tangible_book=fund.price_to_tangible_book,
                short_interest=fund.short_interest,
                ev_to_ebitda=fund.ev_to_ebitda,
                current_ratio=fund.current_ratio,
                burry_score=stock.burry_score,
                composite_score=stock.composite_score,
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

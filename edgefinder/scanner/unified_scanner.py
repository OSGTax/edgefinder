"""EdgeFinder — Unified multi-strategy scanner.

Fetches each ticker's fundamentals ONCE and evaluates every registered
strategy's qualifies_stock() against the shared result, then scores and
persists per-strategy qualification rows.

- Pass 1 API calls: O(tickers) instead of O(tickers × strategies).
- Concurrency: ThreadPoolExecutor parallelizes the network-bound fetches.
- Incremental commits: persists every N tickers so a deploy mid-scan
  doesn't lose progress.

Combined with the universe pre-filter (top 1000 by dollar volume), total
scan time is ~30-60 seconds.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.interfaces import DataProvider
from edgefinder.core.profile import StockProfile
from edgefinder.db.models import Fundamental, Ticker, TickerStrategyQualification
from edgefinder.scanner.scoring import compute_score, compute_universe_stats
from edgefinder.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class TickerScanData:
    """Pass 1 result for a single ticker — fundamentals + per-strategy qualification flags."""
    symbol: str
    profile: StockProfile
    qualified_by: dict[str, bool]  # strategy_name -> qualified
    score_by: dict[str, float]      # strategy_name -> score (set in scoring phase)


class UnifiedScanner:
    """Scan the universe ONCE, qualify against all strategies in parallel.

    Usage:
        scanner = UnifiedScanner(strategies, provider, session_factory)
        summary = scanner.run(tickers)
        # summary = {"alpha": 30, "bravo": 12, "charlie": 8, "degenerate": 45}
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        provider: DataProvider,
        session_factory: Callable[[], Session],
        max_workers: int | None = None,
        commit_batch_size: int | None = None,
    ) -> None:
        self._strategies = list(strategies)
        self._provider = provider
        self._session_factory = session_factory
        self._max_workers = max_workers or settings.scanner_concurrent_workers
        self._commit_batch_size = commit_batch_size or settings.scanner_commit_batch_size

    def run(self, tickers: list[str]) -> dict[str, int]:
        """Execute the unified scan. Returns per-strategy qualified counts."""
        total = len(tickers)
        logger.info(
            "Unified scan starting: %d tickers x %d strategies (workers=%d, batch=%d)",
            total, len(self._strategies), self._max_workers, self._commit_batch_size,
        )
        start_ts = datetime.now()

        # ── Pass 1: Concurrent fundamentals fetch + qualification ──
        scan_data: list[TickerScanData] = []
        completed = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._fetch_and_qualify, ticker): ticker
                for ticker in tickers
            }
            for fut in as_completed(futures):
                completed += 1
                if completed % 100 == 0 or completed == 1:
                    logger.info("Unified scan: pass1 %d/%d", completed, total)
                try:
                    data = fut.result()
                except Exception:
                    logger.exception("Pass1 fetch failed for %s", futures[fut])
                    continue
                if data is not None:
                    scan_data.append(data)

        elapsed_pass1 = (datetime.now() - start_ts).total_seconds()
        logger.info(
            "Unified scan: Pass 1 complete in %.1fs — %d valid profiles of %d",
            elapsed_pass1, len(scan_data), total,
        )

        # ── Pass 2: Concurrent enrichment of qualified tickers ──
        # A ticker is "enriched" if ANY strategy qualified it. Enrichment is
        # strategy-agnostic — full_refresh=True fetches the same TIER 2 data
        # regardless of which strategy asked.
        enrich_set = {
            d.symbol for d in scan_data
            if any(d.qualified_by.values())
        }
        if enrich_set:
            logger.info("Unified scan: enriching %d qualified tickers", len(enrich_set))
            self._enrich_concurrent(scan_data, enrich_set)

        # ── Score per strategy (in-memory, fast) ──
        for strategy in self._strategies:
            qualified_for_strategy = [
                d for d in scan_data if d.qualified_by.get(strategy.name)
            ]
            if not qualified_for_strategy:
                continue
            if strategy.scoring_profile:
                profiles = [d.profile for d in qualified_for_strategy]
                stats = compute_universe_stats(profiles, strategy.scoring_profile.factors)
                for d in qualified_for_strategy:
                    d.score_by[strategy.name] = compute_score(
                        d.profile, strategy.scoring_profile, stats,
                    )
            else:
                for d in qualified_for_strategy:
                    d.score_by[strategy.name] = 100.0

        # ── Pass 3: Persist with incremental commits ──
        summary = self._persist_incremental(scan_data)

        elapsed_total = (datetime.now() - start_ts).total_seconds()
        logger.info(
            "Unified scan complete in %.1fs: %s",
            elapsed_total,
            ", ".join(f"{k}={v}" for k, v in summary.items()),
        )
        return summary

    # ── Pass 1: per-ticker fetch + qualify ──

    def _fetch_and_qualify(self, ticker: str) -> TickerScanData | None:
        """Fetch fundamentals once, evaluate every strategy's qualifies_stock."""
        try:
            fund = self._provider.get_fundamentals(ticker, full_refresh=False)
        except Exception:
            logger.exception("get_fundamentals failed for %s", ticker)
            return None

        if fund is None or fund.company_name is None:
            return None

        profile = StockProfile(symbol=ticker, fundamentals=fund)
        qualified_by: dict[str, bool] = {}
        for strategy in self._strategies:
            try:
                qualified_by[strategy.name] = strategy.qualifies_stock(fund)
            except Exception:
                logger.exception(
                    "[%s] qualifies_stock failed for %s", strategy.name, ticker,
                )
                qualified_by[strategy.name] = False
        return TickerScanData(
            symbol=ticker,
            profile=profile,
            qualified_by=qualified_by,
            score_by={},
        )

    # ── Pass 2: concurrent enrichment of qualified set ──

    def _enrich_concurrent(
        self,
        scan_data: list[TickerScanData],
        enrich_set: set[str],
    ) -> None:
        """Replace fundamentals on qualified tickers with full-refresh enriched data.

        Mutates scan_data in place. Concurrent like Pass 1.
        """
        # Build symbol -> TickerScanData index for fast lookup
        index = {d.symbol: d for d in scan_data if d.symbol in enrich_set}

        completed = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._enrich_one, symbol): symbol
                for symbol in enrich_set
            }
            for fut in as_completed(futures):
                completed += 1
                if completed % 25 == 0 or completed == 1:
                    logger.info("Unified scan: enrich %d/%d", completed, len(enrich_set))
                symbol = futures[fut]
                try:
                    fund = fut.result()
                except Exception:
                    logger.exception("Enrich failed for %s", symbol)
                    continue
                if fund is None:
                    continue
                # Replace the fundamentals on the existing scan_data entry
                target = index.get(symbol)
                if target is not None:
                    target.profile = StockProfile(
                        symbol=symbol,
                        fundamentals=fund,
                    )

    def _enrich_one(self, ticker: str):
        return self._provider.get_fundamentals(ticker, full_refresh=True)

    # ── Pass 3: incremental persistence ──

    def _persist_incremental(
        self, scan_data: list[TickerScanData],
    ) -> dict[str, int]:
        """Persist scan results in batches, committing every N to survive deploys."""
        summary = {s.name: 0 for s in self._strategies}
        scanned_symbols: set[str] = set()
        now = datetime.now(timezone.utc)

        session = self._session_factory()
        try:
            in_batch = 0
            for data in scan_data:
                fund = data.profile.fundamentals
                if fund is None or fund.company_name is None:
                    continue
                scanned_symbols.add(data.symbol)
                any_qualified = any(data.qualified_by.values())

                # ── Upsert ticker ──
                ticker_row = (
                    session.query(Ticker)
                    .filter_by(symbol=data.symbol)
                    .first()
                )
                if ticker_row is None:
                    ticker_row = Ticker(
                        symbol=data.symbol,
                        company_name=fund.company_name,
                        sector=fund.sector,
                        industry=fund.industry,
                        market_cap=fund.market_cap,
                        last_price=fund.price,
                        source="scanner",
                        is_active=any_qualified,
                    )
                    session.add(ticker_row)
                    session.flush()
                else:
                    ticker_row.company_name = fund.company_name or ticker_row.company_name
                    ticker_row.sector = fund.sector or ticker_row.sector
                    ticker_row.industry = fund.industry or ticker_row.industry
                    ticker_row.market_cap = fund.market_cap or ticker_row.market_cap
                    ticker_row.last_price = fund.price or ticker_row.last_price
                    if any_qualified:
                        ticker_row.is_active = True

                # ── Upsert fundamental ──
                fund_data = dict(
                    ticker_id=ticker_row.id,
                    symbol=data.symbol,
                    peg_ratio=fund.peg_ratio,
                    earnings_growth=fund.earnings_growth,
                    debt_to_equity=fund.debt_to_equity,
                    revenue_growth=fund.revenue_growth,
                    institutional_pct=fund.institutional_pct,
                    fcf_yield=fund.fcf_yield,
                    price_to_tangible_book=fund.price_to_tangible_book,
                    ev_to_ebitda=fund.ev_to_ebitda,
                    current_ratio=fund.current_ratio,
                    price_to_earnings=fund.price_to_earnings,
                    price_to_book=fund.price_to_book,
                    return_on_equity=fund.return_on_equity,
                    return_on_assets=fund.return_on_assets,
                    dividend_yield=fund.dividend_yield,
                    free_cash_flow=fund.free_cash_flow,
                    quick_ratio=fund.quick_ratio,
                    short_interest=fund.short_interest,
                    short_shares=fund.short_shares,
                    days_to_cover=fund.days_to_cover,
                    dividend_amount=fund.dividend_amount,
                    ex_dividend_date=fund.ex_dividend_date,
                    news_sentiment=fund.news_sentiment,
                    rsi_14=fund.rsi_14,
                    ema_21=fund.ema_21,
                    sma_50=fund.sma_50,
                    macd_value=fund.macd_value,
                    macd_signal=fund.macd_signal,
                    macd_histogram=fund.macd_histogram,
                    raw_data=fund.raw_data,
                    scan_date=now,
                )
                tier1_fields = {
                    "symbol", "peg_ratio", "earnings_growth", "debt_to_equity",
                    "revenue_growth", "fcf_yield", "price_to_tangible_book",
                    "ev_to_ebitda", "current_ratio", "price_to_earnings",
                    "price_to_book", "return_on_equity", "return_on_assets",
                    "scan_date",
                }
                existing_fund = (
                    session.query(Fundamental)
                    .filter_by(ticker_id=ticker_row.id)
                    .first()
                )
                if existing_fund is None:
                    session.add(Fundamental(**fund_data))
                elif any_qualified:
                    for key, val in fund_data.items():
                        if key != "ticker_id":
                            setattr(existing_fund, key, val)
                else:
                    for key in tier1_fields:
                        if key in fund_data:
                            setattr(existing_fund, key, fund_data[key])

                # ── Upsert one qualification row per strategy ──
                for strategy in self._strategies:
                    qualified = data.qualified_by.get(strategy.name, False)
                    score = data.score_by.get(strategy.name) if qualified else None
                    existing_qual = (
                        session.query(TickerStrategyQualification)
                        .filter_by(ticker_id=ticker_row.id, strategy_name=strategy.name)
                        .first()
                    )
                    if existing_qual is None:
                        session.add(TickerStrategyQualification(
                            ticker_id=ticker_row.id,
                            symbol=data.symbol,
                            strategy_name=strategy.name,
                            qualified=qualified,
                            score=score,
                            scan_date=now,
                        ))
                    else:
                        existing_qual.qualified = qualified
                        existing_qual.score = score
                        existing_qual.scan_date = now

                    if qualified:
                        summary[strategy.name] += 1

                in_batch += 1
                if in_batch >= self._commit_batch_size:
                    session.commit()
                    in_batch = 0
                    logger.debug(
                        "Unified scan: incremental commit (running totals: %s)",
                        summary,
                    )

            # Final commit + bulk deactivation
            session.commit()

            # Deactivate tickers no strategy qualified for (1 query, not N)
            if scanned_symbols:
                qualified_ids = (
                    session.query(TickerStrategyQualification.ticker_id)
                    .filter(TickerStrategyQualification.qualified == True)
                    .subquery()
                )
                session.query(Ticker).filter(
                    Ticker.symbol.in_(scanned_symbols),
                    Ticker.is_active == True,
                    ~Ticker.id.in_(qualified_ids),
                ).update({"is_active": False}, synchronize_session="fetch")
                session.commit()
        except Exception:
            logger.exception("Unified scan persist failed")
            session.rollback()
        finally:
            session.close()

        return summary

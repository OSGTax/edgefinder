"""EdgeFinder v2 — Unified nightly DATA scanner.

The scanner is a pure data collector (the per-strategy qualification layer
was retired with the old arena): it fetches each ticker's fundamentals ONCE
(concurrently) and persists tickers + fundamentals so the permanent data
asset keeps growing. The nightly PIT fundamentals snapshot
(edgefinder/data/pit_fundamentals.snapshot_fundamentals) reads what this
writes — that is how the honest fundamental-strategy history accumulates.

Performance properties kept from the original design:
- Concurrency: ThreadPoolExecutor parallelizes the network-bound fetches.
- Incremental commits: persists every N tickers so a deploy mid-scan
  doesn't lose all progress.
- Combined with the universe pre-filter (top N by dollar volume), a full
  nightly scan completes in seconds-to-minutes, not hours.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import TickerFundamentals
from edgefinder.db.models import Fundamental, Ticker

logger = logging.getLogger(__name__)


@dataclass
class TickerScanData:
    """Pass 1 result for a single ticker — its fetched fundamentals."""
    symbol: str
    fundamentals: TickerFundamentals


class UnifiedScanner:
    """Scan the universe once: fetch fundamentals, persist tickers + rows.

    Usage:
        scanner = UnifiedScanner(provider, session_factory)
        summary = scanner.run(tickers)
        # summary = {"scanned": 980, "persisted": 975}
    """

    def __init__(
        self,
        provider: DataProvider,
        session_factory: Callable[[], Session],
        max_workers: int | None = None,
        commit_batch_size: int | None = None,
    ) -> None:
        self._provider = provider
        self._session_factory = session_factory
        self._max_workers = max_workers or settings.scanner_concurrent_workers
        self._commit_batch_size = commit_batch_size or settings.scanner_commit_batch_size

    def run(self, tickers: list[str]) -> dict[str, int]:
        """Execute the data scan. Returns {"scanned": N, "persisted": M}."""
        total = len(tickers)
        logger.info(
            "Unified scan starting: %d tickers (workers=%d, batch=%d)",
            total, self._max_workers, self._commit_batch_size,
        )
        start_ts = datetime.now()

        # ── Pass 1: concurrent fundamentals fetch ──
        scan_data: list[TickerScanData] = []
        completed = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._fetch_one, ticker): ticker
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

        # ── Pass 2: concurrent TIER-2 enrichment (technicals, short
        # interest, news sentiment). The old scanner enriched only
        # strategy-qualified tickers; with qualification retired the whole
        # scanned set is enriched so the data asset keeps its richness.
        if scan_data:
            logger.info("Unified scan: enriching %d tickers", len(scan_data))
            self._enrich_concurrent(scan_data)

        # ── Pass 3: persist with incremental commits ──
        persisted = self._persist_incremental(scan_data)

        summary = {"scanned": len(scan_data), "persisted": persisted}
        elapsed_total = (datetime.now() - start_ts).total_seconds()
        logger.info(
            "Unified scan complete in %.1fs: %s", elapsed_total, summary,
        )
        return summary

    # ── Pass 1: per-ticker fetch ──

    def _fetch_one(self, ticker: str) -> TickerScanData | None:
        """Fetch fundamentals once for ``ticker``."""
        try:
            fund = self._provider.get_fundamentals(ticker, full_refresh=False)
        except Exception:
            logger.exception("get_fundamentals failed for %s", ticker)
            return None
        if fund is None or fund.company_name is None:
            return None
        return TickerScanData(symbol=ticker, fundamentals=fund)

    # ── Pass 2: concurrent enrichment ──

    def _enrich_concurrent(self, scan_data: list[TickerScanData]) -> None:
        """Replace fundamentals with full-refresh enriched data (in place)."""
        index = {d.symbol: d for d in scan_data}
        completed = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._enrich_one, symbol): symbol
                for symbol in index
            }
            for fut in as_completed(futures):
                completed += 1
                if completed % 25 == 0 or completed == 1:
                    logger.info("Unified scan: enrich %d/%d", completed, len(index))
                symbol = futures[fut]
                try:
                    fund = fut.result()
                except Exception:
                    logger.exception("Enrich failed for %s", symbol)
                    continue
                if fund is not None:
                    index[symbol].fundamentals = fund

    def _enrich_one(self, ticker: str):
        return self._provider.get_fundamentals(ticker, full_refresh=True)

    # ── Pass 3: incremental persistence ──

    def _persist_incremental(self, scan_data: list[TickerScanData]) -> int:
        """Persist scan results in batches, committing every N to survive deploys."""
        persisted = 0
        now = datetime.now(timezone.utc)

        session = self._session_factory()
        try:
            in_batch = 0
            for data in scan_data:
                fund = data.fundamentals
                if fund is None or fund.company_name is None:
                    continue

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
                        is_active=True,
                    )
                    session.add(ticker_row)
                    session.flush()
                else:
                    ticker_row.company_name = fund.company_name or ticker_row.company_name
                    ticker_row.sector = fund.sector or ticker_row.sector
                    ticker_row.industry = fund.industry or ticker_row.industry
                    ticker_row.market_cap = fund.market_cap or ticker_row.market_cap
                    ticker_row.last_price = fund.price or ticker_row.last_price
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
                existing_fund = (
                    session.query(Fundamental)
                    .filter_by(ticker_id=ticker_row.id)
                    .first()
                )
                if existing_fund is None:
                    session.add(Fundamental(**fund_data))
                else:
                    for key, val in fund_data.items():
                        if key != "ticker_id":
                            setattr(existing_fund, key, val)

                persisted += 1
                in_batch += 1
                if in_batch >= self._commit_batch_size:
                    session.commit()
                    in_batch = 0
                    logger.debug(
                        "Unified scan: incremental commit (%d persisted)", persisted,
                    )

            session.commit()
        except Exception:
            logger.exception("Unified scan persist failed")
            session.rollback()
        finally:
            session.close()

        return persisted

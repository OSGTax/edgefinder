"""EdgeFinder — Background data accumulator.

Slowly builds a persistent database of news, dividends, splits, and
market holidays. Runs on scheduler alongside signal checks and scans.
Key principle: don't re-fetch what's already in the DB.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from edgefinder.db.models import (
    Ticker,
    TickerDividend,
    TickerNews,
    TickerSplit,
)

logger = logging.getLogger(__name__)


class DataAccumulator:
    """Accumulates market data into the DB over time."""

    def __init__(self, provider, session_factory) -> None:
        self._provider = provider
        self._session_factory = session_factory

    def accumulate_news(self, symbols: list[str] | None = None) -> int:
        """Fetch and store recent news for watchlist tickers.

        Returns count of new articles stored.
        """
        session = self._session_factory()
        try:
            tickers = symbols or self._get_active_symbols(session)
            total_new = 0
            for sym in tickers:
                try:
                    articles = self._provider.get_news(sym, limit=5)
                    for a in articles:
                        title = a.get("title")
                        if not title:
                            continue
                        exists = (
                            session.query(TickerNews)
                            .filter_by(
                                symbol=sym,
                                title=title,
                                published_utc=a.get("published_utc"),
                            )
                            .first()
                        )
                        if exists:
                            continue
                        session.add(TickerNews(
                            symbol=sym,
                            title=title,
                            author=a.get("author"),
                            published_utc=a.get("published_utc"),
                            article_url=a.get("article_url"),
                            description=a.get("description"),
                            publisher_name=a.get("publisher"),
                        ))
                        total_new += 1
                except Exception:
                    logger.debug("News fetch failed for %s", sym)
            session.commit()
            if total_new:
                logger.info("News accumulator: %d new articles stored", total_new)
            return total_new
        except Exception:
            logger.exception("News accumulation failed")
            return 0
        finally:
            session.close()

    def accumulate_dividends(self, symbols: list[str] | None = None) -> int:
        """Fetch and store dividend history for watchlist tickers.

        Returns count of new dividend records stored.
        """
        session = self._session_factory()
        try:
            tickers = symbols or self._get_active_symbols(session)
            total_new = 0
            for sym in tickers:
                try:
                    divs = self._provider.get_dividends(sym, limit=10)
                    for d in divs:
                        ex_date = d.get("ex_dividend_date")
                        if not ex_date:
                            continue
                        exists = (
                            session.query(TickerDividend)
                            .filter_by(symbol=sym, ex_dividend_date=ex_date)
                            .first()
                        )
                        if exists:
                            continue
                        session.add(TickerDividend(
                            symbol=sym,
                            ex_dividend_date=ex_date,
                            pay_date=d.get("pay_date"),
                            cash_amount=d.get("cash_amount"),
                            declaration_date=d.get("declaration_date"),
                            frequency=d.get("frequency"),
                        ))
                        total_new += 1
                except Exception:
                    logger.debug("Dividend fetch failed for %s", sym)
            session.commit()
            if total_new:
                logger.info("Dividend accumulator: %d new records stored", total_new)
            return total_new
        except Exception:
            logger.exception("Dividend accumulation failed")
            return 0
        finally:
            session.close()

    def accumulate_splits(self, symbols: list[str] | None = None) -> int:
        """Fetch and store split history for watchlist tickers.

        Returns count of new split records stored.
        """
        session = self._session_factory()
        try:
            tickers = symbols or self._get_active_symbols(session)
            total_new = 0
            for sym in tickers:
                try:
                    splits = self._provider.get_splits(sym, limit=5)
                    for s in splits:
                        exec_date = s.get("execution_date")
                        if not exec_date:
                            continue
                        exists = (
                            session.query(TickerSplit)
                            .filter_by(symbol=sym, execution_date=exec_date)
                            .first()
                        )
                        if exists:
                            continue
                        session.add(TickerSplit(
                            symbol=sym,
                            execution_date=exec_date,
                            split_from=s.get("split_from"),
                            split_to=s.get("split_to"),
                        ))
                        total_new += 1
                except Exception:
                    logger.debug("Split fetch failed for %s", sym)
            session.commit()
            if total_new:
                logger.info("Split accumulator: %d new records stored", total_new)
            return total_new
        except Exception:
            logger.exception("Split accumulation failed")
            return 0
        finally:
            session.close()

    @staticmethod
    def _get_active_symbols(session: Session) -> list[str]:
        """Get all active ticker symbols from DB."""
        rows = session.query(Ticker.symbol).filter(Ticker.is_active == True).all()
        return [r[0] for r in rows]

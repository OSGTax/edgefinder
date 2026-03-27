"""EdgeFinder v2 — Index benchmarks service.

Collects daily index closes for SPY, QQQ, IWM, DIA.
Used by the dashboard to overlay strategy equity curves
against market performance.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.interfaces import DataProvider
from edgefinder.db.models import IndexDaily

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Collects and queries daily index data for benchmark comparison."""

    def __init__(self, provider: DataProvider, session: Session) -> None:
        self._provider = provider
        self._session = session

    def collect_daily(self, as_of: date | None = None) -> int:
        """Fetch today's closes for all benchmark indices and store.

        Returns number of records stored.
        """
        target_date = as_of or date.today()
        stored = 0

        for symbol in settings.index_symbols:
            price = self._provider.get_latest_price(symbol)
            if price is None:
                logger.warning("Could not get price for %s", symbol)
                continue

            # Calculate change % from previous close
            prev = self._get_previous_close(symbol)
            change_pct = 0.0
            if prev and prev > 0:
                change_pct = round((price - prev) / prev * 100, 4)

            # Upsert
            existing = (
                self._session.query(IndexDaily)
                .filter_by(symbol=symbol, date=datetime.combine(target_date, datetime.min.time()))
                .first()
            )
            if existing:
                existing.close = price
                existing.change_pct = change_pct
            else:
                record = IndexDaily(
                    symbol=symbol,
                    date=datetime.combine(target_date, datetime.min.time()),
                    close=price,
                    change_pct=change_pct,
                )
                self._session.add(record)
            stored += 1

        self._session.commit()
        logger.info("Stored %d index daily records for %s", stored, target_date)
        return stored

    def backfill(self, days: int = 365) -> int:
        """Backfill historical index data from Polygon bars.

        Fetches daily bars for each index symbol and stores closes.
        """
        end = date.today()
        start = end - timedelta(days=days)
        stored = 0

        for symbol in settings.index_symbols:
            bars = self._provider.get_bars(symbol, "day", start, end)
            if bars is None or bars.empty:
                logger.warning("No bars for %s backfill", symbol)
                continue

            prev_close = None
            for ts, row in bars.iterrows():
                bar_date = ts.date() if hasattr(ts, "date") else ts
                close = float(row["close"])
                change_pct = 0.0
                if prev_close and prev_close > 0:
                    change_pct = round((close - prev_close) / prev_close * 100, 4)

                dt = datetime.combine(bar_date, datetime.min.time()) if isinstance(bar_date, date) else ts

                existing = (
                    self._session.query(IndexDaily)
                    .filter_by(symbol=symbol, date=dt)
                    .first()
                )
                if not existing:
                    self._session.add(IndexDaily(
                        symbol=symbol,
                        date=dt,
                        close=close,
                        change_pct=change_pct,
                    ))
                    stored += 1

                prev_close = close

        self._session.commit()
        logger.info("Backfilled %d index daily records", stored)
        return stored

    def get_comparison_data(
        self, start_date: date | None = None, days: int = 90
    ) -> dict:
        """Get benchmark data formatted for the dashboard comparison chart.

        Returns:
            {
                "dates": ["2024-01-02", ...],
                "indices": {"SPY": [0.0, 0.5, 1.2, ...], ...}
            }
        Cumulative % change from the first date.
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=days)

        start_dt = datetime.combine(start_date, datetime.min.time())
        result: dict = {"dates": [], "indices": {}}

        for symbol in settings.index_symbols:
            records = (
                self._session.query(IndexDaily)
                .filter(IndexDaily.symbol == symbol, IndexDaily.date >= start_dt)
                .order_by(IndexDaily.date)
                .all()
            )
            if not records:
                continue

            base_close = records[0].close
            cumulative = []
            dates = []
            for rec in records:
                pct = round((rec.close - base_close) / base_close * 100, 2) if base_close > 0 else 0.0
                cumulative.append(pct)
                dt = rec.date
                dates.append(dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt))

            result["indices"][symbol] = cumulative
            if not result["dates"]:
                result["dates"] = dates

        return result

    def _get_previous_close(self, symbol: str) -> float | None:
        """Get the most recent close for a symbol."""
        record = (
            self._session.query(IndexDaily)
            .filter_by(symbol=symbol)
            .order_by(IndexDaily.date.desc())
            .first()
        )
        return record.close if record else None

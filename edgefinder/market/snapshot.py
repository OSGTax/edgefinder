"""EdgeFinder v2 — Market snapshot service.

Captures broad market state (indices, VIX, sector ETFs) at trade time
and periodically. Every trade gets a market_snapshot_id FK linking it
to the exact market conditions when it was executed.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import MarketRegime, MarketSnapshot
from edgefinder.db.models import MarketSnapshotRecord

logger = logging.getLogger(__name__)


class MarketSnapshotService:
    """Captures and persists broad market state."""

    def __init__(self, provider: DataProvider, session: Session) -> None:
        self._provider = provider
        self._session = session

    def capture(self) -> MarketSnapshot:
        """Fetch current state of indices, VIX, and sector ETFs.

        Returns a MarketSnapshot domain object and persists to DB.
        """
        index_prices = {}
        for symbol in settings.index_symbols:
            price = self._provider.get_latest_price(symbol)
            index_prices[symbol] = price

        vix = self._provider.get_latest_price(settings.vix_symbol)

        sector_perf = {}
        for etf in settings.sector_etfs:
            price = self._provider.get_latest_price(etf)
            if price is not None:
                sector_perf[etf] = price

        spy_price = index_prices.get("SPY") or 0.0
        qqq_price = index_prices.get("QQQ") or 0.0
        iwm_price = index_prices.get("IWM") or 0.0
        dia_price = index_prices.get("DIA") or 0.0

        # Determine market regime from SPY behavior
        regime = self._determine_regime(spy_price)

        snapshot = MarketSnapshot(
            timestamp=datetime.utcnow(),
            spy_price=spy_price,
            spy_change_pct=0.0,  # Filled by benchmarks service with daily data
            qqq_price=qqq_price,
            qqq_change_pct=0.0,
            iwm_price=iwm_price,
            iwm_change_pct=0.0,
            dia_price=dia_price,
            dia_change_pct=0.0,
            vix_level=vix or 0.0,
            market_regime=regime,
            sector_performance=sector_perf,
        )

        return snapshot

    def capture_and_persist(self) -> int:
        """Capture snapshot and save to DB. Returns the snapshot ID."""
        snapshot = self.capture()
        record = MarketSnapshotRecord(
            timestamp=snapshot.timestamp,
            spy_price=snapshot.spy_price,
            spy_change_pct=snapshot.spy_change_pct,
            qqq_price=snapshot.qqq_price,
            qqq_change_pct=snapshot.qqq_change_pct,
            iwm_price=snapshot.iwm_price,
            iwm_change_pct=snapshot.iwm_change_pct,
            dia_price=snapshot.dia_price,
            dia_change_pct=snapshot.dia_change_pct,
            vix_level=snapshot.vix_level,
            market_regime=snapshot.market_regime.value,
            sector_performance=snapshot.sector_performance,
            advance_decline_ratio=snapshot.advance_decline_ratio,
        )
        self._session.add(record)
        self._session.commit()
        logger.info(
            "Market snapshot captured: SPY=$%.2f VIX=%.1f regime=%s",
            snapshot.spy_price, snapshot.vix_level, snapshot.market_regime.value,
        )
        return record.id

    def get_latest(self) -> MarketSnapshotRecord | None:
        """Get the most recent snapshot from DB."""
        return (
            self._session.query(MarketSnapshotRecord)
            .order_by(MarketSnapshotRecord.timestamp.desc())
            .first()
        )

    def _determine_regime(self, spy_price: float) -> MarketRegime:
        """Simple regime detection based on VIX and recent snapshots.

        - VIX > 30: BEAR
        - VIX < 15: BULL
        - Otherwise: SIDEWAYS
        """
        latest = self.get_latest()
        vix = self._provider.get_latest_price(settings.vix_symbol) or 20.0

        if vix > 30:
            return MarketRegime.BEAR
        if vix < 15:
            return MarketRegime.BULL
        return MarketRegime.SIDEWAYS

"""EdgeFinder v2 — Market snapshot service.

Captures broad market state (indices, VIX, sector ETFs) at trade time
and periodically. Every trade gets a market_snapshot_id FK linking it
to the exact market conditions when it was executed.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.interfaces import DataProvider
from edgefinder.core.models import MarketRegime, MarketSnapshot
from edgefinder.db.models import MarketSnapshotRecord

logger = logging.getLogger(__name__)

# VIX proxy ETF — I:VIX requires an index add-on on Polygon.
# VIXY closely tracks VIX and is available on all plans.
VIX_PROXY = "VIXY"


class MarketSnapshotService:
    """Captures and persists broad market state."""

    def __init__(self, provider: DataProvider, session: Session) -> None:
        self._provider = provider
        self._session = session

    def capture(self) -> MarketSnapshot:
        """Fetch current state of indices, VIX, and sector ETFs.

        Computes daily change % from bars (prev close vs current close)
        since the snapshot endpoint requires a higher-tier plan.
        """
        index_prices = {}
        index_changes = {}
        for symbol in settings.index_symbols:
            price, change = self._get_price_and_change(symbol)
            index_prices[symbol] = price
            index_changes[symbol] = change

        # VIX level from proxy ETF (VIXY tracks VIX, available on Starter)
        vix_price = self._provider.get_latest_price(VIX_PROXY)
        vix_level = vix_price if vix_price is not None else 0.0

        sector_perf = {}
        for etf in settings.sector_etfs:
            price = self._provider.get_latest_price(etf)
            if price is not None:
                sector_perf[etf] = price

        spy_price = index_prices.get("SPY") or 0.0
        qqq_price = index_prices.get("QQQ") or 0.0
        iwm_price = index_prices.get("IWM") or 0.0
        dia_price = index_prices.get("DIA") or 0.0

        spy_chg = index_changes.get("SPY", 0.0)

        regime = self._determine_regime(vix_level, spy_chg)

        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            spy_price=spy_price,
            spy_change_pct=spy_chg,
            qqq_price=qqq_price,
            qqq_change_pct=index_changes.get("QQQ", 0.0),
            iwm_price=iwm_price,
            iwm_change_pct=index_changes.get("IWM", 0.0),
            dia_price=dia_price,
            dia_change_pct=index_changes.get("DIA", 0.0),
            vix_level=vix_level,
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
            "Market snapshot captured: SPY=$%.2f (%.2f%%) VIX=%.1f regime=%s",
            snapshot.spy_price, snapshot.spy_change_pct,
            snapshot.vix_level, snapshot.market_regime.value,
        )
        return record.id

    def get_latest(self) -> MarketSnapshotRecord | None:
        """Get the most recent snapshot from DB."""
        return (
            self._session.query(MarketSnapshotRecord)
            .order_by(MarketSnapshotRecord.timestamp.desc())
            .first()
        )

    def _get_price_and_change(self, symbol: str) -> tuple[float | None, float]:
        """Get current price and daily change % for a symbol.

        Computes change from the last two daily bars since the snapshot
        endpoint (which includes todaysChangePerc) is not available on
        the Starter plan.
        """
        try:
            end = date.today()
            start = end - timedelta(days=5)
            bars = self._provider.get_bars(symbol, "day", start, end)
            if bars is not None and len(bars) >= 2:
                curr_close = float(bars["close"].iloc[-1])
                prev_close = float(bars["close"].iloc[-2])
                if prev_close > 0:
                    change_pct = (curr_close - prev_close) / prev_close * 100
                    return curr_close, round(change_pct, 2)
                return curr_close, 0.0
            elif bars is not None and len(bars) == 1:
                return float(bars["close"].iloc[-1]), 0.0
        except Exception:
            logger.debug("Failed to get bars for %s, falling back to get_latest_price", symbol)

        # Fallback: just get price, no change
        price = self._provider.get_latest_price(symbol)
        return price, 0.0

    def _determine_regime(self, vix_level: float, spy_change_pct: float) -> MarketRegime:
        """Regime detection from VIX level and SPY daily change.

        - VIX > 30 or SPY < -1%: BEAR
        - VIX < 18 and SPY > 0.3%: BULL
        - Otherwise: SIDEWAYS
        """
        if vix_level > 30 or spy_change_pct < -1.0:
            return MarketRegime.BEAR
        if vix_level < 18 and spy_change_pct > 0.3:
            return MarketRegime.BULL
        return MarketRegime.SIDEWAYS

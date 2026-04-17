"""EdgeFinder v2 — Trade analytics: flat feature table from trade history.

Joins trades with market snapshots to build an analysis-ready dataset.
Each row represents one closed trade with all context features needed
for the Delta meta-strategy to learn which conditions produce wins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from edgefinder.analytics.regime import MarketCondition, classify_regime
from edgefinder.db.models import MarketSnapshotRecord, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class TradeFeatures:
    """Flat feature row for a single closed trade."""

    trade_id: str
    strategy_name: str
    symbol: str
    direction: str
    trade_type: str

    # Outcome
    won: bool
    pnl_dollars: float
    pnl_percent: float
    r_multiple: float
    exit_reason: str
    hold_minutes: float  # duration from entry to exit

    # Market context at entry
    regime: MarketCondition
    vix_level: float
    spy_change_pct: float

    # Signal features
    signals_fired: list[str] = field(default_factory=list)
    confidence: float = 0.0

    # Time features
    entry_hour: int = 0
    entry_day_of_week: int = 0  # 0=Monday, 6=Sunday

    # Entry parameters
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0


def build_trade_features(session: Session) -> list[TradeFeatures]:
    """Query all closed trades with their market snapshots and build feature rows.

    Returns a list of TradeFeatures, one per closed trade that has a
    market snapshot attached.
    """
    trades = (
        session.query(TradeRecord)
        .filter(TradeRecord.status == "CLOSED")
        .filter(TradeRecord.pnl_dollars.isnot(None))
        .order_by(TradeRecord.entry_time)
        .all()
    )

    # Pre-fetch all snapshot IDs we need
    snapshot_ids = {t.market_snapshot_id for t in trades if t.market_snapshot_id}
    snapshots: dict[int, MarketSnapshotRecord] = {}
    if snapshot_ids:
        rows = (
            session.query(MarketSnapshotRecord)
            .filter(MarketSnapshotRecord.id.in_(snapshot_ids))
            .all()
        )
        snapshots = {s.id: s for s in rows}

    features: list[TradeFeatures] = []
    for trade in trades:
        snap = snapshots.get(trade.market_snapshot_id) if trade.market_snapshot_id else None

        # Regime classification
        if snap:
            regime = classify_regime(
                vix_level=snap.vix_level,
                spy_change_pct=snap.spy_change_pct,
                market_regime=snap.market_regime,
            )
            vix = snap.vix_level
            spy_chg = snap.spy_change_pct
        else:
            regime = MarketCondition.SIDEWAYS_CALM
            vix = 0.0
            spy_chg = 0.0

        # Hold duration
        hold_minutes = 0.0
        if trade.exit_time and trade.entry_time:
            delta = trade.exit_time - trade.entry_time
            hold_minutes = delta.total_seconds() / 60.0

        # Extract signal patterns from technical_signals JSON
        signals_fired = _extract_signal_patterns(trade.technical_signals)

        features.append(TradeFeatures(
            trade_id=trade.trade_id,
            strategy_name=trade.strategy_name,
            symbol=trade.symbol,
            direction=trade.direction,
            trade_type=trade.trade_type,
            won=(trade.pnl_dollars or 0) > 0,
            pnl_dollars=trade.pnl_dollars or 0.0,
            pnl_percent=trade.pnl_percent or 0.0,
            r_multiple=trade.r_multiple or 0.0,
            exit_reason=trade.exit_reason or "",
            hold_minutes=hold_minutes,
            regime=regime,
            vix_level=vix,
            spy_change_pct=spy_chg,
            signals_fired=signals_fired,
            confidence=trade.confidence or 0.0,
            entry_hour=trade.entry_time.hour if trade.entry_time else 0,
            entry_day_of_week=trade.entry_time.weekday() if trade.entry_time else 0,
            entry_price=trade.entry_price,
            stop_loss=trade.stop_loss,
            target=trade.target,
        ))

    logger.info("Built %d trade feature rows from %d closed trades", len(features), len(trades))
    return features


def _extract_signal_patterns(technical_signals: dict | None) -> list[str]:
    """Extract signal pattern names from the technical_signals JSON blob."""
    if not technical_signals:
        return []

    patterns = []
    # technical_signals can store indicator values or pattern names
    # Check common structures
    if isinstance(technical_signals, dict):
        # If it has a "patterns" key, use that
        if "patterns" in technical_signals:
            p = technical_signals["patterns"]
            if isinstance(p, list):
                patterns = p
        # Otherwise look for pattern-like keys
        else:
            for key, val in technical_signals.items():
                if isinstance(val, bool) and val:
                    patterns.append(key)
    return patterns

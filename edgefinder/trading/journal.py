"""EdgeFinder v2 — Trade journal with market context.

Logs every trade to the database with full context:
strategy, fundamentals, signals, sentiment, and market snapshot.
Provides stats queries for the optimizer and dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy.orm import Session

from edgefinder.core.models import Trade, TradeStatus
from edgefinder.db.models import TradeRecord

logger = logging.getLogger(__name__)


class TradeJournal:
    """Persists trades to database and provides stats queries."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def log_trade(self, trade: Trade, commit: bool = True) -> None:
        """Persist a trade (open or closed) to the database.

        Pass commit=False when the caller wants to batch this write with
        other updates (e.g. the account-state row) into a single atomic
        transaction.
        """
        existing = (
            self._session.query(TradeRecord)
            .filter_by(trade_id=trade.trade_id)
            .first()
        )

        if existing:
            # Update existing trade (e.g., closing an open trade)
            existing.exit_price = trade.exit_price
            existing.exit_time = trade.exit_time
            existing.status = trade.status.value
            existing.pnl_dollars = trade.pnl_dollars
            existing.pnl_percent = trade.pnl_percent
            existing.r_multiple = trade.r_multiple
            existing.exit_reason = trade.exit_reason
            existing.market_snapshot_id = trade.market_snapshot_id
        else:
            record = TradeRecord(
                trade_id=trade.trade_id,
                strategy_name=trade.strategy_name,
                symbol=trade.symbol,
                direction=trade.direction.value,
                trade_type=trade.trade_type.value,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                shares=trade.shares,
                stop_loss=trade.stop_loss,
                target=trade.target,
                confidence=trade.confidence,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                status=trade.status.value,
                pnl_dollars=trade.pnl_dollars,
                pnl_percent=trade.pnl_percent,
                r_multiple=trade.r_multiple,
                exit_reason=trade.exit_reason,
                market_snapshot_id=trade.market_snapshot_id,
                sentiment_data=trade.sentiment_data,
                technical_signals=trade.technical_signals,
                sequence_num=trade.sequence_num,
                integrity_hash=trade.integrity_hash,
            )
            self._session.add(record)

        if commit:
            self._session.commit()

    def get_trades(
        self,
        strategy_name: str | None = None,
        status: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[TradeRecord]:
        """Query trades with optional filters."""
        q = self._session.query(TradeRecord)
        if strategy_name:
            q = q.filter(TradeRecord.strategy_name == strategy_name)
        if status:
            q = q.filter(TradeRecord.status == status)
        if symbol:
            q = q.filter(TradeRecord.symbol == symbol)
        return q.order_by(TradeRecord.created_at.desc()).limit(limit).all()

    def get_open_trades(self, strategy_name: str | None = None) -> list[TradeRecord]:
        return self.get_trades(strategy_name=strategy_name, status="OPEN")

    def get_closed_trades(self, strategy_name: str | None = None) -> list[TradeRecord]:
        return self.get_trades(strategy_name=strategy_name, status="CLOSED")

    def compute_stats(self, strategy_name: str | None = None) -> dict:
        """Compute trading statistics for a strategy (or all)."""
        closed = self.get_closed_trades(strategy_name)
        if not closed:
            return {"total_trades": 0}

        wins = [t for t in closed if t.pnl_dollars and t.pnl_dollars > 0]
        losses = [t for t in closed if t.pnl_dollars and t.pnl_dollars <= 0]

        total_pnl = sum(t.pnl_dollars or 0 for t in closed)
        gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0

        avg_r = (
            sum(t.r_multiple or 0 for t in closed) / len(closed) if closed else 0
        )

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(closed), 2),
            "avg_r_multiple": round(avg_r, 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
            "largest_win": round(max((t.pnl_dollars or 0) for t in closed), 2),
            "largest_loss": round(min((t.pnl_dollars or 0) for t in closed), 2),
        }

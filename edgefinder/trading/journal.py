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
            # Update existing trade (e.g., closing an open trade).
            # NOTE: sequence_num/integrity_hash are intentionally never
            # touched here — the hash chain covers row INSERTION order and
            # must stay immutable for verification (see _next_chain_link).
            existing.exit_price = trade.exit_price
            existing.exit_time = trade.exit_time
            # v2 rebalance lot-splits close a REDUCED share count (the
            # remainder reopens as its own row); old callers pass shares
            # unchanged, so this is a no-op for them.
            existing.shares = trade.shares
            existing.status = trade.status.value
            existing.pnl_dollars = trade.pnl_dollars
            existing.pnl_percent = trade.pnl_percent
            existing.r_multiple = trade.r_multiple
            existing.exit_reason = trade.exit_reason
            existing.market_snapshot_id = trade.market_snapshot_id
            existing.exit_reasoning = getattr(trade, 'exit_reasoning', None)
            existing.indicators_at_exit = getattr(trade, 'indicators_at_exit', None)
            existing.pdt_flag = getattr(trade, 'pdt_flag', False)
            existing.hold_duration_hours = getattr(trade, 'hold_duration_hours', None)
        else:
            sequence_num, integrity_hash = self._next_chain_link(trade)
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
                sentiment_score=trade.sentiment_score,
                sentiment_data=trade.sentiment_data,
                technical_signals=trade.technical_signals,
                sequence_num=sequence_num,
                integrity_hash=integrity_hash,
                entry_reasoning=getattr(trade, 'entry_reasoning', None),
                indicators_at_entry=getattr(trade, 'indicators_at_entry', None),
                fundamentals_at_entry=getattr(trade, 'fundamentals_at_entry', None),
                market_context_at_entry=getattr(trade, 'market_context_at_entry', None),
            )
            self._session.add(record)

        if commit:
            self._session.commit()

    def _next_chain_link(self, trade: Trade) -> tuple[int, str]:
        """Next (sequence_num, integrity_hash) for this strategy's hash chain.

        v2 scheme (2026-06-05): the chain is computed at PERSISTENCE time,
        per strategy, anchored to the previous STORED row — so it survives
        process restarts and every link is verifiable from the DB alone:

            hash_n = sha256(f"{trade_id}:{seq_n}:{hash_(n-1)}")   (hash_0 = "")

        The old scheme chained in-memory executor state: close events
        advanced the chain but their hashes were never stored, and the
        anchor reset every boot — making stored rows structurally
        unverifiable. Rows written before this fix remain a legacy segment;
        the first v2 row chains onto the last legacy row's stored hash,
        which also freezes the legacy tail against silent edits.
        """
        import hashlib

        prev = (
            self._session.query(
                TradeRecord.sequence_num, TradeRecord.integrity_hash
            )
            .filter(
                TradeRecord.strategy_name == trade.strategy_name,
                TradeRecord.sequence_num.is_not(None),
            )
            .order_by(TradeRecord.sequence_num.desc())
            .first()
        )
        seq = (prev[0] + 1) if prev and prev[0] else 1
        prev_hash = (prev[1] or "") if prev else ""
        digest = hashlib.sha256(
            f"{trade.trade_id}:{seq}:{prev_hash}".encode()
        ).hexdigest()
        return seq, digest

    def get_trades(
        self,
        strategy_name: str | None = None,
        status: str | None = None,
        symbol: str | None = None,
        limit: int | None = 100,
    ) -> list[TradeRecord]:
        """Query trades with optional filters (``limit=None`` = no cap)."""
        q = self._session.query(TradeRecord)
        if strategy_name:
            q = q.filter(TradeRecord.strategy_name == strategy_name)
        if status:
            q = q.filter(TradeRecord.status == status)
        if symbol:
            q = q.filter(TradeRecord.symbol == symbol)
        # the model defers the rich-context columns (reasoning, indicator
        # JSON, ...); callers here serialize them, and leaving them deferred
        # meant FIVE lazy round-trips PER ROW — ~50s for the fleet's 148
        # open lots over the production pooler (the dashboard trades page
        # hang, 2026-06-11). One query, all columns.
        from sqlalchemy.orm import undefer

        q = q.options(undefer("*"))
        q = q.order_by(TradeRecord.created_at.desc())
        if limit is not None:
            q = q.limit(limit)
        return q.all()

    # UNCAPPED on purpose: these back the live engine's lot accounting and
    # the stats — a default limit=100 silently dropped lots once the fleet
    # held ~146 open positions (the dashboard read "100 open")
    def get_open_trades(self, strategy_name: str | None = None) -> list[TradeRecord]:
        return self.get_trades(strategy_name=strategy_name, status="OPEN", limit=None)

    def get_closed_trades(self, strategy_name: str | None = None) -> list[TradeRecord]:
        return self.get_trades(strategy_name=strategy_name, status="CLOSED", limit=None)

    def compute_stats(self, strategy_name: str | None = None) -> dict:
        """Compute trading statistics for a strategy (or all)."""
        open_count = len(self.get_open_trades(strategy_name))
        closed = self.get_closed_trades(strategy_name)
        if not closed:
            # total = ALL trades; the old "total_trades: 0" with open lots
            # on the books rendered as "TRADES 0" on the dashboard
            return {"total_trades": open_count, "open_trades": open_count,
                    "closed_trades": 0}

        wins = [t for t in closed if t.pnl_dollars and t.pnl_dollars > 0]
        losses = [t for t in closed if t.pnl_dollars and t.pnl_dollars <= 0]

        total_pnl = sum(t.pnl_dollars or 0 for t in closed)
        gross_profit = sum(t.pnl_dollars for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 0

        avg_r = (
            sum(t.r_multiple or 0 for t in closed) / len(closed) if closed else 0
        )
        avg_pnl_pct = (
            sum(t.pnl_percent or 0 for t in closed) / len(closed) if closed else 0
        )

        return {
            "total_trades": open_count + len(closed),
            "open_trades": open_count,
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / len(closed), 2),
            "avg_pnl_percent": round(avg_pnl_pct, 2),
            "avg_r_multiple": round(avg_r, 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else None,
            "largest_win": round(max((t.pnl_dollars or 0) for t in closed), 2),
            "largest_loss": round(min((t.pnl_dollars or 0) for t in closed), 2),
        }

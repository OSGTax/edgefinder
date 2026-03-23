"""
EdgeFinder Module 4: Trade Journal
====================================
Logs every trade and skipped signal with full context.
Provides querying, statistics, and reporting:
- Trade log with full entry/exit details
- Win rate, average R-multiple, profit factor
- Equity curve from account snapshots
- Skipped signal tracking for strategy optimization

Every trade and signal is persisted for later analysis by the optimizer.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import desc, func

from config import settings
from modules.database import (
    Trade as TradeRecord,
    Signal as SignalRecord,
    AccountSnapshot,
    get_session,
)
from modules.trader import TradeResult, Position

logger = logging.getLogger(__name__)


# ── DATA CLASSES ─────────────────────────────────────────────

@dataclass
class TradeStats:
    """Aggregated trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    avg_r_multiple: float = 0.0
    profit_factor: float = 0.0      # gross_profit / gross_loss
    total_signals: int = 0
    traded_signals: int = 0
    skipped_signals: int = 0
    day_trades: int = 0
    swing_trades: int = 0


@dataclass
class JournalEntry:
    """A single journal entry combining trade result with context."""
    trade_id: str
    ticker: str
    direction: str
    trade_type: str
    entry_price: float
    exit_price: float
    shares: int
    pnl_dollars: float
    pnl_percent: float
    r_multiple: float
    exit_reason: str
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    fundamental_score: float = 0.0
    confidence_score: float = 0.0
    news_sentiment: float = 0.0
    technical_signals: dict = field(default_factory=dict)


# ── TRADE JOURNAL ────────────────────────────────────────────

class TradeJournal:
    """
    Records, queries, and analyzes all trades and signals.

    Works with the database to persist trade records and provide
    statistics for the strategy optimizer.
    """

    # ── LOGGING TRADES ───────────────────────────────────────

    def log_trade(
        self,
        result: TradeResult,
        position: Optional[Position] = None,
    ) -> None:
        """
        Log a completed trade to the database.

        Args:
            result: The TradeResult from closing a position.
            position: The original Position (for context fields).
        """
        try:
            session = get_session()
            record = TradeRecord(
                trade_id=result.trade_id,
                ticker=result.ticker,
                direction=result.direction,
                trade_type=result.trade_type,
                entry_price=result.entry_price,
                exit_price=result.exit_price,
                shares=result.shares,
                stop_loss=result.stop_loss,
                target=result.target,
                entry_time=result.entry_time,
                exit_time=result.exit_time,
                status="CLOSED",
                pnl_dollars=result.pnl_dollars,
                pnl_percent=result.pnl_percent,
                r_multiple=result.r_multiple,
                exit_reason=result.exit_reason,
                fundamental_score=position.fundamental_score if position else None,
                technical_signals=position.technical_signals if position else None,
                news_sentiment=position.news_sentiment if position else None,
                confidence_score=position.confidence_score if position else None,
            )
            session.add(record)
            session.commit()
            logger.info(
                f"Journal: {result.ticker} | {result.exit_reason} | "
                f"P&L: ${result.pnl_dollars:+.2f} | R: {result.r_multiple:+.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
            session.rollback()
        finally:
            session.close()

    def log_skipped_signal(
        self,
        ticker: str,
        signal_type: str,
        trade_type: str,
        confidence: float,
        reason: str,
        indicators: Optional[dict] = None,
    ) -> None:
        """
        Log a signal that was detected but not traded.

        Args:
            ticker: Stock ticker.
            signal_type: "BUY" or "SELL".
            trade_type: "DAY" or "SWING".
            confidence: Signal confidence score.
            reason: Why the signal was skipped.
            indicators: Which indicators fired.
        """
        try:
            session = get_session()
            record = SignalRecord(
                ticker=ticker,
                signal_type=signal_type,
                trade_type=trade_type,
                confidence=confidence,
                indicators=indicators or {},
                was_traded=False,
                reason_skipped=reason,
                timestamp=datetime.now(timezone.utc),
            )
            session.add(record)
            session.commit()
            logger.info(f"Journal: Skipped {ticker} {signal_type} — {reason}")
        except Exception as e:
            logger.error(f"Failed to log skipped signal: {e}")
            session.rollback()
        finally:
            session.close()

    # ── QUERYING ─────────────────────────────────────────────

    def get_trades(
        self,
        ticker: Optional[str] = None,
        trade_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[JournalEntry]:
        """
        Query trade records from the database.

        Args:
            ticker: Filter by ticker (None = all).
            trade_type: Filter by "DAY" or "SWING" (None = all).
            limit: Max records to return.

        Returns:
            List of JournalEntry objects, most recent first.
        """
        try:
            session = get_session()
            query = session.query(TradeRecord).filter(TradeRecord.status == "CLOSED")

            if ticker:
                query = query.filter(TradeRecord.ticker == ticker)
            if trade_type:
                query = query.filter(TradeRecord.trade_type == trade_type)

            records = query.order_by(desc(TradeRecord.exit_time)).limit(limit).all()

            entries = []
            for r in records:
                entries.append(JournalEntry(
                    trade_id=r.trade_id,
                    ticker=r.ticker,
                    direction=r.direction or "LONG",
                    trade_type=r.trade_type or "DAY",
                    entry_price=r.entry_price or 0,
                    exit_price=r.exit_price or 0,
                    shares=r.shares or 0,
                    pnl_dollars=r.pnl_dollars or 0,
                    pnl_percent=r.pnl_percent or 0,
                    r_multiple=r.r_multiple or 0,
                    exit_reason=r.exit_reason or "",
                    entry_time=r.entry_time,
                    exit_time=r.exit_time,
                    fundamental_score=r.fundamental_score or 0,
                    confidence_score=r.confidence_score or 0,
                    news_sentiment=r.news_sentiment or 0,
                    technical_signals=r.technical_signals or {},
                ))
            return entries
        except Exception as e:
            logger.error(f"Failed to query trades: {e}")
            return []
        finally:
            session.close()

    def get_skipped_signals(
        self,
        ticker: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """
        Query skipped signals from the database.

        Returns:
            List of dicts with signal details and skip reason.
        """
        try:
            session = get_session()
            query = session.query(SignalRecord).filter(
                SignalRecord.was_traded == False  # noqa: E712
            )
            if ticker:
                query = query.filter(SignalRecord.ticker == ticker)

            records = query.order_by(desc(SignalRecord.timestamp)).limit(limit).all()

            return [
                {
                    "ticker": r.ticker,
                    "signal_type": r.signal_type,
                    "trade_type": r.trade_type,
                    "confidence": r.confidence,
                    "reason_skipped": r.reason_skipped,
                    "indicators": r.indicators,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in records
            ]
        except Exception as e:
            logger.error(f"Failed to query skipped signals: {e}")
            return []
        finally:
            session.close()

    # ── STATISTICS ───────────────────────────────────────────

    def compute_stats(
        self,
        days: Optional[int] = None,
    ) -> TradeStats:
        """
        Compute trading statistics from the trade log.

        Args:
            days: Only include trades from the last N days (None = all time).

        Returns:
            TradeStats with aggregated metrics.
        """
        stats = TradeStats()

        try:
            session = get_session()

            # Trade query
            query = session.query(TradeRecord).filter(TradeRecord.status == "CLOSED")
            if days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                query = query.filter(TradeRecord.exit_time >= cutoff)

            trades = query.all()
            stats.total_trades = len(trades)

            if not trades:
                # Signal stats
                stats.total_signals = session.query(SignalRecord).count()
                stats.traded_signals = session.query(SignalRecord).filter(
                    SignalRecord.was_traded == True  # noqa: E712
                ).count()
                stats.skipped_signals = stats.total_signals - stats.traded_signals
                return stats

            # Classify trades
            winners = [t for t in trades if t.pnl_dollars and t.pnl_dollars > 0]
            losers = [t for t in trades if t.pnl_dollars and t.pnl_dollars < 0]
            breakevens = [t for t in trades if not t.pnl_dollars or t.pnl_dollars == 0]

            stats.winning_trades = len(winners)
            stats.losing_trades = len(losers)
            stats.breakeven_trades = len(breakevens)
            stats.win_rate = len(winners) / len(trades) if trades else 0.0

            # P&L stats
            all_pnl = [t.pnl_dollars or 0 for t in trades]
            stats.total_pnl = round(sum(all_pnl), 2)
            stats.avg_pnl = round(stats.total_pnl / len(trades), 2) if trades else 0.0

            if winners:
                winner_pnls = [t.pnl_dollars for t in winners]
                stats.avg_winner = round(sum(winner_pnls) / len(winners), 2)
                stats.largest_winner = round(max(winner_pnls), 2)

            if losers:
                loser_pnls = [t.pnl_dollars for t in losers]
                stats.avg_loser = round(sum(loser_pnls) / len(losers), 2)
                stats.largest_loser = round(min(loser_pnls), 2)

            # R-multiple
            r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
            if r_multiples:
                stats.avg_r_multiple = round(sum(r_multiples) / len(r_multiples), 2)

            # Profit factor
            gross_profit = sum(t.pnl_dollars for t in winners) if winners else 0
            gross_loss = abs(sum(t.pnl_dollars for t in losers)) if losers else 0
            stats.profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

            # Trade type breakdown
            stats.day_trades = sum(1 for t in trades if t.trade_type == "DAY")
            stats.swing_trades = sum(1 for t in trades if t.trade_type == "SWING")

            # Signal stats
            stats.total_signals = session.query(SignalRecord).count()
            stats.traded_signals = session.query(SignalRecord).filter(
                SignalRecord.was_traded == True  # noqa: E712
            ).count()
            stats.skipped_signals = stats.total_signals - stats.traded_signals

            return stats

        except Exception as e:
            logger.error(f"Failed to compute stats: {e}")
            return stats
        finally:
            session.close()

    def get_equity_curve(self, limit: int = 365) -> list[dict]:
        """
        Get account snapshots for equity curve plotting.

        Returns:
            List of dicts with date, total_value, drawdown_pct.
        """
        try:
            session = get_session()
            snapshots = session.query(AccountSnapshot).order_by(
                desc(AccountSnapshot.date)
            ).limit(limit).all()

            # Reverse to chronological order
            snapshots.reverse()

            return [
                {
                    "date": s.date.isoformat() if s.date else None,
                    "cash": s.cash,
                    "positions_value": s.positions_value,
                    "total_value": s.total_value,
                    "open_positions": s.open_positions,
                    "peak_value": s.peak_value,
                    "drawdown_pct": s.drawdown_pct,
                }
                for s in snapshots
            ]
        except Exception as e:
            logger.error(f"Failed to get equity curve: {e}")
            return []
        finally:
            session.close()

    # ── REPORTING ────────────────────────────────────────────

    def print_summary(self, days: Optional[int] = None) -> str:
        """
        Generate a human-readable trade summary.

        Args:
            days: Period to summarize (None = all time).

        Returns:
            Formatted summary string.
        """
        stats = self.compute_stats(days=days)
        period = f"Last {days} days" if days else "All time"

        lines = [
            "=" * 60,
            f"  EDGEFINDER TRADE JOURNAL — {period}",
            "=" * 60,
            f"  Total trades:     {stats.total_trades}",
            f"  Winners:          {stats.winning_trades} ({stats.win_rate:.1%})",
            f"  Losers:           {stats.losing_trades}",
            f"  Breakeven:        {stats.breakeven_trades}",
            "-" * 60,
            f"  Total P&L:        ${stats.total_pnl:+,.2f}",
            f"  Avg P&L:          ${stats.avg_pnl:+,.2f}",
            f"  Avg Winner:       ${stats.avg_winner:+,.2f}",
            f"  Avg Loser:        ${stats.avg_loser:+,.2f}",
            f"  Largest Winner:   ${stats.largest_winner:+,.2f}",
            f"  Largest Loser:    ${stats.largest_loser:+,.2f}",
            "-" * 60,
            f"  Avg R-Multiple:   {stats.avg_r_multiple:+.2f}R",
            f"  Profit Factor:    {stats.profit_factor:.2f}",
            "-" * 60,
            f"  Day Trades:       {stats.day_trades}",
            f"  Swing Trades:     {stats.swing_trades}",
            f"  Signals (total):  {stats.total_signals}",
            f"  Signals traded:   {stats.traded_signals}",
            f"  Signals skipped:  {stats.skipped_signals}",
            "=" * 60,
        ]

        summary = "\n".join(lines)
        logger.info(f"\n{summary}")
        return summary

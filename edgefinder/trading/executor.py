"""EdgeFinder v2 — Trade executor with honest slippage modeling.

Sizes positions based on risk management rules, applies slippage,
and manages the lifecycle of trades in virtual accounts.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy.orm import Session

from config.settings import settings
from edgefinder.core.events import event_bus
from edgefinder.core.models import Direction, Signal, Trade, TradeStatus, TradeType
from edgefinder.trading.account import Position, VirtualAccount

logger = logging.getLogger(__name__)


class ChainIntegrityError(Exception):
    """Raised when the integrity hash chain doesn't verify on boot."""


def _compute_chain_hash(trade_id: str, sequence_num: int, prev_hash: str) -> str:
    """Pure function used by both live writes and chain verification."""
    data = f"{trade_id}:{sequence_num}:{prev_hash}"
    return hashlib.sha256(data.encode()).hexdigest()


class Executor:
    """Executes signals into positions within a virtual account."""

    def __init__(self, account: VirtualAccount) -> None:
        self.account = account
        self._sequence_num = 0
        self._prev_hash = ""

    def restore_hash_chain(self, session: Session) -> None:
        """Seed sequence_num / prev_hash from this strategy's last trade.

        Without this, a deploy or restart resets both to 0 / "" and the
        next trade's hash chains to an empty string — a permanent gap in
        the audit trail. Call once per executor after account state is
        restored.
        """
        from edgefinder.db.models import TradeRecord

        last = (
            session.query(TradeRecord.sequence_num, TradeRecord.integrity_hash)
            .filter(
                TradeRecord.strategy_name == self.account.strategy_name,
                TradeRecord.sequence_num.isnot(None),
            )
            .order_by(TradeRecord.sequence_num.desc())
            .first()
        )
        if last is None:
            return
        seq, hsh = last
        self._sequence_num = int(seq or 0)
        self._prev_hash = hsh or ""
        logger.info(
            "[%s] Restored hash chain: sequence_num=%d prev_hash=%s",
            self.account.strategy_name, self._sequence_num,
            self._prev_hash[:12] + "…" if self._prev_hash else "<empty>",
        )

    def verify_chain(self, session: Session) -> tuple[bool, int]:
        """Recompute every row's hash for this strategy and compare.

        Returns (ok, checked_count). When a mismatch is found, logs
        pointedly (strategy, sequence_num, trade_id) and returns False
        without raising — callers choose whether to halt trading.
        """
        from edgefinder.db.models import TradeRecord

        rows = (
            session.query(
                TradeRecord.trade_id,
                TradeRecord.sequence_num,
                TradeRecord.integrity_hash,
            )
            .filter(
                TradeRecord.strategy_name == self.account.strategy_name,
                TradeRecord.sequence_num.isnot(None),
            )
            .order_by(TradeRecord.sequence_num.asc())
            .all()
        )

        prev = ""
        checked = 0
        for trade_id, seq, stored_hash in rows:
            expected = _compute_chain_hash(trade_id, int(seq), prev)
            if expected != (stored_hash or ""):
                logger.error(
                    "[%s] Chain integrity mismatch at sequence=%d trade_id=%s: "
                    "expected %s, stored %s",
                    self.account.strategy_name, seq, trade_id,
                    expected[:12], (stored_hash or "")[:12],
                )
                return False, checked
            prev = stored_hash or ""
            checked += 1
        return True, checked

    def execute_signal(
        self,
        signal: Signal,
        fresh_price: float | None = None,
        sector: str | None = None,
    ) -> Trade | None:
        """Execute a signal: size, slippage, open position, return Trade.

        Args:
            signal: The signal to execute. signal.entry_price is the close
                of the bar where the pattern was detected — usually a few
                minutes stale by the time we get here.
            fresh_price: If provided, use this as the base execution price
                instead of signal.entry_price. The arena fetches a fresh
                quote at execution time so the entry is anchored to the
                current market price (not the historical pattern bar),
                which keeps entry/exit price sources consistent and
                prevents instant target-hits driven by timeframe mismatch.

        Returns None if the signal is rejected (insufficient funds, risk checks, etc.).
        """
        # Use fresh market price if available, fall back to the pattern bar close
        base_price = fresh_price if fresh_price is not None else signal.entry_price
        # Apply slippage first so sizing uses the actual execution price
        execution_price = self._apply_slippage(base_price, signal.action.value)

        # Size the position using slippage-adjusted price
        shares, cost = self._size_position(signal, execution_price)
        if shares <= 0:
            logger.debug("Signal rejected: position size is 0 for %s", signal.ticker)
            return None

        # Check account rules with actual cost
        allowed, reason = self.account.can_open_position(
            cost, signal.trade_type.value, symbol=signal.ticker, sector=sector,
        )
        if not allowed:
            logger.info(
                "[%s] Signal rejected for %s: %s",
                self.account.strategy_name, signal.ticker, reason,
            )
            return None

        # Create position
        trade_id = str(uuid.uuid4())
        position = Position(
            symbol=signal.ticker,
            shares=shares,
            entry_price=execution_price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            direction="LONG" if signal.action.value == "BUY" else "SHORT",
            trade_type=signal.trade_type.value,
            entry_time=datetime.now(timezone.utc),
            trade_id=trade_id,
            sector=sector,
        )

        self.account.open_position(position)
        self._sequence_num += 1

        trade = Trade(
            trade_id=trade_id,
            strategy_name=signal.strategy_name or self.account.strategy_name,
            symbol=signal.ticker,
            direction=Direction.LONG if signal.action.value == "BUY" else Direction.SHORT,
            trade_type=TradeType(signal.trade_type.value),
            entry_price=execution_price,
            shares=shares,
            stop_loss=signal.stop_loss,
            target=signal.target,
            confidence=signal.confidence,
            status=TradeStatus.OPEN,
            technical_signals=signal.indicators,
            entry_time=datetime.now(timezone.utc),
            sequence_num=self._sequence_num,
            integrity_hash=self._compute_hash(trade_id),
        )

        event_bus.publish("trade.opened", trade)
        return trade

    def check_positions(self, prices: dict[str, float]) -> list[Trade]:
        """Check all open positions against current prices.

        Stamps each position's `current_price` with the latest mark so
        total_equity, market_value, and drawdown_pct are accurate
        mark-to-market. Returns list of closed Trade objects.
        """
        closed_trades: list[Trade] = []

        for position in list(self.account.positions):
            price = prices.get(position.symbol)
            if price is None:
                continue

            # Mark-to-market stamp — feeds account.total_equity and
            # drawdown_pct regardless of whether this tick closes.
            position.current_price = price

            reason = None
            if position.should_stop_out(price):
                reason = "STOP_HIT"
            elif position.should_take_profit(price):
                reason = "TARGET_HIT"

            if reason:
                result = self.account.close_position(position, price, reason)
                self._sequence_num += 1

                trade = Trade(
                    trade_id=result["trade_id"],
                    strategy_name=self.account.strategy_name,
                    symbol=result["symbol"],
                    direction=Direction(result["direction"]),
                    trade_type=TradeType(result["trade_type"]),
                    entry_price=result["entry_price"],
                    exit_price=result["exit_price"],
                    shares=result["shares"],
                    stop_loss=position.stop_loss,
                    target=position.target,
                    confidence=0,
                    status=TradeStatus.CLOSED,
                    pnl_dollars=result["pnl_dollars"],
                    pnl_percent=result["pnl_percent"],
                    r_multiple=result["r_multiple"],
                    exit_reason=reason,
                    exit_time=datetime.now(timezone.utc),
                    sequence_num=self._sequence_num,
                    integrity_hash=self._compute_hash(result["trade_id"]),
                )
                closed_trades.append(trade)
                event_bus.publish("trade.closed", trade)

        # After this tick's marks have been applied and any closes
        # recorded, update peak_equity so drawdown tracks real market
        # value, not just values seen at close time.
        self.account.update_peak_equity()

        return closed_trades

    def close_on_signal(self, position: Position, price: float, signal_pattern: str) -> Trade:
        """Close an open position due to a bearish exit signal."""
        reason = f"SIGNAL_EXIT:{signal_pattern}"
        result = self.account.close_position(position, price, reason)
        self._sequence_num += 1

        trade = Trade(
            trade_id=result["trade_id"],
            strategy_name=self.account.strategy_name,
            symbol=result["symbol"],
            direction=Direction(result["direction"]),
            trade_type=TradeType(result["trade_type"]),
            entry_price=result["entry_price"],
            exit_price=result["exit_price"],
            shares=result["shares"],
            stop_loss=position.stop_loss,
            target=position.target,
            confidence=0,
            status=TradeStatus.CLOSED,
            pnl_dollars=result["pnl_dollars"],
            pnl_percent=result["pnl_percent"],
            r_multiple=result["r_multiple"],
            exit_reason=reason,
            exit_time=datetime.now(timezone.utc),
            sequence_num=self._sequence_num,
            integrity_hash=self._compute_hash(result["trade_id"]),
        )
        event_bus.publish("trade.closed", trade)
        return trade

    # ── Private ──────────────────────────────────────

    def _size_position(self, signal: Signal, execution_price: float) -> tuple[int, float]:
        """Calculate position size based on risk management.

        Uses the slippage-adjusted execution_price so cost checks are accurate.
        Max risk per trade = account equity * max_risk_per_trade_pct.
        Shares = max_risk / risk_per_share.
        Cost = shares * execution_price.
        """
        if signal.shares > 0:
            return signal.shares, signal.shares * execution_price

        risk_per_share = signal.risk_per_share
        if risk_per_share <= 0:
            return 0, 0

        # Per-strategy risk config (set via strategy.risk_config)
        risk_pct = self.account.max_risk_pct or 0.02  # fallback 2%
        max_risk = self.account.total_equity * risk_pct
        shares = int(max_risk / risk_per_share)

        # Cap by buying power
        max_by_cash = int(self.account.buying_power / execution_price)
        shares = min(shares, max_by_cash)

        # Cap by concentration (per-strategy)
        conc_pct = self.account.max_concentration_pct or 0.20  # fallback 20%
        max_concentration = self.account.total_equity * conc_pct
        max_by_concentration = int(max_concentration / execution_price)
        shares = min(shares, max_by_concentration)

        if shares <= 0:
            return 0, 0

        # Minimum position cost — reject micro-positions that waste a slot
        cost = shares * execution_price
        min_cost = self.account.starting_capital * 0.01  # 1% of starting capital ($50)
        if cost < min_cost:
            logger.info(
                "[%s] Position too small for %s: %d shares @ $%.2f = $%.2f (min $%.2f)",
                self.account.strategy_name, signal.ticker,
                shares, execution_price, cost, min_cost,
            )
            return 0, 0

        return shares, cost

    @staticmethod
    def _apply_slippage(price: float, action: str) -> float:
        """Apply realistic slippage to execution price."""
        slip = price * settings.slippage_base_rate
        if action == "BUY":
            return round(price + slip, 2)
        return round(price - slip, 2)

    def _compute_hash(self, trade_id: str) -> str:
        """SHA-256 hash chaining for audit trail."""
        h = _compute_chain_hash(trade_id, self._sequence_num, self._prev_hash)
        self._prev_hash = h
        return h

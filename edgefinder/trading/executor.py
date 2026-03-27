"""EdgeFinder v2 — Trade executor with honest slippage modeling.

Sizes positions based on risk management rules, applies slippage,
and manages the lifecycle of trades in virtual accounts.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime

from config.settings import settings
from edgefinder.core.events import event_bus
from edgefinder.core.models import Direction, Signal, Trade, TradeStatus, TradeType
from edgefinder.trading.account import Position, VirtualAccount

logger = logging.getLogger(__name__)


class Executor:
    """Executes signals into positions within a virtual account."""

    def __init__(self, account: VirtualAccount) -> None:
        self.account = account
        self._sequence_num = 0
        self._prev_hash = ""

    def execute_signal(self, signal: Signal) -> Trade | None:
        """Execute a signal: size, slippage, open position, return Trade.

        Returns None if the signal is rejected (insufficient funds, risk checks, etc.).
        """
        # Size the position
        shares, cost = self._size_position(signal)
        if shares <= 0:
            logger.debug("Signal rejected: position size is 0 for %s", signal.ticker)
            return None

        # Check account rules
        allowed, reason = self.account.can_open_position(cost, signal.trade_type.value)
        if not allowed:
            logger.info(
                "[%s] Signal rejected for %s: %s",
                self.account.strategy_name, signal.ticker, reason,
            )
            return None

        # Apply slippage
        execution_price = self._apply_slippage(signal.entry_price, signal.action.value)

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
            entry_time=datetime.utcnow(),
            trade_id=trade_id,
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
            entry_time=datetime.utcnow(),
            sequence_num=self._sequence_num,
            integrity_hash=self._compute_hash(trade_id),
        )

        event_bus.publish("trade.opened", trade)
        return trade

    def check_positions(self, prices: dict[str, float]) -> list[Trade]:
        """Check all open positions against current prices.

        Returns list of closed Trade objects.
        """
        closed_trades: list[Trade] = []

        for position in list(self.account.positions):
            price = prices.get(position.symbol)
            if price is None:
                continue

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
                    exit_time=datetime.utcnow(),
                    sequence_num=self._sequence_num,
                    integrity_hash=self._compute_hash(result["trade_id"]),
                )
                closed_trades.append(trade)
                event_bus.publish("trade.closed", trade)

        return closed_trades

    # ── Private ──────────────────────────────────────

    def _size_position(self, signal: Signal) -> tuple[int, float]:
        """Calculate position size based on risk management.

        Max risk per trade = account equity * max_risk_per_trade_pct.
        Shares = max_risk / risk_per_share.
        Cost = shares * entry_price.
        """
        if signal.shares > 0:
            return signal.shares, signal.shares * signal.entry_price

        risk_per_share = signal.risk_per_share
        if risk_per_share <= 0:
            return 0, 0

        max_risk = self.account.total_equity * settings.max_risk_per_trade_pct
        shares = int(max_risk / risk_per_share)

        # Cap by buying power
        max_by_cash = int(self.account.buying_power / signal.entry_price)
        shares = min(shares, max_by_cash)

        # Cap by concentration
        max_concentration = self.account.total_equity * settings.max_portfolio_concentration_pct
        max_by_concentration = int(max_concentration / signal.entry_price)
        shares = min(shares, max_by_concentration)

        if shares <= 0:
            return 0, 0

        return shares, shares * signal.entry_price

    @staticmethod
    def _apply_slippage(price: float, action: str) -> float:
        """Apply realistic slippage to execution price."""
        slip = price * settings.slippage_base_rate
        if action == "BUY":
            return round(price + slip, 2)
        return round(price - slip, 2)

    def _compute_hash(self, trade_id: str) -> str:
        """SHA-256 hash chaining for audit trail."""
        data = f"{trade_id}:{self._sequence_num}:{self._prev_hash}"
        h = hashlib.sha256(data.encode()).hexdigest()
        self._prev_hash = h
        return h

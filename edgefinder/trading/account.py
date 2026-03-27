"""EdgeFinder v2 — Per-strategy virtual account.

Each strategy gets an isolated $5,000 account with:
- Cash-only buying power (no margin/leverage)
- Optional PDT tracking (per-strategy toggle)
- Position management with unrealized P&L
- Drawdown tracking and circuit breaker
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position in a virtual account."""

    symbol: str
    shares: int
    entry_price: float
    stop_loss: float
    target: float
    direction: str  # LONG or SHORT
    trade_type: str  # DAY or SWING
    entry_time: datetime = field(default_factory=datetime.utcnow)
    trade_id: str = ""

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.shares
        return (self.entry_price - current_price) * self.shares

    def unrealized_pnl_pct(self, current_price: float) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.cost_basis * 100

    def should_stop_out(self, current_price: float) -> bool:
        if self.direction == "LONG":
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def should_take_profit(self, current_price: float) -> bool:
        if self.direction == "LONG":
            return current_price >= self.target
        return current_price <= self.target


class VirtualAccount:
    """Isolated virtual trading account for a single strategy."""

    def __init__(
        self,
        strategy_name: str,
        starting_capital: float | None = None,
        pdt_enabled: bool = False,
    ) -> None:
        self.strategy_name = strategy_name
        self.starting_capital = starting_capital or settings.starting_capital
        self.cash = self.starting_capital
        self.positions: list[Position] = []
        self.pdt_enabled = pdt_enabled
        self.peak_equity = self.starting_capital
        self.is_paused = False

        # PDT tracking
        self._day_trades: list[datetime] = []

        # Revenge trade cooldown
        self._last_stop_out: datetime | None = None

    # ── Properties ───────────────────────────────────

    @property
    def buying_power(self) -> float:
        """Available cash. No margin — cash only."""
        return self.cash

    @property
    def open_positions_value(self) -> float:
        """Sum of cost basis for all open positions."""
        return sum(p.cost_basis for p in self.positions)

    @property
    def total_equity(self) -> float:
        """Cash + open position cost basis (use mark-to-market for real equity)."""
        return self.cash + self.open_positions_value

    @property
    def drawdown_pct(self) -> float:
        equity = self.total_equity
        if self.peak_equity <= 0:
            return 0.0
        return max(0, (self.peak_equity - equity) / self.peak_equity)

    @property
    def position_count(self) -> int:
        return len(self.positions)

    # ── Account Checks ───────────────────────────────

    def can_open_position(self, cost: float, trade_type: str = "SWING") -> tuple[bool, str]:
        """Check if a new position can be opened. Returns (allowed, reason)."""
        if self.is_paused:
            return False, "Account is paused"

        if cost > self.buying_power:
            return False, f"Insufficient buying power: need ${cost:.2f}, have ${self.buying_power:.2f}"

        if self.position_count >= settings.max_open_positions:
            return False, f"Max positions reached ({settings.max_open_positions})"

        if self.drawdown_pct >= settings.drawdown_circuit_breaker_pct:
            self.is_paused = True
            return False, f"Drawdown circuit breaker triggered ({self.drawdown_pct:.1%})"

        if self.pdt_enabled and trade_type == "DAY" and not self._can_day_trade():
            return False, "PDT limit reached (3 day trades in 5 business days)"

        if self._is_revenge_trade():
            return False, "Revenge trade cooldown active"

        return True, "OK"

    def _can_day_trade(self) -> bool:
        """Check PDT compliance: max 3 day trades per 5 rolling business days."""
        cutoff = datetime.utcnow() - timedelta(days=settings.pdt_window_days)
        recent = [dt for dt in self._day_trades if dt > cutoff]
        return len(recent) < settings.pdt_day_trade_limit

    def _is_revenge_trade(self) -> bool:
        if self._last_stop_out is None:
            return False
        cooldown = timedelta(minutes=settings.revenge_trade_cooldown_minutes)
        return datetime.utcnow() - self._last_stop_out < cooldown

    # ── Position Management ──────────────────────────

    def open_position(self, position: Position) -> None:
        """Open a new position and deduct cash."""
        cost = position.cost_basis
        self.cash -= cost
        self.positions.append(position)
        logger.info(
            "[%s] Opened %s %s: %d shares @ $%.2f (cost: $%.2f)",
            self.strategy_name, position.direction, position.symbol,
            position.shares, position.entry_price, cost,
        )

    def close_position(
        self, position: Position, exit_price: float, reason: str
    ) -> dict:
        """Close a position and return cash. Returns trade result dict."""
        pnl = position.unrealized_pnl(exit_price)
        proceeds = position.cost_basis + pnl
        self.cash += proceeds

        risk_per_share = abs(position.entry_price - position.stop_loss)
        r_multiple = pnl / (risk_per_share * position.shares) if risk_per_share > 0 else 0.0

        self.positions.remove(position)

        # Update peak equity
        equity = self.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Track day trades for PDT
        if position.trade_type == "DAY":
            self._day_trades.append(datetime.utcnow())

        # Track stop-outs for revenge trade cooldown
        if reason in ("STOP_HIT", "STOP_LOSS"):
            self._last_stop_out = datetime.utcnow()

        logger.info(
            "[%s] Closed %s %s: %d shares @ $%.2f | P&L: $%.2f (%.1fR) | Reason: %s",
            self.strategy_name, position.direction, position.symbol,
            position.shares, exit_price, pnl, r_multiple, reason,
        )

        return {
            "symbol": position.symbol,
            "direction": position.direction,
            "trade_type": position.trade_type,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "shares": position.shares,
            "pnl_dollars": round(pnl, 2),
            "pnl_percent": round(position.unrealized_pnl_pct(exit_price), 2),
            "r_multiple": round(r_multiple, 2),
            "exit_reason": reason,
            "trade_id": position.trade_id,
        }

    def get_position(self, symbol: str) -> Position | None:
        """Find an open position by symbol."""
        for p in self.positions:
            if p.symbol == symbol:
                return p
        return None

    def get_sector_count(self, sector: str) -> int:
        """Count positions in a given sector (placeholder — needs sector data)."""
        # In the full system, positions would carry sector metadata
        return 0

    def to_dict(self) -> dict:
        """Serialize account state."""
        return {
            "strategy_name": self.strategy_name,
            "starting_capital": self.starting_capital,
            "cash": round(self.cash, 2),
            "open_positions_value": round(self.open_positions_value, 2),
            "total_equity": round(self.total_equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "position_count": self.position_count,
            "pdt_enabled": self.pdt_enabled,
            "is_paused": self.is_paused,
        }

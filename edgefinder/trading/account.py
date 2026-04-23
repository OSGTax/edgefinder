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
from datetime import datetime, timedelta, timezone

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position in a virtual account.

    `current_price` is the latest mark the position monitor has seen for
    this ticker. It is updated in Executor.check_positions() every tick
    and feeds VirtualAccount.total_equity for mark-to-market accounting.
    If the monitor hasn't ticked yet (e.g. immediately after open) the
    field stays None and equity calculations fall back to entry_price.
    """

    symbol: str
    shares: int
    entry_price: float
    stop_loss: float
    target: float
    direction: str  # LONG or SHORT
    trade_type: str  # DAY or SWING
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trade_id: str = ""
    current_price: float | None = None
    sector: str | None = None

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    @property
    def market_value(self) -> float:
        """Current mark-to-market value for this position.

        For a LONG: shares * current_price (falls back to entry_price).
        For a SHORT: the cash-equivalent a close-out would take, i.e.
          cost_basis + unrealized_pnl = cost_basis + (entry - mark) * shares.
        Treating a short's "market value" as the equity the account would
        hold in its stead keeps VirtualAccount.total_equity (cash + sum of
        market_value) consistent between longs and shorts.
        """
        mark = self.current_price if self.current_price is not None else self.entry_price
        if self.direction == "LONG":
            return self.shares * mark
        # SHORT: entry proceeds minus cost to buy back at mark
        return self.cost_basis + (self.entry_price - mark) * self.shares

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
        max_risk_pct: float | None = None,
        max_concentration_pct: float | None = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.starting_capital = starting_capital or settings.starting_capital
        self.cash = self.starting_capital
        self.positions: list[Position] = []
        self.pdt_enabled = pdt_enabled
        self.peak_equity = self.starting_capital
        self.is_paused = False
        self.realized_pnl = 0.0
        # Per-strategy risk overrides (None = use global settings)
        self.max_risk_pct = max_risk_pct
        self.max_concentration_pct = max_concentration_pct

        # PDT tracking
        self._day_trades: list[datetime] = []

        # Revenge trade cooldown
        self._last_stop_out: datetime | None = None

        # Per-ticker re-entry cooldown — prevents instant rebuy after a close.
        # Maps symbol -> close timestamp. Checked in can_open_position via
        # the (cost, trade_type, symbol) overload below.
        self._last_close_per_ticker: dict[str, datetime] = {}

    # ── Properties ───────────────────────────────────

    @property
    def buying_power(self) -> float:
        """Available cash. No margin — cash only."""
        return self.cash

    @property
    def open_positions_value(self) -> float:
        """Mark-to-market value of all open positions.

        Uses Position.market_value (current_price when the monitor has
        ticked, falling back to entry_price). Replaces the older
        cost-basis formulation that made drawdown and risk sizing lie.
        """
        return sum(p.market_value for p in self.positions)

    @property
    def open_positions_cost_basis(self) -> float:
        """Sum of cost basis (entry_price * shares). Used by
        _recalculate_account_balances — cash on open is entry_cost
        regardless of current mark.
        """
        return sum(p.cost_basis for p in self.positions)

    @property
    def total_equity(self) -> float:
        """Cash + mark-to-market value of all open positions."""
        return self.cash + self.open_positions_value

    def update_peak_equity(self) -> None:
        """Snap peak_equity to current total_equity if it's a new high.

        Called after every price mark so drawdown tracks real account
        value, not just values observed at trade-close time.
        """
        equity = self.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity

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

    def can_open_position(
        self,
        cost: float,
        trade_type: str = "SWING",
        symbol: str | None = None,
        sector: str | None = None,
    ) -> tuple[bool, str]:
        """Check if a new position can be opened. Returns (allowed, reason)."""
        if self.is_paused:
            return False, "Account is paused"

        # Max open positions hard limit
        if self.position_count >= settings.max_open_positions:
            return False, f"Max open positions reached ({settings.max_open_positions})"

        # Duplicate ticker check — only one position per symbol at a time
        if symbol and self.get_position(symbol):
            return False, f"Already have open position in {symbol}"

        # Sector concentration cap — prevents 5 semiconductors simultaneously.
        # Unknown sector (None) passes; that just means fundamentals didn't
        # supply one for this ticker.
        if sector and self.get_sector_count(sector) >= settings.max_same_sector_positions:
            return (
                False,
                f"Sector '{sector}' at concentration limit "
                f"({settings.max_same_sector_positions} positions)",
            )

        if cost > self.buying_power:
            return False, f"Insufficient buying power: need ${cost:.2f}, have ${self.buying_power:.2f}"

        # Ensure cash won't go negative
        if self.cash - cost < 0:
            return False, f"Would result in negative cash: ${self.cash:.2f} - ${cost:.2f}"

        if self.drawdown_pct >= settings.drawdown_circuit_breaker_pct:
            self.is_paused = True
            return False, f"Drawdown circuit breaker triggered ({self.drawdown_pct:.1%})"

        if self.pdt_enabled and trade_type == "DAY" and not self._can_day_trade():
            return False, "PDT limit reached (3 day trades in 5 business days)"

        if self._is_revenge_trade():
            return False, "Revenge trade cooldown active"

        # Per-ticker re-entry cooldown: prevents the position monitor from
        # closing a winning trade and the next signal check immediately
        # reopening the same ticker (which produced the NVDA infinite-loop
        # phantom-wins bug under stale-bar conditions).
        if symbol and self._is_in_reentry_cooldown(symbol):
            cooldown = settings.ticker_reentry_cooldown_minutes
            return False, f"Ticker {symbol} in re-entry cooldown ({cooldown}m after last close)"

        return True, "OK"

    def _is_in_reentry_cooldown(self, symbol: str) -> bool:
        last_close = self._last_close_per_ticker.get(symbol)
        if last_close is None:
            return False
        cooldown = timedelta(minutes=settings.ticker_reentry_cooldown_minutes)
        return datetime.now(timezone.utc) - last_close < cooldown

    def _can_day_trade(self) -> bool:
        """Check PDT compliance: max 3 day trades per 5 rolling business days.

        SEC Rule 4210 counts business days, not calendar days. Using a
        calendar-day cutoff (previous implementation) made Monday trades
        "age out" by the next Monday's clock, 2 business days early; it
        also created false positives after three-day weekends.
        """
        import numpy as np

        today = np.datetime64(datetime.now(timezone.utc).date())
        cutoff_date = np.busday_offset(
            today, -settings.pdt_window_days, roll="backward"
        )
        cutoff = datetime.combine(
            cutoff_date.astype("datetime64[D]").astype(object),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )
        recent = [dt for dt in self._day_trades if dt > cutoff]
        return len(recent) < settings.pdt_day_trade_limit

    def _is_revenge_trade(self) -> bool:
        if self._last_stop_out is None:
            return False
        cooldown = timedelta(minutes=settings.revenge_trade_cooldown_minutes)
        return datetime.now(timezone.utc) - self._last_stop_out < cooldown

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
        self.realized_pnl += pnl

        risk_per_share = abs(position.entry_price - position.stop_loss)
        r_multiple = pnl / (risk_per_share * position.shares) if risk_per_share > 0 else 0.0

        self.positions.remove(position)

        # Update peak equity
        equity = self.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Track day trades for PDT
        if position.trade_type == "DAY":
            self._day_trades.append(datetime.now(timezone.utc))

        # Track stop-outs for revenge trade cooldown
        if reason in ("STOP_HIT", "STOP_LOSS"):
            self._last_stop_out = datetime.now(timezone.utc)

        # Per-ticker re-entry cooldown: gates the next open on this symbol
        self._last_close_per_ticker[position.symbol] = datetime.now(timezone.utc)

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

    def get_sector_count(self, sector: str | None) -> int:
        """Count open positions in a given sector.

        Returns 0 for unknown (None) sectors so the concentration check
        doesn't silently block trades on tickers that didn't supply a
        sector (rare but possible for low-coverage fundamentals).
        """
        if not sector:
            return 0
        return sum(1 for p in self.positions if p.sector == sector)

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
            "realized_pnl": round(self.realized_pnl, 2),
            "position_count": self.position_count,
            "pdt_enabled": self.pdt_enabled,
            "is_paused": self.is_paused,
        }

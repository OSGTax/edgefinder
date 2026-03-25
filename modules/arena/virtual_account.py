"""
EdgeFinder Virtual Account — Per-Strategy Isolated Trading Account
===================================================================
Each strategy gets its own virtual account with separate cash, positions,
and P&L tracking. No interference between strategies.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """An open position within a virtual account."""
    trade_id: str
    ticker: str
    direction: str              # "LONG"
    trade_type: str             # "DAY" or "SWING"
    entry_price: float
    shares: int
    stop_loss: float
    target: float
    entry_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    sector: str = ""
    confidence: float = 0.0
    trailing_stop: Optional[float] = None
    high_water_mark: float = 0.0
    last_known_price: float = 0.0
    slippage_applied: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.shares

    @property
    def unrealized_pnl(self) -> float:
        price = self.last_known_price or self.entry_price
        return (price - self.entry_price) * self.shares

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis * 100

    @property
    def market_value(self) -> float:
        price = self.last_known_price or self.entry_price
        return price * self.shares

    @property
    def r_multiple(self) -> float:
        if self.risk_per_share == 0:
            return 0.0
        price = self.last_known_price or self.entry_price
        return (price - self.entry_price) / self.risk_per_share


class VirtualAccount:
    """Isolated virtual trading account for a single strategy.

    Tracks cash, positions, equity curve, day trade count, and drawdown.
    No interaction with other strategy accounts.
    """

    def __init__(
        self,
        strategy_name: str,
        starting_capital: float | None = None,
        max_positions: int | None = None,
        max_risk_pct: float | None = None,
    ):
        self.strategy_name = strategy_name
        self.starting_capital = starting_capital or settings.STARTING_CAPITAL
        self.cash = self.starting_capital
        self.max_positions = max_positions or settings.MAX_OPEN_POSITIONS
        self.max_risk_pct = max_risk_pct or settings.MAX_RISK_PER_TRADE_PCT

        self.positions: dict[str, Position] = {}  # trade_id -> Position
        self.closed_trades: list[dict] = []
        self.equity_history: list[dict] = []
        self.peak_equity: float = self.starting_capital
        self.is_paused: bool = False
        self.pause_reason: str = ""

        # Day trade tracking (PDT compliance)
        self._day_trades: list[datetime] = []

        # Revenge trade cooldown: track last stop-out per ticker
        self._last_stop_out: dict[str, datetime] = {}

        # Sector tracking for concentration limits
        self._position_sectors: dict[str, str] = {}  # trade_id -> sector

    # ── ACCOUNT STATE ────────────────────────────────────────

    @property
    def positions_value(self) -> float:
        """Total market value of open positions."""
        return sum(
            (p.last_known_price or p.entry_price) * p.shares
            for p in self.positions.values()
        )

    @property
    def total_equity(self) -> float:
        """Cash + positions value."""
        return self.cash + self.positions_value

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(t.get("pnl_dollars", 0) for t in self.closed_trades)

    @property
    def total_return_pct(self) -> float:
        if self.starting_capital == 0:
            return 0.0
        return (self.total_equity - self.starting_capital) / self.starting_capital * 100

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak equity."""
        if self.peak_equity == 0:
            return 0.0
        return (self.total_equity - self.peak_equity) / self.peak_equity * 100

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.get("pnl_dollars", 0) > 0)
        return wins / len(self.closed_trades) * 100

    # ── POSITION MANAGEMENT ──────────────────────────────────

    def can_open_position(
        self, ticker: str = "", sector: str = ""
    ) -> tuple[bool, str]:
        """Check if account can open a new position.

        Returns:
            Tuple of (allowed, reason). For backward compatibility,
            bool(result) works because tuple is truthy when first element is True.
        """
        if self.is_paused:
            return False, f"Strategy paused: {self.pause_reason}"
        if self.open_position_count >= self.max_positions:
            return False, f"Max positions reached ({self.max_positions})"

        # Already holding this ticker
        if ticker:
            for p in self.positions.values():
                if p.ticker == ticker:
                    return False, f"Already holding {ticker}"

        # Revenge trade cooldown
        if ticker and ticker in self._last_stop_out:
            cooldown = timedelta(minutes=settings.REVENGE_TRADE_COOLDOWN_MINUTES)
            elapsed = datetime.now(timezone.utc) - self._last_stop_out[ticker]
            if elapsed < cooldown:
                remaining = int((cooldown - elapsed).total_seconds() // 60)
                return False, f"Revenge cooldown on {ticker} ({remaining}m remaining)"

        # Sector concentration
        if sector:
            sector_count = sum(
                1 for s in self._position_sectors.values() if s == sector
            )
            if sector_count >= settings.MAX_SAME_SECTOR_POSITIONS:
                return False, f"Sector concentration limit for {sector} ({sector_count}/{settings.MAX_SAME_SECTOR_POSITIONS})"

        # Aggregate position value cap
        capital_deployed = sum(p.cost_basis for p in self.positions.values())
        if capital_deployed >= settings.ARENA_MAX_TOTAL_POSITION_VALUE:
            return False, f"Aggregate position cap reached (${capital_deployed:.2f} / ${settings.ARENA_MAX_TOTAL_POSITION_VALUE:.2f})"

        return True, "OK"

    def max_position_dollars(self) -> float:
        """Maximum dollar amount for next position based on risk rules."""
        return self.total_equity * settings.MAX_PORTFOLIO_CONCENTRATION_PCT

    def max_risk_dollars(self) -> float:
        """Maximum dollar risk for next trade."""
        return self.total_equity * self.max_risk_pct

    def calculate_shares(self, entry_price: float, stop_loss: float) -> int:
        """Calculate position size based on risk per trade.

        Uses the 2% risk rule: max_risk / risk_per_share = shares.
        Also respects max position concentration.
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0 or entry_price == 0:
            return 0

        # Risk-based sizing
        risk_shares = int(self.max_risk_dollars() / risk_per_share)

        # Concentration limit
        max_dollar = self.max_position_dollars()
        concentration_shares = int(max_dollar / entry_price)

        # Cash limit
        cash_shares = int(self.cash / entry_price)

        # Aggregate position value cap
        capital_deployed = sum(p.cost_basis for p in self.positions.values())
        remaining_budget = settings.ARENA_MAX_TOTAL_POSITION_VALUE - capital_deployed
        budget_shares = int(remaining_budget / entry_price) if remaining_budget > 0 else 0

        shares = min(risk_shares, concentration_shares, cash_shares, budget_shares)
        return max(shares, 0)

    def open_position(self, position: Position) -> bool:
        """Add a position to the account. Deducts cash.

        Returns:
            True if position opened, False if rejected.
        """
        allowed, reason = self.can_open_position(
            ticker=position.ticker, sector=position.sector
        )
        if not allowed:
            logger.warning(
                f"[{self.strategy_name}] Cannot open {position.ticker}: {reason}"
            )
            return False

        cost = position.entry_price * position.shares
        if cost > self.cash:
            logger.warning(
                f"[{self.strategy_name}] Insufficient cash: "
                f"need ${cost:.2f}, have ${self.cash:.2f}"
            )
            return False

        self.cash -= cost
        position.high_water_mark = position.entry_price
        position.last_known_price = position.entry_price
        self.positions[position.trade_id] = position
        if position.sector:
            self._position_sectors[position.trade_id] = position.sector
        logger.info(
            f"[{self.strategy_name}] Opened {position.direction} "
            f"{position.shares} {position.ticker} @ ${position.entry_price:.2f}"
        )
        return True

    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        exit_time: Optional[datetime] = None,
        slippage: float = 0.0,
    ) -> Optional[dict]:
        """Close a position and return trade result.

        Returns:
            Trade result dict or None if position not found.
        """
        if trade_id not in self.positions:
            logger.warning(f"[{self.strategy_name}] Position {trade_id} not found")
            return None

        position = self.positions.pop(trade_id)
        self._position_sectors.pop(trade_id, None)
        adjusted_exit = exit_price - slippage if position.direction == "LONG" else exit_price + slippage

        pnl_dollars = (adjusted_exit - position.entry_price) * position.shares
        pnl_percent = (adjusted_exit - position.entry_price) / position.entry_price * 100
        r_multiple = 0.0
        if position.risk_per_share > 0:
            r_multiple = (adjusted_exit - position.entry_price) / position.risk_per_share

        # Return cash
        self.cash += adjusted_exit * position.shares

        now = exit_time or datetime.now(timezone.utc)

        result = {
            "trade_id": position.trade_id,
            "strategy_name": self.strategy_name,
            "ticker": position.ticker,
            "direction": position.direction,
            "trade_type": position.trade_type,
            "entry_price": position.entry_price,
            "exit_price": adjusted_exit,
            "shares": position.shares,
            "stop_loss": position.stop_loss,
            "target": position.target,
            "entry_time": position.entry_time,
            "exit_time": now,
            "pnl_dollars": round(pnl_dollars, 2),
            "pnl_percent": round(pnl_percent, 4),
            "r_multiple": round(r_multiple, 2),
            "exit_reason": exit_reason,
            "confidence": position.confidence,
            "slippage_applied": slippage,
            "metadata": position.metadata,
        }

        self.closed_trades.append(result)

        # Track day trades for PDT
        if position.trade_type == "DAY":
            self._day_trades.append(now)

        # Track stop-outs for revenge trade cooldown
        if exit_reason in ("STOP_HIT", "TRAILING_STOP_HIT"):
            self._last_stop_out[position.ticker] = now

        # Update peak equity
        equity = self.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        logger.info(
            f"[{self.strategy_name}] Closed {position.ticker}: "
            f"${pnl_dollars:+.2f} ({pnl_percent:+.2f}%) "
            f"R={r_multiple:+.2f} [{exit_reason}]"
        )
        return result

    def update_position_price(self, trade_id: str, price: float) -> None:
        """Update a position's current price for P&L tracking."""
        if trade_id in self.positions:
            pos = self.positions[trade_id]
            pos.last_known_price = price
            if price > pos.high_water_mark:
                pos.high_water_mark = price

    # ── EQUITY & SNAPSHOTS ───────────────────────────────────

    def take_snapshot(self) -> dict:
        """Record current account state for equity curve."""
        equity = self.total_equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        snapshot = {
            "timestamp": datetime.now(timezone.utc),
            "strategy_name": self.strategy_name,
            "cash": round(self.cash, 2),
            "positions_value": round(self.positions_value, 2),
            "total_equity": round(equity, 2),
            "peak_equity": round(self.peak_equity, 2),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "open_positions": self.open_position_count,
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "is_paused": self.is_paused,
        }
        self.equity_history.append(snapshot)
        return snapshot

    # ── CIRCUIT BREAKERS ─────────────────────────────────────

    def check_drawdown_breaker(
        self, limit_pct: float | None = None
    ) -> bool:
        """Check if drawdown exceeds limit. Returns True if breaker tripped.

        Args:
            limit_pct: Drawdown limit as negative percentage (e.g., -15.0).
                       Defaults to settings.DRAWDOWN_CIRCUIT_BREAKER_PCT * 100.
        """
        limit = limit_pct or -(settings.DRAWDOWN_CIRCUIT_BREAKER_PCT * 100)
        if self.drawdown_pct <= limit:
            self.is_paused = True
            self.pause_reason = (
                f"Drawdown breaker: {self.drawdown_pct:.1f}% "
                f"(limit: {limit:.1f}%)"
            )
            logger.warning(
                f"[{self.strategy_name}] PAUSED — {self.pause_reason}"
            )
            return True
        return False

    def unpause(self) -> None:
        """Manually re-enable the strategy."""
        self.is_paused = False
        self.pause_reason = ""
        logger.info(f"[{self.strategy_name}] Unpaused")

    # ── PDT COMPLIANCE ───────────────────────────────────────

    def day_trades_remaining(self) -> int:
        """How many day trades left in the rolling window."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=settings.PDT_WINDOW_DAYS
        )
        recent = [dt for dt in self._day_trades if dt > cutoff]
        self._day_trades = recent  # Clean old entries
        return max(0, settings.PDT_DAY_TRADE_LIMIT - len(recent))

    def can_day_trade(self) -> bool:
        return self.day_trades_remaining() > 0

    # ── SERIALIZATION ────────────────────────────────────────

    def reset_account(self) -> None:
        """Reset account to starting state — full cash, no positions."""
        self.cash = self.starting_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.equity_history.clear()
        self.peak_equity = self.starting_capital
        self.is_paused = False
        self.pause_reason = ""
        self._day_trades.clear()
        self._last_stop_out.clear()
        self._position_sectors.clear()

    def to_dict(self) -> dict:
        """Serialize account state for persistence or API response."""
        return {
            "strategy_name": self.strategy_name,
            "starting_capital": self.starting_capital,
            "cash": round(self.cash, 2),
            "total_equity": round(self.total_equity, 2),
            "positions_value": round(self.positions_value, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "drawdown_pct": round(self.drawdown_pct, 4),
            "peak_equity": round(self.peak_equity, 2),
            "open_positions": self.open_position_count,
            "closed_trades": len(self.closed_trades),
            "win_rate": round(self.win_rate, 2),
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "day_trades_remaining": self.day_trades_remaining(),
            "positions": {
                tid: {
                    "ticker": p.ticker,
                    "direction": p.direction,
                    "shares": p.shares,
                    "entry_price": p.entry_price,
                    "last_price": p.last_known_price,
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                    "r_multiple": round(p.r_multiple, 2),
                }
                for tid, p in self.positions.items()
            },
        }

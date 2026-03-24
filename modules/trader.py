"""
EdgeFinder: Trade Data Classes
===============================
Data classes used by the trading system (Position, TradeResult, AccountState).
The active trading logic lives in modules/arena/.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


# ── DATA CLASSES ─────────────────────────────────────────────

@dataclass
class Position:
    """An open paper trade position."""
    trade_id: str
    ticker: str
    direction: str              # "LONG"
    trade_type: str             # "DAY" or "SWING"
    entry_price: float
    shares: int
    stop_loss: float
    target: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sector: str = ""
    fundamental_score: float = 0.0
    technical_signals: dict = field(default_factory=dict)
    news_sentiment: float = 0.0
    confidence_score: float = 0.0
    trailing_stop: Optional[float] = None
    high_water_mark: float = 0.0  # Highest price since entry (for trailing)
    last_known_price: float = 0.0  # Updated every monitoring cycle

    @property
    def cost_basis(self) -> float:
        """Total cost of position."""
        return self.entry_price * self.shares

    @property
    def risk_per_share(self) -> float:
        """Dollar risk per share (entry - stop)."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def total_risk(self) -> float:
        """Total dollar risk on position."""
        return self.risk_per_share * self.shares


@dataclass
class TradeResult:
    """Result of closing a position."""
    trade_id: str
    ticker: str
    direction: str
    trade_type: str
    entry_price: float
    exit_price: float
    shares: int
    stop_loss: float
    target: float
    pnl_dollars: float
    pnl_percent: float
    r_multiple: float
    exit_reason: str
    entry_time: datetime
    exit_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccountState:
    """Current state of the paper trading account."""
    cash: float = settings.STARTING_CAPITAL
    positions: dict = field(default_factory=dict)  # trade_id → Position
    peak_value: float = settings.STARTING_CAPITAL
    day_trades_timestamps: list = field(default_factory=list)  # datetime list
    last_stop_out_time: Optional[datetime] = None
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    daily_pnl_reset_date: Optional[datetime] = None
    weekly_pnl_reset_date: Optional[datetime] = None

    @property
    def open_position_count(self) -> int:
        """Number of currently open positions."""
        return len(self.positions)

    @property
    def positions_value(self) -> float:
        """Total value of open positions at entry price (approximate)."""
        return sum(p.cost_basis for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """Total account value (cash + positions)."""
        return self.cash + self.positions_value

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value

    def sector_count(self, sector: str) -> int:
        """Count open positions in a given sector."""
        return sum(1 for p in self.positions.values() if p.sector == sector)

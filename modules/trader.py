"""
EdgeFinder Module 3: Paper Trader
==================================
Simulated trade execution with full risk management:
- Position sizing (2% risk per trade, max 5 open positions)
- Stop-loss and profit target calculation
- PDT (Pattern Day Trader) rule compliance
- Trailing stops
- Daily/weekly loss circuit breakers
- Sector concentration limits

Executes trades based on signals that pass the sentiment gate.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import settings
from modules.database import (
    Trade as TradeRecord,
    AccountSnapshot,
    get_session,
)

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


# ── PAPER TRADER ─────────────────────────────────────────────

class PaperTrader:
    """
    Simulated trade execution engine with risk management.

    Manages an account with cash, open positions, and enforces:
    - Max risk per trade (2%)
    - Max open positions (5)
    - PDT compliance (3 day trades per 5 days)
    - Daily/weekly loss limits
    - Drawdown circuit breaker
    - Sector concentration limits
    - Revenge trade cooldown
    """

    def __init__(self, account: Optional[AccountState] = None):
        """Initialize trader with optional pre-existing account state."""
        self.account = account or AccountState()

    # ── PRE-TRADE CHECKS ─────────────────────────────────────

    def can_trade(self, ticker: str, trade_type: str, sector: str = "") -> tuple[bool, str]:
        """
        Run all pre-trade risk checks.

        Returns:
            Tuple of (can_trade: bool, reason: str).
        """
        # Max open positions
        if self.account.open_position_count >= settings.MAX_OPEN_POSITIONS:
            return False, f"Max open positions reached ({settings.MAX_OPEN_POSITIONS})"

        # Already holding this ticker
        for p in self.account.positions.values():
            if p.ticker == ticker:
                return False, f"Already holding position in {ticker}"

        # PDT check for day trades
        if trade_type == "DAY":
            recent_day_trades = self._count_recent_day_trades()
            if recent_day_trades >= settings.PDT_DAY_TRADE_LIMIT:
                return False, (
                    f"PDT limit reached ({recent_day_trades}/{settings.PDT_DAY_TRADE_LIMIT} "
                    f"day trades in {settings.PDT_WINDOW_DAYS} days)"
                )

        # Sector concentration
        if sector and self.account.sector_count(sector) >= settings.MAX_SAME_SECTOR_POSITIONS:
            return False, f"Max sector concentration reached for {sector}"

        # Daily loss limit
        self._reset_pnl_trackers()
        if self.account.cash < settings.STARTING_CAPITAL:
            daily_loss_pct = abs(self.account.daily_pnl) / settings.STARTING_CAPITAL
            if (self.account.daily_pnl < 0 and
                    daily_loss_pct >= settings.DAILY_LOSS_LIMIT_PCT):
                return False, f"Daily loss limit hit ({daily_loss_pct:.1%})"

        # Weekly loss limit
        if self.account.cash < settings.STARTING_CAPITAL:
            weekly_loss_pct = abs(self.account.weekly_pnl) / settings.STARTING_CAPITAL
            if (self.account.weekly_pnl < 0 and
                    weekly_loss_pct >= settings.WEEKLY_LOSS_LIMIT_PCT):
                return False, f"Weekly loss limit hit ({weekly_loss_pct:.1%})"

        # Drawdown circuit breaker
        if self.account.drawdown_pct >= settings.DRAWDOWN_CIRCUIT_BREAKER_PCT:
            return False, f"Drawdown circuit breaker triggered ({self.account.drawdown_pct:.1%})"

        # Revenge trade cooldown
        if self.account.last_stop_out_time:
            cooldown = timedelta(minutes=settings.REVENGE_TRADE_COOLDOWN_MINUTES)
            now = datetime.now(timezone.utc)
            if now - self.account.last_stop_out_time < cooldown:
                remaining = cooldown - (now - self.account.last_stop_out_time)
                return False, f"Revenge trade cooldown ({remaining.seconds // 60}m remaining)"

        # Sufficient cash
        if self.account.cash <= 0:
            return False, "No cash available"

        return True, "All checks passed"

    # ── POSITION SIZING ──────────────────────────────────────

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 60.0,
    ) -> int:
        """
        Calculate number of shares based on risk management rules.

        Uses the 2% risk rule: risk_amount / risk_per_share = shares.
        Confidence affects whether we take a full or half position.

        Args:
            entry_price: Planned entry price.
            stop_loss: Stop-loss price.
            confidence: Signal confidence (0-100).

        Returns:
            Number of shares to buy (0 if trade is invalid).
        """
        if entry_price <= 0 or stop_loss <= 0:
            return 0

        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        # Max risk amount = 2% of total account value
        max_risk = self.account.total_value * settings.MAX_RISK_PER_TRADE_PCT

        # Half position for moderate confidence
        if confidence < settings.SIGNAL_CONFIDENCE_HIGH:
            max_risk *= 0.5

        # Calculate shares from risk
        shares = int(max_risk / risk_per_share)

        # Max concentration check: position can't exceed 20% of account
        max_position_value = self.account.total_value * settings.MAX_PORTFOLIO_CONCENTRATION_PCT
        max_shares_by_concentration = int(max_position_value / entry_price)
        shares = min(shares, max_shares_by_concentration)

        # Must afford it
        max_shares_by_cash = int(self.account.cash / entry_price)
        shares = min(shares, max_shares_by_cash)

        # Minimum 1 share
        return max(0, shares)

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: Optional[float] = None,
    ) -> float:
        """
        Calculate stop-loss price.

        Uses ATR if available, otherwise defaults to 2% below entry.

        Args:
            entry_price: Planned entry price.
            atr: Average True Range (optional).

        Returns:
            Stop-loss price.
        """
        if atr and atr > 0:
            return round(entry_price - (1.5 * atr), 2)
        # Default: 2% below entry
        return round(entry_price * 0.98, 2)

    def calculate_target(
        self,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """
        Calculate profit target using minimum reward-to-risk ratio.

        Args:
            entry_price: Planned entry price.
            stop_loss: Stop-loss price.

        Returns:
            Target price.
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * settings.MIN_REWARD_TO_RISK_RATIO
        return round(entry_price + reward, 2)

    # ── TRADE EXECUTION ──────────────────────────────────────

    def open_position(
        self,
        ticker: str,
        entry_price: float,
        stop_loss: float,
        target: float,
        shares: int,
        trade_type: str = "DAY",
        sector: str = "",
        fundamental_score: float = 0.0,
        technical_signals: Optional[dict] = None,
        news_sentiment: float = 0.0,
        confidence_score: float = 0.0,
    ) -> Optional[Position]:
        """
        Open a new paper trade position.

        Deducts cash, creates Position, logs the entry.

        Returns:
            Position object if successful, None if rejected.
        """
        if shares <= 0:
            logger.warning(f"{ticker}: Cannot open position with {shares} shares")
            return None

        cost = entry_price * shares
        if cost > self.account.cash:
            logger.warning(f"{ticker}: Insufficient cash (need ${cost:.2f}, have ${self.account.cash:.2f})")
            return None

        trade_id = str(uuid.uuid4())
        position = Position(
            trade_id=trade_id,
            ticker=ticker,
            direction="LONG",
            trade_type=trade_type,
            entry_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            target=target,
            sector=sector,
            fundamental_score=fundamental_score,
            technical_signals=technical_signals or {},
            news_sentiment=news_sentiment,
            confidence_score=confidence_score,
            high_water_mark=entry_price,
        )

        self.account.cash -= cost
        self.account.positions[trade_id] = position

        logger.info(
            f"OPENED {ticker} | {trade_type} | {shares} shares @ ${entry_price:.2f} | "
            f"Stop: ${stop_loss:.2f} | Target: ${target:.2f} | "
            f"Risk: ${position.total_risk:.2f} | ID: {trade_id[:8]}"
        )

        return position

    def close_position(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "MANUAL",
    ) -> Optional[TradeResult]:
        """
        Close an open position and compute P&L.

        Args:
            trade_id: The trade ID to close.
            exit_price: Price at which to close.
            exit_reason: Why the trade was closed (STOP_HIT, TARGET_HIT,
                        TRAILING_STOP, END_OF_DAY, MANUAL).

        Returns:
            TradeResult with P&L details, or None if trade not found.
        """
        position = self.account.positions.get(trade_id)
        if not position:
            logger.warning(f"Trade {trade_id[:8]} not found in open positions")
            return None

        # Calculate P&L
        pnl_dollars = (exit_price - position.entry_price) * position.shares
        pnl_percent = (exit_price - position.entry_price) / position.entry_price if position.entry_price > 0 else 0.0
        risk_per_share = position.risk_per_share
        r_multiple = (exit_price - position.entry_price) / risk_per_share if risk_per_share > 0 else 0.0

        result = TradeResult(
            trade_id=trade_id,
            ticker=position.ticker,
            direction=position.direction,
            trade_type=position.trade_type,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            stop_loss=position.stop_loss,
            target=position.target,
            pnl_dollars=round(pnl_dollars, 2),
            pnl_percent=round(pnl_percent, 4),
            r_multiple=round(r_multiple, 2),
            exit_reason=exit_reason,
            entry_time=position.entry_time,
        )

        # Return cash
        self.account.cash += exit_price * position.shares

        # Track P&L
        self.account.daily_pnl += pnl_dollars
        self.account.weekly_pnl += pnl_dollars

        # Update peak
        if self.account.total_value > self.account.peak_value:
            self.account.peak_value = self.account.total_value

        # Track day trades for PDT
        if position.trade_type == "DAY":
            self.account.day_trades_timestamps.append(datetime.now(timezone.utc))

        # Track stop-outs for revenge cooldown
        if exit_reason == "STOP_HIT":
            self.account.last_stop_out_time = datetime.now(timezone.utc)

        # Remove from open positions
        del self.account.positions[trade_id]

        logger.info(
            f"CLOSED {result.ticker} | {exit_reason} | "
            f"P&L: ${result.pnl_dollars:+.2f} ({result.pnl_percent:+.2%}) | "
            f"R: {result.r_multiple:+.2f} | ID: {trade_id[:8]}"
        )

        return result

    # ── PRICE UPDATE & STOP MANAGEMENT ───────────────────────

    def update_price(self, trade_id: str, current_price: float) -> Optional[str]:
        """
        Update position with current market price. Checks stops and targets.

        Args:
            trade_id: The trade to update.
            current_price: Current market price.

        Returns:
            Action string if triggered: "STOP_HIT", "TARGET_HIT",
            "TRAILING_STOP", or None if no action needed.
        """
        position = self.account.positions.get(trade_id)
        if not position:
            return None

        # Track last known price for honest EOD closes
        position.last_known_price = current_price

        # Update high water mark
        if current_price > position.high_water_mark:
            position.high_water_mark = current_price

        # Check stop loss
        if current_price <= position.stop_loss:
            return "STOP_HIT"

        # Check target
        if current_price >= position.target:
            return "TARGET_HIT"

        # Trailing stop logic
        risk_per_share = position.risk_per_share
        if risk_per_share > 0:
            r_from_entry = (current_price - position.entry_price) / risk_per_share

            # Activate trailing stop at 1R profit
            if r_from_entry >= settings.TRAILING_STOP_ACTIVATION_R:
                if position.trailing_stop is None:
                    position.trailing_stop = position.entry_price  # Move to breakeven
                    logger.debug(f"{position.ticker}: Trailing stop activated at breakeven")

            # Trail by 1R once at 2R
            if r_from_entry >= settings.TRAILING_STOP_TRAIL_R:
                new_trail = position.high_water_mark - risk_per_share
                if position.trailing_stop is None or new_trail > position.trailing_stop:
                    position.trailing_stop = new_trail

            # Check trailing stop
            if position.trailing_stop and current_price <= position.trailing_stop:
                return "TRAILING_STOP"

        return None

    # ── EXECUTE SIGNAL ───────────────────────────────────────

    def execute_signal(
        self,
        ticker: str,
        signal_type: str,
        trade_type: str,
        entry_price: float,
        confidence: float,
        sector: str = "",
        fundamental_score: float = 0.0,
        technical_signals: Optional[dict] = None,
        news_sentiment: float = 0.0,
        sentiment_action: str = "PROCEED",
        atr: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Full trade execution from a signal.

        1. Run pre-trade checks
        2. Calculate stop-loss and target
        3. Size position
        4. Apply sentiment adjustment (REDUCE_50 halves size)
        5. Open position

        Returns:
            Position if trade opened, None if rejected.
        """
        # Only BUY signals for now (long only)
        if signal_type != "BUY":
            logger.info(f"{ticker}: Skipping {signal_type} signal (long-only system)")
            return None

        # Pre-trade checks
        can, reason = self.can_trade(ticker, trade_type, sector)
        if not can:
            logger.info(f"{ticker}: Trade rejected — {reason}")
            return None

        # Calculate levels
        stop_loss = self.calculate_stop_loss(entry_price, atr)
        target = self.calculate_target(entry_price, stop_loss)

        # Check reward-to-risk
        risk = entry_price - stop_loss
        reward = target - entry_price
        if risk > 0 and reward / risk < settings.MIN_REWARD_TO_RISK_RATIO:
            logger.info(f"{ticker}: R:R too low ({reward / risk:.2f})")
            return None

        # Size position
        shares = self.calculate_position_size(entry_price, stop_loss, confidence)

        # Sentiment adjustment: halve position on REDUCE_50
        if sentiment_action == "REDUCE_50":
            shares = max(1, shares // 2)
            logger.info(f"{ticker}: Position halved due to mild negative sentiment")

        if shares <= 0:
            logger.info(f"{ticker}: Position size is 0 — skipping")
            return None

        return self.open_position(
            ticker=ticker,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            shares=shares,
            trade_type=trade_type,
            sector=sector,
            fundamental_score=fundamental_score,
            technical_signals=technical_signals,
            news_sentiment=news_sentiment,
            confidence_score=confidence,
        )

    # ── INTERNAL HELPERS ─────────────────────────────────────

    def _count_recent_day_trades(self) -> int:
        """Count day trades in the PDT rolling window."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=settings.PDT_WINDOW_DAYS)
        recent = [t for t in self.account.day_trades_timestamps if t > cutoff]
        self.account.day_trades_timestamps = recent  # Clean up old entries
        return len(recent)

    def _reset_pnl_trackers(self) -> None:
        """Reset daily/weekly P&L trackers if the period has rolled over."""
        now = datetime.now(timezone.utc)
        today = now.date()

        if self.account.daily_pnl_reset_date is None or self.account.daily_pnl_reset_date != today:
            self.account.daily_pnl = 0.0
            self.account.daily_pnl_reset_date = today

        # Reset weekly on Monday
        week_start = today - timedelta(days=today.weekday())
        if self.account.weekly_pnl_reset_date is None or self.account.weekly_pnl_reset_date != week_start:
            self.account.weekly_pnl = 0.0
            self.account.weekly_pnl_reset_date = week_start

    # ── DATABASE PERSISTENCE ─────────────────────────────────

    def save_trade(self, result: TradeResult, position: Optional[Position] = None) -> None:
        """Save a completed trade to the database."""
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
            logger.info(f"Saved trade {result.trade_id[:8]} to database")
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
            session.rollback()
        finally:
            session.close()

    def save_account_snapshot(self) -> None:
        """Save current account state as a daily snapshot."""
        try:
            session = get_session()
            snapshot = AccountSnapshot(
                date=datetime.now(timezone.utc),
                cash=round(self.account.cash, 2),
                positions_value=round(self.account.positions_value, 2),
                total_value=round(self.account.total_value, 2),
                open_positions=self.account.open_position_count,
                peak_value=round(self.account.peak_value, 2),
                drawdown_pct=round(self.account.drawdown_pct, 4),
            )
            session.add(snapshot)
            session.commit()
            logger.info(
                f"Account snapshot: ${self.account.total_value:.2f} | "
                f"Cash: ${self.account.cash:.2f} | "
                f"Positions: {self.account.open_position_count} | "
                f"Drawdown: {self.account.drawdown_pct:.1%}"
            )
        except Exception as e:
            logger.error(f"Failed to save account snapshot: {e}")
            session.rollback()
        finally:
            session.close()

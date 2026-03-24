"""
EdgeFinder Honest Executor — No-Lookahead Trade Execution
==========================================================
Executes trades with:
- Volume-aware slippage modeling
- Immutable audit trail (every price timestamped and sourced)
- No future data in decision-making
- Trade tagging (regime, overlap, events)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import settings
from modules.strategies.base import Signal, TradeNotification, MarketRegime
from modules.arena.virtual_account import VirtualAccount, Position

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """Immutable record of every execution decision."""
    trade_id: str
    strategy_name: str
    ticker: str
    action: str
    signal_timestamp: datetime
    price_source: str
    price_timestamp: datetime
    execution_timestamp: datetime
    signal_price: float           # Price at signal generation
    execution_price: float        # Price at execution (after slippage)
    slippage: float
    shares: int
    stop_loss: float
    target: float
    confidence: float
    trade_type: str
    bar_data_at_decision: dict    # OHLCV bar used for decision
    market_regime: Optional[str] = None
    signal_overlap: int = 0       # How many other strategies also signaled
    position_overlap: int = 0     # How many other strategies hold this ticker
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "strategy_name": self.strategy_name,
            "ticker": self.ticker,
            "action": self.action,
            "signal_timestamp": self.signal_timestamp.isoformat(),
            "price_source": self.price_source,
            "price_timestamp": self.price_timestamp.isoformat(),
            "execution_timestamp": self.execution_timestamp.isoformat(),
            "signal_price": self.signal_price,
            "execution_price": self.execution_price,
            "slippage": self.slippage,
            "shares": self.shares,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "confidence": self.confidence,
            "trade_type": self.trade_type,
            "bar_data_at_decision": self.bar_data_at_decision,
            "market_regime": self.market_regime,
            "signal_overlap": self.signal_overlap,
            "position_overlap": self.position_overlap,
            "metadata": self.metadata,
        }


class Executor:
    """Honest trade executor with slippage modeling and audit trail.

    Rules:
    1. No lookahead — only uses data available at decision time.
    2. Every price is tagged with source and timestamp.
    3. Slippage scales with order size and stock volume.
    4. Full audit trail for every execution.
    """

    def __init__(
        self,
        base_slippage: float | None = None,
        volume_factor: float | None = None,
    ):
        self.base_slippage = base_slippage or settings.SLIPPAGE_BASE_RATE
        self.volume_factor = volume_factor or settings.SLIPPAGE_VOLUME_FACTOR
        self.audit_log: list[AuditEntry] = []

    def calculate_slippage(
        self,
        price: float,
        shares: int,
        avg_daily_volume: float,
    ) -> float:
        """Calculate realistic slippage based on order size and liquidity.

        Formula: slippage = base_rate * price * (1 + volume_penalty)
        Volume penalty increases when order is large relative to avg volume.

        Args:
            price: Current stock price.
            shares: Number of shares in order.
            avg_daily_volume: Average daily trading volume.

        Returns:
            Dollar slippage per share.
        """
        if avg_daily_volume <= 0:
            avg_daily_volume = settings.SLIPPAGE_MIN_AVG_VOLUME

        # Volume impact: larger orders relative to volume = more slippage
        volume_ratio = shares / avg_daily_volume if avg_daily_volume > 0 else 1.0
        volume_penalty = volume_ratio * self.volume_factor

        # Low-volume stocks get extra penalty
        if avg_daily_volume < settings.SLIPPAGE_MIN_AVG_VOLUME:
            volume_penalty *= 2.0

        slippage_per_share = self.base_slippage * price * (1 + volume_penalty)
        return round(slippage_per_share, 4)

    def execute_signal(
        self,
        signal: Signal,
        account: VirtualAccount,
        avg_daily_volume: float = 1_000_000,
        current_price: Optional[float] = None,
        price_source: str = "unknown",
        bar_data: Optional[dict] = None,
        market_regime: Optional[MarketRegime] = None,
        signal_overlap: int = 0,
        position_overlap: int = 0,
    ) -> Optional[AuditEntry]:
        """Execute a trading signal against a virtual account.

        Args:
            signal: The trading signal to execute.
            account: The strategy's virtual account.
            avg_daily_volume: Average daily volume for slippage calculation.
            current_price: Live price at execution time (if different from signal).
            price_source: Where the price came from (e.g., "alpaca", "yfinance").
            bar_data: OHLCV bar at decision time (for audit).
            market_regime: Current market environment.
            signal_overlap: Number of other strategies with same signal.
            position_overlap: Number of other strategies holding this ticker.

        Returns:
            AuditEntry if executed, None if rejected.
        """
        now = datetime.now(timezone.utc)
        exec_price = current_price or signal.entry_price

        # ── PRE-CHECKS ──────────────────────────────────────
        # Extract sector from signal metadata for concentration checks
        sector = signal.metadata.get("sector", "") if signal.metadata else ""
        allowed, reject_reason = account.can_open_position(
            ticker=signal.ticker, sector=sector
        )
        if not allowed:
            logger.info(
                f"[{account.strategy_name}] Rejected {signal.ticker}: "
                f"{reject_reason}"
            )
            return None

        if signal.action != "BUY":
            # For now, only support long entries. Sells are handled by
            # close_position. Short selling can be added later.
            logger.debug(
                f"[{account.strategy_name}] Skipping {signal.action} "
                f"{signal.ticker} (only BUY supported)"
            )
            return None

        # Check minimum reward:risk
        if signal.reward_to_risk < settings.MIN_REWARD_TO_RISK_RATIO:
            logger.info(
                f"[{account.strategy_name}] Rejected {signal.ticker}: "
                f"R:R {signal.reward_to_risk:.2f} < "
                f"{settings.MIN_REWARD_TO_RISK_RATIO}"
            )
            return None

        # Check day trade PDT limit
        if signal.trade_type == "DAY" and not account.can_day_trade():
            logger.info(
                f"[{account.strategy_name}] Rejected {signal.ticker}: "
                f"PDT limit reached"
            )
            return None

        # ── SIZING ───────────────────────────────────────────
        shares = signal.shares
        if shares <= 0:
            shares = account.calculate_shares(exec_price, signal.stop_loss)
        if shares <= 0:
            logger.info(
                f"[{account.strategy_name}] Rejected {signal.ticker}: "
                f"calculated 0 shares"
            )
            return None

        # ── SLIPPAGE ─────────────────────────────────────────
        slippage = self.calculate_slippage(exec_price, shares, avg_daily_volume)
        final_price = exec_price + slippage  # Buying costs more

        # ── OPEN POSITION ────────────────────────────────────
        trade_id = str(uuid.uuid4())
        position = Position(
            trade_id=trade_id,
            ticker=signal.ticker,
            direction="LONG",
            trade_type=signal.trade_type,
            entry_price=final_price,
            shares=shares,
            stop_loss=signal.stop_loss,
            target=signal.target,
            entry_time=now,
            confidence=signal.confidence,
            slippage_applied=slippage,
            metadata=signal.metadata,
        )

        if not account.open_position(position):
            return None

        # ── AUDIT TRAIL ──────────────────────────────────────
        audit = AuditEntry(
            trade_id=trade_id,
            strategy_name=account.strategy_name,
            ticker=signal.ticker,
            action=signal.action,
            signal_timestamp=signal.timestamp,
            price_source=price_source,
            price_timestamp=now,
            execution_timestamp=now,
            signal_price=signal.entry_price,
            execution_price=final_price,
            slippage=slippage,
            shares=shares,
            stop_loss=signal.stop_loss,
            target=signal.target,
            confidence=signal.confidence,
            trade_type=signal.trade_type,
            bar_data_at_decision=bar_data or {},
            market_regime=market_regime.trend if market_regime else None,
            signal_overlap=signal_overlap,
            position_overlap=position_overlap,
            metadata=signal.metadata,
        )
        self.audit_log.append(audit)

        logger.info(
            f"[{account.strategy_name}] EXECUTED {signal.ticker} "
            f"BUY {shares}x @ ${final_price:.2f} "
            f"(signal: ${signal.entry_price:.2f}, "
            f"slippage: ${slippage:.4f}) "
            f"[{price_source}]"
        )
        return audit

    def close_on_stop_or_target(
        self,
        account: VirtualAccount,
        trade_id: str,
        current_price: float,
        price_source: str = "unknown",
    ) -> Optional[dict]:
        """Check if a position should be closed (stop hit, target hit, trailing).

        Args:
            account: The strategy's virtual account.
            trade_id: The trade to check.
            current_price: Current market price.
            price_source: Price data source.

        Returns:
            Trade result dict if closed, None if still open.
        """
        if trade_id not in account.positions:
            return None

        position = account.positions[trade_id]
        account.update_position_price(trade_id, current_price)

        exit_reason = None

        # Check stop loss
        if current_price <= position.stop_loss:
            exit_reason = "STOP_HIT"

        # Check trailing stop
        elif position.trailing_stop and current_price <= position.trailing_stop:
            exit_reason = "TRAILING_STOP_HIT"

        # Check target
        elif current_price >= position.target:
            exit_reason = "TARGET_HIT"

        # Update trailing stop if in profit
        if exit_reason is None and position.risk_per_share > 0:
            r_current = (current_price - position.entry_price) / position.risk_per_share

            # Activate trailing stop at 1R (breakeven)
            if r_current >= settings.TRAILING_STOP_ACTIVATION_R:
                new_trailing = position.entry_price  # Breakeven
                if position.trailing_stop is None or new_trailing > position.trailing_stop:
                    position.trailing_stop = new_trailing

            # Trail at 2R
            if r_current >= settings.TRAILING_STOP_TRAIL_R:
                trail_price = current_price - (position.risk_per_share * 1.0)
                if position.trailing_stop is None or trail_price > position.trailing_stop:
                    position.trailing_stop = trail_price

        if exit_reason:
            slippage = self.calculate_slippage(
                current_price, position.shares, 1_000_000
            )
            return account.close_position(
                trade_id=trade_id,
                exit_price=current_price,
                exit_reason=exit_reason,
                slippage=slippage,
            )

        return None

    def get_audit_log(self, strategy_name: Optional[str] = None) -> list[dict]:
        """Get audit entries, optionally filtered by strategy."""
        entries = self.audit_log
        if strategy_name:
            entries = [e for e in entries if e.strategy_name == strategy_name]
        return [e.to_dict() for e in entries]

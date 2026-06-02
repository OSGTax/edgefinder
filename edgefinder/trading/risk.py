"""Centralized risk manager — sizing, stops, targets.

Each strategy gets its own RiskManager instance configured with
its risk percentage, stop percentage, and target percentage.
The account system handles cash constraints; this handles the math.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Computes position size, stop loss, and profit target."""

    def __init__(
        self,
        risk_pct: float,
        stop_pct: float = 0.20,
        target_pct: float = 0.25,
    ) -> None:
        self.risk_pct = risk_pct      # max loss per trade as fraction of equity
        self.stop_pct = stop_pct      # stop distance as fraction of entry price
        self.target_pct = target_pct  # target distance as fraction of entry price

    def compute_stop(self, entry_price: float) -> float:
        """Stop loss = entry * (1 - stop_pct)."""
        return round(entry_price * (1 - self.stop_pct), 2)

    def compute_target(self, entry_price: float) -> float:
        """Profit target = entry * (1 + target_pct)."""
        return round(entry_price * (1 + self.target_pct), 2)

    def compute_shares(
        self,
        entry_price: float,
        equity: float,
        available_cash: float | None = None,
        max_concentration_pct: float | None = None,
    ) -> int:
        """Position size based on risk budget and stop distance.

        shares = max_loss / stop_distance, capped by available cash and (when
        given) by the portfolio-concentration ceiling so a single trade can't
        consume the whole account.
        """
        max_loss = equity * self.risk_pct
        stop_distance = entry_price * self.stop_pct
        if stop_distance <= 0:
            return 0

        shares = int(max_loss / stop_distance)

        # Cap by available cash
        if available_cash is not None:
            max_by_cash = int(available_cash / entry_price)
            shares = min(shares, max_by_cash)

        # Cap by concentration (fraction of equity in a single position)
        if max_concentration_pct is not None and max_concentration_pct > 0 and entry_price > 0:
            max_by_conc = int(equity * max_concentration_pct / entry_price)
            shares = min(shares, max_by_conc)

        return max(shares, 0)

    def should_stop_out(self, entry_price: float, current_price: float) -> bool:
        """Check if current price has hit the stop loss level."""
        stop = self.compute_stop(entry_price)
        return current_price <= stop

    def should_take_profit(self, entry_price: float, current_price: float) -> bool:
        """Check if current price has hit the profit target."""
        target = self.compute_target(entry_price)
        return current_price >= target

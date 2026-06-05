"""Gap Drift — post-event continuation after a large held gap-up.

Effect class: post-event drift (PEAD proxy on price/volume only). A >=gap_min
overnight gap that HOLDS into a strong close (close >= open, close in the
upper part of the range) on >=1.5x volume marks underreaction to real news;
the documented drift plays out over the following days-to-weeks.

Uniquely among the candidates, the engine's mechanics MEASURE this effect
correctly instead of taxing it: drift is defined from the post-event-day
price, and our fill is exactly that (signal on the event day's completed
bar, fill at the next open). Pre-registered as the SURVIVORSHIP CONTROL:
gap events are news-driven and far less flattered by a today's-winners
universe than momentum entries.

Exits: the centerpiece is a manufactured tight stop — a gap trade that
retraces fail_pct from the fill has failed (gap-fill proxy), which is the
fix for degenerate's event trades riding toward the fixed -20% stop.
Winners exit at a 10-20% target or a 10-20 day time stop (drift horizon).

Defaults below are PRE-REGISTERED for the research screen (2026-06-05).
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("gap_drift")
class GapDriftStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "gap_drift"

    @property
    def risk_pct(self) -> float:
        # Fixed (not searched) — knob-budget spent on the signal shape.
        return 0.03

    @property
    def target_pct(self) -> float:
        return self._p("target_pct", 0.15)

    @property
    def max_hold_days(self) -> int:
        return self._p("max_hold_days", 15)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        # Research candidate: no fundamental gate yet (see pullback_rider).
        return True

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        prev = data.history.latest  # yesterday (today's bar is NOT in history)
        if prev is None or not prev.close or not ind.open:
            return None

        # 1) Gap event: today opened >= gap_min above yesterday's close.
        gap_min = self._p("gap_min", 0.05)
        if ind.open < prev.close * (1 + gap_min):
            return None

        # 2) Gap HELD into a strong close (filters exhaustion gaps).
        close_loc = self._p("close_loc", 0.65)
        day_range = max(ind.high - ind.low, 0.01)
        if ind.close < ind.open or (ind.close - ind.low) / day_range < close_loc:
            return None

        # 3) Event volume (fixed, not searched).
        if ind.volume_ratio is None or ind.volume_ratio < 1.5:
            return None

        # 4) Optional trend gate (searched: PEAD also works on beaten-down names).
        if self._p("trend_gate", True):
            if ind.ema_200 is None or ind.close <= ind.ema_200:
                return None

        gap_pct = (ind.open - prev.close) / prev.close * 100
        return self.make_intent(
            ticker, data,
            f"Held gap-up +{gap_pct:.1f}% on {ind.volume_ratio:.1f}x volume, "
            f"close in top {100 - close_loc * 100:.0f}% of range",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        # Manufactured tight stop: a gap trade that retraces fail_pct from
        # our fill has failed — never ride an event trade toward -20%.
        fail_pct = self._p("fail_pct", 0.06)
        if entry_price > 0 and ind.close < entry_price * (1 - fail_pct):
            return self.make_exit(
                ticker, data,
                f"Gap failed: close {ind.close:.2f} is "
                f"{fail_pct * 100:.0f}% below fill {entry_price:.2f}",
            )

        return None

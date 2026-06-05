"""Gap Carry — gap_drift v1's entry, exits rebuilt to hold the drift.

Round-1 evidence: gap_drift v1's aggregate profit (PF 1.92, ≈SPY at half
exposure) partly lives in multi-week carry across window boundaries, which
its 15%-target/15-day exits clip. This variant keeps v1's ENTRY untouched
(absolute 5% held gap on 1.5x volume, trend-gated — the best-screening
entry the lab has found) and changes ONLY the exit side:

- trail the drift: hold while the close stays above the 21EMA; losing it
  ends the trade (winners ride for weeks, failed drifts die in days),
- early fail-stop kept from v1 (close < fill*(1-fail_pct)),
- wide fixed target (30%) so the engine's target doesn't clip the ride,
- long max_hold recycles stagnant capital.

One variable at a time: entry identical to v1, exits the only change.

Defaults below are PRE-REGISTERED (2026-06-05, round 2) before any screen.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("gap_carry")
class GapCarryStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "gap_carry"

    @property
    def risk_pct(self) -> float:
        return 0.03  # fixed, as in v1

    @property
    def target_pct(self) -> float:
        # Fixed wide: the 21EMA trail is the real winner exit, not the target.
        return 0.30

    @property
    def max_hold_days(self) -> int:
        return self._p("max_hold_days", 45)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        return True  # lab-only until promoted

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        # ENTRY: identical to gap_drift v1 pre-registered defaults (fixed —
        # not searched — so any fold/screen difference is attributable to
        # the exit redesign alone).
        ind = data.current
        prev = data.history.latest
        if prev is None or not prev.close or not ind.open:
            return None
        if ind.open < prev.close * 1.05:  # v1 gap_min=0.05, fixed
            return None
        day_range = max(ind.high - ind.low, 0.01)
        if ind.close < ind.open or (ind.close - ind.low) / day_range < 0.65:
            return None
        if ind.volume_ratio is None or ind.volume_ratio < 1.5:
            return None
        if ind.ema_200 is None or ind.close <= ind.ema_200:  # v1 trend_gate=True
            return None

        gap_pct = (ind.open - prev.close) / prev.close * 100
        return self.make_intent(
            ticker, data,
            f"Held gap-up +{gap_pct:.1f}% on {ind.volume_ratio:.1f}x volume "
            "(carry exits: ride the 21EMA)",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        # Early failure: gap retraced from the fill (kept from v1).
        fail_pct = self._p("fail_pct", 0.06)
        if entry_price > 0 and ind.close < entry_price * (1 - fail_pct):
            return self.make_exit(
                ticker, data,
                f"Gap failed: close {ind.close:.2f} is "
                f"{fail_pct * 100:.0f}% below fill {entry_price:.2f}",
            )

        # Trail the drift: the ride is over when the 21EMA goes.
        if ind.ema_21 is not None and ind.close < ind.ema_21:
            return self.make_exit(
                ticker, data,
                f"Drift over: close {ind.close:.2f} lost the 21EMA "
                f"{ind.ema_21:.2f}",
            )

        return None

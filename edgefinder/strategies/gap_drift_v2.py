"""Gap Drift v2 — held gap-up continuation, ATR-normalized threshold.

Single isolated change from gap_drift v1: the gap threshold is measured in
units of the stock's OWN normal daily movement (ATR) instead of an absolute
percent. A 4% gap in a quiet name is a major event; an 8% gap in a wild one
is noise — v1's fixed gap_min conflated them, which is the explicit-rule
version of what the per-fold optimizer kept re-fitting (it chose different
thresholds in different regimes). If the v1 effect's lumpiness came from
absolute thresholds drifting in and out of calibration, this should smooth
it; if not, the variant dies and the optimizer's fold pass looks more like
selection bias.

Everything else (held-into-strong-close filter, 1.5x volume, manufactured
fail-stop, drift-horizon time stop) is unchanged from v1 by design — one
variable at a time.

Defaults below are PRE-REGISTERED (2026-06-05, round 2) before any screen.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("gap_drift_v2")
class GapDriftV2Strategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "gap_drift_v2"

    @property
    def risk_pct(self) -> float:
        return 0.03  # fixed, as in v1

    @property
    def target_pct(self) -> float:
        return self._p("target_pct", 0.15)

    @property
    def max_hold_days(self) -> int:
        return self._p("max_hold_days", 15)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        return True  # lab-only until promoted

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        prev = data.history.latest  # yesterday (today's bar is NOT in history)
        if prev is None or not prev.close or not ind.open or not prev.atr:
            return None

        # 1) Gap event in ATR units, with a small absolute floor so micro-gaps
        #    on ultra-quiet names can't sneak under the cost hurdle.
        atr_mult = self._p("atr_mult", 2.5)
        gap_dollars = ind.open - prev.close
        if gap_dollars < atr_mult * prev.atr:
            return None
        if gap_dollars / prev.close < 0.02:  # fixed 2% absolute floor
            return None

        # 2) Gap HELD into a strong close (unchanged from v1).
        close_loc = self._p("close_loc", 0.65)
        day_range = max(ind.high - ind.low, 0.01)
        if ind.close < ind.open or (ind.close - ind.low) / day_range < close_loc:
            return None

        # 3) Event volume (fixed, unchanged).
        if ind.volume_ratio is None or ind.volume_ratio < 1.5:
            return None

        # 4) Optional trend gate (searched, unchanged).
        if self._p("trend_gate", True):
            if ind.ema_200 is None or ind.close <= ind.ema_200:
                return None

        atrs = gap_dollars / prev.atr
        return self.make_intent(
            ticker, data,
            f"Held gap-up of {atrs:.1f} ATRs (+{gap_dollars / prev.close * 100:.1f}%) "
            f"on {ind.volume_ratio:.1f}x volume",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current
        fail_pct = self._p("fail_pct", 0.06)
        if entry_price > 0 and ind.close < entry_price * (1 - fail_pct):
            return self.make_exit(
                ticker, data,
                f"Gap failed: close {ind.close:.2f} is "
                f"{fail_pct * 100:.0f}% below fill {entry_price:.2f}",
            )
        return None

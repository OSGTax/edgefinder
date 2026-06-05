"""Trend Dip — short-horizon oversold stretch, bought ONLY in an uptrend.

Effect class: Connors-family short-horizon mean reversion above the
long-term trend (the most evidence-backed daily-bar pattern family) — and
the corrected version of what coward got wrong: coward bought every dip
including falling knives; this buys a 2-4 day sharp stretch ONLY when the
name is above its 200dma, and exits into the recovery within days.

Round-1 selector verdict was "maybe": the thesis collides with the
engine's next-day-open fill (the bounce is concentrated in the first bar
after the trigger, which the fill gives away — the mechanic that killed
coward). It gets its fair, cheap screen now that the warm lab makes its
ema_200 gate testable at all. Pre-registered kill criteria: avg net per
trade < +0.25%, or PF < 1.10 → dead.

Defaults below are PRE-REGISTERED for the research screen (2026-06-05,
round 2). target_pct is intentionally FIXED (not searched) to keep the
knob count at 5.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("trend_dip")
class TrendDipStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "trend_dip"

    @property
    def risk_pct(self) -> float:
        return self._p("risk_pct", 0.03)

    @property
    def target_pct(self) -> float:
        # Fixed (not searched) — knob budget spent on the stretch definition.
        return 0.08

    @property
    def max_hold_days(self) -> int:
        # Short horizon: the bounce either happens in days or the trade is wrong.
        return self._p("max_hold_days", 6)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        # Research candidate: no fundamental gate yet (lab-only until promoted).
        return True

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        if None in (ind.ema_200, ind.williams_r):
            return None

        # 1) Trend gate — never catch knives below the 200dma.
        if ind.close <= ind.ema_200:
            return None

        # 2) Sharp short-horizon stretch: N consecutive down closes ending
        #    today AND Williams %R deeply oversold.
        down_days_min = self._p("down_days_min", 3)
        closes = [c for c in data.history.get_field_series("close") if c]
        if len(closes) < down_days_min:
            return None
        seq = closes + [ind.close]  # ... yesterday, today
        down = 0
        for i in range(len(seq) - 1, 0, -1):
            if seq[i] < seq[i - 1]:
                down += 1
            else:
                break
        if down < down_days_min:
            return None

        wr_entry = self._p("wr_entry", -90)
        if ind.williams_r > wr_entry:
            return None

        return self.make_intent(
            ticker, data,
            f"{down}-day stretch above 200dma (W%R {ind.williams_r:.0f} <= "
            f"{wr_entry}, close {ind.close:.2f} > ema200 {ind.ema_200:.2f})",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        # Sell into the recovery — the bounce IS the trade; don't overstay.
        rsi_exit = self._p("rsi_exit", 60)
        if ind.rsi is not None and ind.rsi >= rsi_exit:
            return self.make_exit(
                ticker, data,
                f"RSI {ind.rsi:.0f} >= {rsi_exit} — recovery reached",
            )

        return None

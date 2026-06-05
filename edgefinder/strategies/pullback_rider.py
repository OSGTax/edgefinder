"""Pullback Rider — buy the resumption of an uptrend pullback.

Effect class: pullback-in-uptrend continuation. The entry is a CONFIRMED
reclaim of the 21EMA inside an EMA-stack uptrend (50>200, price>200) — we
never buy while the dip is in progress, so the bet is multi-day
continuation rather than first-bar bounce capture (the mechanic that made
coward unviable under next-day-open fills).

Exits that actually fire in this engine: a reachable 6-12% target, a
thesis-invalidation cut at the 50EMA (~-4..-6%, long before the fixed 20%
stop), an RSI sell-into-strength exit, and a 10-20 day time stop. The
trailing stop is inert here (arms at +1R = +20% > target) by design.

Defaults below are PRE-REGISTERED for the research screen — committed
before the first screen run (research protocol, 2026-06-05).
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("pullback_rider")
class PullbackRiderStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "pullback_rider"

    @property
    def risk_pct(self) -> float:
        return self._p("risk_pct", 0.03)

    @property
    def target_pct(self) -> float:
        return self._p("target_pct", 0.08)

    @property
    def max_hold_days(self) -> int:
        return self._p("max_hold_days", 15)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        # Research candidate: no fundamental gate yet (the lab runs with
        # fundamentals off; live qualification is decided at promotion).
        return True

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        prev = data.history.latest  # yesterday (today's bar is NOT in history)
        if prev is None:
            return None
        if None in (ind.ema_21, ind.ema_50, ind.ema_200, ind.rsi, prev.ema_21):
            return None

        # 1) Trend stack: only ride established uptrends.
        if not (ind.ema_50 > ind.ema_200 and ind.close > ind.ema_200):
            return None

        # 2) Reclaim cross event: pullback ENDED yesterday→today (anti-knife).
        if not (prev.close <= prev.ema_21 and ind.close > ind.ema_21):
            return None

        # 3) Depth control: a pullback, not a collapse nor already overheated.
        rsi_floor = self._p("rsi_floor", 40)
        if not (rsi_floor <= ind.rsi <= 60):
            return None

        # 4) Optional trend-strength gate (0 disables).
        adx_min = self._p("adx_min", 0)
        if adx_min and (ind.adx is None or ind.adx < adx_min):
            return None

        return self.make_intent(
            ticker, data,
            f"21EMA reclaim in uptrend (close {ind.close:.2f} > ema21 "
            f"{ind.ema_21:.2f}, RSI {ind.rsi:.0f}, 50>200 stack)",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        # Thesis invalidation: trend pullback failed — cut well before -20%.
        if ind.ema_50 is not None and ind.close < ind.ema_50:
            return self.make_exit(
                ticker, data,
                f"Close {ind.close:.2f} lost the 50EMA {ind.ema_50:.2f} — "
                "pullback failed",
            )

        # Sell into strength instead of letting winners decay back to entry.
        rsi_exit = self._p("rsi_exit", 70)
        if ind.rsi is not None and ind.rsi >= rsi_exit:
            return self.make_exit(
                ticker, data,
                f"RSI {ind.rsi:.0f} >= {rsi_exit} — selling into strength",
            )

        return None

"""Turtle ADX — 30-day breakout cross with a trend-strength gate.

Effect class: classic Donchian/turtle breakout trend-following. Entry is a
CROSS event — today closed at a new 30-day high and yesterday was not
already one — gated by ADX trend strength with +DI dominance, volume
participation, and price above the 50EMA.

This is the one candidate whose exit geometry makes the engine's native
machinery genuinely work: wide targets (25-60%) leave room for the
trailing stop (which only arms at +1R = +20%) to convert big trends into
locked gains; failed breakouts are cut by the 21EMA exit at ~-4..-7%,
long before the fixed 20% stop; a 30-60 day time stop recycles stagnant
capital. Losers die small, winners are owned until the trail or the
21EMA ends the trend — the explicit fix for "winners returned to
break-even".

Defaults below are PRE-REGISTERED for the research screen (2026-06-05).
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("turtle_adx")
class TurtleAdxStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "turtle_adx"

    @property
    def risk_pct(self) -> float:
        return self._p("risk_pct", 0.03)

    @property
    def target_pct(self) -> float:
        return self._p("target_pct", 0.40)

    @property
    def max_hold_days(self) -> int:
        return self._p("max_hold_days", 45)

    @property
    def trailing_stop_pct(self) -> float:
        # Reachable here (targets >= 25% keep positions alive past +20%).
        return self._p("trailing_stop_pct", 0.12)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        # Research candidate: no fundamental gate yet (see pullback_rider).
        return True

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        closes = [c for c in data.history.get_field_series("close") if c]
        # Demand a full-ish window so "30-day high" means what it says.
        if len(closes) < 26:
            return None

        # 1) Breakout CROSS event: new 30d closing high today, and yesterday
        #    was NOT already the high (prevents re-firing all trend long).
        if ind.close < max(closes):
            return None
        if closes[-1] >= max(closes[:-1]):
            return None

        # 2) Trend-strength gate.
        adx_min = self._p("adx_min", 22)
        if ind.adx is None or ind.adx < adx_min:
            return None
        if ind.plus_di is None or ind.minus_di is None or ind.plus_di <= ind.minus_di:
            return None

        # 3) Participation.
        vol_min = self._p("vol_min", 1.2)
        if vol_min > 1.0 and (ind.volume_ratio is None or ind.volume_ratio < vol_min):
            return None

        # 4) Sanity: above the intermediate trend.
        if ind.ema_50 is None or ind.close <= ind.ema_50:
            return None

        return self.make_intent(
            ticker, data,
            f"30d-high breakout (close {ind.close:.2f}, ADX {ind.adx:.0f}, "
            f"+DI>{ind.minus_di:.0f}, vol {ind.volume_ratio or 0:.1f}x)",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        # Failed breakout / mature trend: losing the 21EMA ends the trade
        # at typically -4..-7% — the real stop in this design.
        if ind.ema_21 is not None and ind.close < ind.ema_21:
            return self.make_exit(
                ticker, data,
                f"Close {ind.close:.2f} lost the 21EMA {ind.ema_21:.2f} — "
                "breakout failed or trend over",
            )

        return None

"""Alpha Strategy — Momentum day trading.

Targets stocks with positive earnings and revenue momentum.
Qualifies: earnings_growth > 0, revenue_growth > 0
Signals: EMA crossovers, MACD bullish crosses, volume spikes
Trade type: DAY

NOTE: Mock framework — qualification criteria are placeholders for refinement.
"""

from __future__ import annotations

import pandas as pd

from edgefinder.core.models import Signal, TickerFundamentals
from edgefinder.signals.engine import compute_indicators, detect_signals
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry, TradeNotification


@StrategyRegistry.register("alpha")
class AlphaStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "alpha"

    @property
    def version(self) -> str:
        return "3.0"

    @property
    def preferred_signals(self) -> list[str]:
        return ["ema_crossover_bullish", "macd_bullish_cross", "volume_spike_bullish"]

    @property
    def exit_signals(self) -> list[str]:
        return ["ema_crossover_bearish", "volume_spike_bearish"]

    def init(self) -> None:
        pass

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        eg = fundamentals.earnings_growth
        if eg is None or eg <= 0:
            return False
        rg = fundamentals.revenue_growth
        if rg is None or rg <= 0:
            return False
        return True

    def generate_signals(self, ticker: str, bars: pd.DataFrame) -> list[Signal]:
        indicators = compute_indicators(bars)
        if indicators is None:
            return []
        all_signals = detect_signals(indicators, ticker)
        result = []
        for sig in all_signals:
            pattern = sig.metadata.get("pattern", "")
            if pattern in self.preferred_signals and sig.action.value == "BUY":
                sig.strategy_name = self.name
                result.append(sig)
        return result

    def on_trade_executed(self, notification: TradeNotification) -> None:
        pass

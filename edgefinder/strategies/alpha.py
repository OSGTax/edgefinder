"""Alpha Strategy — Momentum / EMA Crossover day trading.

Targets fundamentally strong stocks with technical momentum.
Qualifies: composite_score >= 60, earnings_growth > 0, PEG < 2.0
Signals: EMA crossovers, MACD bullish crosses, volume spikes
Trade type: DAY
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
        return "2.1"

    @property
    def preferred_signals(self) -> list[str]:
        return ["ema_crossover_bullish", "macd_bullish_cross", "volume_spike_bullish"]

    @property
    def exit_signals(self) -> list[str]:
        return ["ema_crossover_bearish", "volume_spike_bearish"]

    def init(self) -> None:
        pass

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        composite = fundamentals.composite_score
        if composite is None or composite < 60:
            return False
        eg = fundamentals.earnings_growth
        if eg is None or eg <= 0:
            return False
        peg = fundamentals.peg_ratio
        if peg is not None and peg >= 2.0:
            return False
        return True

    def generate_signals(self, ticker: str, bars: pd.DataFrame) -> list[Signal]:
        indicators = compute_indicators(bars)
        if indicators is None:
            return []
        all_signals = detect_signals(indicators, ticker)
        # Filter to preferred patterns and DAY trades
        result = []
        for sig in all_signals:
            pattern = sig.metadata.get("pattern", "")
            if pattern in self.preferred_signals and sig.action.value == "BUY":
                sig.strategy_name = self.name
                result.append(sig)
        return result

    def on_trade_executed(self, notification: TradeNotification) -> None:
        pass

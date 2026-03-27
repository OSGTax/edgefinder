"""Bravo Strategy — Mean Reversion / Bollinger Band swing trading.

Targets value stocks bouncing off oversold levels.
Qualifies: burry_score >= 50, current_ratio > 1.2
Signals: BB lower touch, RSI oversold
Trade type: SWING
"""

from __future__ import annotations

import pandas as pd

from edgefinder.core.models import Signal, SignalAction, TickerFundamentals
from edgefinder.signals.engine import compute_indicators, detect_signals
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry, TradeNotification


@StrategyRegistry.register("bravo")
class BravoStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "bravo"

    @property
    def version(self) -> str:
        return "2.0"

    @property
    def preferred_signals(self) -> list[str]:
        return ["bb_lower_touch", "rsi_oversold"]

    def init(self) -> None:
        pass

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        burry = fundamentals.burry_score
        if burry is None or burry < 50:
            return False
        cr = fundamentals.current_ratio
        if cr is None or cr <= 1.2:
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
            if pattern in self.preferred_signals and sig.action == SignalAction.BUY:
                sig.strategy_name = self.name
                result.append(sig)
        return result

    def on_trade_executed(self, notification: TradeNotification) -> None:
        pass

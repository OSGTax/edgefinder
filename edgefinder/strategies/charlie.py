"""Charlie Strategy — Deep Value / Contrarian swing trading.

Targets high short-interest stocks with strong value fundamentals.
Qualifies: burry_score >= 70, short_interest > 10%, fcf_yield > 5%
Signals: RSI oversold, MACD bullish cross (high confidence only, >= 80)
Trade type: SWING
"""

from __future__ import annotations

import pandas as pd

from edgefinder.core.models import Signal, SignalAction, TickerFundamentals
from edgefinder.signals.engine import compute_indicators, detect_signals
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry, TradeNotification

MIN_CONFIDENCE = 80


@StrategyRegistry.register("charlie")
class CharlieStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "charlie"

    @property
    def version(self) -> str:
        return "2.0"

    @property
    def preferred_signals(self) -> list[str]:
        return ["rsi_oversold", "macd_bullish_cross"]

    def init(self) -> None:
        pass

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        burry = fundamentals.burry_score
        if burry is None or burry < 70:
            return False
        si = fundamentals.short_interest
        if si is None or si <= 0.10:
            return False
        fcf = fundamentals.fcf_yield
        if fcf is None or fcf <= 0.05:
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
            if (
                pattern in self.preferred_signals
                and sig.action == SignalAction.BUY
                and sig.confidence >= MIN_CONFIDENCE
            ):
                sig.strategy_name = self.name
                result.append(sig)
        return result

    def on_trade_executed(self, notification: TradeNotification) -> None:
        pass

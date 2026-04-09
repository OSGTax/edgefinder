"""Bravo Strategy — Mean Reversion swing trading.

Targets stocks with solid balance sheets bouncing off oversold levels.
Qualifies: current_ratio > 1.0, debt_to_equity < 2.0
Signals: BB lower touch, RSI oversold
Trade type: SWING

NOTE: Mock framework — qualification criteria are placeholders for refinement.
"""

from __future__ import annotations

import pandas as pd

from edgefinder.core.models import Signal, TickerFundamentals
from edgefinder.scanner.scoring import ScoringFactor, ScoringProfile
from edgefinder.signals.engine import compute_indicators, detect_signals
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry, TradeNotification


@StrategyRegistry.register("bravo")
class BravoStrategy(BaseStrategy):

    @property
    def name(self) -> str:
        return "bravo"

    @property
    def version(self) -> str:
        return "4.0"

    @property
    def preferred_signals(self) -> list[str]:
        return ["bb_lower_touch", "rsi_oversold"]

    @property
    def exit_signals(self) -> list[str]:
        return ["rsi_overbought"]

    @property
    def scoring_profile(self) -> ScoringProfile:
        """Mean reversion: favors strong balance sheets, low leverage, low valuation."""
        return ScoringProfile(
            factors=[
                ScoringFactor("current_ratio", 0.20, "high"),
                ScoringFactor("quick_ratio", 0.15, "high"),
                ScoringFactor("debt_to_equity", 0.20, "low"),
                ScoringFactor("price_to_book", 0.20, "low"),
                ScoringFactor("short_interest", 0.10, "low"),
                ScoringFactor("dividend_yield", 0.15, "high"),
            ],
            top_n=50,
        )

    @property
    def risk_config(self) -> dict:
        """Conservative value — 2% risk, 20% concentration."""
        return {"max_risk_pct": 0.02, "max_concentration_pct": 0.20}

    def init(self) -> None:
        pass

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        cr = fundamentals.current_ratio
        if cr is None or cr <= 1.0:
            return False
        de = fundamentals.debt_to_equity
        if de is not None and de >= 2.0:
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

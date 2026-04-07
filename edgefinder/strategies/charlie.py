"""Charlie Strategy — Value / Contrarian swing trading.

Targets stocks with strong free cash flow and manageable debt.
Qualifies: fcf_yield > 2%, debt_to_equity < 3.0
Signals: RSI oversold, MACD bullish cross (high confidence only, >= 80)
Trade type: SWING

NOTE: Mock framework — qualification criteria are placeholders for refinement.
"""

from __future__ import annotations

import pandas as pd

from edgefinder.core.models import Signal, TickerFundamentals
from edgefinder.scanner.scoring import ScoringFactor, ScoringProfile
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
        return "4.0"

    @property
    def preferred_signals(self) -> list[str]:
        return ["rsi_oversold", "macd_bullish_cross"]

    @property
    def exit_signals(self) -> list[str]:
        return ["macd_bearish_cross"]

    @property
    def scoring_profile(self) -> ScoringProfile:
        """Deep value: favors high cash flow, low debt, cheap valuation, dividend income."""
        return ScoringProfile(
            factors=[
                ScoringFactor("fcf_yield", 0.25, "high"),
                ScoringFactor("free_cash_flow", 0.15, "high"),
                ScoringFactor("debt_to_equity", 0.20, "low"),
                ScoringFactor("price_to_free_cash_flow", 0.15, "low"),
                ScoringFactor("ev_to_ebitda", 0.10, "low"),
                ScoringFactor("dividend_yield", 0.15, "high"),
            ],
            top_n=50,
        )

    def init(self) -> None:
        pass

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        fcf = fundamentals.fcf_yield
        if fcf is None or fcf <= 0.02:
            return False
        dte = fundamentals.debt_to_equity
        if dte is not None and dte >= 3.0:
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
                and sig.action.value == "BUY"
                and sig.confidence >= MIN_CONFIDENCE
            ):
                sig.strategy_name = self.name
                result.append(sig)
        return result

    def on_trade_executed(self, notification: TradeNotification) -> None:
        pass

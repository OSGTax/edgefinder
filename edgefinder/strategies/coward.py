"""Coward — conservative swing trading.

Watches quality stocks (positive earnings, strong balance sheet).
Enters on oversold dips. Exits early at first sign of a top.
Wins often, wins small.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.strategy_interface import SwingStrategy


class CowardStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "coward"

    @property
    def risk_pct(self) -> float:
        return 0.05

    @property
    def target_pct(self) -> float:
        return 0.15

    @property
    def watchlist_size(self) -> int:
        return 50

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        eg = fundamentals.earnings_growth
        if eg is None or eg <= 0:
            return False
        cr = fundamentals.current_ratio
        if cr is None or cr <= 1.5:
            return False
        return True

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current

        # Entry condition 1: RSI below 35
        if ind.rsi is not None and ind.rsi < 35:
            return self.make_intent(
                ticker, data,
                f"RSI oversold at {ind.rsi:.1f} (threshold: 35)",
            )

        # Entry condition 2: Price within 1% of BB lower band
        if (
            ind.bb_lower is not None
            and ind.close > 0
            and abs(ind.close - ind.bb_lower) / ind.close <= 0.01
        ):
            return self.make_intent(
                ticker, data,
                f"Price ${ind.close:.2f} near BB lower ${ind.bb_lower:.2f} (within 1%)",
            )

        return None

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        # Exit when RSI crosses above 70
        if ind.rsi is not None and ind.rsi > 70:
            return self.make_exit(
                ticker, data,
                f"RSI overbought at {ind.rsi:.1f} (threshold: 70)",
            )

        return None

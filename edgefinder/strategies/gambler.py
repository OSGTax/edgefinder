"""Gambler — balanced swing trading.

Rides momentum in the middle of moves. Enters when MACD crosses
bullish with RSI in neutral territory. Exits when momentum fades.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("gambler")
class GamblerStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "gambler"

    @property
    def risk_pct(self) -> float:
        return 0.10

    @property
    def target_pct(self) -> float:
        return 0.25

    @property
    def watchlist_size(self) -> int:
        return 100

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        eg = fundamentals.earnings_growth
        rg = fundamentals.revenue_growth
        has_earnings = eg is not None and eg > 0
        has_revenue = rg is not None and rg > 0
        return has_earnings or has_revenue

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        prev = data.history.previous

        if prev is None or ind.macd_histogram is None or ind.rsi is None:
            return None

        prev_hist = prev.macd_histogram
        if prev_hist is None:
            return None

        macd_crossed = prev_hist < 0 and ind.macd_histogram >= 0
        rsi_midrange = 40 <= ind.rsi <= 60

        if macd_crossed and rsi_midrange:
            return self.make_intent(
                ticker, data,
                f"MACD histogram crossed positive ({prev_hist:.3f} -> {ind.macd_histogram:.3f}), "
                f"RSI neutral at {ind.rsi:.1f}",
            )

        return None

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current
        prev = data.history.previous

        if prev is None or ind.macd_histogram is None:
            return None

        prev_hist = prev.macd_histogram
        if prev_hist is None:
            return None

        if prev_hist > 0 and ind.macd_histogram <= 0:
            return self.make_exit(
                ticker, data,
                f"MACD histogram crossed negative ({prev_hist:.3f} -> {ind.macd_histogram:.3f})",
            )

        return None

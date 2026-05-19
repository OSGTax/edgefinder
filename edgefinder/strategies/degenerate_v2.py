"""Degenerate — aggressive swing trading.

Jumps into volume spikes with bullish momentum. Rides until the
hype dies (volume fades + overbought). Lives or dies on single trades.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("degenerate")
class DegenerateStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "degenerate"

    @property
    def risk_pct(self) -> float:
        return 0.20

    @property
    def target_pct(self) -> float:
        return 0.50

    @property
    def watchlist_size(self) -> int:
        return 200

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        return fundamentals.market_cap is not None

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current

        if ind.rsi is None or ind.ema_21 is None:
            return None

        volume_spike = data.volume_ratio > 2.0
        bullish = ind.rsi > 50 and ind.close > ind.ema_21

        if volume_spike and bullish:
            return self.make_intent(
                ticker, data,
                f"Volume spike {data.volume_ratio:.1f}x with bullish momentum "
                f"(RSI {ind.rsi:.1f}, price ${ind.close:.2f} > EMA21 ${ind.ema_21:.2f})",
            )

        return None

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current

        if ind.rsi is None:
            return None

        volume_faded = data.volume_ratio < 1.0
        overbought = ind.rsi > 80

        if volume_faded and overbought:
            return self.make_exit(
                ticker, data,
                f"Volume faded ({data.volume_ratio:.1f}x) and overbought (RSI {ind.rsi:.1f})",
            )

        return None

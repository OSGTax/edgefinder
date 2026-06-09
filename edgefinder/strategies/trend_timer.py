"""Trend Timer — hold the index in uptrends, go to cash in downtrends.

Goal class: RISK-ADJUSTED outperformance of SPY (the "SPY-like return with much
less risk" bar), NOT raw-return alpha. This is the canonical, most-replicated,
most retail-friendly way to beat buy-and-hold on Sharpe / drawdown: stay fully
invested in the index while it trends up (capturing the bull), step entirely to
cash when it loses its long-term trend (sidestepping the worst of the bear).
Faber (2007), "A Quantitative Approach to Tactical Asset Allocation."

- entry: index close > its long EMA (ema_200) → hold ~full equity in the index,
- exit: index close < ema_200 → flat (cash) until the trend reasserts.

It deliberately gives up some raw return to whipsaw in choppy markets, in
exchange for cutting the large drawdowns — so it is scored with
``--risk-adjusted`` (beat SPY's Sharpe in a majority of folds AND a smaller max
drawdown), never on mean excess return.

Sizing: risk_pct 0.20 against the fixed 20% catastrophic stop sizes to ~full
equity; max_concentration_pct 1.0 lets a single index position be the whole
book; no profit target / trailing / max-hold (ride the trend until it breaks).
The fixed 20% stop is an inert backstop — the index does not gap 20% intraday.

Knobs: NONE (pure pre-registered trend rule). Run with --fixed.
Defaults PRE-REGISTERED 2026-06-09 (risk-adjusted round 1) before any test.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("trend_timer")
class TrendTimerStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "trend_timer"

    @property
    def risk_pct(self) -> float:
        return 0.20  # with the fixed 20% stop → ~full-equity index position

    @property
    def max_concentration_pct(self) -> float:
        return 1.0  # a single index position may be the whole book

    @property
    def target_pct(self) -> float:
        return 100.0  # effectively no profit target — ride the trend

    @property
    def max_hold_days(self) -> int:
        return 0  # hold through the uptrend; the trend break is the exit

    @property
    def trailing_stop_pct(self):
        return None  # no trailing — the 200-EMA cross is the only exit

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        return True  # lab-only until promoted

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        if ind.ema_200 is None or not ind.close:
            return None
        if ind.close > ind.ema_200:
            return self.make_intent(
                ticker, data,
                f"{ticker} above its 200-EMA "
                f"({ind.close:.2f} > {ind.ema_200:.2f}) — risk-on, hold the index",
            )
        return None

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current
        if ind.ema_200 is not None and ind.close and ind.close < ind.ema_200:
            return self.make_exit(
                ticker, data,
                f"{ticker} lost its 200-EMA "
                f"({ind.close:.2f} < {ind.ema_200:.2f}) — risk-off, go to cash",
            )
        return None

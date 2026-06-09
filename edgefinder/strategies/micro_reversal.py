"""Micro Reversal — short-horizon mean reversion in the small-cap/illiquid band.

Effect class: short-term reversal (Connors / Alvarez family) — the most
evidence-backed daily-bar edge that demonstrably SURVIVES out-of-sample, but
only in small/illiquid names (in liquid large-caps it is arbitraged to ~0).
That is exactly why this strategy is born paired with the realistic cost
engine: the reversal bounce is partly the bid-ask bounce, so a naive backtest
captures spread it can never actually earn. Run WITHOUT --costed this number is
a fantasy; the honest test is microcap + costed.

Thesis: a name in a long-term uptrend (above its 200dma) that suffers a sharp
multi-day washout is oversold and tends to snap back within days. Buy the
washout, sell into the recovery.

- entry: close > ema_200 (not a falling knife) AND a >= drop_pct cumulative
  drop over the last `lookback` days AND RSI oversold (<= rsi_entry),
- exit: RSI recovers to >= rsi_exit, or max_hold, or the 20% stop.

Pre-registered kill criteria (microcap + costed screen): net profit factor
< 1.10 or non-positive mean net return per trade → dead. (The old flat-$5
floor doesn't apply here — friction is now modelled per name, not assumed.)

Defaults below are PRE-REGISTERED (2026-06-09, microcap round 1) before any
screen. risk_pct and target_pct are FIXED (not searched) — the exit is the
RSI recovery / time stop, not a profit target.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("micro_reversal")
class MicroReversalStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "micro_reversal"

    @property
    def risk_pct(self) -> float:
        return 0.03  # fixed

    @property
    def target_pct(self) -> float:
        # Fixed and wide — the RSI-recovery / time exit ends the trade, not a
        # target. A tight target would clip the bounce the strategy exists for.
        return 0.15

    @property
    def max_hold_days(self) -> int:
        # Short horizon: the snap-back happens in days or the thesis was wrong.
        return self._p("max_hold_days", 5)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        return True  # lab-only until promoted

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        ind = data.current
        if ind.ema_200 is None or ind.rsi is None or not ind.close:
            return None

        # 1) Long-term uptrend — never buy a washout below the 200dma (those
        #    are the names that keep going to zero / delist).
        if ind.close <= ind.ema_200:
            return None

        # 2) Sharp multi-day washout: cumulative drop over the lookback window.
        lookback = self._p("lookback", 3)
        closes = [c for c in data.history.get_field_series("close") if c]
        if len(closes) < lookback:
            return None
        ref = closes[-lookback]
        if not ref:
            return None
        drop = ind.close / ref - 1.0
        drop_pct = self._p("drop_pct", 0.10)
        if drop > -drop_pct:  # not enough of a washout
            return None

        # 3) Oversold confirmation.
        rsi_entry = self._p("rsi_entry", 30)
        if ind.rsi > rsi_entry:
            return None

        return self.make_intent(
            ticker, data,
            f"Washout {drop * 100:.0f}% over {lookback}d above 200dma "
            f"(RSI {ind.rsi:.0f} <= {rsi_entry}) — reversion buy",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        ind = data.current
        rsi_exit = self._p("rsi_exit", 55)
        if ind.rsi is not None and ind.rsi >= rsi_exit:
            return self.make_exit(
                ticker, data,
                f"RSI {ind.rsi:.0f} >= {rsi_exit} — reverted, sell the bounce",
            )
        return None

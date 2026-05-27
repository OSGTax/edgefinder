"""Example backtest strategies — demonstrate the BacktestStrategy interface."""

from __future__ import annotations

import math
from collections import defaultdict, deque

from edgefinder.backtest.engine import Bar, BacktestContext, Order


class SmaCrossStrategy:
    """Simple moving-average crossover, long-only, one position per symbol.

    Buys ~``target_dollars`` worth when the fast SMA crosses above the slow
    SMA; sells the whole position on the reverse cross. A minimal but
    fully-working strategy to exercise the engine end to end.
    """

    def __init__(self, fast: int = 10, slow: int = 30, target_dollars: float = 2_000.0) -> None:
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        self.fast = fast
        self.slow = slow
        self.target_dollars = target_dollars
        self._closes: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=slow))
        self._prev_fast_above: dict[str, bool] = {}

    def on_bar(self, bar: Bar, ctx: BacktestContext) -> list[Order] | None:
        closes = self._closes[bar.symbol]
        closes.append(bar.close)
        if len(closes) < self.slow:
            return None

        fast_sma = sum(list(closes)[-self.fast :]) / self.fast
        slow_sma = sum(closes) / self.slow
        fast_above = fast_sma > slow_sma
        prev = self._prev_fast_above.get(bar.symbol)
        self._prev_fast_above[bar.symbol] = fast_above

        if prev is None:
            return None

        held = ctx.position(bar.symbol)
        if fast_above and not prev and held is None:
            qty = int(math.floor(self.target_dollars / bar.close))
            if qty > 0:
                return [Order(bar.symbol, "BUY", qty)]
        elif not fast_above and prev and held is not None:
            return [Order(bar.symbol, "SELL", held.quantity)]
        return None

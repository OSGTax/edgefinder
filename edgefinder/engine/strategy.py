"""The general strategy interface for the EdgeFinder lab.

One interface, every kind of strategy. A strategy sees the WHOLE universe as it
looked on the decision date (point-in-time — no future data exists in the
context, by construction) and returns the portfolio it wants to hold:

    rebalance(ctx) -> {symbol: target_weight}

Weights are fractions of equity in [0, 1]; their sum is the invested fraction
(the rest is cash). That single shape expresses everything:

- technical:   weight 1.0 on SPY while its close > 200-EMA, else {} (cash)
- fundamental: equal-weight the 20 lowest-PEG names with >15% earnings growth
- cross-sec:   the top-k by momentum, equal-weight
- calendar:    the top-traded 'B...' tickers, but only on Tuesdays
- dumb:        literally anything you can compute from an AssetView

The engine (engine/backtest.py) turns target weights into trades at the next
open, with realistic costs — the strategy never touches order mechanics, so a
strategy CANNOT have a 'trading-sequence bug'. That lives in one tested place.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from edgefinder.core.models import TickerFundamentals
from edgefinder.data.market_data import IndicatorSnapshot


@dataclass(frozen=True)
class AssetView:
    """Everything known about ONE asset as of the decision date (point-in-time).

    ``history`` ends on the decision date — there is no later bar to peek at.
    """

    symbol: str
    price: float                       # close on the decision date
    indicators: IndicatorSnapshot      # indicators as of the decision date
    history: pd.DataFrame              # OHLCV up to & including the decision date
    fundamentals: TickerFundamentals | None = None

    def ret(self, lookback: int) -> float | None:
        """Trailing total return over ``lookback`` bars (None if too short)."""
        c = self.history["close"]
        if len(c) <= lookback or c.iloc[-lookback - 1] <= 0:
            return None
        return float(c.iloc[-1] / c.iloc[-lookback - 1] - 1.0)


@dataclass(frozen=True)
class RebalanceContext:
    """The whole universe, point-in-time, handed to a strategy each rebalance.

    ``holdings`` is the strategy's CURRENT book at decision time, expressed as
    a fraction of equity per held symbol (the engine drives it; the strategy
    only reads it). It is computed look-ahead-free from the PRE-fill shares and
    the SAME decision-date close prices the strategy sees in ``assets`` — never
    today's fill. It defaults to ``{}`` so every existing stateless strategy is
    byte-identical (a strategy that ignores ``holdings`` is unaffected). A
    stateful hold/exit strategy reads it via ``held``/``held_symbols`` to decide
    keep/exit/enter per position instead of reselecting the whole book daily.
    """

    date: date
    assets: dict[str, AssetView]
    holdings: dict[str, float] = field(default_factory=dict)

    def symbols(self) -> list[str]:
        return list(self.assets)

    def get(self, symbol: str) -> AssetView | None:
        return self.assets.get(symbol)

    def price(self, symbol: str) -> float | None:
        a = self.assets.get(symbol)
        return a.price if a else None

    def held(self, symbol: str) -> float:
        """Current weight of ``symbol`` in the book (0.0 if not held)."""
        return self.holdings.get(symbol, 0.0)

    def held_symbols(self) -> list[str]:
        """Symbols currently held with a positive weight."""
        return [s for s, w in self.holdings.items() if w > 0]


@runtime_checkable
class Strategy(Protocol):
    """A strategy is anything with a name and a rebalance() returning weights."""

    @property
    def name(self) -> str: ...

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        """Return desired {symbol: weight}; weights in [0,1], sum <= 1 (rest
        cash). Symbols not in ctx.assets, or with non-positive weight, are
        ignored by the engine."""
        ...


# ── reference strategies (used by the engine's own tests / as adapters) ──


class BuyAndHold:
    """Hold one symbol at 100% — the engine's correctness anchor (it must equal
    that symbol's buy-and-hold return, net of one entry cost)."""

    def __init__(self, symbol: str = "SPY") -> None:
        self.symbol = symbol

    @property
    def name(self) -> str:
        return f"buy_and_hold_{self.symbol.lower()}"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        return {self.symbol: 1.0} if self.symbol in ctx.assets else {}


class EqualWeight:
    """Equal-weight every asset in the universe (rebalanced on schedule)."""

    @property
    def name(self) -> str:
        return "equal_weight"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        syms = ctx.symbols()
        return {s: 1.0 / len(syms) for s in syms} if syms else {}


def _resolve_index(dates: list, decision_date) -> int:
    """Index of the last bar dated <= decision_date (-1 if none)."""
    return bisect.bisect_right(dates, decision_date) - 1

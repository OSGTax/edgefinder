"""Turn-of-Month Seasonality — hold equities only through the month turn.

Effect class: turn-of-month (TOM) — equities historically earn a
disproportionate share of returns in the last few and first few trading
days of each month (pension/401k flow rebalancing being the usual story).
One of the oldest documented calendar anomalies; first calendar candidate
this lab can test now that strategies know the session date
(MarketContext.as_of, v5.19.0).

Mechanics: queue entries late in the month (signals on day T fill at
T+1's open, so an entry_day of 25 means fills land ~the 26th onward);
exit at the close once the calendar leaves the turn window. The engine's
20% stop backstops; the profit target is set wide so the calendar, not a
target, ends the trade.

DISCLOSED DESIGN CAVEAT: every watchlist name qualifies simultaneously,
so which ~5 names fill the position slots is watchlist-order (effectively
arbitrary). For an index-level calendar effect any basket of liquid
large-caps approximates the market, but this is a known bias source and
is disclosed up front rather than engineered around.

Knobs (2): entry_day, exit_day (calendar days; trading-day precision is
not available to strategies and the approximation is part of the test).
Pre-registered kill criteria: screen avg net/trade < $5 or PF < 1.10 →
dead.

Defaults below are PRE-REGISTERED (2026-06-05, round 3 ws2) before any
screen.
"""

from __future__ import annotations

from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData
from edgefinder.strategies.base import StrategyRegistry
from edgefinder.strategies.strategy_interface import SwingStrategy


@StrategyRegistry.register("tom_seasonality")
class TomSeasonalityStrategy(SwingStrategy):

    @property
    def name(self) -> str:
        return "tom_seasonality"

    @property
    def risk_pct(self) -> float:
        return 0.03  # fixed

    @property
    def target_pct(self) -> float:
        return 0.30  # fixed wide — the calendar ends the trade, not a target

    @property
    def max_hold_days(self) -> int:
        return 15  # fixed safety net; the exit_day rule fires well before

    @property
    def entry_day(self) -> int:
        return self._p("entry_day", 25)

    @property
    def exit_day(self) -> int:
        return self._p("exit_day", 5)

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        return True  # lab-only until promoted

    def _in_window(self, d) -> bool:
        """True while the calendar sits inside the turn-of-month window."""
        return d.day >= self.entry_day or d.day <= self.exit_day

    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        d = data.context.as_of
        if d is None or d.day < self.entry_day:
            # Enter only on the back side of the window: a fill on the 2nd
            # of the month would capture almost none of the turn.
            return None
        return self.make_intent(
            ticker, data,
            f"Turn-of-month window (day {d.day} >= {self.entry_day})",
        )

    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        d = data.context.as_of
        if d is None:
            return None
        if not self._in_window(d):
            return self.make_exit(
                ticker, data,
                f"Turn-of-month over (day {d.day})",
            )
        return None

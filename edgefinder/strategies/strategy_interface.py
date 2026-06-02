"""New strategy interface — strategies evaluate raw data and return intents.

All strategies are swing-oriented. Day trades only happen as damage control
(stop hit on entry day). Each strategy defines its own entry/exit logic
using the full MarketData object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from config.settings import settings
from edgefinder.core.models import ExitIntent, TickerFundamentals, TradeIntent
from edgefinder.data.market_data import MarketData


class SwingStrategy(ABC):
    """Base class for all new strategies.

    Subclasses define:
    - name: unique identifier
    - risk_pct: fraction of equity to risk per trade
    - target_pct: profit target as fraction of entry price
    - watchlist_size: max tickers to watch
    - qualifies_stock(): fundamental watchlist filter
    - evaluate(): entry decision
    - should_exit(): exit decision
    """

    stop_pct: float = 0.20  # fixed for all strategies, non-negotiable

    @property
    @abstractmethod
    def name(self) -> str: ...

    # ── Risk / exit parameters (overridable; safe defaults) ──
    # These keep capital recycling so trades complete and produce realized
    # P&L. They are the knobs the Phase-2 validation lab will tune per strategy.

    @property
    def max_concentration_pct(self) -> float:
        """Max fraction of equity in a single position (hard ceiling)."""
        return settings.max_portfolio_concentration_pct

    @property
    def max_hold_days(self) -> int:
        """Force-exit a position after this many calendar days, so capital
        doesn't sit idle when neither stop nor target triggers. 0 disables."""
        return 20

    @property
    def trailing_stop_pct(self) -> float | None:
        """Once a position is up >= 1R, trail the stop this far below the peak
        price. None disables trailing."""
        return 0.10

    @property
    @abstractmethod
    def risk_pct(self) -> float: ...

    @property
    @abstractmethod
    def target_pct(self) -> float: ...

    @property
    def watchlist_size(self) -> int:
        return 50

    @abstractmethod
    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        """Watchlist filter — applied during nightly scan."""
        ...

    @abstractmethod
    def evaluate(self, ticker: str, data: MarketData) -> TradeIntent | None:
        """Decide whether to enter a trade. Return TradeIntent or None."""
        ...

    @abstractmethod
    def should_exit(
        self, ticker: str, data: MarketData, entry_price: float
    ) -> ExitIntent | None:
        """Decide whether to exit an open position. Return ExitIntent or None."""
        ...

    # ── Helpers for subclasses ──

    def make_intent(
        self, ticker: str, data: MarketData, reasoning: str
    ) -> TradeIntent:
        return TradeIntent(
            ticker=ticker,
            direction="LONG",
            reasoning=reasoning,
            strategy_name=self.name,
            indicators_snapshot=data.current.to_dict(),
            fundamentals_snapshot=(
                data.fundamentals.model_dump(exclude_none=True)
                if data.fundamentals else {}
            ),
            market_context_snapshot={
                "spy_price": data.context.spy_price,
                "spy_change_pct": data.context.spy_change_pct,
                "vix_level": data.context.vix_level,
                "market_regime": data.context.market_regime,
            },
        )

    def make_exit(
        self, ticker: str, data: MarketData, reasoning: str
    ) -> ExitIntent:
        return ExitIntent(
            ticker=ticker,
            reasoning=reasoning,
            indicators_snapshot=data.current.to_dict(),
        )

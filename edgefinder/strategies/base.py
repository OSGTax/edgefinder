"""EdgeFinder v2 — Strategy plugin system.

BaseStrategy ABC defines the contract all strategies implement.
StrategyRegistry provides auto-discovery via @register decorator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from edgefinder.core.models import MarketSnapshot, Signal, TickerFundamentals, Trade


@dataclass
class TradeNotification:
    """Immutable notification sent to strategies when trades execute."""

    trade: Trade
    event: str  # "opened", "closed", "cancelled"


class BaseStrategy(ABC):
    """Abstract base all trading strategies must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Strategy version string."""
        ...

    @property
    @abstractmethod
    def preferred_signals(self) -> list[str]:
        """Signal pattern names this strategy is interested in."""
        ...

    @abstractmethod
    def init(self) -> None:
        """One-time setup called when the system starts."""
        ...

    @abstractmethod
    def generate_signals(self, ticker: str, bars: pd.DataFrame) -> list[Signal]:
        """Given OHLCV bars for a ticker, return 0-N trading signals."""
        ...

    @abstractmethod
    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        """Should this stock stay in the active watchlist for this strategy?"""
        ...

    @abstractmethod
    def on_trade_executed(self, notification: TradeNotification) -> None:
        """Called when a trade opens, closes, or is cancelled."""
        ...

    # ── Optional hooks (default no-ops) ──────────────

    def get_watchlist(self) -> list[str]:
        """Return tickers this strategy currently wants data for."""
        return []

    def on_market_regime_change(self, regime: str) -> None:
        """React to bull/bear/sideways regime changes."""
        pass

    def on_strategy_pause(self, reason: str) -> None:
        """Handle auto-pause due to drawdown or other triggers."""
        pass

    def on_market_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Called with latest market context before signal generation."""
        pass

    # ── AI agent hooks (future) ──────────────────────

    def get_state(self) -> dict[str, Any]:
        """Return serializable internal state for AI agent inspection."""
        return {}

    def apply_suggestion(self, suggestion: dict) -> bool:
        """Accept a parameter/behavior suggestion from the AI agent.
        Returns True if applied. Default: reject all."""
        return False


class StrategyRegistry:
    """Class-level registry for auto-discovered strategy plugins.

    Usage:
        @StrategyRegistry.register("alpha")
        class AlphaStrategy(BaseStrategy):
            ...
    """

    _strategies: dict[str, type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy class."""
        def decorator(strategy_cls: type[BaseStrategy]) -> type[BaseStrategy]:
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseStrategy] | None:
        """Get a strategy class by name."""
        return cls._strategies.get(name)

    @classmethod
    def get_all(cls) -> dict[str, type[BaseStrategy]]:
        """Get all registered strategy classes."""
        return dict(cls._strategies)

    @classmethod
    def get_instances(cls) -> list[BaseStrategy]:
        """Instantiate and return all registered strategies."""
        instances = []
        for name, cls_ in cls._strategies.items():
            try:
                instances.append(cls_())
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "Failed to instantiate strategy '%s'", name
                )
        return instances

    @classmethod
    def list_names(cls) -> list[str]:
        """List all registered strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered strategies. For testing."""
        cls._strategies.clear()

"""Echo Strategy — Meta-strategy that learns from other strategies' history.

Echo doesn't generate its own signals. Instead, it:
1. Observes the current market regime (VIX + SPY trend)
2. Queries historical performance of all other strategies under similar conditions
3. Forwards only signals from the strategy with the best track record
   in the current regime

This creates a "strategy of strategies" that adapts to market conditions
by delegating to whichever approach has historically worked best.

When there's insufficient history, Echo falls back to forwarding signals
from ALL strategies that have a positive overall expectancy.
"""

from __future__ import annotations

import logging

import pandas as pd

from edgefinder.analytics.conditional_stats import (
    StrategyProfile,
    build_strategy_profiles,
    get_best_strategy_for_regime,
    get_strategy_scores_for_regime,
)
from edgefinder.analytics.regime import MarketCondition, classify_regime
from edgefinder.analytics.trade_analytics import TradeFeatures, build_trade_features
from edgefinder.core.models import Signal, TickerFundamentals
from edgefinder.signals.engine import compute_indicators, detect_signals
from edgefinder.strategies.base import BaseStrategy, StrategyRegistry, TradeNotification

logger = logging.getLogger(__name__)


@StrategyRegistry.register("echo")
class EchoStrategy(BaseStrategy):
    """Meta-strategy that selects signals based on historical regime performance."""

    def __init__(self) -> None:
        self._profiles: dict[str, StrategyProfile] = {}
        self._current_regime: MarketCondition = MarketCondition.SIDEWAYS_CALM
        self._active_strategy: str | None = None
        self._vix_level: float = 20.0
        self._spy_change_pct: float = 0.0
        self._db_session = None

    @property
    def name(self) -> str:
        return "echo"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def preferred_signals(self) -> list[str]:
        """Echo accepts all signal types — filtering happens at strategy level."""
        return [
            "ema_crossover_bullish",
            "macd_bullish_cross",
            "rsi_oversold",
            "bb_lower_touch",
            "volume_spike_bullish",
        ]

    @property
    def exit_signals(self) -> list[str]:
        """Use the active strategy's exit signals, or conservative defaults."""
        return ["ema_crossover_bearish", "volume_spike_bearish"]

    @property
    def scoring_profile(self):
        """Broad scoring: balanced across growth, value, and momentum factors.

        Echo casts a wide net since it delegates signal selection to whichever
        source strategy performs best in the current regime. The scoring just
        ensures the watchlist covers stocks that any strategy might want.
        """
        from edgefinder.scanner.scoring import ScoringFactor, ScoringProfile
        return ScoringProfile(
            factors=[
                ScoringFactor("earnings_growth", 0.20, "high"),
                ScoringFactor("revenue_growth", 0.20, "high"),
                ScoringFactor("fcf_yield", 0.15, "high"),
                ScoringFactor("return_on_equity", 0.15, "high"),
                ScoringFactor("ev_to_ebitda", 0.15, "low"),
                ScoringFactor("peg_ratio", 0.15, "low"),
            ],
            top_n=50,
        )

    @property
    def risk_config(self) -> dict:
        """Conservative risk — Echo should be careful since it's learning."""
        return {"max_risk_pct": 0.02, "max_concentration_pct": 0.20}

    def init(self) -> None:
        """Load historical profiles on startup."""
        self._refresh_profiles()

    def set_db_session(self, session) -> None:
        """Inject DB session for querying trade history."""
        self._db_session = session
        self._refresh_profiles()

    def _refresh_profiles(self) -> None:
        """Rebuild strategy profiles from trade history."""
        if self._db_session is None:
            logger.info("[echo] No DB session — profiles empty until set_db_session()")
            return

        try:
            features = build_trade_features(self._db_session)
            if features:
                self._profiles = build_strategy_profiles(features)
                self._update_active_strategy()
                logger.info(
                    "[echo] Loaded profiles for %d strategies from %d trades",
                    len(self._profiles), len(features),
                )
            else:
                logger.info("[echo] No closed trades yet — will forward all signals")
        except Exception:
            logger.exception("[echo] Failed to load profiles")

    def _update_active_strategy(self) -> None:
        """Select the best-performing strategy for current regime."""
        self._current_regime = classify_regime(
            vix_level=self._vix_level,
            spy_change_pct=self._spy_change_pct,
        )

        best = get_best_strategy_for_regime(self._profiles, self._current_regime)
        if best != self._active_strategy:
            logger.info(
                "[echo] Regime=%s -> active strategy: %s (was %s)",
                self._current_regime.value, best or "ALL", self._active_strategy,
            )
            self._active_strategy = best

    def on_market_snapshot(self, snapshot) -> None:
        """Update regime classification when new market data arrives."""
        self._vix_level = snapshot.vix_level
        self._spy_change_pct = snapshot.spy_change_pct
        self._update_active_strategy()

    def qualifies_stock(self, fundamentals: TickerFundamentals) -> bool:
        """Echo is permissive — it qualifies anything that any strategy would.

        The filtering happens at signal selection time, not qualification time.
        This ensures Echo's watchlist is the union of all other strategies.
        """
        # Minimum viability: must have market cap and some data
        if fundamentals.market_cap is not None and fundamentals.market_cap < 300_000_000:
            return False
        # Accept if there's any growth or value data
        has_data = any([
            fundamentals.earnings_growth is not None,
            fundamentals.revenue_growth is not None,
            fundamentals.fcf_yield is not None,
        ])
        return has_data

    def generate_signals(self, ticker: str, bars: pd.DataFrame) -> list[Signal]:
        """Generate signals by filtering through the best-performing strategy's lens.

        If we have a clear best strategy for the current regime, only emit
        signals that match that strategy's preferred patterns. Otherwise,
        emit signals that match ANY strategy with positive expectancy.
        """
        indicators = compute_indicators(bars)
        if indicators is None:
            return []

        all_signals = detect_signals(indicators, ticker)
        if not all_signals:
            return []

        # Determine which signal patterns to accept
        allowed_patterns = self._get_allowed_patterns()

        result = []
        for sig in all_signals:
            pattern = sig.metadata.get("pattern", "")
            if pattern in allowed_patterns and sig.action.value == "BUY":
                sig.strategy_name = self.name
                result.append(sig)

        if result:
            logger.debug(
                "[echo] %s: %d signals (regime=%s, active=%s)",
                ticker, len(result), self._current_regime.value,
                self._active_strategy or "ALL",
            )

        return result

    def _get_allowed_patterns(self) -> set[str]:
        """Determine which signal patterns to accept based on current regime performance."""
        if not self._profiles:
            # No history — use all preferred signals (exploratory mode)
            return set(self.preferred_signals)

        if self._active_strategy:
            # We have a clear winner — use that strategy's signal preferences
            return self._get_strategy_preferred_signals(self._active_strategy)

        # No clear winner — use signals from all strategies with positive expectancy
        allowed: set[str] = set()
        scores = get_strategy_scores_for_regime(self._profiles, self._current_regime)
        for name, expectancy, reliable in scores:
            if expectancy > 0:
                allowed.update(self._get_strategy_preferred_signals(name))

        # If nothing has positive expectancy, fall back to our defaults
        return allowed if allowed else set(self.preferred_signals)

    @staticmethod
    def _get_strategy_preferred_signals(strategy_name: str) -> set[str]:
        """Look up a strategy's preferred signals from the registry."""
        strategy_cls = StrategyRegistry.get(strategy_name)
        if strategy_cls:
            try:
                instance = strategy_cls()
                return set(instance.preferred_signals)
            except Exception:
                pass
        # Fallback: common bullish signals
        return {"ema_crossover_bullish", "macd_bullish_cross", "volume_spike_bullish"}

    def on_trade_executed(self, notification: TradeNotification) -> None:
        """After each trade closes, refresh profiles to learn from new data."""
        if notification.event == "closed" and self._db_session:
            self._refresh_profiles()

    # ── AI agent hooks ──────────────────────────────────

    def get_state(self) -> dict:
        """Expose internal state for inspection."""
        scores = []
        if self._profiles:
            scores = [
                {"strategy": name, "expectancy": exp, "reliable": rel}
                for name, exp, rel in get_strategy_scores_for_regime(
                    self._profiles, self._current_regime
                )
            ]

        return {
            "current_regime": self._current_regime.value,
            "active_strategy": self._active_strategy,
            "vix_level": self._vix_level,
            "spy_change_pct": self._spy_change_pct,
            "strategy_scores": scores,
            "profiles_loaded": len(self._profiles),
        }

"""EdgeFinder v2 — Conditional win-rate tracker.

Computes per-strategy performance broken down by market regime,
signal type, and other conditions. This is the core intelligence
that the Delta meta-strategy uses to decide which source strategy
to trust in current conditions.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from edgefinder.analytics.regime import MarketCondition
from edgefinder.analytics.trade_analytics import TradeFeatures

logger = logging.getLogger(__name__)

# Minimum trades needed in a condition bucket to be statistically meaningful
MIN_TRADES_FOR_CONFIDENCE = 5


@dataclass
class ConditionStats:
    """Performance stats for a strategy under specific conditions."""

    strategy_name: str
    condition: str  # regime name, signal pattern, etc.
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_r_multiple: float = 0.0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    expectancy: float = 0.0  # avg_win * win_rate - avg_loss * loss_rate

    @property
    def is_reliable(self) -> bool:
        """Whether we have enough data to trust these stats."""
        return self.total_trades >= MIN_TRADES_FOR_CONFIDENCE


@dataclass
class StrategyProfile:
    """Complete conditional performance profile for one strategy."""

    strategy_name: str
    overall_win_rate: float = 0.0
    overall_expectancy: float = 0.0
    total_trades: int = 0
    by_regime: dict[str, ConditionStats] = field(default_factory=dict)
    by_signal: dict[str, ConditionStats] = field(default_factory=dict)
    by_exit_reason: dict[str, ConditionStats] = field(default_factory=dict)


def compute_condition_stats(
    trades: list[TradeFeatures],
    strategy_name: str,
    group_key: str,
) -> dict[str, ConditionStats]:
    """Group trades by a condition key and compute stats for each bucket.

    Args:
        trades: Filtered list of TradeFeatures for one strategy.
        strategy_name: Strategy name for labeling.
        group_key: Which attribute to group by ("regime", "signal", "exit_reason").
    """
    buckets: dict[str, list[TradeFeatures]] = defaultdict(list)

    for t in trades:
        if group_key == "regime":
            buckets[t.regime.value].append(t)
        elif group_key == "signal":
            for sig in t.signals_fired:
                buckets[sig].append(t)
            if not t.signals_fired:
                buckets["unknown"].append(t)
        elif group_key == "exit_reason":
            buckets[t.exit_reason or "unknown"].append(t)

    result: dict[str, ConditionStats] = {}
    for condition, group in buckets.items():
        result[condition] = _compute_stats(strategy_name, condition, group)

    return result


def _compute_stats(
    strategy_name: str,
    condition: str,
    trades: list[TradeFeatures],
) -> ConditionStats:
    """Compute stats for a single condition bucket."""
    if not trades:
        return ConditionStats(strategy_name=strategy_name, condition=condition)

    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    total_pnl = sum(t.pnl_dollars for t in trades)
    avg_r = sum(t.r_multiple for t in trades) / len(trades)
    win_rate = len(wins) / len(trades)

    # Expectancy: avg_win * win_rate - avg_loss * loss_rate
    avg_win = sum(t.pnl_dollars for t in wins) / len(wins) if wins else 0
    avg_loss = abs(sum(t.pnl_dollars for t in losses) / len(losses)) if losses else 0
    loss_rate = 1 - win_rate
    expectancy = avg_win * win_rate - avg_loss * loss_rate

    return ConditionStats(
        strategy_name=strategy_name,
        condition=condition,
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        total_pnl=round(total_pnl, 2),
        avg_r_multiple=round(avg_r, 2),
        win_rate=round(win_rate, 4),
        avg_pnl=round(total_pnl / len(trades), 2),
        expectancy=round(expectancy, 2),
    )


def build_strategy_profiles(
    all_features: list[TradeFeatures],
) -> dict[str, StrategyProfile]:
    """Build conditional performance profiles for all strategies.

    Args:
        all_features: Complete list of TradeFeatures from trade_analytics.

    Returns:
        Dict mapping strategy_name -> StrategyProfile.
    """
    # Group features by strategy
    by_strategy: dict[str, list[TradeFeatures]] = defaultdict(list)
    for f in all_features:
        by_strategy[f.strategy_name].append(f)

    profiles: dict[str, StrategyProfile] = {}
    for name, trades in by_strategy.items():
        overall = _compute_stats(name, "overall", trades)
        profile = StrategyProfile(
            strategy_name=name,
            overall_win_rate=overall.win_rate,
            overall_expectancy=overall.expectancy,
            total_trades=overall.total_trades,
            by_regime=compute_condition_stats(trades, name, "regime"),
            by_signal=compute_condition_stats(trades, name, "signal"),
            by_exit_reason=compute_condition_stats(trades, name, "exit_reason"),
        )
        profiles[name] = profile
        logger.info(
            "Strategy '%s' profile: %d trades, %.1f%% win rate, $%.2f expectancy",
            name, overall.total_trades, overall.win_rate * 100, overall.expectancy,
        )

    return profiles


def get_best_strategy_for_regime(
    profiles: dict[str, StrategyProfile],
    regime: MarketCondition,
) -> str | None:
    """Return the strategy with the highest expectancy for a given regime.

    Only considers strategies with reliable data (>= MIN_TRADES_FOR_CONFIDENCE).
    Returns None if no strategy has reliable data for this regime.
    """
    best_name: str | None = None
    best_expectancy = float("-inf")

    for name, profile in profiles.items():
        # Skip the echo strategy itself to avoid circular dependency
        if name == "echo":
            continue

        stats = profile.by_regime.get(regime.value)
        if stats and stats.is_reliable and stats.expectancy > best_expectancy:
            best_expectancy = stats.expectancy
            best_name = name

    return best_name


def get_strategy_scores_for_regime(
    profiles: dict[str, StrategyProfile],
    regime: MarketCondition,
) -> list[tuple[str, float, bool]]:
    """Return all strategies scored by expectancy for a regime.

    Returns list of (strategy_name, expectancy, is_reliable) sorted by
    expectancy descending. Includes unreliable strategies but flags them.
    """
    scores: list[tuple[str, float, bool]] = []

    for name, profile in profiles.items():
        if name == "echo":
            continue
        stats = profile.by_regime.get(regime.value)
        if stats:
            scores.append((name, stats.expectancy, stats.is_reliable))
        else:
            scores.append((name, 0.0, False))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

"""EdgeFinder v2 — Market regime tagger.

Classifies market conditions into discrete buckets based on VIX level
and index trend. Used by the analytics layer and Delta meta-strategy
to understand which strategies perform best under which conditions.
"""

from __future__ import annotations

from enum import Enum


class MarketCondition(str, Enum):
    """Discrete market regime buckets combining trend + volatility."""

    BULL_CALM = "bull_calm"        # SPY up, VIX < 20
    BULL_VOLATILE = "bull_volatile"  # SPY up, VIX >= 20
    BEAR_CALM = "bear_calm"        # SPY down, VIX < 20
    BEAR_VOLATILE = "bear_volatile"  # SPY down, VIX >= 20
    SIDEWAYS_CALM = "sideways_calm"
    SIDEWAYS_VOLATILE = "sideways_volatile"


# VIX threshold separating calm from volatile
VIX_VOLATILITY_THRESHOLD = 20.0

# SPY change % thresholds for trend classification
SPY_BULL_THRESHOLD = 0.3    # > +0.3% = bullish
SPY_BEAR_THRESHOLD = -0.3   # < -0.3% = bearish


def classify_regime(
    vix_level: float,
    spy_change_pct: float = 0.0,
    market_regime: str = "sideways",
) -> MarketCondition:
    """Classify market conditions into a discrete bucket.

    Args:
        vix_level: Current VIX level.
        spy_change_pct: SPY daily change percentage.
        market_regime: The regime string from MarketSnapshot ("bull", "bear", "sideways").

    Returns:
        A MarketCondition enum value.
    """
    is_volatile = vix_level >= VIX_VOLATILITY_THRESHOLD

    # Use spy_change_pct as primary trend signal, fall back to regime string
    if spy_change_pct > SPY_BULL_THRESHOLD or market_regime == "bull":
        return MarketCondition.BULL_VOLATILE if is_volatile else MarketCondition.BULL_CALM
    elif spy_change_pct < SPY_BEAR_THRESHOLD or market_regime == "bear":
        return MarketCondition.BEAR_VOLATILE if is_volatile else MarketCondition.BEAR_CALM
    else:
        return MarketCondition.SIDEWAYS_VOLATILE if is_volatile else MarketCondition.SIDEWAYS_CALM


def classify_from_snapshot_record(snapshot) -> MarketCondition:
    """Classify regime from a MarketSnapshotRecord ORM object."""
    return classify_regime(
        vix_level=snapshot.vix_level,
        spy_change_pct=snapshot.spy_change_pct,
        market_regime=snapshot.market_regime,
    )

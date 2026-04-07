"""EdgeFinder v2 — Multi-factor scoring engine.

Replaces binary qualifies_stock() pass/fail with a 0-100 composite score
per strategy per stock. Each strategy defines a ScoringProfile with weighted
factors. The scanner computes scores for qualifying stocks and takes the
top N per strategy for the watchlist.

Scoring algorithm:
1. Collect all qualifying stocks' metric values across the universe
2. Compute universe min/max per metric (for normalization)
3. For each stock, normalize each metric to 0-1 using universe min/max
4. Apply directional weighting (high=keep, low=invert, range=proximity)
5. Multiply each sub-score by its weight, sum, scale to 0-100
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from edgefinder.core.models import TickerFundamentals

logger = logging.getLogger(__name__)


@dataclass
class ScoringFactor:
    """A single factor in a strategy's scoring profile."""

    metric: str       # field name on TickerFundamentals (e.g., "earnings_growth")
    weight: float     # 0.0-1.0, all weights in a profile should sum to ~1.0
    ideal: str        # "high" (higher is better), "low" (lower is better), "range"
    range_min: float = 0.0  # for ideal="range" only
    range_max: float = 0.0


@dataclass
class ScoringProfile:
    """Defines how a strategy scores and ranks stock candidates."""

    factors: list[ScoringFactor] = field(default_factory=list)
    top_n: int = 50  # max watchlist size for this strategy


def compute_universe_stats(
    stocks: list[TickerFundamentals],
    factors: list[ScoringFactor],
) -> dict[str, tuple[float, float]]:
    """Compute min/max for each metric across the universe.

    Returns dict mapping metric_name -> (min_value, max_value).
    Stocks with None values for a metric are excluded from that metric's stats.
    """
    stats: dict[str, tuple[float, float]] = {}
    for factor in factors:
        values = []
        for fund in stocks:
            val = getattr(fund, factor.metric, None)
            if val is not None and isinstance(val, (int, float)):
                values.append(float(val))
        if values:
            stats[factor.metric] = (min(values), max(values))
        else:
            stats[factor.metric] = (0.0, 0.0)
    return stats


def compute_score(
    fund: TickerFundamentals,
    profile: ScoringProfile,
    universe_stats: dict[str, tuple[float, float]],
) -> float:
    """Compute a 0-100 composite score for a stock against a scoring profile.

    Returns 0.0 if no factors can be evaluated (all metrics are None).
    """
    total_score = 0.0
    total_weight = 0.0

    for factor in profile.factors:
        val = getattr(fund, factor.metric, None)
        if val is None or not isinstance(val, (int, float)):
            continue

        val = float(val)
        min_val, max_val = universe_stats.get(factor.metric, (0.0, 0.0))
        spread = max_val - min_val

        # Normalize to 0-1
        if spread > 0:
            normalized = (val - min_val) / spread
        else:
            normalized = 0.5  # all values identical, neutral score

        # Apply direction
        if factor.ideal == "high":
            sub_score = normalized
        elif factor.ideal == "low":
            sub_score = 1.0 - normalized
        elif factor.ideal == "range":
            # Score 1.0 if within range, decay linearly outside
            if factor.range_min <= val <= factor.range_max:
                sub_score = 1.0
            elif val < factor.range_min:
                dist = factor.range_min - val
                range_spread = factor.range_max - factor.range_min
                sub_score = max(0.0, 1.0 - dist / (range_spread or 1.0))
            else:
                dist = val - factor.range_max
                range_spread = factor.range_max - factor.range_min
                sub_score = max(0.0, 1.0 - dist / (range_spread or 1.0))
        else:
            sub_score = normalized

        # Clamp to [0, 1]
        sub_score = max(0.0, min(1.0, sub_score))

        total_score += sub_score * factor.weight
        total_weight += factor.weight

    if total_weight == 0:
        return 0.0

    # Scale to 0-100
    return round((total_score / total_weight) * 100, 1)


def rank_and_filter(
    stocks: list[tuple[TickerFundamentals, float]],
    top_n: int,
) -> list[tuple[TickerFundamentals, float]]:
    """Sort stocks by score descending and return top N."""
    sorted_stocks = sorted(stocks, key=lambda x: x[1], reverse=True)
    return sorted_stocks[:top_n]

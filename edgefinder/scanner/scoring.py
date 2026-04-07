"""EdgeFinder v2 — Multi-factor scoring engine.

Replaces binary qualifies_stock() pass/fail with a 0-100 composite score
per strategy per stock. Each strategy defines a ScoringProfile with weighted
factors. The scanner computes scores for qualifying stocks and takes the
top N per strategy for the watchlist.

Scoring works with StockProfile (unified data) — factors can reference
any field from fundamentals, technical indicators, or relative strength.

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
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScoringFactor:
    """A single factor in a strategy's scoring profile.

    The metric can be any field available on StockProfile.get():
    - Fundamental: "earnings_growth", "debt_to_equity", "fcf_yield", etc.
    - Technical: "rsi", "adx", "volume_ratio", "bb_width", etc.
    - Supplemental: "eps_surprise_pct", "analyst_target_price", etc.
    - Relative strength: "rs_vs_spy", "rs_vs_sector"
    """

    metric: str       # field name accessible via StockProfile.get()
    weight: float     # 0.0-1.0, all weights in a profile should sum to ~1.0
    ideal: str        # "high" (higher is better), "low" (lower is better), "range"
    range_min: float = 0.0  # for ideal="range" only
    range_max: float = 0.0


@dataclass
class ScoringProfile:
    """Defines how a strategy scores and ranks stock candidates."""

    factors: list[ScoringFactor] = field(default_factory=list)
    top_n: int = 50  # max watchlist size for this strategy


def _get_value(obj: Any, field_name: str) -> float | None:
    """Get a numeric value from an object — supports StockProfile.get() or getattr."""
    if hasattr(obj, "get") and callable(obj.get):
        return obj.get(field_name)
    val = getattr(obj, field_name, None)
    if val is not None and isinstance(val, (int, float)):
        return float(val)
    return None


def compute_universe_stats(
    stocks: list,
    factors: list[ScoringFactor],
) -> dict[str, tuple[float, float]]:
    """Compute min/max for each metric across the universe.

    Returns dict mapping metric_name -> (min_value, max_value).
    Accepts list of StockProfile, TickerFundamentals, or any object with get()/getattr.
    """
    stats: dict[str, tuple[float, float]] = {}
    for factor in factors:
        values = []
        for stock in stocks:
            val = _get_value(stock, factor.metric)
            if val is not None:
                values.append(val)
        if values:
            stats[factor.metric] = (min(values), max(values))
        else:
            stats[factor.metric] = (0.0, 0.0)
    return stats


def compute_score(
    stock: Any,
    profile: ScoringProfile,
    universe_stats: dict[str, tuple[float, float]],
) -> float:
    """Compute a 0-100 composite score for a stock against a scoring profile.

    Accepts StockProfile, TickerFundamentals, or any object with get()/getattr.
    Returns 0.0 if no factors can be evaluated (all metrics are None).
    """
    total_score = 0.0
    total_weight = 0.0

    for factor in profile.factors:
        val = _get_value(stock, factor.metric)
        if val is None:
            continue

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
    stocks: list[tuple[Any, float]],
    top_n: int,
) -> list[tuple[Any, float]]:
    """Sort stocks by score descending and return top N."""
    sorted_stocks = sorted(stocks, key=lambda x: x[1], reverse=True)
    return sorted_stocks[:top_n]

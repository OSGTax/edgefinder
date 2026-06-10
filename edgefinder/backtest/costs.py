"""Realistic transaction-cost model for illiquid / microcap backtesting.

The liquid lab charges a flat 5 bps of slippage. For a name that trades five
figures a day with a multi-percent bid-ask spread, that is ~100x too
optimistic — and pretending costs are tiny is exactly how a naive microcap
backtest manufactures phantom alpha. This module estimates the real frictions
from daily OHLCV alone (we have no quote data), via three published, citable
pieces:

  1. Half bid-ask spread, estimated from daily high/low with the
     Corwin & Schultz (2012, Journal of Finance) two-day high-low estimator.
  2. Square-root market impact (Almgren et al. 2005; the "square-root law"):
     pushing size through a thin book moves the price against you, scaling with
     the square root of your participation in the day's volume.
  3. A participation cap: you cannot be an unbounded fraction of a day's
     volume, so order size is clipped to a fraction of average daily volume.

Design rule — these are FIXED COST ASSUMPTIONS, never tunable knobs. They must
never be fed to the optimizer, or a backtest would happily "discover" that
assuming costs are low makes money. They are deliberately conservative
(pessimistic) so that a surviving edge is more likely to be real. The whole
point of this module is to tell the truth about cost, even when the truth is
"there is no net edge here."
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SQRT2 = math.sqrt(2.0)
# Corwin-Schultz constant 3 - 2*sqrt(2).
_CS_DENOM = 3.0 - 2.0 * _SQRT2


def corwin_schultz_spread(h0: float, l0: float, h1: float, l1: float) -> float:
    """Proportional bid-ask spread from two consecutive days' high & low.

    Returns a fraction of price (0.02 == a 2% spread). The estimator can go
    negative for low-volatility days; by convention those clamp to 0 (Corwin &
    Schultz suggest treating negatives as zero spread). Returns 0 on degenerate
    input (non-positive prices).

    Reference: Corwin, S. A., & Schultz, P. (2012), "A Simple Way to Estimate
    Bid-Ask Spreads from Daily High and Low Prices", Journal of Finance 67(2).
    """
    if min(h0, l0, h1, l1) <= 0.0:
        return 0.0
    # Sum of the two single-day squared log high/low ranges (beta).
    beta = math.log(h0 / l0) ** 2 + math.log(h1 / l1) ** 2
    # Squared log range of the high/low taken over BOTH days (gamma).
    hi2 = max(h0, h1)
    lo2 = min(l0, l1)
    gamma = math.log(hi2 / lo2) ** 2
    alpha = (
        (math.sqrt(2.0 * beta) - math.sqrt(beta)) / _CS_DENOM
        - math.sqrt(gamma / _CS_DENOM)
    )
    spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return spread if spread > 0.0 else 0.0


@dataclass(frozen=True)
class CostModel:
    """A conservative, FIXED transaction-cost model for thin names.

    ``impact_coef``        square-root-impact coefficient (eta); ~1 is a
                           standard, mildly conservative choice.
    ``max_participation``  hard cap on order size as a fraction of average
                           daily *dollar* volume (0.05 == take at most 5% of a
                           day's volume).
    ``spread_floor``       minimum proportional spread to assume even when the
                           high-low estimate is ~0 (microcaps are never truly
                           frictionless); applied before halving.
    ``min_adv_dollars``    names whose trailing ADV is below this are treated
                           as untradeable (cap → 0 shares); guards against
                           "filling" thousands of dollars in a name that trades
                           hundreds.
    """

    impact_coef: float = 1.0
    max_participation: float = 0.05
    spread_floor: float = 0.005  # 0.5% minimum spread for thin names
    min_adv_dollars: float = 25_000.0

    # Liquidity-tiered spread floors (FIXED constants, never optimized): the
    # 0.5% microcap floor is ~50-300x a mega-cap's true spread — applied to a
    # liquid top-N universe it would be a systematic FAIL bias, not honesty.
    # Tiers keyed by trailing dollar ADV, descending; names below the last
    # cutoff (and legacy callers that pass no ADV) keep the microcap floor.
    SPREAD_FLOOR_TIERS: tuple = ((1e9, 0.0002), (50e6, 0.0005), (5e6, 0.002))

    # The sqrt impact law is only valid for participation << 1; unclamped, a
    # forced exit into collapsed liquidity can exceed 1.0 — a NEGATIVE sell
    # fill (cash falling on a sale). Clamp the total one-way fraction.
    MAX_COST_FRACTION: float = 0.5

    # ── individual cost components (all fractions of price) ──

    def spread_floor_for(self, adv_dollars: float | None) -> float:
        """The liquidity-appropriate minimum spread for a name."""
        if adv_dollars:
            for cutoff, floor in self.SPREAD_FLOOR_TIERS:
                if adv_dollars >= cutoff:
                    return floor
        return self.spread_floor

    def half_spread(self, spread_frac: float, adv_dollars: float | None = None) -> float:
        """Half the (floored) proportional spread — paid on each side."""
        return max(spread_frac, self.spread_floor_for(adv_dollars)) / 2.0

    def impact(self, order_dollars: float, adv_dollars: float, volatility: float) -> float:
        """Square-root market impact as a fraction of price.

        ``impact = eta * sigma * sqrt(order / ADV)`` where sigma is the name's
        daily return volatility and order/ADV is the participation rate.
        Returns 0 when ADV is unknown/zero (sizing should already have refused
        the trade via :meth:`cap_shares`).
        """
        if adv_dollars <= 0.0 or order_dollars <= 0.0:
            return 0.0
        participation = order_dollars / adv_dollars
        return self.impact_coef * max(volatility, 0.0) * math.sqrt(participation)

    def cost_fraction(
        self,
        order_dollars: float,
        adv_dollars: float,
        volatility: float,
        spread_frac: float,
    ) -> float:
        """Total one-way cost as a fraction of price = half-spread + impact,
        clamped at MAX_COST_FRACTION (the sqrt law breaks down far above full
        participation, and a fraction > 1 would mean negative sell proceeds)."""
        frac = self.half_spread(spread_frac, adv_dollars) + self.impact(
            order_dollars, adv_dollars, volatility
        )
        return min(frac, self.MAX_COST_FRACTION)

    def fill_price(
        self,
        mid: float,
        side: str,
        *,
        order_dollars: float,
        adv_dollars: float,
        volatility: float,
        spread_frac: float,
    ) -> float:
        """Cost-adjusted fill price. BUY pays up, SELL receives less."""
        frac = self.cost_fraction(order_dollars, adv_dollars, volatility, spread_frac)
        if side.upper() == "BUY":
            return mid * (1.0 + frac)
        return mid * (1.0 - frac)

    # ── liquidity-capped sizing ──

    def cap_shares(self, desired_shares: int, price: float, adv_dollars: float) -> int:
        """Clip an intended share count to the participation limit.

        Names below ``min_adv_dollars`` of trailing volume are untradeable → 0.
        Otherwise the cap is ``max_participation * ADV / price`` shares.
        """
        if desired_shares <= 0 or price <= 0:
            return 0
        if adv_dollars < self.min_adv_dollars:
            return 0
        max_dollars = self.max_participation * adv_dollars
        max_shares = int(max_dollars / price)
        return max(min(desired_shares, max_shares), 0)

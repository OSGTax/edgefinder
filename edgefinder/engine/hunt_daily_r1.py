"""DAILY-DECISION HUNT ROUND 1 — pre-registered roster (2026-06-14).

PRE-REGISTRATION: this file is committed BEFORE any candidate's first
validation run; the parameters below are FROZEN. A candidate that "needs"
different parameters is a NEW candidate, registered fresh in a later round.
Verdicts come from edgefinder.engine.validate (fixed-param walk-forward,
PIT universes, realistic costs, total-return, sealed holdout 2026-04-01).

WHAT THIS LANE IS. The owner wants the system to analyze EVERY trading day
whether to buy / sell / hold, while letting each strategy hold for whatever
its signal dictates — quick flips, swings, or long-term positions. The
engine already separates DECISION cadence from TRADE frequency:
- ``--schedule daily`` asks the strategy for target weights every trading
  day (the daily analysis the owner wants);
- ``--rebalance-band`` makes "hold" FREE — if today's target ≈ what is
  already held, nothing trades, no toll. A position meant to be held for
  months is re-evaluated daily and simply kept, trading zero times until
  the signal flips.
So holding horizon is a per-strategy property (set by its signal), not the
schedule. This roster deliberately SPANS horizons on one daily-decision loop.

LOOK-AHEAD HONESTY. The context is point-in-time: ``ctx`` only carries data
through the decision date's close, and the engine fills at the NEXT open. So
any "gap" signal here is the LAGGED gap (yesterday's open vs the prior
close), already known at decision time — never today's open. Decisions use
only ``a.history`` / ``a.indicators`` / ``a.ret(n)``, all ≤ decision date.

Lane (run config lives in hunt/queue.json; classes are lane-agnostic):
``--universe top:500 --start 2021-06-01 --schedule daily
--rebalance-band 0.01 --costed --div-adjust --bars-from r2
--holdout-start 2026-04-01 --record --total-return``. Labels daily-r1:*.
Null control (buy_and_hold:SPY) runs in-batch; two fresh-seed RandomK
baskets continue the false-positive yardstick (the band is the churn/toll
control).

All baskets are LONG-ONLY, cross-sectional, equal-weight K=20 (matching the
daily hunt's cross-sectional convention). Helpers and the RandomK control
are reused verbatim from hunt_r1.
"""

from __future__ import annotations

from edgefinder.engine.hunt_r1 import (
    RandomK,
    _dollar_vol,
    _high_ratio,
    _topk_ew,
    _vol,
)
from edgefinder.engine.strategy import RebalanceContext

K = 20


def _uptrend(a) -> bool:
    return bool(a.indicators.ema_200 and a.price > a.indicators.ema_200)


def _lagged_gap(a) -> float | None:
    """Yesterday's gap = open[-1] / close[-2] - 1 (LAGGED, known at decision
    time — never today's open). None if fewer than 2 bars."""
    h = a.history
    if len(h) < 2:
        return None
    prev_close = float(h["close"].iloc[-2])
    last_open = float(h["open"].iloc[-1])
    if prev_close <= 0:
        return None
    return last_open / prev_close - 1.0


# ════ QUICK-FLIP / SHORT HORIZON ════════════════════════════════════════


class DrReversal1:
    """1-day short-term reversal: buy the K most-negative trailing 1-day
    returns. Turns over in days — the fastest flip in the lane."""

    k = K

    @property
    def name(self) -> str:
        return "dr_reversal_1"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            r = a.ret(1)
            if r is not None:
                scored.append((r, s))
        return _topk_ew(scored, self.k, reverse=False)


class DrReversal5:
    """5-day short-term reversal: buy the K most-negative trailing 5-day
    returns (the weekly-reversal effect, decided daily)."""

    k = K

    @property
    def name(self) -> str:
        return "dr_reversal_5"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            r = a.ret(5)
            if r is not None:
                scored.append((r, s))
        return _topk_ew(scored, self.k, reverse=False)


class DrGapFade:
    """Lagged-gap FADE: buy the K most-negative yesterday-gaps (gap-down
    names), betting the overnight dislocation reverts."""

    k = K

    @property
    def name(self) -> str:
        return "dr_gap_fade"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            g = _lagged_gap(a)
            if g is not None:
                scored.append((g, s))
        return _topk_ew(scored, self.k, reverse=False)


class DrGapCont:
    """Lagged-gap CONTINUATION: among uptrending names, buy the K most-
    positive yesterday-gaps (gap-up names), betting strength persists."""

    k = K

    @property
    def name(self) -> str:
        return "dr_gap_cont"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            g = _lagged_gap(a)
            if g is not None:
                scored.append((g, s))
        return _topk_ew(scored, self.k, reverse=True)


# ════ SWING HORIZON ═════════════════════════════════════════════════════


class DrMom20:
    """1-month momentum swing: among uptrending names, buy the top-K by
    trailing 20-day return."""

    k = K

    @property
    def name(self) -> str:
        return "dr_mom_20"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            r = a.ret(20)
            if r is not None:
                scored.append((r, s))
        return _topk_ew(scored, self.k, reverse=True)


class DrHighBreak:
    """52-week-high breakout: buy the top-K nearest their trailing-252-day
    high (proximity ratio closest to 1.0)."""

    k = K

    @property
    def name(self) -> str:
        return "dr_high_break"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            hr = _high_ratio(a, 252)
            if hr is not None:
                scored.append((hr, s))
        return _topk_ew(scored, self.k, reverse=True)


class DrVolBreakout:
    """Range-expansion breakout: among uptrending names, buy the top-K by
    today's range (high-low)/close relative to the trailing-20-day mean
    range — the 'volatility expansion' setup, long-only."""

    k = K

    @property
    def name(self) -> str:
        return "dr_vol_breakout"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            h = a.history
            if len(h) < 21:
                continue
            close = float(h["close"].iloc[-1])
            if close <= 0:
                continue
            rng = (float(h["high"].iloc[-1]) - float(h["low"].iloc[-1])) / close
            rngs = (h["high"].iloc[-21:-1] - h["low"].iloc[-21:-1]) \
                / h["close"].iloc[-21:-1]
            mean_rng = float(rngs.mean())
            if mean_rng <= 0:
                continue
            scored.append((rng / mean_rng, s))
        return _topk_ew(scored, self.k, reverse=True)


# ════ POSITION HORIZON ══════════════════════════════════════════════════


class DrTrendHold:
    """Slow trend/quality hold: equal-weight EVERY name above its 200-EMA
    with positive trailing 126-day return. Decided daily but held for
    months via the no-trade band — proof the loop covers long-term too."""

    @property
    def name(self) -> str:
        return "dr_trend_hold"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        picks = []
        for s, a in ctx.assets.items():
            r = a.ret(126)
            if _uptrend(a) and r is not None and r > 0:
                picks.append(s)
        picks = sorted(picks)
        return {s: 1.0 / len(picks) for s in picks} if picks else {}


# ── spec registry (consumed by engine.strategies.make_strategy_factory) ──

HUNT_DAILY_R1_SPECS = {
    # quick-flip / short horizon
    "dr_reversal_1": DrReversal1,
    "dr_reversal_5": DrReversal5,
    "dr_gap_fade": DrGapFade,
    "dr_gap_cont": DrGapCont,
    # swing horizon
    "dr_mom_20": DrMom20,
    "dr_high_break": DrHighBreak,
    "dr_vol_breakout": DrVolBreakout,
    # position horizon
    "dr_trend_hold": DrTrendHold,
    # controls (false-positive yardstick — fresh seeds; band is the toll control)
    "dr_random_61": lambda: RandomK(61),
    "dr_random_67": lambda: RandomK(67),
}

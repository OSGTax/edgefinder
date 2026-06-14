"""DAILY-DECISION HUNT ROUND 2 — pre-registered STATEFUL roster (2026-06-14).

PRE-REGISTRATION: this file is committed BEFORE any candidate's first
validation run; the parameters below are FROZEN. A candidate that "needs"
different parameters is a NEW candidate, registered fresh in a later round.
Verdicts come from edgefinder.engine.validate (fixed-param walk-forward,
PIT universes, realistic costs, total-return, sealed holdout 2026-04-01).

WHY STATEFUL. Daily-round-1 (reviews/DAILY-ROUND-1.md) returned 0/9 with a
decisive STRUCTURAL cause: every candidate was a STATELESS cross-sectional
top-K basket RESELECTED daily, so its membership rotates by construction
(today's "top-20 momentum" is a largely different 20 names than yesterday's).
The no-trade band only spares names you KEEP — but a daily reselection keeps
almost nothing, so it rotates and pays the toll 20-in / 20-out every day:
~25k trades and -60 to -74pp to tolls, monotonic in turnover. The only
low-turnover thing all session was the FIXED-book SPY null (6 trades).

THE FIX (entry/exit hysteresis). Give each strategy STATE — it reads
``ctx.held_symbols()`` / ``ctx.held(s)`` (the current book as weights, fed by
the engine look-ahead-free) and decides KEEP / EXIT / ENTER per position, so
turnover is SIGNAL-driven (enter on a signal, hold until an exit signal),
not reselection-driven. The canonical turnover-killer is a wide EXIT band
around a narrow ENTRY band (the "buffer zone"): rank the universe by the
signal; KEEP currently-held names that are still inside the WIDE exit band;
ADD non-held names in the NARROW entry band, filling up to the cap K;
equal-weight the resulting set. A name that drifts into the buffer (e.g. rank
11-29 for a top-10 enter / top-30 exit rule) is HELD, not churned.

LOOK-AHEAD HONESTY. The context is point-in-time: ``ctx`` carries only data
through the decision date's close (and the engine fills at the NEXT open).
``ctx.holdings`` is the PRE-fill book valued at decision-date closes — never
today's fill. Decisions use only ``a.history`` / ``a.indicators`` / ``a.ret(n)``,
all <= the decision date.

Lane (run config lives in hunt/queue.json; classes are lane-agnostic):
``--universe top:500 --start 2021-06-01 --schedule daily
--rebalance-band 0.01 --costed --div-adjust --bars-from r2
--holdout-start 2026-04-01 --record --total-return --warmup-days 290``.
Labels daily-r2:*. Null control (buy_and_hold:SPY) runs in-batch; the two
seeded hold-with-hysteresis RandomHold baskets are the TURNOVER-MATCHED
false-positive yardstick. The FIRST thing to confirm is that turnover
COLLAPSES vs daily-r1 (the hysteresis working).

All baskets are LONG-ONLY, FIXED params, daily schedule, look-ahead-honest.
The 52w-high helper (_high_ratio) and AssetView.ret are reused verbatim from
hunt_r1 / the strategy interface; ranking goes through the local _ranked_by.
"""

from __future__ import annotations

import hashlib

from edgefinder.engine.hunt_r1 import _high_ratio
from edgefinder.engine.strategy import RebalanceContext


def _hysteresis(
    ctx: RebalanceContext,
    ranked: list[str],
    *,
    cap: int,
    enter_rank: int,
    exit_rank: int,
) -> dict[str, float]:
    """The shared turnover-killer.

    ``ranked`` is the universe sorted BEST-first by the strategy's signal.
    KEEP every currently-held name whose rank is still inside the WIDE exit
    band (``rank < exit_rank``); then ADD non-held names from the NARROW entry
    band (``rank < enter_rank``) in rank order until the book reaches ``cap``.
    Equal-weight the result. A held name in the buffer (``enter_rank <= rank <
    exit_rank``) is retained — that is the no-churn property.
    """
    rank = {s: i for i, s in enumerate(ranked)}
    held = ctx.held_symbols()
    keep = [s for s in held if s in rank and rank[s] < exit_rank]
    kept = set(keep)
    book = list(keep)
    for s in ranked:                         # rank order = best-first
        if len(book) >= cap:
            break
        if s in kept:
            continue
        if rank[s] < enter_rank:
            book.append(s)
            kept.add(s)
    if len(book) > cap:                       # held names can exceed the cap
        book = book[:cap]                     #   if many survive — trim worst
    if not book:
        return {}
    w = 1.0 / len(book)
    return {s: w for s in book}


def _ranked_by(ctx: RebalanceContext, key, *, reverse: bool,
               eligible=None) -> list[str]:
    """Universe symbols sorted BEST-first by ``key(asset)`` (None scores drop).

    ``reverse=True`` => higher score is better; ties broken by symbol (stable).
    ``eligible(asset)`` optionally pre-filters the universe.
    """
    scored: list[tuple[float, str]] = []
    for s, a in ctx.assets.items():
        if eligible is not None and not eligible(a):
            continue
        v = key(a)
        if v is not None:
            scored.append((v, s))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=reverse)
    return [s for _, s in scored]


def _uptrend(a) -> bool:
    return bool(a.indicators.ema_200 and a.price > a.indicators.ema_200)


# ════ STATEFUL HYSTERESIS FAMILIES ══════════════════════════════════════


class ShMom20:
    """20-day momentum with hysteresis: ENTER the top-10 by trailing 20-day
    return; HOLD a name until it falls out of the top-30 (a 10/30 buffer)."""

    cap, enter_rank, exit_rank = 10, 10, 30

    @property
    def name(self) -> str:
        return "sh_mom_20"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        ranked = _ranked_by(ctx, lambda a: a.ret(20), reverse=True)
        return _hysteresis(ctx, ranked, cap=self.cap,
                           enter_rank=self.enter_rank, exit_rank=self.exit_rank)


class ShTrendHold:
    """Trend hold with a CLEAN STATE EXIT (the daily-r1 churn killer): ENTER
    names with price>200-EMA AND ret(126)>0; EXIT a held name only when its
    price falls below its 200-EMA — no reselection churn. New entries are
    capped to keep the book at ~20 names, preferring the highest ret(126)."""

    cap = 20

    @property
    def name(self) -> str:
        return "sh_trend_hold"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        held = set(ctx.held_symbols())
        # keep held names that are STILL above their 200-EMA (clean exit only
        # when price<ema_200) — no rank/reselection involved
        keep = [s for s in held
                if (a := ctx.get(s)) is not None and _uptrend(a)]
        book = list(keep)
        kept = set(keep)
        # entry candidates: not held, uptrending, positive 126d return,
        # ranked by ret(126) desc
        cands = _ranked_by(
            ctx, lambda a: a.ret(126), reverse=True,
            eligible=lambda a: _uptrend(a)
            and (r := a.ret(126)) is not None and r > 0)
        for s in cands:
            if len(book) >= self.cap:
                break
            if s in kept:
                continue
            book.append(s)
            kept.add(s)
        if len(book) > self.cap:
            book = book[:self.cap]
        if not book:
            return {}
        w = 1.0 / len(book)
        return {s: w for s in book}


class ShHighBreak:
    """52-week-high breakout with hysteresis: ENTER when at/near the trailing
    252-day high (_high_ratio >= 0.99); EXIT a held name when its price falls
    below its 50-EMA. Caps the book at 15 names (highest ratio first)."""

    cap = 15
    enter_floor = 0.99

    @property
    def name(self) -> str:
        return "sh_high_break"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        held = set(ctx.held_symbols())
        # keep held names still above their 50-EMA (clean exit when price<ema_50)
        keep = []
        for s in held:
            a = ctx.get(s)
            if a is not None and a.indicators.ema_50 and a.price > a.indicators.ema_50:
                keep.append(s)
        book = list(keep)
        kept = set(keep)
        # entry candidates: not held, at/near the 252d high; best ratio first
        cands = _ranked_by(
            ctx, lambda a: _high_ratio(a, 252), reverse=True,
            eligible=lambda a: (hr := _high_ratio(a, 252)) is not None
            and hr >= self.enter_floor)
        for s in cands:
            if len(book) >= self.cap:
                break
            if s in kept:
                continue
            book.append(s)
            kept.add(s)
        if len(book) > self.cap:
            book = book[:self.cap]
        if not book:
            return {}
        w = 1.0 / len(book)
        return {s: w for s in book}


class ShReversal5:
    """5-day reversal with hysteresis: ENTER the top-10 by MOST-NEGATIVE
    trailing 5-day return; HOLD until the name's 5-day-return rank recovers
    above 30 (the reversion has played out). A 10/30 buffer on the reversal
    rank (most-negative ranks BEST)."""

    cap, enter_rank, exit_rank = 10, 10, 30

    @property
    def name(self) -> str:
        return "sh_reversal_5"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        # most-negative ret(5) ranks first => reverse=False
        ranked = _ranked_by(ctx, lambda a: a.ret(5), reverse=False)
        return _hysteresis(ctx, ranked, cap=self.cap,
                           enter_rank=self.enter_rank, exit_rank=self.exit_rank)


# ════ CONTROLS — turnover-MATCHED hold-with-hysteresis ═══════════════════


class RandomHold:
    """Turnover-matched false-positive yardstick: rank the universe by a
    deterministic hash of (seed, symbol) — a FIXED pseudo-random order, no
    monthly redraw — then run the SAME hysteresis machinery as the candidates
    (enter top-10, hold until out of top-30). Matched churn means any "edge"
    a candidate shows over this control is signal, not the hysteresis floor."""

    cap, enter_rank, exit_rank = 10, 10, 30

    def __init__(self, seed: int) -> None:
        self.seed = seed

    @property
    def name(self) -> str:
        return f"sh_random_{self.seed}"

    def _hash(self, symbol: str) -> int:
        return int(hashlib.md5(f"{self.seed}:{symbol}".encode())
                   .hexdigest()[:8], 16)

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        ranked = sorted(ctx.assets, key=lambda s: (self._hash(s), s))
        return _hysteresis(ctx, ranked, cap=self.cap,
                           enter_rank=self.enter_rank, exit_rank=self.exit_rank)


# ── spec registry (consumed by engine.strategies.make_strategy_factory) ──

HUNT_DAILY_R2_SPECS = {
    # stateful hysteresis families
    "sh_mom_20": ShMom20,
    "sh_trend_hold": ShTrendHold,
    "sh_high_break": ShHighBreak,
    "sh_reversal_5": ShReversal5,
    # controls (turnover-matched hold-with-hysteresis; band/null also in-batch)
    "sh_random_71": lambda: RandomHold(71),
    "sh_random_73": lambda: RandomHold(73),
}

"""INTRADAY HUNT ROUND 1 — pre-registered strategy roster (2026-06-14).

PRE-REGISTRATION: this file is committed BEFORE any candidate's first
intraday validation run; the parameters below are FROZEN. A candidate that
"needs" different parameters is a NEW candidate, registered fresh in a later
round. Verdicts come from ``edgefinder.engine.intraday_validate`` (the
minute-bar sibling of the daily walk-forward) — fixed-param walk-forward vs
SPY, realistic intraday costs, the same sealed-holdout discipline.

THE LANE (one config for the whole round, run via intraday/queue.json):
- Universe: the liquid mega-cap + index subset frozen in
  ``intraday/menu.json`` (top-50 by trailing-126d dollar volume + the
  protected ETF menu, as of 2026-06-11). Strategies here are
  lane-agnostic — they see whatever universe the run hands them.
- ``--costed`` (the FIXED intraday cost model: spread + impact + caps).
- ``--flatten-at-close`` (default): no overnight risk; every position is
  flattened MOC and re-entered next session — these are pure intraday ideas.
- Decision every 5 bars (``--decision-interval 5``): one decision per 5
  minutes, not every minute (keeps churn — and tolls — honest while still
  reacting within the session).
- Benchmark: SPY, scored over the SAME sessions. Walk-forward folds planned
  on DAYS; sealed holdout pinned at ``--holdout-start 2026-04-01`` (never
  evaluated without ``--burn-holdout`` and owner sign-off).
- Total-return bar is PRIMARY (``--total-return``); risk-adjusted recorded
  alongside. Any passer must clear all THREE adversarial re-checks (shifted
  in-sample windows / late start) before a "finalist" claim, exactly like
  the daily hunt.

LONG-ONLY (CRITICAL): the intraday engine has NO shorting — weights are in
[0, 1], the rest is cash. So the "reversal" and "gap-fade" families express
their thesis by going LONG the names they expect to bounce (the intraday
losers / the gapped-down names), NOT by shorting the winners. The momentum
family is the deliberate OPPOSITE of reversal — both are pre-registered so
the round measures which (if either) survives; at most one should.

FAMILIES (FIXED params are class attributes — NEVER tunable; K=5 everywhere
unless a class notes otherwise):
- REVERSAL: ir_reversal, ir_vwap_rev — buy the intraday losers.
- MOMENTUM: ir_momentum, ir_high_break — buy the intraday winners
  (the deliberate opposite of REVERSAL).
- BREAKOUT: ir_orb — buy names breaking above the opening range.
- GAP: ir_gap_fade (fade open gap-downs, early-session only),
  ir_gap_go (ride open gap-ups, first-hour only).
- CLOSE-EFFECT: ir_late_mom — last-hour winners.
- CONTROLS: ir_random_101, ir_random_103 — pre-registered coin flips that
  buy tolls without signal, calibrating the round's false-positive floor.
  (The ``flat`` null and ``buy_hold_open:SPY`` anchor already live in
  ``make_intraday_factory`` and the queue includes them — not re-added here.)
"""

from __future__ import annotations

import hashlib

from edgefinder.engine.intraday_strategy import IntradayContext

# FIXED basket size for the whole round (no per-strategy override unless a
# class explicitly documents one).
K = 5


def _topk_ew(scored: list[tuple[float, str]], k: int,
             reverse: bool = True) -> dict[str, float]:
    """Equal-weight the top ``k`` by score (ties broken by symbol — stable).

    ``reverse=True`` (default) picks the LARGEST scores; ``reverse=False``
    picks the SMALLEST. Fewer than ``k`` candidates ⇒ partially invested
    (1/k each over those present); empty ⇒ {} (all-cash)."""
    scored.sort(key=lambda t: (t[0], t[1]), reverse=reverse)
    pick = scored[:k]
    return {s: 1.0 / k for _, s in pick} if pick else {}


# ════ REVERSAL family — buy the intraday losers (expect a bounce) ════════


class IntradayReversal:
    """Hold the K names with the most-NEGATIVE trailing-30-bar return — the
    intraday losers, betting on mean reversion. Skips names whose ret(30)
    is None (not enough session history yet)."""

    lookback = 30

    @property
    def name(self) -> str:
        return "ir_reversal"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            r = a.ret(self.lookback)
            if r is not None:
                scored.append((r, s))
        # most-negative return = smallest -> reverse=False
        return _topk_ew(scored, K, reverse=False)


class VwapReversion:
    """Among names trading at/below ``vwap*(1-band)``, hold the K furthest
    BELOW VWAP by (price-vwap)/vwap (most negative). Long-only reversion to
    VWAP. Requires session_vwap and at least 10 bars since open (let VWAP
    settle)."""

    band = 0.003

    @property
    def name(self) -> str:
        return "ir_vwap_rev"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar or ctx.bars_since_open < 10:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            vwap = a.session_vwap
            if vwap is None or vwap <= 0:
                continue
            if a.price <= vwap * (1.0 - self.band):
                scored.append(((a.price - vwap) / vwap, s))
        # most below VWAP = smallest (most negative) -> reverse=False
        return _topk_ew(scored, K, reverse=False)


# ════ MOMENTUM family — the deliberate OPPOSITE of REVERSAL ══════════════


class IntradayMomentum:
    """Hold the K names with the most-POSITIVE trailing-30-bar return — the
    intraday winners, betting on continuation. The mirror image of
    IntradayReversal (at most one of the two should survive)."""

    lookback = 30

    @property
    def name(self) -> str:
        return "ir_momentum"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            r = a.ret(self.lookback)
            if r is not None:
                scored.append((r, s))
        return _topk_ew(scored, K, reverse=True)


class HighBreak:
    """Hold the K names making NEW session highs (price >= session_high —
    continuation), ranked by trailing-30-bar return among those. Requires at
    least 10 bars since open (a meaningful session high to break)."""

    lookback = 30

    @property
    def name(self) -> str:
        return "ir_high_break"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar or ctx.bars_since_open < 10:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            if a.price >= a.session_high:
                r = a.ret(self.lookback)
                if r is not None:
                    scored.append((r, s))
        return _topk_ew(scored, K, reverse=True)


# ════ BREAKOUT family — opening-range breakout ════════════════════════════


class OpeningRangeBreakout:
    """After the opening range forms (bars_since_open >= m), hold names whose
    price > the opening-range HIGH, ranked by how far above (price/OR_high-1)
    descending, top-K. Returns {} before the range forms."""

    m = 30

    @property
    def name(self) -> str:
        return "ir_orb"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar or ctx.bars_since_open < self.m:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            rng = a.opening_range(self.m)
            if rng is None:
                continue
            or_high, _ = rng
            if or_high > 0 and a.price > or_high:
                scored.append((a.price / or_high - 1.0, s))
        return _topk_ew(scored, K, reverse=True)


# ════ GAP family ══════════════════════════════════════════════════════════


class GapFade:
    """Fade open gap-DOWNS: among names whose session_open gapped down
    >= thresh vs prev_close (session_open <= prev_close*(1-thresh)), hold the
    K with the LARGEST gap-down (most negative), expecting reversion UP.
    Requires prev_close.

    This is an EARLY-SESSION strategy: it only acts in the first half-hour
    (bars_since_open <= 30). It re-evaluates the qualifying set each decision
    bar — the simplest correct behavior; the gap-down condition naturally
    sustains early-session (session_open is fixed; the gate is on the FIXED
    open vs the FIXED prev_close, so a qualifying name keeps qualifying while
    the half-hour window is open) and the strategy then goes all-cash once
    bars_since_open passes 30."""

    thresh = 0.005

    @property
    def name(self) -> str:
        return "ir_gap_fade"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar or ctx.bars_since_open > 30:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            pc = a.prev_close
            if pc is None or pc <= 0:
                continue
            if a.session_open <= pc * (1.0 - self.thresh):
                gap = a.session_open / pc - 1.0   # negative
                scored.append((gap, s))
        # largest gap-down = most negative -> reverse=False
        return _topk_ew(scored, K, reverse=False)


class GapGo:
    """Ride open gap-UPS: among names that gapped up >= thresh vs prev_close
    AND are still holding the gain (price > session_open), hold the K with the
    LARGEST gap, top-K. First-hour only (bars_since_open <= 60). Requires
    prev_close."""

    thresh = 0.005

    @property
    def name(self) -> str:
        return "ir_gap_go"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar or ctx.bars_since_open > 60:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            pc = a.prev_close
            if pc is None or pc <= 0:
                continue
            if a.session_open >= pc * (1.0 + self.thresh) and a.price > a.session_open:
                gap = a.session_open / pc - 1.0   # positive
                scored.append((gap, s))
        return _topk_ew(scored, K, reverse=True)


# ════ CLOSE-EFFECT family ═════════════════════════════════════════════════


class LateDayMomentum:
    """Only in the last hour (bars_until_close <= 60), hold the K names with
    the highest day-return so far (price/session_open - 1); else all-cash."""

    @property
    def name(self) -> str:
        return "ir_late_mom"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar or ctx.bars_until_close > 60:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            so = a.session_open
            if so > 0:
                scored.append((a.price / so - 1.0, s))
        return _topk_ew(scored, K, reverse=True)


# ════ CONTROLS — pre-registered coin flips (the honesty instruments) ══════


class RandomBasket:
    """Each decision, hold K names chosen by a DETERMINISTIC hash of
    (seed, session_date, symbol) — a pre-registered coin flip. Same
    (seed, date) ⇒ same basket all session; different seed ⇒ different draw.
    Buys tolls without any signal, calibrating the round's false-positive
    floor. Flattens at close like everything else."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return f"ir_random_{self.seed}"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar:
            return {}
        d = ctx.session_date.isoformat()
        scored: list[tuple[int, str]] = []
        for s in ctx.assets:
            h = hashlib.md5(f"{self.seed}:{d}:{s}".encode()).hexdigest()
            scored.append((int(h[:8], 16), s))
        return _topk_ew(scored, K, reverse=False)


# ── spec registry (consumed by intraday_validate.make_intraday_factory) ──

INTRADAY_R1_SPECS: dict[str, callable] = {
    # REVERSAL
    "ir_reversal": IntradayReversal,
    "ir_vwap_rev": VwapReversion,
    # MOMENTUM
    "ir_momentum": IntradayMomentum,
    "ir_high_break": HighBreak,
    # BREAKOUT
    "ir_orb": OpeningRangeBreakout,
    # GAP
    "ir_gap_fade": GapFade,
    "ir_gap_go": GapGo,
    # CLOSE-EFFECT
    "ir_late_mom": LateDayMomentum,
    # CONTROLS
    "ir_random_101": lambda: RandomBasket(101),
    "ir_random_103": lambda: RandomBasket(103),
}

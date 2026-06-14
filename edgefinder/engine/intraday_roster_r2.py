"""INTRADAY HUNT ROUND 2 — pre-registered roster (2026-06-14): the
LOW-TURNOVER pivot.

PRE-REGISTRATION: this file is committed BEFORE any Round-2 candidate's first
intraday validation run; the parameters below are FROZEN. A candidate that
"needs" different parameters is a NEW candidate, registered fresh in a later
round. Verdicts come from ``edgefinder.engine.intraday_validate`` — fixed-param
walk-forward vs SPY, realistic intraday costs, the same sealed-holdout
discipline (holdout pinned at 2026-04-01, never evaluated without
``--burn-holdout`` + owner sign-off).

═══ THE ROUND-1 LESSON (why this round exists) ═══════════════════════════════
Round 1 returned 0/10 (reviews/INTRADAY-ROUND-1.md). The single finding:
**TURNOVER IS THE WHOLE STORY.** Losses were nearly monotonic in trade count —
the candidates that churned most (momentum 169k trades, reversal 164k) lost
most (~-60pp); the ones that barely traded sat at the do-nothing floor. The
5-minute full-basket RESELECTION cadence paid the bid-ask + impact toll
hundreds of times per day, and no fast signal on liquid names out-earned that
toll. A thing and its mirror (momentum/reversal) BOTH lost ~-60pp — proof the
signal isn't the driver, the turnover is.

═══ THE ROUND-2 HYPOTHESIS (test turnover directly) ═════════════════════════
A viable intraday strategy must trade RARELY. Round 2 therefore tests
ENTER-ONCE, HOLD-TO-CLOSE families: pick a basket from a signal that is FIXED
for the session, then HOLD it (no mid-session reselection). The gap/ORB/
morning-trend THESES were never really tested in Round 1 because 5-min
reselection churned them; an enter-once version pays ~2 tolls/day (entry +
MOC flatten), not hundreds.

THE ENTER-ONCE CONTRACT (CRITICAL — the whole point of the round):
every strategy here returns the SAME stable basket on every decision bar from
its entry trigger onward. The signal inputs must be STABLE intra-session so the
basket does not drift. The engine's new ``--rebalance-band`` (0.01 for this
round) is what turns "same target every bar" into "actually hold": with the
band, a held basket re-trues only on MEANINGFUL drift, so integer-share
re-trues (as the equity mark wanders) don't manufacture churn. WITHOUT the band,
even an identical target re-trues every bar as equity drifts — that would
re-create Round 1's churn and defeat the experiment. So Round 2 = these
enter-once specs + ``--rebalance-band 0.01``.

═══ THE TWO LANES (a RUN flag, NOT a strategy difference) ═══════════════════
Each strategy is registered ONCE and run TWICE:
- FLATTEN-AT-CLOSE lane (``--flatten-at-close``): pure intraday, MOC flatten,
  ~2 tolls/day. Forfeits overnight drift (Round 1's anchor showed that costs
  ~2.45pp/fold) — a stiff but honest bar.
- OVERNIGHT-HOLD lane (``--hold-overnight``): a THESIS change — carry winners
  across the close to reclaim overnight drift; ~1 toll/day on hold days. No
  longer "pure intraday" but a legitimate low-turnover variant. The flat-vs-
  overnight distinction lives entirely in the queue's run flag; the strategies
  are lane-agnostic (they just keep returning the same basket — the engine's
  flatten flag decides whether the close flattens it).

LONG-ONLY (the intraday engine has NO shorting): weights in [0, 1], the rest
cash. Equal-weight K=5 everywhere.

═══ FAMILIES (FIXED params are class attributes — NEVER tunable) ════════════
- iro_gap_fade  (thresh=0.005): hold the K names that gapped DOWN >= 0.5% at
  the open, largest gap-down first. Fixed all session.
- iro_gap_go    (thresh=0.005): hold the K names that gapped UP >= 0.5%,
  largest gap first. Fixed all session.
- iro_orb       (m=30): hold the K names that have broken above their FROZEN
  opening-range high at some point this session (monotone — see class doc).
- iro_morning_trend (m=30): hold the K names with the highest FIXED morning
  return (m-th bar close vs session open). Fixed once the m-th bar exists.
- CONTROLS iro_random_201, iro_random_203: hold K names by hash(seed, date,
  symbol) — fixed per session. The low-turnover floor RECALIBRATED for this
  cadence (Round 1's randoms churned every 5 min; these hold all day).
"""

from __future__ import annotations

import hashlib

from edgefinder.engine.intraday_strategy import IntradayContext

# FIXED basket size for the whole round.
K = 5


def _topk_ew(scored: list[tuple[float, str]], k: int,
             reverse: bool = True) -> dict[str, float]:
    """Equal-weight the top ``k`` by score (ties broken by symbol — stable).

    ``reverse=True`` (default) picks the LARGEST scores; ``reverse=False``
    picks the SMALLEST. Fewer than ``k`` candidates => partially invested
    (1/k each over those present); empty => {} (all-cash). Identical to the
    Round-1 helper so baskets are directly comparable across rounds."""
    scored.sort(key=lambda t: (t[0], t[1]), reverse=reverse)
    pick = scored[:k]
    return {s: 1.0 / k for _, s in pick} if pick else {}


# ════ GAP family — fixed at the open, held all session ════════════════════


class IroGapFade:
    """ENTER-ONCE: hold the K names that gapped DOWN >= ``thresh`` at the open
    (``session_open <= prev_close*(1-thresh)``), ranked by LARGEST gap-down
    (most negative), expecting reversion UP.

    STABLE BY CONSTRUCTION: both ``session_open`` and ``prev_close`` are
    SESSION CONSTANTS (the open never changes; prev_close is the prior session's
    last close). So the qualifying set AND the ranking score
    (``session_open/prev_close - 1``) are identical on every decision bar from
    the open onward => the same basket every bar => held to close (or carried
    overnight in that lane). NO time gate — the fixed signal + the rebalance
    band ARE the hold (unlike Round-1 GapFade, which had an early-session gate
    and re-evaluated each bar). Skips names with no prev_close (first loaded
    session)."""

    thresh = 0.005

    @property
    def name(self) -> str:
        return "iro_gap_fade"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            pc = a.prev_close
            if pc is None or pc <= 0:
                continue
            so = a.session_open
            if so > 0 and so <= pc * (1.0 - self.thresh):
                scored.append((so / pc - 1.0, s))   # negative; biggest gap-down first
        # largest gap-down = most negative -> reverse=False
        return _topk_ew(scored, K, reverse=False)


class IroGapGo:
    """ENTER-ONCE: hold the K names that gapped UP >= ``thresh`` at the open
    (``session_open >= prev_close*(1+thresh)``), ranked by LARGEST gap-up,
    expecting continuation.

    STABLE BY CONSTRUCTION: same as IroGapFade — ``session_open`` and
    ``prev_close`` are session constants, so the set and the ranking score
    (``session_open/prev_close - 1``) are fixed all session => held. NOTE this
    DROPS Round-1 GapGo's ``price > session_open`` confirmation gate: that gate
    was a live (non-stable) condition that flipped intraday and would force
    reselection, defeating enter-once. The pure fixed-gap rule is the
    conservative stable form."""

    thresh = 0.005

    @property
    def name(self) -> str:
        return "iro_gap_go"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            pc = a.prev_close
            if pc is None or pc <= 0:
                continue
            so = a.session_open
            if so >= pc * (1.0 + self.thresh):
                scored.append((so / pc - 1.0, s))   # positive; biggest gap-up first
        return _topk_ew(scored, K, reverse=True)


# ════ BREAKOUT family — opening-range breakout, held once broken ══════════


class IroOrb:
    """ENTER-ONCE: after the opening range forms (``bars_since_open >= m``),
    hold the K names that have broken above their opening-range high AT SOME
    POINT this session, ranked by how far the SESSION HIGH sits above the OR
    high (``session_high/OR_high - 1``), descending. {} before the range forms.

    THE STABILITY PROBLEM AND ITS RESOLUTION (the trickiest spec):
    The naive ORB rule ``price > OR_high`` is NOT stable — price flips back
    below the OR high intraday, so membership churns (and decide() is stateless,
    so we can't just "remember" who already broke out). Round 1 churned ORB to
    -15.85pp for exactly this reason.

    THE FIXED-SIGNAL PROXY: ``opening_range(m)`` is FROZEN once m bars exist
    (it reads only the first m bars of the session — a session constant from
    bar m onward). And ``session_high`` is MONOTONE NON-DECREASING through the
    session. So the predicate ``session_high > OR_high`` is MONOTONE: once a
    name's session high clears the OR high it STAYS cleared for the rest of the
    session => the membership set only grows, never drops a name => stable hold
    of everyone who has broken out. The rank score ``session_high/OR_high - 1``
    is likewise monotone non-decreasing per name. This is the most conservative
    stateless rule that captures "broke out and we hold it." It can ADD a name
    mid-session (a late breakout) — that is a genuine new entry, not churn, and
    the rebalance band still suppresses dust re-trues on the names already held.
    Look-ahead-free: opening_range(m) and session_high both read only indices
    <= the decision bar."""

    m = 30

    @property
    def name(self) -> str:
        return "iro_orb"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar or ctx.bars_since_open < self.m:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            rng = a.opening_range(self.m)
            if rng is None:
                continue
            or_high, _ = rng
            if or_high > 0 and a.session_high > or_high:
                scored.append((a.session_high / or_high - 1.0, s))
        return _topk_ew(scored, K, reverse=True)


# ════ MORNING-TREND family — fixed morning return, held all day ═══════════


class IroMorningTrend:
    """ENTER-ONCE: after the opening window forms (``bars_since_open >= m``),
    hold the K names with the highest MORNING RETURN — the m-th bar's close vs
    the session open, ``closes(bars_since_open)[m-1]/session_open - 1`` — top-K.
    {} before the window forms.

    THE STABILITY PROBLEM AND ITS RESOLUTION:
    The intuitive "morning return" (``price_at_bar_m / session_open - 1``)
    needs the m-th bar's close, but decide() is stateless and ``price`` is the
    CURRENT bar (which drifts), so a naive (price/open) ranking would churn all
    day. The FIXED proxy: ``closes(bars_since_open)`` returns the slice
    ``_c[session_start : i+1]`` (this session's closes through the decision
    bar), so index ``[m-1]`` is ALWAYS ``_c[session_start + m - 1]`` — the
    m-th bar's close, a SESSION CONSTANT once that bar exists. Dividing by the
    (constant) session_open gives a morning-return score that is IDENTICAL on
    every decision bar from bar m onward => the same top-K basket all session
    => held. (This deliberately freezes the signal at bar m and ignores later
    price action — that is the enter-once thesis, not a bug.) Look-ahead-free:
    only reads closes at/before the decision index."""

    m = 30

    @property
    def name(self) -> str:
        return "iro_morning_trend"

    def decide(self, ctx: IntradayContext) -> dict[str, float]:
        if ctx.is_last_decision_bar or ctx.bars_since_open < self.m:
            return {}
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            so = a.session_open
            if so <= 0:
                continue
            sess_closes = a.closes(ctx.bars_since_open)
            if len(sess_closes) < self.m:
                continue
            ret = float(sess_closes[self.m - 1]) / so - 1.0   # fixed m-th-bar return
            scored.append((ret, s))
        return _topk_ew(scored, K, reverse=True)


# ════ CONTROLS — low-turnover coin flips (the recalibrated floor) ═════════


class IroRandomHold:
    """ENTER-ONCE control: hold K names chosen by a DETERMINISTIC hash of
    (seed, session_date, symbol) — a pre-registered coin flip that is FIXED for
    the whole session (same seed+date => same basket every bar => held all day).

    This RECALIBRATES the false-positive floor for the Round-2 cadence: Round
    1's randoms re-drew every 5 min and bled ~6pp to tolls; these hold all day,
    so the random floor is now ~2 tolls/day — the honest "no-signal,
    enter-once" baseline every candidate must beat."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return f"iro_random_{self.seed}"

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

INTRADAY_R2_SPECS: dict[str, callable] = {
    # GAP
    "iro_gap_fade": IroGapFade,
    "iro_gap_go": IroGapGo,
    # BREAKOUT
    "iro_orb": IroOrb,
    # MORNING-TREND
    "iro_morning_trend": IroMorningTrend,
    # CONTROLS
    "iro_random_201": lambda: IroRandomHold(201),
    "iro_random_203": lambda: IroRandomHold(203),
}

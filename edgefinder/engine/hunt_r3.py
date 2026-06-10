"""HUNT ROUND 3 — pre-registered roster (2026-06-10).

PRE-REGISTRATION: committed BEFORE any candidate's first validation run;
parameters frozen. Informed by round-2 FAMILY learnings (never params):
- the value×momentum INTERACTION was the find of round 2 (the barbell
  beat both sleeves run alone and re-checked widest) → probe the
  interaction space STRUCTURALLY (trend-gated value sleeve, integrated
  rank blend) — different mechanisms, not sleeve-ratio tunes;
- every confirmed finalist carries a drawdown caveat and a hard regime
  gate destroys the edge → register SOFT overlays (vol-scaled exposure,
  half-book brake) and a different momentum signal (52-week-high);
- 3 of 4 finalists share the 12-1 engine → diversify with families that
  use NO price momentum and NO P/E: quality, Lynch fast growers, FCF
  yield, dividend value, the equal-weight size premium, seasonality.

Lanes (queue in hunt/queue.json): top:500 PIT, 2021-06→now, --costed
--div-adjust --bars-from r2, sealed holdout 2025-12-05. Labels hunt-r3:*.
Null control runs in-batch; two fresh-seed randoms continue the
false-positive yardstick.
"""

from __future__ import annotations

from edgefinder.engine.hunt_r1 import RandomK, _dollar_vol, _pe, _topk_ew, _vol
from edgefinder.engine.hunt_r2 import _f, _mom_12_1, _uptrend
from edgefinder.engine.strategy import RebalanceContext


def _mom_book(ctx: RebalanceContext, k: int, skip: str | None = None) -> list[str]:
    """Top-k 12-1 momentum names in uptrends — the round-1/2 confirmed engine."""
    scored = []
    for s, a in ctx.assets.items():
        if s == skip or not _uptrend(a):
            continue
        m = _mom_12_1(a)
        if m is not None:
            scored.append((m, s))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [s for _, s in scored[:k]]


# ── momentum with SOFT risk overlays (the drawdown-caveat attack) ────────


class MomSpyVolBrake:
    """12-1 momentum top-20, book exposure scaled by min(1, 15% / SPY 60d
    realized vol) — vol-targeting the BOOK, not the weights. Round 2 showed
    a hard regime gate destroys the edge; this brake never fully exits."""

    k = 20
    target = 0.15

    @property
    def name(self) -> str:
        return "mom_spy_vol_brake"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        book = _mom_book(ctx, self.k, skip="SPY")
        if not book:
            return {}
        scale = 1.0
        spy = ctx.get("SPY")
        if spy:
            v = _vol(spy, 60)
            if v and v > 0:
                scale = min(1.0, self.target / v)
        return {s: scale / self.k for s in book}


class MomSoftGate:
    """12-1 momentum top-20; FULL book while SPY holds its 200-EMA, HALF
    book below it — the registered intermediate between round 2's
    edge-killing hard gate and the drawdown-eating ungated book."""

    k = 20

    @property
    def name(self) -> str:
        return "mom_soft_gate"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        book = _mom_book(ctx, self.k, skip="SPY")
        if not book:
            return {}
        spy = ctx.get("SPY")
        risk_on = bool(spy and spy.indicators.ema_200
                       and spy.price > spy.indicators.ema_200)
        w = (1.0 if risk_on else 0.5) / self.k
        return {s: w for s in book}


class Mom52wHigh:
    """George–Hwang: rank by closeness to the 52-week high (price / 252d
    max close), top-20 in uptrends — a momentum SIGNAL the 12-1 family
    doesn't use."""

    k = 20

    @property
    def name(self) -> str:
        return "mom_52w_high"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            c = a.history["close"].iloc[-252:]
            if len(c) < 252:
                continue
            hi = float(c.max())
            if hi > 0:
                scored.append((a.price / hi, s))
        return _topk_ew(scored, self.k)


# ── value×momentum interaction space (structural probes) ────────────────


class BarbellTrendValue:
    """Round 2's confirmed barbell with ONE structural change: the value
    sleeve also requires an uptrend (cheap AND already turning). 10
    cheapest profitable P/E<10 in uptrends + 10 strongest 12-1 momentum,
    each name 5%."""

    @property
    def name(self) -> str:
        return "barbell_trend_value"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        value, mom = [], []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            up = _uptrend(a)
            if pe is not None and pe < 10.0 and up:
                value.append((pe, s))
            if up:
                m = _mom_12_1(a)
                if m is not None:
                    mom.append((m, s))
        value.sort(key=lambda t: (t[0], t[1]))
        mom.sort(key=lambda t: (t[0], t[1]), reverse=True)
        out: dict[str, float] = {}
        for _, s in value[:10]:
            out[s] = out.get(s, 0) + 0.05
        for _, s in mom[:10]:
            out[s] = out.get(s, 0) + 0.05
        return out


class ValueMomRankBlend:
    """The INTEGRATED value×momentum blend: among profitable uptrend
    names, average the P/E rank (cheap best) and the 12-1 momentum rank
    (strong best); hold the top-20 combined — one book scoring both
    signals per name, vs the barbell's two separate sleeves."""

    k = 20

    @property
    def name(self) -> str:
        return "value_mom_rank_blend"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        rows = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            if pe is None or not _uptrend(a):
                continue
            m = _mom_12_1(a)
            if m is not None:
                rows.append((s, pe, m))
        if not rows:
            return {}
        pe_rank = {s: i for i, (s, _, _) in
                   enumerate(sorted(rows, key=lambda r: (r[1], r[0])))}
        mom_rank = {s: i for i, (s, _, _) in
                    enumerate(sorted(rows, key=lambda r: (-r[2], r[0])))}
        scored = [(-(pe_rank[s] + mom_rank[s]), s) for s, _, _ in rows]
        return _topk_ew(scored, self.k)


# ── families with NO price momentum and NO P/E (engine diversification) ─


class QualityRoeTop:
    """Quality standalone: ROE > 15%, D/E < 1, current ratio > 1.5;
    top-20 by ROE. No price signal at all."""

    k = 20

    @property
    def name(self) -> str:
        return "quality_roe_top"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            roe = _f(a, "return_on_equity")
            de = _f(a, "debt_to_equity")
            cr = _f(a, "current_ratio")
            if (roe is not None and roe > 0.15 and de is not None
                    and de < 1.0 and cr is not None and cr > 1.5):
                scored.append((roe, s))
        return _topk_ew(scored, self.k)


class QualityMomentum:
    """The quality screen of quality_roe_top, ranked by 12-1 momentum in
    uptrends — a HARDER quality cut than round 2's EPS>0 earnings tilt."""

    k = 20

    @property
    def name(self) -> str:
        return "quality_momentum"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            roe = _f(a, "return_on_equity")
            de = _f(a, "debt_to_equity")
            if (roe is None or roe <= 0.15 or de is None or de >= 1.0
                    or not _uptrend(a)):
                continue
            m = _mom_12_1(a)
            if m is not None:
                scored.append((m, s))
        return _topk_ew(scored, self.k)


class FastGrowers:
    """Lynch fast growers, systematized: profitable, earnings growth >20%,
    revenue growth >10%, in an uptrend; top-20 by earnings growth."""

    k = 20

    @property
    def name(self) -> str:
        return "fast_growers"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            eps = _f(a, "earnings_per_share")
            eg = _f(a, "earnings_growth")
            rg = _f(a, "revenue_growth")
            if (eps is not None and eps > 0 and eg is not None and eg > 0.20
                    and rg is not None and rg > 0.10 and _uptrend(a)):
                scored.append((eg, s))
        return _topk_ew(scored, self.k)


class FcfYieldTop:
    """Cash-flow value: top-20 by free-cash-flow yield among profitable
    names — cheapness measured where P/E can't be massaged."""

    k = 20

    @property
    def name(self) -> str:
        return "fcf_yield_top"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            eps = _f(a, "earnings_per_share")
            fy = _f(a, "fcf_yield")
            if eps is not None and eps > 0 and fy is not None and fy > 0:
                scored.append((fy, s))
        return _topk_ew(scored, self.k)


class DividendValue:
    """Equity income: top-20 by dividend yield among profitable names.
    Bars are total-return, so the yield is fully credited."""

    k = 20

    @property
    def name(self) -> str:
        return "dividend_value"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            eps = _f(a, "earnings_per_share")
            dy = _f(a, "dividend_yield")
            if eps is not None and eps > 0 and dy is not None and dy > 0:
                scored.append((dy, s))
        return _topk_ew(scored, self.k)


class EwTop100:
    """The equal-weight size premium: EW the 100 highest-dollar-volume
    names monthly — does plain de-cap-weighting beat SPY after costs?"""

    k = 100

    @property
    def name(self) -> str:
        return "ew_top100"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            dv = _dollar_vol(a, 20)
            if dv is not None and dv > 0:
                scored.append((dv, s))
        return _topk_ew(scored, self.k)


class SeasonalSpy:
    """Pure seasonality: SPY November–April, cash May–October (the
    Halloween effect at monthly cadence). Zero other signals."""

    @property
    def name(self) -> str:
        return "seasonal_spy"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        if ctx.date.month in (11, 12, 1, 2, 3, 4) and ctx.get("SPY"):
            return {"SPY": 1.0}
        return {}


HUNT_R3_SPECS = {
    "mom_spy_vol_brake": MomSpyVolBrake,
    "mom_soft_gate": MomSoftGate,
    "mom_52w_high": Mom52wHigh,
    "barbell_trend_value": BarbellTrendValue,
    "value_mom_rank_blend": ValueMomRankBlend,
    "quality_roe_top": QualityRoeTop,
    "quality_momentum": QualityMomentum,
    "fast_growers": FastGrowers,
    "fcf_yield_top": FcfYieldTop,
    "dividend_value": DividendValue,
    "ew_top100": EwTop100,
    "seasonal_spy": SeasonalSpy,
    # false-positive yardstick continuity (fresh seeds)
    "random_20_s41": lambda: RandomK(41),
    "random_20_s43": lambda: RandomK(43),
}

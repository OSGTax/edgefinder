"""HUNT ROUND 2 — pre-registered roster (2026-06-10).

PRE-REGISTRATION: committed BEFORE any candidate's first validation run;
parameters frozen. Informed by round-1 FAMILY learnings (never params):
- momentum's return premium is real in the top-500 but violent →
  variants that widen the book, tilt to quality, gate on regime, or
  weight by inverse vol;
- value-with-profitability is the Lynch lane's live vein (deep_value_pe10
  +8.7pp; three cousins each ONE fold short of majority) → cheapness ∩
  profitability ∩ trend combinations;
- churn dies to costs → everything here is monthly.

Lanes (queue in hunt/queue.json): stock/Lynch = top:500, 2021-06→now,
--costed --div-adjust --bars-from r2 (post-slim, breadth lives in R2),
sealed holdout 2025-12-05. Labels hunt-r2:*. Null control runs in-batch;
two fixed-seed randoms continue the false-positive yardstick.
"""

from __future__ import annotations

from edgefinder.engine.hunt_r1 import RandomK, _pe, _topk_ew, _vol
from edgefinder.engine.strategy import RebalanceContext


def _mom_12_1(a):
    r252, r21 = a.ret(252), a.ret(21)
    if r252 is None or r21 is None or (1 + r21) <= 0:
        return None
    return (1 + r252) / (1 + r21) - 1.0


def _uptrend(a):
    return bool(a.indicators.ema_200 and a.price > a.indicators.ema_200)


def _f(a, attr):
    return getattr(a.fundamentals, attr, None) if a.fundamentals else None


# ── momentum family ──────────────────────────────────────────────────────


class Mom12_1K40:
    """Round-1 finalist's mechanism with a 2x wider book (top-40): same
    return engine, diluted single-name risk and drawdown."""

    k = 40

    @property
    def name(self) -> str:
        return "mom_12_1_k40"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            m = _mom_12_1(a)
            if m is not None:
                scored.append((m, s))
        return _topk_ew(scored, self.k)


class Mom6M:
    """6-month momentum variant (shorter formation window), top-20."""

    k = 20

    @property
    def name(self) -> str:
        return "mom_6m_k20"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            r = a.ret(126)
            if r is not None:
                scored.append((r, s))
        return _topk_ew(scored, self.k)


class MomInverseVol:
    """12-1 momentum top-20, weighted INVERSE to 60d vol instead of equal —
    the cheapest drawdown-taming overlay that keeps the return engine."""

    k = 20

    @property
    def name(self) -> str:
        return "mom_inverse_vol"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            m = _mom_12_1(a)
            if m is not None:
                scored.append((m, s))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        inv = {}
        for _, s in scored[: self.k]:
            v = _vol(ctx.assets[s], 60)
            if v and v > 0:
                inv[s] = 1.0 / v
        total = sum(inv.values())
        return {s: w / total for s, w in inv.items()} if total > 0 else {}


class MomRegimeGated:
    """12-1 momentum top-20, but the BOOK exists only while SPY holds its
    200-EMA — momentum crashes live in broken regimes; cash otherwise."""

    k = 20

    @property
    def name(self) -> str:
        return "mom_regime_gated"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        spy = ctx.get("SPY")
        if not (spy and spy.indicators.ema_200 and spy.price > spy.indicators.ema_200):
            return {}
        scored = []
        for s, a in ctx.assets.items():
            if s == "SPY" or not _uptrend(a):
                continue
            m = _mom_12_1(a)
            if m is not None:
                scored.append((m, s))
        return _topk_ew(scored, self.k)


class MomEarningsTilt:
    """12-1 momentum restricted to PROFITABLE names (PIT EPS > 0) — the
    quality tilt that classically tames momentum's junk rallies."""

    k = 20

    @property
    def name(self) -> str:
        return "mom_earnings_tilt"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            eps = _f(a, "earnings_per_share")
            if eps is None or eps <= 0 or not _uptrend(a):
                continue
            m = _mom_12_1(a)
            if m is not None:
                scored.append((m, s))
        return _topk_ew(scored, self.k)


# ── value-with-profitability family ─────────────────────────────────────


class ValuePE12:
    """Deep value's sibling: profitable names under 12x earnings, top-20
    cheapest. A NEW pre-registration, not a tune of pe10."""

    k = 20
    ceiling = 12.0

    @property
    def name(self) -> str:
        return "value_pe12"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            if pe is not None and pe < self.ceiling:
                scored.append((pe, s))
        return _topk_ew(scored, self.k, reverse=False)


class ValueROE:
    """Cheap AND good: P/E < 15 with ROE > 10%; top-20 cheapest."""

    k = 20

    @property
    def name(self) -> str:
        return "value_roe"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            roe = _f(a, "return_on_equity")
            if pe is not None and pe < 15.0 and roe is not None and roe > 0.10:
                scored.append((pe, s))
        return _topk_ew(scored, self.k, reverse=False)


class ValueMomentum:
    """Cheap, profitable, AND in an uptrend; top-20 by 6-month momentum —
    the classic value+momentum interaction."""

    k = 20

    @property
    def name(self) -> str:
        return "value_momentum"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            r = a.ret(126)
            if (pe is not None and pe < 15.0 and _uptrend(a) and r is not None):
                scored.append((r, s))
        return _topk_ew(scored, self.k)


class EarningsYieldTop:
    """Top-20 by earnings yield (EPS/price) among profitable names — the
    Greenblatt-style cheapness rank without a hard P/E ceiling."""

    k = 20

    @property
    def name(self) -> str:
        return "earnings_yield_top"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            eps = _f(a, "earnings_per_share")
            if eps is None or eps <= 0 or a.price <= 0:
                continue
            scored.append((eps / a.price, s))
        return _topk_ew(scored, self.k)


class ValueLowDebt:
    """Cheap + fortress balance sheet: P/E < 12 and D/E < 1; top-20."""

    k = 20

    @property
    def name(self) -> str:
        return "value_low_debt"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            de = _f(a, "debt_to_equity")
            if pe is not None and pe < 12.0 and de is not None and de < 1.0:
                scored.append((pe, s))
        return _topk_ew(scored, self.k, reverse=False)


# ── blends ───────────────────────────────────────────────────────────────


class ValueMomBarbell:
    """Half deep-value, half momentum: 10 cheapest profitable (P/E<10) +
    10 strongest 12-1 momentum, each sleeve equal-weight 5%."""

    @property
    def name(self) -> str:
        return "value_mom_barbell"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        value, mom = [], []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            if pe is not None and pe < 10.0:
                value.append((pe, s))
            if _uptrend(a):
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


class LowVolValue:
    """The 50 least-volatile names, kept only when reasonably priced
    (P/E < 20) — low-vol's drawdown profile with a value screen."""

    k = 50

    @property
    def name(self) -> str:
        return "low_vol_value"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            v = _vol(a, 60)
            if pe is not None and pe < 20.0 and v is not None:
                scored.append((v, s))
        return _topk_ew(scored, self.k, reverse=False)


HUNT_R2_SPECS = {
    "mom_12_1_k40": Mom12_1K40,
    "mom_6m_k20": Mom6M,
    "mom_inverse_vol": MomInverseVol,
    "mom_regime_gated": MomRegimeGated,
    "mom_earnings_tilt": MomEarningsTilt,
    "value_pe12": ValuePE12,
    "value_roe": ValueROE,
    "value_momentum": ValueMomentum,
    "earnings_yield_top": EarningsYieldTop,
    "value_low_debt": ValueLowDebt,
    "value_mom_barbell": ValueMomBarbell,
    "low_vol_value": LowVolValue,
    # false-positive yardstick continuity (fresh seeds)
    "random_20_s23": lambda: RandomK(23),
    "random_20_s31": lambda: RandomK(31),
}

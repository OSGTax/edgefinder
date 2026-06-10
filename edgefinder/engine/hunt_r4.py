"""HUNT ROUND 4 — pre-registered roster (2026-06-10).

PRE-REGISTRATION: committed BEFORE any candidate's first validation run;
parameters frozen. Informed by round-3 FAMILY learnings (never params):
- growth fundamentals are a real, independent engine (fast_growers
  confirmed with the stablest re-checks recorded) → probe the vein
  structurally: rank by revenue instead of earnings, growth×momentum,
  PEG-constrained growth, Lynch stalwarts;
- the value×momentum interaction needs SEPARATE sleeves (rank blends
  die) → pair the three confirmed engines as barbells: growth+value,
  growth+momentum, and the tri-sleeve;
- momentum formation cousins not yet tried: intermediate 3-12 (skip the
  last quarter), risk-adjusted rank (momentum / vol);
- PIT field coverage audit (round 3): eg/rg/roe/cr/de/eps usable;
  fcf_yield, dividend_yield, price_to_sales, ev_to_ebitda, price_to_book
  are ~0% — every screen below uses covered fields only.

Lanes: top:500 PIT, 2021-06→now, --costed --div-adjust --bars-from r2,
sealed holdout 2025-12-05. Labels hunt-r4:*. Null + two fresh randoms.
"""

from __future__ import annotations

from edgefinder.engine.hunt_r1 import RandomK, _pe, _topk_ew, _vol
from edgefinder.engine.hunt_r2 import _f, _mom_12_1, _uptrend
from edgefinder.engine.strategy import RebalanceContext


def _is_fast_grower(a) -> bool:
    """The exact confirmed fast_growers screen (round 3)."""
    eps = _f(a, "earnings_per_share")
    eg = _f(a, "earnings_growth")
    rg = _f(a, "revenue_growth")
    return (eps is not None and eps > 0 and eg is not None and eg > 0.20
            and rg is not None and rg > 0.10 and _uptrend(a))


def _sleeve(scored: list[tuple[float, str]], n: int, w: float,
            out: dict[str, float], reverse: bool = True) -> None:
    scored.sort(key=lambda t: (t[0], t[1]), reverse=reverse)
    for _, s in scored[:n]:
        out[s] = out.get(s, 0.0) + w


# ── growth-vein structural probes ────────────────────────────────────────


class FastGrowersRev:
    """The confirmed fast-growers screen ranked by REVENUE growth —
    is the signal the screen or the earnings-growth rank?"""

    k = 20

    @property
    def name(self) -> str:
        return "fast_growers_rev"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = [(_f(a, "revenue_growth"), s) for s, a in ctx.assets.items()
                  if _is_fast_grower(a)]
        return _topk_ew([(v, s) for v, s in scored if v is not None], self.k)


class FastGrowersMom:
    """The fast-growers screen ranked by 12-1 momentum — the
    growth×momentum interaction, sleeve-free."""

    k = 20

    @property
    def name(self) -> str:
        return "fast_growers_mom"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _is_fast_grower(a):
                continue
            m = _mom_12_1(a)
            if m is not None:
                scored.append((m, s))
        return _topk_ew(scored, self.k)


class PegGrowers:
    """Growth at a constrained price: earnings growth >20%, PEG < 1,
    uptrend; top-20 by earnings growth."""

    k = 20

    @property
    def name(self) -> str:
        return "peg_growers"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            eg = _f(a, "earnings_growth")
            if (pe is None or eg is None or eg <= 0.20 or not _uptrend(a)):
                continue
            if pe / (eg * 100.0) < 1.0:
                scored.append((eg, s))
        return _topk_ew(scored, self.k)


class SteadyCompounders:
    """Lynch stalwarts: profitable, earnings growth 5–20%, D/E < 1,
    uptrend; top-20 by earnings growth. The boring-compounder lane."""

    k = 20

    @property
    def name(self) -> str:
        return "steady_compounders"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            eps = _f(a, "earnings_per_share")
            eg = _f(a, "earnings_growth")
            de = _f(a, "debt_to_equity")
            if (eps is not None and eps > 0 and eg is not None
                    and 0.05 <= eg <= 0.20 and de is not None and de < 1.0
                    and _uptrend(a)):
                scored.append((eg, s))
        return _topk_ew(scored, self.k)


# ── sleeve pairings of the three confirmed engines ───────────────────────


class GrowthValueBarbell:
    """10 fast growers (by earnings growth) + 10 cheapest profitable
    P/E<10 in uptrends, 5% each — growth sleeve replaces momentum."""

    @property
    def name(self) -> str:
        return "growth_value_barbell"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        growth, value = [], []
        for s, a in ctx.assets.items():
            if _is_fast_grower(a):
                eg = _f(a, "earnings_growth")
                if eg is not None:
                    growth.append((eg, s))
            pe = _pe(a)
            if pe is not None and pe < 10.0 and _uptrend(a):
                value.append((pe, s))
        out: dict[str, float] = {}
        _sleeve(growth, 10, 0.05, out)
        _sleeve(value, 10, 0.05, out, reverse=False)
        return out


class GrowthMomBarbell:
    """10 fast growers + 10 strongest 12-1 momentum in uptrends, 5% each."""

    @property
    def name(self) -> str:
        return "growth_mom_barbell"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        growth, mom = [], []
        for s, a in ctx.assets.items():
            if _is_fast_grower(a):
                eg = _f(a, "earnings_growth")
                if eg is not None:
                    growth.append((eg, s))
            if _uptrend(a):
                m = _mom_12_1(a)
                if m is not None:
                    mom.append((m, s))
        out: dict[str, float] = {}
        _sleeve(growth, 10, 0.05, out)
        _sleeve(mom, 10, 0.05, out)
        return out


class TriSleeve:
    """The three confirmed engines in one book: 10 trend-gated deep value
    (P/E<10) + 10 12-1 momentum + 10 fast growers, each name 1/30."""

    @property
    def name(self) -> str:
        return "tri_sleeve"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        value, mom, growth = [], [], []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            up = _uptrend(a)
            if pe is not None and pe < 10.0 and up:
                value.append((pe, s))
            if up:
                m = _mom_12_1(a)
                if m is not None:
                    mom.append((m, s))
            if _is_fast_grower(a):
                eg = _f(a, "earnings_growth")
                if eg is not None:
                    growth.append((eg, s))
        out: dict[str, float] = {}
        w = 1.0 / 30.0
        _sleeve(value, 10, w, out, reverse=False)
        _sleeve(mom, 10, w, out)
        _sleeve(growth, 10, w, out)
        return out


# ── momentum formation cousins ───────────────────────────────────────────


class Mom3_12:
    """Intermediate momentum: months 4–12 ((1+r252)/(1+r63)−1), skipping
    the most recent QUARTER instead of the month; uptrend, top-20."""

    k = 20

    @property
    def name(self) -> str:
        return "mom_3_12"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            r252, r63 = a.ret(252), a.ret(63)
            if r252 is None or r63 is None or (1 + r63) <= 0:
                continue
            scored.append(((1 + r252) / (1 + r63) - 1.0, s))
        return _topk_ew(scored, self.k)


class MomSharpeRank:
    """Risk-adjusted momentum SIGNAL: rank by 12-1 return / 60d vol
    (volatility in the rank, not the weights — r2's inverse-vol confirmed
    the weights version); uptrend, top-20 equal-weight."""

    k = 20

    @property
    def name(self) -> str:
        return "mom_sharpe_rank"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            m = _mom_12_1(a)
            v = _vol(a, 60)
            if m is not None and v and v > 0:
                scored.append((m / v, s))
        return _topk_ew(scored, self.k)


# ── covered-field value cousins ──────────────────────────────────────────


class ValueCrFortress:
    """Cheap + liquid balance sheet: P/E < 12 and current ratio > 2;
    top-20 cheapest (the fortress metric round 2 didn't try)."""

    k = 20

    @property
    def name(self) -> str:
        return "value_cr_fortress"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            cr = _f(a, "current_ratio")
            if pe is not None and pe < 12.0 and cr is not None and cr > 2.0:
                scored.append((pe, s))
        return _topk_ew(scored, self.k, reverse=False)


class RoeValue:
    """Quality-per-price: rank profitable names by ROE / P/E (Greenblatt
    spirit on covered fields); top-20."""

    k = 20

    @property
    def name(self) -> str:
        return "roe_value"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            roe = _f(a, "return_on_equity")
            if pe is not None and pe > 0 and roe is not None and roe > 0:
                scored.append((roe / pe, s))
        return _topk_ew(scored, self.k)


# ── defensive structural probe ───────────────────────────────────────────


class MinVolUptrend:
    """The 20 least-volatile names that are IN UPTRENDS — low-vol with
    the trend gate (round 2's low_vol_value used a P/E gate and died)."""

    k = 20

    @property
    def name(self) -> str:
        return "min_vol_uptrend"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not _uptrend(a):
                continue
            v = _vol(a, 60)
            if v is not None:
                scored.append((v, s))
        return _topk_ew(scored, self.k, reverse=False)


HUNT_R4_SPECS = {
    "fast_growers_rev": FastGrowersRev,
    "fast_growers_mom": FastGrowersMom,
    "peg_growers": PegGrowers,
    "steady_compounders": SteadyCompounders,
    "growth_value_barbell": GrowthValueBarbell,
    "growth_mom_barbell": GrowthMomBarbell,
    "tri_sleeve": TriSleeve,
    "mom_3_12": Mom3_12,
    "mom_sharpe_rank": MomSharpeRank,
    "value_cr_fortress": ValueCrFortress,
    "roe_value": RoeValue,
    "min_vol_uptrend": MinVolUptrend,
    # false-positive yardstick continuity (fresh seeds)
    "random_20_s47": lambda: RandomK(47),
    "random_20_s53": lambda: RandomK(53),
}

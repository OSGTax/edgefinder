"""HUNT ROUND 1 — pre-registered strategy roster (2026-06-10).

PRE-REGISTRATION: this file is committed BEFORE any candidate's first
validation run; the parameters below are frozen. A candidate that "needs"
different parameters is a NEW candidate, registered fresh in a later round.
Verdicts come from edgefinder.engine.validate (fixed-param walk-forward,
PIT universes, realistic costs, total-return, sealed holdout 2025-12-05).

Lanes (run configs live in hunt/queue.json; the strategy classes here are
lane-agnostic — they see whatever universe the run hands them):
- ETF defensive/regime: 21yr deep menu (2005->now), monthly, flat 2bps.
- Stock cross-sectional: --universe top:500 --start 2021-06-01 --costed
  --div-adjust, monthly/weekly.
- Lynch/fundamental: stock lane + --pit-fundamentals (PIT snapshots;
  strategies compute P/E, PEG from AssetView.price at decision time;
  fundamentals are None before a symbol's coverage — gates must handle it).
- Deliberately dumb: stock lane; these double as the empirical
  false-positive measurement for the whole round.

Implementation notes:
- All scales are FRACTIONS (verified against fundamentals_snapshots medians
  2023: earnings_growth -0.029, return_on_equity 0.040, debt_to_equity 0.85).
- PEG convention: (price/EPS) / (earnings_growth * 100) — P/E over growth
  expressed in percent, the standard Lynch form.
- Every gate treats None as "not eligible" (missing data never qualifies).
"""

from __future__ import annotations

import hashlib
import math

from edgefinder.engine.strategy import AssetView, RebalanceContext

# The 9 ETFs with full 2005->now history in daily_bars (verified): the etf7
# risk menu + the two deep bond sleeves.
ETF_MENU = ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "EFA", "AGG", "LQD")
RISK_ETFS = ("SPY", "QQQ", "IWM", "DIA", "EFA")


# ── shared helpers (pure functions of an AssetView) ─────────────────────


def _vol(a: AssetView, lookback: int = 20) -> float | None:
    """Annualized close-to-close volatility over ``lookback`` bars."""
    c = a.history["close"].iloc[-(lookback + 1):]
    if len(c) < lookback + 1:
        return None
    r = c.pct_change().dropna()
    sd = float(r.std())
    return sd * math.sqrt(252) if sd and sd > 0 else None


def _dollar_vol(a: AssetView, lookback: int = 20) -> float | None:
    """Mean daily dollar volume over ``lookback`` bars."""
    h = a.history.iloc[-lookback:]
    if len(h) < lookback:
        return None
    return float((h["close"] * h["volume"]).mean())


def _high_ratio(a: AssetView, lookback: int = 252, min_bars: int = 200) -> float | None:
    """price / trailing-``lookback``-bar high (1.0 = at the high)."""
    c = a.history["close"].iloc[-lookback:]
    if len(c) < min_bars:
        return None
    hi = float(c.max())
    return a.price / hi if hi > 0 else None


def _topk_ew(scored: list[tuple[float, str]], k: int, reverse: bool = True) -> dict[str, float]:
    """Equal-weight the top ``k`` by score (ties broken by symbol — stable)."""
    scored.sort(key=lambda t: (t[0], t[1]), reverse=reverse)
    pick = scored[:k]
    return {s: 1.0 / k for _, s in pick} if pick else {}


def _pe(a: AssetView) -> float | None:
    f = a.fundamentals
    eps = getattr(f, "earnings_per_share", None) if f else None
    if eps is None or eps <= 0 or a.price <= 0:
        return None
    return a.price / eps


def _peg(a: AssetView) -> float | None:
    f = a.fundamentals
    pe = _pe(a)
    growth = getattr(f, "earnings_growth", None) if f else None
    if pe is None or growth is None or growth <= 0:
        return None
    return pe / (growth * 100.0)


# ════ ETF DEFENSIVE / REGIME LANE (10) ═══════════════════════════════════


class VolTargetSpy:
    """Scale SPY exposure to a 10% annualized vol target; rest in cash
    (or a defensive asset). Classic vol-managed portfolio."""

    def __init__(self, target: float = 0.10, defensive: str | None = None,
                 name_override: str | None = None) -> None:
        self.target = target
        self.defensive = defensive
        self._name = name_override or (
            "vol_target_spy" if defensive is None else
            f"vol_target_spy_{defensive.lower()}")

    @property
    def name(self) -> str:
        return self._name

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        a = ctx.get("SPY")
        if not a:
            return {}
        v = _vol(a, 20)
        if v is None or v <= 0:
            return {}
        w = max(0.0, min(1.0, self.target / v))
        out = {"SPY": w}
        if self.defensive and w < 1.0 and ctx.get(self.defensive):
            out[self.defensive] = 1.0 - w
        return out


class CanaryMomentum:
    """Keller-style canary: risk-on only while BOTH canaries (EFA, IWM) have
    positive 6-month momentum; otherwise long bonds."""

    canaries = ("EFA", "IWM")

    @property
    def name(self) -> str:
        return "canary_efa_iwm"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        ok = True
        for c in self.canaries:
            a = ctx.get(c)
            r = a.ret(126) if a else None
            if r is None or r <= 0:
                ok = False
                break
        if ok and ctx.get("SPY"):
            return {"SPY": 1.0}
        return {"TLT": 1.0} if ctx.get("TLT") else {}


class GoldenCross:
    """SPY 50-EMA above 200-EMA -> SPY; else cash (or a defensive asset)."""

    def __init__(self, defensive: str | None = None) -> None:
        self.defensive = defensive

    @property
    def name(self) -> str:
        return ("golden_cross_spy" if self.defensive is None
                else f"golden_cross_spy_{self.defensive.lower()}")

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        a = ctx.get("SPY")
        ind = a.indicators if a else None
        if ind and ind.ema_50 and ind.ema_200 and ind.ema_50 > ind.ema_200:
            return {"SPY": 1.0}
        if self.defensive and ctx.get(self.defensive):
            return {self.defensive: 1.0}
        return {}


class InverseVol:
    """Inverse-volatility weights over a fixed menu (fully invested)."""

    def __init__(self, symbols: tuple[str, ...], lookback: int = 60,
                 name_override: str = "inverse_vol_etf7") -> None:
        self.symbols = symbols
        self.lookback = lookback
        self._name = name_override

    @property
    def name(self) -> str:
        return self._name

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        inv: dict[str, float] = {}
        for s in self.symbols:
            a = ctx.get(s)
            v = _vol(a, self.lookback) if a else None
            if v and v > 0:
                inv[s] = 1.0 / v
        total = sum(inv.values())
        return {s: w / total for s, w in inv.items()} if total > 0 else {}


class TrendBreadthGate:
    """Equal-weight the equity sleeve while SPY is above its 200-EMA AND most
    of the menu is above its own 200-EMA; otherwise long bonds."""

    @property
    def name(self) -> str:
        return "trend_breadth_gate"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        spy = ctx.get("SPY")
        spy_ok = bool(spy and spy.indicators.ema_200
                      and spy.price > spy.indicators.ema_200)
        above = 0
        for s in ETF_MENU:
            a = ctx.get(s)
            if a and a.indicators.ema_200 and a.price > a.indicators.ema_200:
                above += 1
        if spy_ok and above >= 5:
            held = [s for s in RISK_ETFS if ctx.get(s)]
            return {s: 1.0 / len(held) for s in held} if held else {}
        return {"TLT": 1.0} if ctx.get("TLT") else {}


class DualMomentum9:
    """The pre-registered dual momentum, widened to the 9-asset deep menu
    (adds AGG + LQD defensive slots). Same absolute+relative filters."""

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "dual_momentum_9"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored: list[tuple[float, str]] = []
        for s in ETF_MENU:
            a = ctx.get(s)
            if a and a.indicators.ema_200 and a.price > a.indicators.ema_200:
                scored.append((a.price / a.indicators.ema_200 - 1.0, s))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return {s: 1.0 / self.top_k for _, s in scored[: self.top_k]}


class RiskParityLite:
    """Inverse-vol over the classic 3-asset diversifier set (SPY/TLT/GLD)."""

    @property
    def name(self) -> str:
        return "risk_parity_lite"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        return InverseVol(("SPY", "TLT", "GLD"), lookback=60,
                          name_override="").rebalance(ctx)


class Mom6Top2:
    """Top-2 of the deep menu by 6-month return, absolute-filtered (only
    positive momentum gets a slot; unfilled slots are cash)."""

    @property
    def name(self) -> str:
        return "mom_6m_top2"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored: list[tuple[float, str]] = []
        for s in ETF_MENU:
            a = ctx.get(s)
            r = a.ret(126) if a else None
            if r is not None and r > 0:
                scored.append((r, s))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return {s: 0.5 for _, s in scored[:2]}


class BreadthTiltSpy:
    """Continuous exposure: SPY weight = fraction of the menu above its own
    200-EMA (breadth as a risk dial, no hard regime switch)."""

    @property
    def name(self) -> str:
        return "breadth_tilt_spy"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        known = above = 0
        for s in ETF_MENU:
            a = ctx.get(s)
            if a and a.indicators.ema_200:
                known += 1
                if a.price > a.indicators.ema_200:
                    above += 1
        if not known or not ctx.get("SPY"):
            return {}
        w = above / known
        return {"SPY": w} if w > 0 else {}


# ════ STOCK CROSS-SECTIONAL LANE (10) ════════════════════════════════════


class XsecMom12_1:
    """Jegadeesh-Titman 12-1 momentum: trailing 12-month return excluding the
    last month, top-20 equal weight, trend-gated (price above 200-EMA)."""

    k = 20

    @property
    def name(self) -> str:
        return "xsec_mom_12_1"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored: list[tuple[float, str]] = []
        for s, a in ctx.assets.items():
            if not (a.indicators.ema_200 and a.price > a.indicators.ema_200):
                continue
            r252, r21 = a.ret(252), a.ret(21)
            if r252 is None or r21 is None or (1 + r21) <= 0:
                continue
            scored.append(((1 + r252) / (1 + r21) - 1.0, s))
        return _topk_ew(scored, self.k)


class LowVol50:
    """The 50 least-volatile names (60-bar), equal weight — the textbook
    low-volatility anomaly portfolio."""

    k = 50

    @property
    def name(self) -> str:
        return "low_vol_50"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            v = _vol(a, 60)
            if v is not None:
                scored.append((v, s))
        return _topk_ew(scored, self.k, reverse=False)


class Near52wkHigh:
    """Top-20 by proximity to the 52-week high, trend-gated."""

    k = 20

    @property
    def name(self) -> str:
        return "near_52wk_high"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not (a.indicators.ema_200 and a.price > a.indicators.ema_200):
                continue
            hr = _high_ratio(a)
            if hr is not None:
                scored.append((hr, s))
        return _topk_ew(scored, self.k)


class Near52wkLow:
    """The deliberate contrarian mirror: bottom-20 by the same ratio (closest
    to 52-week lows). If losers rebound, this finds it; if momentum rules,
    this documents the other side."""

    k = 20

    @property
    def name(self) -> str:
        return "near_52wk_low"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            hr = _high_ratio(a)
            if hr is not None:
                scored.append((hr, s))
        return _topk_ew(scored, self.k, reverse=False)


class StReversalWeekly:
    """Short-term reversal: among uptrending names, buy the 20 worst trailing
    5-day returns (weekly schedule)."""

    k = 20

    @property
    def name(self) -> str:
        return "st_reversal_w"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            if not (a.indicators.ema_200 and a.price > a.indicators.ema_200):
                continue
            r5 = a.ret(5)
            if r5 is not None:
                scored.append((r5, s))
        return _topk_ew(scored, self.k, reverse=False)


class QualMomBlend:
    """Combined rank: 6-month momentum (desc) + volatility (asc), top-20.
    The standard 'quality momentum' construction."""

    k = 20

    @property
    def name(self) -> str:
        return "qual_mom_blend"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        moms, vols = {}, {}
        for s, a in ctx.assets.items():
            r, v = a.ret(126), _vol(a, 60)
            if r is not None and v is not None:
                moms[s], vols[s] = r, v
        if not moms:
            return {}
        mom_rank = {s: i for i, s in enumerate(
            sorted(moms, key=lambda x: (moms[x], x), reverse=True))}
        vol_rank = {s: i for i, s in enumerate(
            sorted(vols, key=lambda x: (vols[x], x)))}
        scored = [(-(mom_rank[s] + vol_rank[s]), s) for s in moms]
        return _topk_ew(scored, self.k)


class DollarVolFade:
    """The 20 LEAST-traded names inside the (already liquid) menu — does a
    liquidity premium survive realistic costs at the top-500 boundary?"""

    k = 20

    @property
    def name(self) -> str:
        return "dollar_vol_fade"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            dv = _dollar_vol(a, 20)
            if dv is not None and dv > 0:
                scored.append((dv, s))
        return _topk_ew(scored, self.k, reverse=False)


class RsiOversoldQuality:
    """Mean reversion with a quality gate: uptrending names with RSI < 30,
    equal weight (up to 25), weekly. Cash when nothing qualifies."""

    max_k = 25

    @property
    def name(self) -> str:
        return "rsi_oversold_q"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        picks = []
        for s, a in ctx.assets.items():
            ind = a.indicators
            if (ind.ema_200 and a.price > ind.ema_200
                    and ind.rsi is not None and ind.rsi < 30):
                picks.append(s)
        picks = sorted(picks)[: self.max_k]
        return {s: 1.0 / len(picks) for s in picks} if picks else {}


class VolSqueeze:
    """Volatility compression: top-20 narrowest Bollinger width among names
    above their 50-EMA (the 'squeeze then go' setup, long-only)."""

    k = 20

    @property
    def name(self) -> str:
        return "vol_squeeze"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            ind = a.indicators
            if (ind.ema_50 and a.price > ind.ema_50
                    and ind.bb_width is not None and ind.bb_width > 0):
                scored.append((ind.bb_width, s))
        return _topk_ew(scored, self.k, reverse=False)


class InvVolTop50:
    """Defensive construction on the most liquid half: the top-50 by dollar
    volume, weighted inverse to volatility."""

    k = 50

    @property
    def name(self) -> str:
        return "inv_vol_top50"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        by_dv = []
        for s, a in ctx.assets.items():
            dv = _dollar_vol(a, 20)
            if dv is not None:
                by_dv.append((dv, s))
        by_dv.sort(key=lambda t: (t[0], t[1]), reverse=True)
        inv: dict[str, float] = {}
        for _, s in by_dv[: self.k]:
            v = _vol(ctx.assets[s], 60)
            if v and v > 0:
                inv[s] = 1.0 / v
        total = sum(inv.values())
        return {s: w / total for s, w in inv.items()} if total > 0 else {}


# ════ LYNCH / FUNDAMENTAL LANE (8) ═══════════════════════════════════════


class GarpClassic:
    """The Lynch screen: earnings growth > 15%, D/E < 1, profitable; buy the
    20 cheapest by PEG. P/E and PEG computed at decision time from price."""

    k = 20

    @property
    def name(self) -> str:
        return "garp_classic"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            f = a.fundamentals
            if not f:
                continue
            g = getattr(f, "earnings_growth", None)
            de = getattr(f, "debt_to_equity", None)
            peg = _peg(a)
            if (g is not None and g > 0.15 and de is not None and de < 1.0
                    and peg is not None and peg > 0):
                scored.append((peg, s))
        return _topk_ew(scored, self.k, reverse=False)


class GarpQuality:
    """Quality growth: ROE > 15%, growth > 10%, current ratio > 1.5; top-20
    by ROE. Also run on the small-cap band as ``quality_smallcap``."""

    k = 20

    def __init__(self, name_override: str = "garp_quality") -> None:
        self._name = name_override

    @property
    def name(self) -> str:
        return self._name

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            f = a.fundamentals
            if not f:
                continue
            roe = getattr(f, "return_on_equity", None)
            g = getattr(f, "earnings_growth", None)
            cr = getattr(f, "current_ratio", None)
            if (roe is not None and roe > 0.15 and g is not None and g > 0.10
                    and cr is not None and cr > 1.5):
                scored.append((roe, s))
        return _topk_ew(scored, self.k)


class EarningsGrowers:
    """Pure growth: positive earnings AND revenue growth; top-20 by earnings
    growth (no valuation discipline — the contrast case to GARP)."""

    k = 20

    @property
    def name(self) -> str:
        return "earnings_growers"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            f = a.fundamentals
            if not f:
                continue
            g = getattr(f, "earnings_growth", None)
            rg = getattr(f, "revenue_growth", None)
            if g is not None and g > 0 and rg is not None and rg > 0:
                scored.append((g, s))
        return _topk_ew(scored, self.k)


class LowDebtValue:
    """Balance-sheet value: D/E < 0.5, profitable; top-20 cheapest by P/E."""

    k = 20

    @property
    def name(self) -> str:
        return "low_debt_value"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            f = a.fundamentals
            de = getattr(f, "debt_to_equity", None) if f else None
            pe = _pe(a)
            if de is not None and de < 0.5 and pe is not None:
                scored.append((pe, s))
        return _topk_ew(scored, self.k, reverse=False)


class CashRichGrowth:
    """Fortress balance sheet + growth: current ratio > 2, D/E < 0.3,
    growth > 10%; top-20 by growth."""

    k = 20

    @property
    def name(self) -> str:
        return "cash_rich_growth"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            f = a.fundamentals
            if not f:
                continue
            cr = getattr(f, "current_ratio", None)
            de = getattr(f, "debt_to_equity", None)
            g = getattr(f, "earnings_growth", None)
            if (cr is not None and cr > 2.0 and de is not None and de < 0.3
                    and g is not None and g > 0.10):
                scored.append((g, s))
        return _topk_ew(scored, self.k)


class DeepValuePE10:
    """Old-school deep value: profitable names at P/E < 10, top-20 cheapest.
    No quality gates — tests whether raw cheapness alone still pays."""

    k = 20

    @property
    def name(self) -> str:
        return "deep_value_pe10"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            pe = _pe(a)
            if pe is not None and pe < 10.0:
                scored.append((pe, s))
        return _topk_ew(scored, self.k, reverse=False)


class GarpMomentum:
    """Lynch + trend: GARP-eligible (PEG < 1.5, growth > 10%) names that are
    ALSO in uptrends; top-20 by 6-month momentum."""

    k = 20

    @property
    def name(self) -> str:
        return "garp_momentum"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for s, a in ctx.assets.items():
            f = a.fundamentals
            g = getattr(f, "earnings_growth", None) if f else None
            peg = _peg(a)
            r = a.ret(126)
            if (g is not None and g > 0.10 and peg is not None
                    and 0 < peg < 1.5 and r is not None
                    and a.indicators.ema_200 and a.price > a.indicators.ema_200):
                scored.append((r, s))
        return _topk_ew(scored, self.k)


# ════ DELIBERATELY-DUMB SWEEP (6) — the false-positive yardstick ═════════


class LetterB:
    """Equal-weight every name starting with 'B', weekly. If this 'works',
    the bar is broken — that is the point of the dumb sweep."""

    @property
    def name(self) -> str:
        return "letter_b_weekly"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        picks = sorted(s for s in ctx.assets if s.startswith("B"))
        return {s: 1.0 / len(picks) for s in picks} if picks else {}


class EarlyMonthHold:
    """Fully invested (equal-weight menu) only during the first 7 calendar
    days of each month; cash otherwise. Daily schedule."""

    @property
    def name(self) -> str:
        return "early_month_hold"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        if ctx.date.day > 7:
            return {}
        syms = ctx.symbols()
        return {s: 1.0 / len(syms) for s in syms} if syms else {}


class RandomK:
    """20 names chosen by a deterministic hash of (seed, symbol, month) —
    a pre-registered coin flip. Two seeds = two independent draws."""

    k = 20

    def __init__(self, seed: int) -> None:
        self.seed = seed

    @property
    def name(self) -> str:
        return f"random_20_s{self.seed}"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        ym = f"{ctx.date.year}-{ctx.date.month:02d}"
        scored = []
        for s in ctx.assets:
            h = hashlib.md5(f"{self.seed}:{s}:{ym}".encode()).hexdigest()
            scored.append((int(h[:8], 16), s))
        return _topk_ew(scored, self.k, reverse=False)


class AlphabetRotation:
    """A–M names in odd months, N–Z in even months. Pure calendar noise."""

    @property
    def name(self) -> str:
        return "alphabet_rotation"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        first_half = ctx.date.month % 2 == 1
        picks = sorted(s for s in ctx.assets
                       if (s[:1] <= "M") == first_half)
        return {s: 1.0 / len(picks) for s in picks} if picks else {}


class TuesdayHold:
    """Hold the equal-weight menu only on Tuesdays (enter Tuesday's open via
    Monday's decision, exit Wednesday). Daily schedule, maximal churn —
    deliberately the costliest dumb idea in the sweep."""

    @property
    def name(self) -> str:
        return "tuesday_hold"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        # decision date is the day BEFORE the fill: Monday's decision fills
        # Tuesday's open, so hold when tomorrow-ish is Tuesday (weekday 0)
        if ctx.date.weekday() != 0:
            return {}
        syms = ctx.symbols()
        return {s: 1.0 / len(syms) for s in syms} if syms else {}


# ── spec registry (consumed by engine.strategies.make_strategy_factory) ──

HUNT_SPECS = {
    # ETF defensive/regime
    "vol_target_spy": lambda: VolTargetSpy(),
    "vol_target_spy_tlt": lambda: VolTargetSpy(defensive="TLT"),
    "canary_efa_iwm": CanaryMomentum,
    "golden_cross_spy": lambda: GoldenCross(),
    "golden_cross_spy_tlt": lambda: GoldenCross(defensive="TLT"),
    "inverse_vol_etf7": lambda: InverseVol(
        ("SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "EFA")),
    "trend_breadth_gate": TrendBreadthGate,
    "dual_momentum_9": DualMomentum9,
    "risk_parity_lite": RiskParityLite,
    "mom_6m_top2": Mom6Top2,
    "breadth_tilt_spy": BreadthTiltSpy,
    # stock cross-sectional
    "xsec_mom_12_1": XsecMom12_1,
    "low_vol_50": LowVol50,
    "near_52wk_high": Near52wkHigh,
    "near_52wk_low": Near52wkLow,
    "st_reversal_w": StReversalWeekly,
    "qual_mom_blend": QualMomBlend,
    "dollar_vol_fade": DollarVolFade,
    "rsi_oversold_q": RsiOversoldQuality,
    "vol_squeeze": VolSqueeze,
    "inv_vol_top50": InvVolTop50,
    # Lynch/fundamental
    "garp_classic": GarpClassic,
    "garp_quality": lambda: GarpQuality(),
    "quality_smallcap": lambda: GarpQuality(name_override="quality_smallcap"),
    "earnings_growers": EarningsGrowers,
    "low_debt_value": LowDebtValue,
    "cash_rich_growth": CashRichGrowth,
    "deep_value_pe10": DeepValuePE10,
    "garp_momentum": GarpMomentum,
    # dumb sweep
    "letter_b_weekly": LetterB,
    "early_month_hold": EarlyMonthHold,
    "random_20_s7": lambda: RandomK(7),
    "random_20_s13": lambda: RandomK(13),
    "alphabet_rotation": AlphabetRotation,
    "tuesday_hold": TuesdayHold,
}

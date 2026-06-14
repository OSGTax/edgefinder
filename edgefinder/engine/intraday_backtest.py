"""The intraday (minute-bar) backtest engine — one correct trading sequence.

The daily sibling (engine/backtest.py) is the template; every honesty invariant
is copied EXACTLY:

- No look-ahead: the decision context at bar ``i`` is built from data through
  bar ``i``; the resulting target is filled at bar ``i+1``'s OPEN (the next bar
  in calendar order). Cost context is read STRICTLY BEFORE the fill bar.
- No wall-clock: everything is keyed to the bar calendar; ``bars_until_close``
  comes from the ET CLOCK (minutes to 16:00 ET / bar interval), so it is
  live-replicable and not derived by counting future bars.
- Sells settle before buys (cash-only, no leverage); buys capped by cash;
  integer shares; weights summing > 1 scale to 1.0.
- ``trade_start_day`` gates BOTH trading and equity marks, so stats cover only
  the scored region (mirrors daily ``trade_start``).

DESIGN — DAILY equity curve from intraday mechanics. The engine steps bar by
bar within each ET session but emits ONE (session_date, equity) point at each
session close. That daily curve is what gets scored, so Sharpe annualizes
sqrt(252) and reuses the daily _aggregate unchanged — directly comparable to
SPY and the monthly fleet. Intraday mechanics only set each day's return. We
ALSO track each session's intra-session low-water equity for an honest intraday
max-drawdown (close-to-close returns, but drawdown from intra-session lows).

THE FLATTEN-AT-CLOSE MOC PROXY (the one place a fill uses the same bar's
close): when ``flatten_at_close`` is on, the engine forces the target to {} on
each session's last decision bar. But there is no bar at/after the close to fill
into within the session. We resolve this honestly by filling the flatten at the
session's LAST bar's CLOSE (a market-on-close proxy), WITH tolls. Justified:
MOC is a real order type, and the alternative — gapping out at the next
session's open — would let an overnight move flatter (or worsen) a flat-by-
design strategy's daily return, overstating its mechanics. Non-flatten
positions carry to the next session's open as usual.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from edgefinder.backtest.costs import corwin_schultz_spread
from edgefinder.data.minutestore import to_et
from edgefinder.engine.intraday_strategy import (
    _RTH_CLOSE_MIN,
    IntradayAssetView,
    IntradayContext,
    IntradayStrategy,
)

_COST_WINDOW = 30   # FIXED trailing bars for ADV / volatility (never optimized)


@dataclass
class IntradayBacktestResult:
    daily_equity_curve: list[tuple]    # [(session_date, equity)]
    trades: list[dict]
    stats: dict
    decisions_log: list[dict] = field(default_factory=list)


def _sharpe(equity: list[float]) -> float | None:
    if len(equity) < 3:
        return None
    rets = [equity[i] / equity[i - 1] - 1.0
            for i in range(1, len(equity)) if equity[i - 1] > 0]
    if len(rets) < 2:
        return None
    sd = statistics.stdev(rets)
    return (statistics.mean(rets) / sd) * math.sqrt(252) if sd > 0 else None


def _max_drawdown(equity: list[float]) -> float:
    peak, mdd = (equity[0] if equity else 0.0), 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


@dataclass
class _Prep:
    """Per-symbol numpy arrays + index maps for O(1) bar access."""
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    ts: np.ndarray
    sess: np.ndarray             # ET session-date per bar (object array of date)
    ts_to_idx: dict              # ts -> array index
    sess_start: np.ndarray       # per bar: index of that session's first bar
    last_idx_of_session: dict    # session_date -> index of its last bar
    adv: np.ndarray              # trailing-N mean dollar volume (look-ahead-free)
    vol: np.ndarray              # trailing-N per-bar return stdev
    last_idx: int                # final loaded index


def _prepare(bars_by_symbol: dict[str, pd.DataFrame]) -> tuple[dict, list]:
    """Build per-symbol numpy prep + the unified sorted minute calendar."""
    prep: dict[str, _Prep] = {}
    all_ts: set = set()
    for sym, df in bars_by_symbol.items():
        d = df.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
        ts = d["ts"].to_numpy(dtype="int64")
        o = d["open"].to_numpy(dtype="float64")
        h = d["high"].to_numpy(dtype="float64")
        l = d["low"].to_numpy(dtype="float64")
        c = d["close"].to_numpy(dtype="float64")
        v = d["volume"].to_numpy(dtype="float64")
        sess = to_et(d["ts"]).dt.date.to_numpy()
        n = len(ts)
        # per-bar session start index + last index per session (one pass)
        sess_start = np.empty(n, dtype="int64")
        last_idx_of_session: dict = {}
        start = 0
        for i in range(n):
            if i == 0 or sess[i] != sess[i - 1]:
                start = i
            sess_start[i] = start
            last_idx_of_session[sess[i]] = i
        # look-ahead-free liquidity stats (index j read by the loop is < fill)
        dollar = c * v
        adv = pd.Series(dollar).rolling(_COST_WINDOW, min_periods=2).mean().to_numpy()
        rets = pd.Series(c).pct_change()
        vol = rets.rolling(_COST_WINDOW, min_periods=2).std().to_numpy()
        prep[sym] = _Prep(
            o=o, h=h, l=l, c=c, v=v, ts=ts, sess=sess,
            ts_to_idx={int(t): k for k, t in enumerate(ts)},
            sess_start=sess_start, last_idx_of_session=last_idx_of_session,
            adv=adv, vol=vol, last_idx=n - 1)
        all_ts.update(int(t) for t in ts)
    return prep, sorted(all_ts)


def _cost_ctx(p: _Prep, j: int) -> tuple[float, float, float]:
    """Look-ahead-free (adv_dollars, volatility, spread_frac) computed from bars
    STRICTLY BEFORE fill index ``j`` (so it reads j-1 and j-2)."""
    if j < 2:
        return (0.0, 0.0, 0.0)
    adv = p.adv[j - 1]
    vol = p.vol[j - 1]
    spread = corwin_schultz_spread(p.h[j - 2], p.l[j - 2], p.h[j - 1], p.l[j - 1])
    return (float(adv) if adv == adv else 0.0,
            float(vol) if vol == vol else 0.0,
            spread)


def _fill_price(px, side, shares, cost_model, cost_rate, ctx):
    if cost_model is None:
        return px * (1 + cost_rate) if side == "BUY" else px * (1 - cost_rate)
    adv, vol, spread = ctx
    return cost_model.fill_price(
        px, side, order_dollars=shares * px,
        adv_dollars=adv, volatility=vol, spread_frac=spread)


def run_intraday_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: IntradayStrategy,
    *,
    start_cash: float = 1_000_000.0,
    decision_interval: int = 1,
    cost_model=None,
    cost_bps: float = 2.0,
    warmup_bars: int = 0,
    trade_start_day=None,
    flatten_at_close: bool = True,
    benchmark: pd.DataFrame | None = None,
    bar_seconds: int = 60,
    log_decisions: bool = False,
) -> IntradayBacktestResult:
    """Replay ``strategy`` bar-by-bar over a minute calendar; return a DAILY
    equity curve (one point per ET session close), trades, and stats.

    See the module docstring for the honesty invariants and the flatten-at-close
    MOC proxy. ``bar_seconds`` (default 60) drives ``bars_until_close`` from the
    ET clock; ``decision_interval`` thins decisions to every Nth bar; costs use
    ``cost_model`` (the FIXED intraday model) or flat ``cost_bps`` otherwise.
    ``trade_start_day`` gates both trading and marks (sessions before it feed
    history only)."""
    cost_rate = cost_bps / 1e4
    bar_min = max(1, bar_seconds // 60)

    prep, calendar = _prepare(bars_by_symbol)
    if not calendar:
        return IntradayBacktestResult([], [], _summarize([], [], start_cash, {}, benchmark, []), [])

    # ts -> ET session date / minute-of-day (single vectorized pass).
    cal_ts = pd.Series(calendar, dtype="int64")
    cal_et = to_et(cal_ts)
    ts_sess = {int(t): d for t, d in zip(calendar, cal_et.dt.date)}
    ts_minute = {int(t): int(m) for t, m in
                 zip(calendar, (cal_et.dt.hour * 60 + cal_et.dt.minute))}

    holdings: dict[str, int] = {}
    cash = float(start_cash)
    trades: list[dict] = []
    decisions: list[dict] = []
    daily_curve: list[tuple] = []
    intraday_min_by_day: dict = {}

    # pending target to execute at the NEXT bar's open, per symbol set.
    pending: dict | None = None      # {"weights": {...}, "decision_idx_by_sym": {sym: i}}

    sessions = sorted({d for d in ts_sess.values()})
    scored = (sessions if trade_start_day is None
              else [d for d in sessions if d >= trade_start_day])
    scored_set = set(scored)

    def _mark(ts_int: int) -> float:
        eq = cash
        for sym, sh in holdings.items():
            p = prep[sym]
            j = p.ts_to_idx.get(ts_int)
            px = p.c[j] if j is not None else _last_known_close(p, ts_int)
            eq += sh * px
        return eq

    def _last_known_close(p: _Prep, ts_int: int) -> float:
        # most recent bar at or before ts_int for this symbol
        idx = np.searchsorted(p.ts, ts_int, side="right") - 1
        return float(p.c[idx]) if idx >= 0 else 0.0

    n_cal = len(calendar)
    for ci in range(n_cal):
        ts_int = calendar[ci]
        sess_date = ts_sess[ts_int]
        minute = ts_minute[ts_int]

        # ── 1. execute any pending target at THIS bar's open ──
        if pending is not None:
            cash = _execute(
                pending["weights"], prep, ts_int, holdings, cash,
                cost_model, cost_rate, trades, sess_date,
                pending["decision_idx_by_sym"])
            pending = None

        # ── 2. flatten-at-close: at each session's last bar, sell to MOC ──
        # Detect the session's last bar from the calendar (next ts is a new day
        # or the calendar ends). The flatten fills at THIS bar's CLOSE.
        is_session_last = (ci == n_cal - 1) or (ts_sess[calendar[ci + 1]] != sess_date)
        if flatten_at_close and is_session_last and holdings and sess_date in scored_set:
            for sym in sorted(list(holdings)):
                p = prep[sym]
                j = p.ts_to_idx.get(ts_int)
                if j is None:
                    j = int(np.searchsorted(p.ts, ts_int, side="right") - 1)
                if j < 0:
                    continue
                px = p.c[j]                                   # MOC proxy: same-bar close
                fill = _fill_price(px, "SELL", holdings[sym], cost_model,
                                   cost_rate, _cost_ctx(p, j))
                cash += holdings[sym] * fill
                trades.append({"ts": ts_int, "date": sess_date, "symbol": sym,
                               "side": "SELL", "shares": holdings[sym],
                               "price": round(fill, 4), "reason": "MOC_FLATTEN"})
            holdings.clear()

        # ── 3. decision (only on scored sessions, every Nth bar, past warmup) ──
        if (sess_date in scored_set and ci >= warmup_bars
                and (ci % decision_interval == 0)):
            # don't decide on the session-last bar UNLESS not flattening (a
            # last-bar decision would have no in-session fill bar; flatten
            # already handled the exit above)
            if not (is_session_last and flatten_at_close):
                ctx = _build_context(prep, ts_int, sess_date, minute, bar_min,
                                     is_session_last)
                weights = strategy.decide(ctx) or {}
                total = sum(w for w in weights.values() if w and w > 0)
                if total > 1.0:
                    weights = {s: w / total for s, w in weights.items()}
                # record the decision index per symbol so the fill guard can
                # assert fill_idx > decision_idx
                didx = {sym: prep[sym].ts_to_idx[ts_int]
                        for sym in prep if ts_int in prep[sym].ts_to_idx}
                pending = {"weights": weights, "decision_idx_by_sym": didx}
                if log_decisions:
                    decisions.append({"ts": ts_int, "date": sess_date,
                                      "minute_of_day": minute,
                                      "weights": dict(weights)})

        # ── 4. intra-session low-water mark (scored sessions only) ──
        if sess_date in scored_set:
            eq_now = _mark(ts_int)
            cur = intraday_min_by_day.get(sess_date)
            intraday_min_by_day[sess_date] = eq_now if cur is None else min(cur, eq_now)

        # ── 5. daily curve point at each session close ──
        if is_session_last and sess_date in scored_set:
            daily_curve.append((sess_date, round(_mark(ts_int), 2)))

    # honest intraday drawdown from intra-session lows interleaved with closes.
    intra_series: list[float] = []
    for d, eq in daily_curve:
        lo = intraday_min_by_day.get(d)
        if lo is not None:
            intra_series.append(round(lo, 2))
        intra_series.append(eq)

    bench_curve = _benchmark_daily_curve(benchmark, scored_set) if benchmark is not None else None
    stats = _summarize([e for _, e in daily_curve], trades, start_cash,
                       holdings, bench_curve, intra_series)
    stats["avg_trades_per_day"] = round(len(trades) / len(daily_curve), 2) if daily_curve else 0.0
    return IntradayBacktestResult(daily_curve, trades, stats, decisions)


def _build_context(prep, ts_int, sess_date, minute, bar_min, is_session_last):
    """Snapshot the universe at the decision bar (data through this bar only)."""
    assets: dict[str, IntradayAssetView] = {}
    bars_since_open = 0
    for sym, p in prep.items():
        i = p.ts_to_idx.get(ts_int)
        if i is None:
            continue
        price = p.c[i]
        if price <= 0:
            continue
        ss = int(p.sess_start[i])
        assets[sym] = IntradayAssetView(
            symbol=sym, _o=p.o, _h=p.h, _l=p.l, _c=p.c, _v=p.v, _ts=p.ts,
            i=i, session_start=ss)
        bars_since_open = max(bars_since_open, i - ss + 1)
    # bars_until_close from the ET CLOCK (minutes to 16:00 / interval) — NOT by
    # counting future bars, so it's live-replicable and look-ahead-free.
    minutes_left = max(0, _RTH_CLOSE_MIN - minute)
    bars_until_close = max(0, (minutes_left // bar_min) - 1)
    return IntradayContext(
        ts=ts_int, session_date=sess_date, minute_of_day=minute,
        bars_since_open=bars_since_open, bars_until_close=bars_until_close,
        assets=assets, is_last_decision_bar=is_session_last)


def _execute(weights, prep, ts_int, holdings, cash, cost_model, cost_rate,
             trades, sess_date, decision_idx_by_sym):
    """Trade toward target weights at THIS bar's open. Sells settle before buys;
    buys capped by cash; integer shares. Guards raise on look-ahead violations.
    """
    open_px: dict[str, float] = {}
    fill_idx: dict[str, int] = {}
    ctx_by_sym: dict[str, tuple] = {}
    for sym, p in prep.items():
        j = p.ts_to_idx.get(ts_int)
        if j is None or p.o[j] <= 0:
            continue
        # GUARD: a fill bar must be strictly AFTER the decision bar.
        di = decision_idx_by_sym.get(sym)
        if di is not None and j <= di:
            raise AssertionError(
                f"intraday look-ahead: fill idx {j} <= decision idx {di} for {sym}")
        open_px[sym] = float(p.o[j])
        fill_idx[sym] = j
        ctx_by_sym[sym] = _cost_ctx(p, j)

    equity = cash + sum(sh * open_px[s] for s, sh in holdings.items() if s in open_px)
    target: dict[str, int] = {}
    frozen: set = set()
    for s, w in weights.items():
        if w and w > 0 and s in open_px and open_px[s] > 0:
            n = int((w * equity) / open_px[s])
            if cost_model is not None:
                adv = ctx_by_sym[s][0]
                capped = cost_model.cap_shares(n, open_px[s], adv)
                if capped == 0 and n > 0 and holdings.get(s):
                    frozen.add(s)
                n = capped
            target[s] = n

    deltas: dict[str, int] = {}
    for s in sorted(set(holdings) | set(target)):
        if s not in open_px or s in frozen:
            continue
        d = target.get(s, 0) - holdings.get(s, 0)
        if d != 0:
            deltas[s] = d

    for s, d in deltas.items():                 # sells first (raise cash)
        if d < 0:
            fill = _fill_price(open_px[s], "SELL", -d, cost_model, cost_rate, ctx_by_sym[s])
            cash += (-d) * fill
            holdings[s] = holdings.get(s, 0) + d
            trades.append({"ts": ts_int, "date": sess_date, "symbol": s,
                           "side": "SELL", "shares": -d, "price": round(fill, 4)})
    for s, d in deltas.items():                 # then buys (capped by cash)
        if d > 0:
            unit = _fill_price(open_px[s], "BUY", d, cost_model, cost_rate, ctx_by_sym[s])
            buy = min(d, int(cash / unit) if unit > 0 else 0)
            if buy > 0:
                cash -= buy * unit
                holdings[s] = holdings.get(s, 0) + buy
                trades.append({"ts": ts_int, "date": sess_date, "symbol": s,
                               "side": "BUY", "shares": buy, "price": round(unit, 4)})
    for s in [s for s, sh in holdings.items() if sh == 0]:
        del holdings[s]
    return cash


def _benchmark_daily_curve(benchmark: pd.DataFrame, scored_set: set) -> dict | None:
    """Build SPY's session-close daily curve over the scored sessions, anchored
    at the first scored session's first decision-bar OPEN (mirrors daily
    _summarize's open-anchored benchmark). Returns {"opens0", "closes": [...]}.
    """
    if benchmark is None or not len(benchmark):
        return None
    b = benchmark.sort_values("ts").drop_duplicates(subset="ts").reset_index(drop=True)
    et = to_et(b["ts"])
    b = b.assign(_d=et.dt.date.to_numpy())
    b = b[b["_d"].isin(scored_set)]
    if not len(b):
        return None
    closes: list[float] = []
    first_open = None
    for d, grp in b.groupby("_d", sort=True):
        grp = grp.sort_values("ts")
        if first_open is None:
            first_open = float(grp["open"].iloc[0])
        closes.append(float(grp["close"].iloc[-1]))
    return {"open0": first_open, "closes": closes}


def _summarize(eq, trades, start_cash, holdings, bench_curve, intra_series) -> dict:
    eq = eq or [start_cash]
    final = eq[-1]
    stats = {
        "return_pct": round((final - start_cash) / start_cash * 100, 2) if start_cash else 0.0,
        "sharpe": round(_sharpe(eq), 2) if _sharpe(eq) is not None else None,
        "max_drawdown_pct": round(_max_drawdown(eq) * 100, 2),
        "final_equity": round(final, 2),
        "num_trades": len(trades),
        "days": len(eq),
        "open_positions": len(holdings),
    }
    # honest intraday drawdown from intra-session lows (>= close-to-close dd).
    stats["intraday_max_drawdown_pct"] = round(
        _max_drawdown(intra_series or eq) * 100, 2)
    if bench_curve is not None:
        bc = bench_curve["closes"]
        if len(bc) > 1:
            base = bench_curve["open0"] if bench_curve["open0"] > 0 else bc[0]
            bret = (bc[-1] - base) / base * 100
            brets = [bc[i] / bc[i - 1] - 1 for i in range(1, len(bc))]
            bsh = (statistics.mean(brets) / statistics.stdev(brets) * math.sqrt(252)
                   if len(brets) > 1 and statistics.stdev(brets) > 0 else None)
            stats["benchmark_return_pct"] = round(bret, 2)
            stats["excess_return_pct"] = round(stats["return_pct"] - bret, 2)
            if bsh is not None:
                stats["benchmark_sharpe"] = round(bsh, 2)
                if stats["sharpe"] is not None:
                    stats["excess_sharpe"] = round(stats["sharpe"] - bsh, 2)
            stats["benchmark_max_drawdown_pct"] = round(_max_drawdown(bc) * 100, 2)
            stats["drawdown_reduction_pct"] = round(
                stats["benchmark_max_drawdown_pct"] - stats["max_drawdown_pct"], 2)
    return stats

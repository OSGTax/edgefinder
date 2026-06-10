"""The portfolio backtest engine — one correct trading sequence, fully tested.

A backtest is a pure function: (bars, strategy, costs) -> equity curve. There is
NO account state machine, NO cooldown/PDT/revenge logic, NO wall-clock — those
live-trading concerns are exactly where the old engine's bugs hid. Here the only
job is: ask the strategy for target weights using data through yesterday, trade
to them at today's open with realistic costs, mark to market, repeat.

Correct by construction:
- No look-ahead: the rebalance context is built as of the PREVIOUS bar; fills
  happen at TODAY's open.
- No wall-clock: everything is keyed to the simulated calendar.
- Bad prints can't fire phantom trades (OHLC is sanitized).
- A delisted holding is closed at its last real price, never frozen.
- Costs (spread/impact in bps of traded notional) are charged on every fill.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

# Reuse the vetted precompute + OHLC sanitizer; rebuild only the trade loop.
from edgefinder.backtest.costs import corwin_schultz_spread
from edgefinder.backtest.daily_backtest import _sanitize_ohlcv, precompute_snapshots

_COST_WINDOW = 20   # trailing days for ADV / volatility (matches the old lab)
from edgefinder.engine.strategy import (
    AssetView,
    RebalanceContext,
    Strategy,
    _resolve_index,
)


@dataclass
class BacktestResult:
    equity_curve: list[tuple]          # [(date, equity)]
    trades: list[dict]
    stats: dict
    weights_log: list[dict] = field(default_factory=list)  # [{date, weights}]


def _sharpe(equity: list[float]) -> float | None:
    if len(equity) < 3:
        return None
    rets = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity)) if equity[i - 1] > 0]
    if len(rets) < 2:
        return None
    sd = statistics.pstdev(rets) if len(rets) < 2 else statistics.stdev(rets)
    return (statistics.mean(rets) / sd) * math.sqrt(252) if sd > 0 else None


def _max_drawdown(equity: list[float]) -> float:
    peak, mdd = equity[0] if equity else 0.0, 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _is_rebalance(i: int, cal: list, schedule: str) -> bool:
    if i == 0:
        return False
    if schedule == "daily":
        return True
    prev, cur = cal[i - 1], cal[i]
    if schedule == "weekly":
        return cur.isocalendar()[1] != prev.isocalendar()[1] or cur.year != prev.year
    if schedule == "monthly":
        return cur.month != prev.month or cur.year != prev.year
    raise ValueError(f"unknown schedule {schedule!r}")


def prepare_bars(bars_by_symbol: dict[str, pd.DataFrame]) -> tuple[dict, list]:
    """Sanitize bars and precompute indicator snapshots per symbol.

    Returns ``(prep, calendar)`` — the per-symbol prep dicts and the sorted
    union trading calendar. Shared by the backtest loop and the live
    portfolio runner so lab and live decisions are built by the SAME code.
    """
    prep: dict[str, dict] = {}
    all_dates: set = set()
    for sym, df in bars_by_symbol.items():
        d = df.sort_values("date").reset_index(drop=True)
        ohlcv = _sanitize_ohlcv(
            d[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
        dates = list(d["date"])
        prep[sym] = {"dates": dates, "ohlcv": ohlcv,
                     "snaps": precompute_snapshots(ohlcv),
                     "idx": {dt: i for i, dt in enumerate(dates)},
                     "last_date": dates[-1] if dates else None,
                     "last_close": float(ohlcv["close"].iloc[-1]) if len(ohlcv) else 0.0}
        all_dates.update(dates)
    return prep, sorted(all_dates)


def _build_context(prep: dict, decision_date, fundamentals) -> RebalanceContext:
    """Snapshot the whole universe as of ``decision_date`` (inclusive).

    ``fundamentals`` is either a static ``{symbol: TickerFundamentals}`` dict
    (CAUTION: applies the same values to every date — look-ahead unless the
    run is disclosed as such) or a point-in-time source exposing
    ``asof(symbol, date)`` (e.g. data.pit_fundamentals.PITFundamentals),
    which is the honest path for fundamental strategies.
    """
    pit = getattr(fundamentals, "asof", None)
    assets: dict[str, AssetView] = {}
    for sym, p in prep.items():
        di = _resolve_index(p["dates"], decision_date)
        if di < 0:
            continue
        price = float(p["ohlcv"]["close"].iloc[di])
        if price <= 0:
            continue
        assets[sym] = AssetView(
            symbol=sym,
            price=price,
            indicators=p["snaps"][di],
            history=p["ohlcv"].iloc[: di + 1],
            fundamentals=(pit(sym, decision_date) if pit
                          else (fundamentals or {}).get(sym)),
        )
    return RebalanceContext(date=decision_date, assets=assets)


def _execute_to_target(
    weights: dict, open_px: dict, holdings: dict, cash: float,
    cost_rate: float, trades: list, dt,
    cost_model=None, cost_ctx: dict | None = None,
    rebalance_band: float = 0.0,
) -> float:
    """Trade current holdings toward the target weights at today's open.

    Returns the new cash. Sells settle before buys (cash-only, no leverage);
    buys are capped by available cash. Costs: with ``cost_model`` (and the
    per-symbol look-ahead-free ``cost_ctx`` of (adv_dollars, volatility,
    spread_frac)), every fill pays spread + impact and buys are capped at the
    participation limit; otherwise every fill pays flat ``cost_rate``.

    ``rebalance_band``: skip re-true deltas smaller than this fraction of
    equity unless they open or fully close a position — the live runner's
    dust/churn guard (its REBALANCE_BAND is 0.01). Default 0.0 trades to
    exact weights, preserving every pre-band result bit-for-bit.
    """
    equity = cash + sum(sh * open_px[s] for s, sh in holdings.items() if s in open_px)
    target: dict[str, int] = {}
    frozen: set = set()
    for s, w in weights.items():
        if w and w > 0 and s in open_px and open_px[s] > 0:
            n = int((w * equity) / open_px[s])
            if cost_model is not None:
                adv, _, _ = (cost_ctx or {}).get(s, (0.0, 0.0, 0.0))
                capped = cost_model.cap_shares(n, open_px[s], adv)
                if capped == 0 and n > 0 and holdings.get(s):
                    # the strategy WANTS this name but its liquidity collapsed
                    # below the tradeable floor: freeze the existing holding —
                    # force-dumping a position into a name we just declared
                    # untradeable would be self-contradictory
                    frozen.add(s)
                n = capped
            target[s] = n

    def _fill(s: str, side: str, shares: int) -> float:
        px = open_px[s]
        if cost_model is None:
            return px * (1 + cost_rate) if side == "BUY" else px * (1 - cost_rate)
        adv, vol, spread = (cost_ctx or {}).get(s, (0.0, 0.0, 0.0))
        return cost_model.fill_price(
            px, side, order_dollars=shares * px,
            adv_dollars=adv, volatility=vol, spread_frac=spread)

    deltas = {}
    for s in sorted(set(holdings) | set(target)):   # deterministic fill order
        if s not in open_px or s in frozen:   # untradeable today — hold as-is
            continue
        d = target.get(s, 0) - holdings.get(s, 0)
        if d == 0:
            continue
        opens_or_closes = holdings.get(s, 0) == 0 or target.get(s, 0) == 0
        if (rebalance_band > 0.0 and not opens_or_closes
                and abs(d) * open_px[s] < rebalance_band * equity):
            continue
        deltas[s] = d

    for s, d in deltas.items():        # sells first (raise cash)
        if d < 0:
            fill = _fill(s, "SELL", -d)
            cash += (-d) * fill
            holdings[s] = holdings.get(s, 0) + d
            trades.append({"date": dt, "symbol": s, "side": "SELL",
                           "shares": -d, "price": round(fill, 4)})
    for s, d in deltas.items():        # then buys (capped by cash)
        if d > 0:
            # impact is priced on the intended order; if cash then truncates
            # the executed size, the fill conservatively keeps the larger
            # order's impact (overstates cost — can never manufacture alpha)
            unit = _fill(s, "BUY", d)
            buy = min(d, int(cash / unit) if unit > 0 else 0)
            if buy > 0:
                cash -= buy * unit
                holdings[s] = holdings.get(s, 0) + buy
                trades.append({"date": dt, "symbol": s, "side": "BUY",
                               "shares": buy, "price": round(unit, 4)})

    for s in [s for s, sh in holdings.items() if sh == 0]:
        del holdings[s]
    return cash


def run_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    strategy: Strategy,
    *,
    start_cash: float = 10_000.0,
    schedule: str = "weekly",
    cost_bps: float = 2.0,
    cost_model=None,
    warmup_days: int = 200,
    trade_start: date | None = None,
    benchmark: pd.DataFrame | None = None,
    fundamentals: dict | None = None,
    log_weights: bool = False,
    rebalance_band: float = 0.0,
) -> BacktestResult:
    """Replay ``strategy`` over ``bars_by_symbol`` and return the equity curve,
    trades, and stats. ``benchmark`` is an optional (date, open, close) frame;
    its return/Sharpe/drawdown are computed from raw prices (frictionless
    buy-and-hold, return anchored at the first bar's open) — strategy stats
    carry costs and integer-share flooring, the benchmark does not. Use a
    ``start_cash`` large enough that flooring is negligible relative to the
    effect being measured.

    ``trade_start`` (a date) overrides ``warmup_days`` (a calendar index): bars
    before it feed indicators/history only — no trading AND no equity marks, so
    stats cover exactly the scored region. This is the fold-warmup discipline a
    walk-forward harness needs (a fold sliced ``[start-warmup .. end]`` scores
    ``[start .. end]``); without it, flat warmup days dilute Sharpe."""
    cost_rate = cost_bps / 1e4

    prep, calendar = prepare_bars(bars_by_symbol)
    if cost_model is not None:
        # look-ahead-free liquidity stats per symbol, aligned to bar index;
        # the loop reads index j-1 (data strictly before today) — same
        # semantics as the old lab's microcap cost mode. Dollar-ADV prefers
        # the RAW close when the caller dividend-adjusted the bars (an
        # adjusted close embeds factors for FUTURE ex-dates, so a name's
        # tradeability at t would otherwise depend on dividends paid after t).
        for sym, p in prep.items():
            o = p["ohlcv"]
            src = bars_by_symbol[sym]
            if "close_raw" in src.columns:
                # same sort prepare_bars applied, so rows align with o
                close_for_adv = (src.sort_values("date")
                                 .reset_index(drop=True)["close_raw"])
            else:
                close_for_adv = o["close"]
            dollar = close_for_adv * o["volume"]
            p["adv"] = dollar.rolling(_COST_WINDOW, min_periods=2).mean().to_numpy()
            p["vol"] = (o["close"].pct_change()
                        .rolling(_COST_WINDOW, min_periods=2).std().to_numpy())
            p["hi"] = o["high"].to_numpy()
            p["lo"] = o["low"].to_numpy()
    warm_idx = (bisect.bisect_left(calendar, trade_start)
                if trade_start is not None else warmup_days)

    holdings: dict[str, int] = {}
    cash = float(start_cash)
    equity_curve: list[tuple] = []
    trades: list[dict] = []
    weights_log: list[dict] = []
    last_price: dict[str, float] = {}

    for i, dt in enumerate(calendar):
        # today's tradable bars (+ look-ahead-free cost context when costed)
        today_open: dict[str, float] = {}
        today_close: dict[str, float] = {}
        cost_ctx: dict[str, tuple] = {}
        for sym, p in prep.items():
            j = p["idx"].get(dt)
            if j is not None:
                today_open[sym] = float(p["ohlcv"]["open"].iloc[j])
                today_close[sym] = float(p["ohlcv"]["close"].iloc[j])
                last_price[sym] = today_close[sym]
                if cost_model is not None and j >= 2:
                    adv, vol = p["adv"][j - 1], p["vol"][j - 1]
                    spread = corwin_schultz_spread(
                        p["hi"][j - 2], p["lo"][j - 2],
                        p["hi"][j - 1], p["lo"][j - 1])
                    cost_ctx[sym] = (
                        float(adv) if adv == adv else 0.0,   # NaN-safe
                        float(vol) if vol == vol else 0.0,
                        spread)

        # delist: a holding whose data has ended is closed at its last real
        # price — through the cost model when costed (a forced full-position
        # liquidation of a dying name is the most cost-hostile fill in the
        # whole backtest; charging it flat bps would flatter exactly the
        # microcap band the cost model exists for)
        for sym in list(holdings):
            if holdings[sym] and dt > prep[sym]["last_date"]:
                px = prep[sym]["last_close"]
                if cost_model is not None:
                    p = prep[sym]
                    last = len(p["dates"]) - 1
                    adv = float(p["adv"][last]) if p["adv"][last] == p["adv"][last] else 0.0
                    vol = float(p["vol"][last]) if p["vol"][last] == p["vol"][last] else 0.0
                    spread = (corwin_schultz_spread(
                        p["hi"][last - 1], p["lo"][last - 1],
                        p["hi"][last], p["lo"][last]) if last >= 1 else 0.0)
                    fill = cost_model.fill_price(
                        px, "SELL", order_dollars=holdings[sym] * px,
                        adv_dollars=adv, volatility=vol, spread_frac=spread)
                else:
                    fill = px * (1 - cost_rate)
                cash += holdings[sym] * fill
                trades.append({"date": dt, "symbol": sym, "side": "SELL",
                               "shares": holdings[sym], "price": round(fill, 4),
                               "reason": "DELISTED"})
                del holdings[sym]

        # rebalance (decision uses data through YESTERDAY; fill at today's open)
        # The first scored bar always rebalances — without this, a weekly/
        # monthly fold sits in cash until the next schedule boundary while its
        # benchmark is invested from day 1 (a measured ~-1.8pp/fold artifact).
        if i >= warm_idx and (
                i == max(warm_idx, 1) or _is_rebalance(i, calendar, schedule)):
            ctx = _build_context(prep, calendar[i - 1], fundamentals)
            weights = strategy.rebalance(ctx) or {}
            total = sum(w for w in weights.values() if w and w > 0)
            if total > 1.0:                      # never lever; scale to 100%
                weights = {s: w / total for s, w in weights.items()}
            cash = _execute_to_target(weights, today_open, holdings, cash,
                                      cost_rate, trades, dt,
                                      cost_model=cost_model, cost_ctx=cost_ctx,
                                      rebalance_band=rebalance_band)
            if log_weights:
                weights_log.append({"date": dt, "weights": dict(weights)})

        # mark to market at close (with trade_start, only the scored region)
        if trade_start is None or i >= warm_idx:
            eq = cash
            for sym, sh in holdings.items():
                eq += sh * today_close.get(sym, last_price.get(sym, 0.0))
            equity_curve.append((dt, round(eq, 2)))

    stats = _summarize(equity_curve, trades, start_cash, holdings, benchmark)
    return BacktestResult(equity_curve, trades, stats, weights_log)


def _summarize(equity_curve, trades, start_cash, holdings, benchmark) -> dict:
    eq = [e for _, e in equity_curve] or [start_cash]
    final = eq[-1]
    n = len(eq)
    stats = {
        "return_pct": round((final - start_cash) / start_cash * 100, 2) if start_cash else 0.0,
        "sharpe": round(_sharpe(eq), 2) if _sharpe(eq) is not None else None,
        "max_drawdown_pct": round(_max_drawdown(eq) * 100, 2),
        "final_equity": round(final, 2),
        "num_trades": len(trades),
        "days": n,
        "open_positions": len(holdings),
    }
    if benchmark is not None and len(benchmark) > 1:
        b = benchmark.sort_values("date")
        b = b[b["close"] > 0]
        bc = b["close"].tolist()
        if len(bc) > 1:
            # Return is anchored at the first bar's OPEN (falling back to its
            # close) because the strategy's first fill is at that open — a
            # close-anchored benchmark would skip day one's open->close move
            # that the strategy curve includes.
            first_open = float(b["open"].iloc[0]) if "open" in b.columns else 0.0
            base = first_open if first_open > 0 else bc[0]
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

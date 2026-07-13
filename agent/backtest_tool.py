"""Backtest-as-a-tool: the agent grounds its ideas in history before betting.

A small set of parametric rules the agent can compose, run through the KEPT
pure backtest engine (``edgefinder.engine.backtest.run_backtest``) with the
realistic cost model and total-return bars, benchmarked against SPY. The
agent doesn't write Python strategies — it picks a rule + symbols + window
and reads back honest return / Sharpe / drawdown / excess-vs-SPY.

Rules (``--rule``):
  buyhold:SYM         hold one symbol at 100%
  equal_weight        equal-weight every symbol, rebalanced on schedule
  momentum:K          hold the top-K symbols by trailing 6m return, equal-weight
  trend:SYM           hold SYM while its close > 200-EMA, else cash
  momo_trend:K        top-K by 6m return AMONG names above their 200-EMA
  meanrev:K           top-K most oversold (RSI) names, above-200-EMA only
  breakout:K          top-K closest to their 252-bar high with positive 3m return
  regime_momentum:K   momentum:K while SPY > its 200-EMA, else cash (needs SPY
                      in the symbol list; SPY itself is never selected)
"""

from __future__ import annotations

import json
from datetime import date

from edgefinder.engine.backtest import run_backtest
from edgefinder.engine.strategy import BuyAndHold, EqualWeight, RebalanceContext


class _Momentum:
    """Top-K by trailing 6-month return, equal-weight, refreshed each rebalance."""

    def __init__(self, k: int = 5, lookback: int = 126) -> None:
        self.k = k
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"momentum_k{self.k}"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for sym, a in ctx.assets.items():
            r = a.ret(self.lookback)
            if r is not None:
                scored.append((r, sym))
        scored.sort(reverse=True)
        winners = [s for r, s in scored[: self.k] if r > 0]
        if not winners:
            return {}
        return {s: 1.0 / len(winners) for s in winners}


class _Trend:
    """Hold one symbol while it trades above its 200-EMA, else go to cash."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()

    @property
    def name(self) -> str:
        return f"trend_{self.symbol.lower()}"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        a = ctx.get(self.symbol)
        if a and a.indicators.ema_200 and a.price > a.indicators.ema_200:
            return {self.symbol: 1.0}
        return {}


class _MomoTrend:
    """Top-K by 6m return, selected only from names above their 200-EMA —
    momentum with a trend-quality gate (skips falling knives that still
    carry stale trailing returns)."""

    def __init__(self, k: int = 5, lookback: int = 126) -> None:
        self.k = k
        self.lookback = lookback

    @property
    def name(self) -> str:
        return f"momo_trend_k{self.k}"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for sym, a in ctx.assets.items():
            e200 = a.indicators.ema_200
            if not e200 or a.price <= e200:
                continue
            r = a.ret(self.lookback)
            if r is not None and r > 0:
                scored.append((r, sym))
        scored.sort(reverse=True)
        winners = [s for _, s in scored[: self.k]]
        return {s: 1.0 / len(winners) for s in winners} if winners else {}


class _MeanRev:
    """Top-K most oversold by RSI, but only names still above their 200-EMA —
    buying dips in uptrends, never catching knives in downtrends."""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return f"meanrev_k{self.k}"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for sym, a in ctx.assets.items():
            e200, rsi = a.indicators.ema_200, a.indicators.rsi
            if not e200 or a.price <= e200 or rsi is None:
                continue
            scored.append((rsi, sym))
        scored.sort()  # lowest RSI = most oversold first
        winners = [s for _, s in scored[: self.k]]
        return {s: 1.0 / len(winners) for s in winners} if winners else {}


class _Breakout:
    """Top-K names trading closest to their 252-bar high, with positive 3m
    return — the 52-week-high effect."""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return f"breakout_k{self.k}"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        scored = []
        for sym, a in ctx.assets.items():
            c = a.history["close"]
            if len(c) < 252:
                continue
            high = float(c.iloc[-252:].max())
            r3m = a.ret(63)
            if high <= 0 or r3m is None or r3m <= 0:
                continue
            scored.append((a.price / high, sym))  # 1.0 = at the high
        scored.sort(reverse=True)
        winners = [s for _, s in scored[: self.k]]
        return {s: 1.0 / len(winners) for s in winners} if winners else {}


class _RegimeMomentum:
    """momentum:K while SPY trades above its 200-EMA, else 100% cash.
    SPY must be in the symbol list as the regime gauge; it is never held."""

    def __init__(self, k: int = 5, lookback: int = 126) -> None:
        self.k = k
        self.inner = _Momentum(k=k, lookback=lookback)

    @property
    def name(self) -> str:
        return f"regime_momentum_k{self.k}"

    def rebalance(self, ctx: RebalanceContext) -> dict[str, float]:
        spy = ctx.get("SPY")
        if not spy or not spy.indicators.ema_200 \
                or spy.price <= spy.indicators.ema_200:
            return {}
        weights = self.inner.rebalance(RebalanceContext(
            date=ctx.date,
            assets={s: a for s, a in ctx.assets.items() if s != "SPY"},
            holdings=ctx.holdings))
        return weights


def build_strategy(rule: str):
    rule = rule.strip()
    if rule.startswith("buyhold:"):
        return BuyAndHold(rule.split(":", 1)[1].upper())
    if rule == "equal_weight":
        return EqualWeight()
    if rule.startswith("momentum:"):
        return _Momentum(k=int(rule.split(":", 1)[1]))
    if rule.startswith("trend:"):
        return _Trend(rule.split(":", 1)[1])
    if rule.startswith("momo_trend:"):
        return _MomoTrend(k=int(rule.split(":", 1)[1]))
    if rule.startswith("meanrev:"):
        return _MeanRev(k=int(rule.split(":", 1)[1]))
    if rule.startswith("breakout:"):
        return _Breakout(k=int(rule.split(":", 1)[1]))
    if rule.startswith("regime_momentum:"):
        return _RegimeMomentum(k=int(rule.split(":", 1)[1]))
    raise ValueError(f"unknown rule {rule!r}")


def run_prepared(bars: dict, bench, rule: str, *, schedule: str = "monthly",
                 start: date | None = None, costed: bool = True) -> dict:
    """Backtest ``rule`` over ALREADY-LOADED bars + benchmark.

    The seam the Strategy Lab sweeps through: loading bars dominates the cost
    of a backtest, so a sweep loads each universe once and runs many rules
    against it. ``run()`` below is the single-shot convenience wrapper."""
    from edgefinder.backtest.costs import CostModel

    strategy = build_strategy(rule)
    b = bench
    if start is not None and b is not None and len(b):
        b = b[b["date"] >= start]
    result = run_backtest(
        bars, strategy,
        start_cash=100_000.0,
        schedule=schedule,
        cost_model=CostModel() if costed else None,
        cost_bps=2.0,
        trade_start=start,
        benchmark=b if b is not None and len(b) else None,
    )
    s = result.stats
    return {
        "rule": rule,
        "symbols": sorted(bars),
        "schedule": schedule,
        "start": str(start) if start else None,
        "costed": costed,
        "return_pct": s.get("return_pct"),
        "sharpe": s.get("sharpe"),
        "max_drawdown_pct": s.get("max_drawdown_pct"),
        "benchmark_return_pct": s.get("benchmark_return_pct"),
        "excess_return_pct": s.get("excess_return_pct"),
        "num_trades": s.get("num_trades"),
        "days": s.get("days"),
        "final_equity": s.get("final_equity"),
    }


def run(symbols: list[str], rule: str, *, schedule: str = "monthly",
        start: date | None = None, end: date | None = None,
        costed: bool = True, div_adjust: bool = True,
        source: str = "auto") -> dict:
    """Backtest ``rule`` over ``symbols`` and return an honest scorecard dict."""
    from agent.data import load_bars, spy_series_df

    bars = load_bars(symbols, start=None, end=end, div_adjust=div_adjust, source=source)
    bars = {s: df for s, df in bars.items() if df is not None and len(df) > 210}
    if not bars:
        return {"error": "no bars with enough history for the requested symbols"}

    bench = spy_series_df()
    out = run_prepared(bars, bench, rule, schedule=schedule, start=start,
                       costed=costed)
    out["end"] = str(end) if end else None
    out["div_adjust"] = div_adjust
    return out


def _parse_date(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", required=True, help="comma-separated tickers")
    p.add_argument("--rule", required=True, help="buyhold:SYM | equal_weight | momentum:K | trend:SYM")
    p.add_argument("--schedule", default="monthly", choices=["daily", "weekly", "monthly"])
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--no-costs", action="store_true")
    p.add_argument("--no-div", action="store_true")
    p.add_argument("--source", default="auto", choices=["auto", "r2", "db"])
    p.add_argument("--save", action="store_true", help="persist to desk_backtests")
    p.add_argument("--run-id", default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    out = run(symbols, args.rule, schedule=args.schedule,
              start=_parse_date(args.start), end=_parse_date(args.end),
              costed=not args.no_costs, div_adjust=not args.no_div,
              source=args.source)
    if args.save and "error" not in out:
        from agent.ledger import save_backtest
        save_backtest(args.label or f"{args.rule} [{args.symbols}]", out,
                      run_id=args.run_id)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

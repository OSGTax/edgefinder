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
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

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
    raise ValueError(f"unknown rule {rule!r}")


def run(symbols: list[str], rule: str, *, schedule: str = "monthly",
        start: date | None = None, end: date | None = None,
        costed: bool = True, div_adjust: bool = True,
        source: str = "auto") -> dict:
    """Backtest ``rule`` over ``symbols`` and return an honest scorecard dict."""
    from agent.data import load_bars, session_factory
    from edgefinder.backtest.costs import CostModel
    from edgefinder.engine.data import spy_series

    bars = load_bars(symbols, start=None, end=end, div_adjust=div_adjust, source=source)
    bars = {s: df for s, df in bars.items() if df is not None and len(df) > 210}
    if not bars:
        return {"error": "no bars with enough history for the requested symbols"}

    strategy = build_strategy(rule)

    sess = session_factory()()
    try:
        bench = spy_series(sess)
    finally:
        sess.close()
    if start is not None and bench is not None and len(bench):
        bench = bench[bench["date"] >= start]

    result = run_backtest(
        bars, strategy,
        start_cash=100_000.0,
        schedule=schedule,
        cost_model=CostModel() if costed else None,
        cost_bps=2.0,
        trade_start=start,
        benchmark=bench if bench is not None and len(bench) else None,
    )
    s = result.stats
    return {
        "rule": rule,
        "symbols": sorted(bars),
        "schedule": schedule,
        "start": str(start) if start else None,
        "end": str(end) if end else None,
        "costed": costed,
        "div_adjust": div_adjust,
        "return_pct": s.get("return_pct"),
        "sharpe": s.get("sharpe"),
        "max_drawdown_pct": s.get("max_drawdown_pct"),
        "benchmark_return_pct": s.get("benchmark_return_pct"),
        "excess_return_pct": s.get("excess_return_pct"),
        "num_trades": s.get("num_trades"),
        "days": s.get("days"),
        "final_equity": s.get("final_equity"),
    }


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

"""CLI: walk-forward validation on the portfolio engine.

The committed, reproducible runner for engine-v2 validation (the Phase-1
headline numbers were ad-hoc REPL runs — this replaces that). Examples:

    # the 21-year ETF lane, holdout reserved (sealed, not burned)
    python -m edgefinder.engine.validate --strategy equal_weight \
        --symbols SPY,QQQ,IWM,DIA,GLD,TLT,EFA --schedule monthly \
        --holdout-days 126 --no-holdout-eval --record

    # null control: buy-and-hold SPY must NOT pass (harness honesty check)
    python -m edgefinder.engine.validate --strategy buy_and_hold:SPY \
        --symbols SPY --holdout-days 126 --no-holdout-eval
"""

from __future__ import annotations

import argparse
import json
import logging

from edgefinder.db.engine import get_engine, get_session_factory
from edgefinder.engine.data import load_bars, spy_series
from edgefinder.engine.record import current_git_sha, record_validation_run
from edgefinder.engine.strategies import DualMomentum, TrendTimer
from edgefinder.engine.strategy import BuyAndHold, EqualWeight
from edgefinder.engine.walkforward import run_walkforward

logger = logging.getLogger(__name__)


def make_strategy_factory(spec: str):
    """Strategy spec -> a fresh-instance factory.

    Specs: ``equal_weight`` | ``buy_and_hold:SYM`` | ``trend_timer:SYM`` |
    ``dual_momentum`` (pre-registered 7-ETF menu, top_k=3).
    """
    if spec == "equal_weight":
        return EqualWeight
    if spec == "dual_momentum":
        return DualMomentum
    if spec.startswith("buy_and_hold:"):
        sym = spec.split(":", 1)[1].upper()
        return lambda: BuyAndHold(sym)
    if spec.startswith("trend_timer:"):
        sym = spec.split(":", 1)[1].upper()
        return lambda: TrendTimer(sym)
    raise SystemExit(
        f"unknown strategy spec {spec!r} (use equal_weight, dual_momentum, "
        "buy_and_hold:SYM, or trend_timer:SYM)")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True,
                   help="equal_weight | buy_and_hold:SYM")
    p.add_argument("--symbols", required=True,
                   help="comma-separated universe, e.g. SPY,QQQ,IWM")
    p.add_argument("--schedule", default="monthly",
                   choices=["daily", "weekly", "monthly"])
    p.add_argument("--is-days", type=int, default=378)
    p.add_argument("--oos-days", type=int, default=126)
    p.add_argument("--step-days", type=int, default=126)
    p.add_argument("--holdout-days", type=int, default=0)
    p.add_argument("--no-holdout-eval", action="store_true",
                   help="reserve the holdout without burning the one look")
    p.add_argument("--warmup-days", type=int, default=210)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--start-cash", type=float, default=10_000.0)
    p.add_argument("--total-return", action="store_true",
                   help="score on the total-return bar instead of risk-adjusted")
    p.add_argument("--record", action="store_true",
                   help="persist the scorecard to validation_runs")
    p.add_argument("--universe-label", default=None,
                   help="label stored with the run (default: <n>syms+v2)")
    args = p.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    factory = make_strategy_factory(args.strategy)

    session = get_session_factory(get_engine())()
    try:
        bars = load_bars(session, symbols)
        spy = spy_series(session)
    finally:
        session.close()   # never hold a pooler connection through the folds
    missing = [s for s in symbols if s not in bars]
    if missing:
        raise SystemExit(f"no bars for: {', '.join(missing)}")

    scorecard = run_walkforward(
        bars, factory,
        spy_bars=spy,
        is_days=args.is_days, oos_days=args.oos_days, step_days=args.step_days,
        holdout_days=args.holdout_days, holdout_eval=not args.no_holdout_eval,
        warmup_days=args.warmup_days, start_cash=args.start_cash,
        schedule=args.schedule, cost_bps=args.cost_bps,
        risk_adjusted=not args.total_return,
    )
    print(json.dumps(scorecard, indent=2, default=str))

    if args.record:
        universe = args.universe_label or f"{len(symbols)}syms+v2"
        try:
            session = get_session_factory(get_engine())()
            try:
                run_id = record_validation_run(
                    session, scorecard, universe=universe,
                    git_sha=current_git_sha())
            finally:
                session.close()
            print(f"\nrecorded validation_runs id={run_id} universe={universe}")
        except Exception:   # recording must never void a completed run
            logger.exception("failed to record validation run")
    return scorecard


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

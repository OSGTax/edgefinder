"""CLI: walk-forward validation on the portfolio engine.

The committed, reproducible runner for engine-v2 validation (the Phase-1
headline numbers were ad-hoc REPL runs — this replaces that). Examples:

    # the 21-year ETF lane — the holdout is RESERVED (sealed) by default;
    # burning the one look requires the explicit --burn-holdout flag
    python -m edgefinder.engine.validate --strategy equal_weight \
        --symbols SPY,QQQ,IWM,DIA,GLD,TLT,EFA --schedule monthly \
        --holdout-days 126 --record

    # null control: buy-and-hold SPY must NOT pass (harness honesty check)
    python -m edgefinder.engine.validate --strategy buy_and_hold:SPY \
        --symbols SPY --holdout-days 126
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

from edgefinder.db.engine import get_engine, get_session_factory
from edgefinder.engine.data import (
    adjust_for_dividends,
    load_bars,
    load_dividends,
    spy_series,
)
from edgefinder.engine.record import current_git_sha, record_validation_run
from edgefinder.engine.strategies import make_strategy_factory
from edgefinder.engine.walkforward import run_walkforward

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True,
                   help="equal_weight | buy_and_hold:SYM")
    p.add_argument("--symbols", default=None,
                   help="comma-separated FIXED universe, e.g. SPY,QQQ,IWM")
    p.add_argument("--universe", default=None, metavar="top:N[+OFFSET]",
                   help="POINT-IN-TIME universe: per fold, the top N by "
                        "dollar volume ranked using only data through the day "
                        "before that fold's first scored day (e.g. top:500 or "
                        "top:2000+1000 for the rank 1000-3000 band)")
    p.add_argument("--start", default=None,
                   help="clip history to dates >= this (YYYY-MM-DD); use "
                        "2021-06-01 for stock universes (bars start there)")
    p.add_argument("--costed", action="store_true",
                   help="realistic costs (Corwin-Schultz spread + sqrt impact "
                        "+ participation caps; FIXED params, never optimized) "
                        "instead of flat --cost-bps")
    p.add_argument("--div-adjust", action="store_true",
                   help="total-return prices: back-adjust all bars AND the "
                        "SPY benchmark for cash dividends")
    p.add_argument("--schedule", default="monthly",
                   choices=["daily", "weekly", "monthly"])
    p.add_argument("--is-days", type=int, default=378)
    p.add_argument("--oos-days", type=int, default=126)
    p.add_argument("--step-days", type=int, default=126)
    p.add_argument("--holdout-days", type=int, default=0)
    p.add_argument("--holdout-start", default=None,
                   help="pin the sealed holdout to a fixed DATE (YYYY-MM-DD) "
                        "instead of a rolling last-N-days count — required "
                        "discipline for any multi-run research program")
    p.add_argument("--burn-holdout", action="store_true",
                   help="EVALUATE the sealed holdout — spends the one "
                        "look-per-round; without this flag the holdout is "
                        "reserved (carved off, never scored)")
    p.add_argument("--warmup-days", type=int, default=210)
    p.add_argument("--cost-bps", type=float, default=2.0)
    p.add_argument("--start-cash", type=float, default=1_000_000.0)
    p.add_argument("--total-return", action="store_true",
                   help="score on the total-return bar instead of risk-adjusted")
    p.add_argument("--record", action="store_true",
                   help="persist the scorecard to validation_runs")
    p.add_argument("--universe-label", default=None,
                   help="label stored with the run (default: <n>syms+v2)")
    p.add_argument("--bars-from", default="db", choices=["db", "r2"],
                   help="bar source: the DB (default) or the verified R2 store")
    args = p.parse_args(argv)

    if bool(args.symbols) == bool(args.universe):
        raise SystemExit("exactly one of --symbols or --universe is required")
    if args.universe and not args.start:
        raise SystemExit("--universe requires --start (stock bars begin "
                         "2021-06; without a start the SPY calendar would "
                         "plan years of empty pre-2021 universes)")
    if args.universe and args.bars_from != "db":
        raise SystemExit("--bars-from r2 is not supported with --universe yet")
    try:
        factory = make_strategy_factory(args.strategy)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    start = date.fromisoformat(args.start) if args.start else None
    holdout_start = (date.fromisoformat(args.holdout_start)
                     if args.holdout_start else None)
    universe_fn = None

    if args.universe:
        # POINT-IN-TIME universe: plan the fold geometry on SPY's calendar
        # (the NYSE trading calendar), resolve each window's top-N as of the
        # trading day BEFORE its first scored day, then load the union once.
        import re

        from edgefinder.backtest.jobs import resolve_universe
        from edgefinder.engine.walkforward import plan_folds

        m = re.fullmatch(r"top:(\d+)(?:\+(\d+))?", args.universe)
        if not m:
            raise SystemExit(f"bad --universe {args.universe!r} (use top:N[+OFFSET])")
        top_n, rank_offset = int(m.group(1)), int(m.group(2) or 0)

        session = get_session_factory(get_engine())()
        try:
            spy = spy_series(session)
            if start is not None:
                spy = spy[spy["date"] >= start].reset_index(drop=True)
            days = list(spy["date"])
            folds, holdout = plan_folds(
                days, is_days=args.is_days, oos_days=args.oos_days,
                step_days=args.step_days, holdout_days=args.holdout_days,
                holdout_start=holdout_start)
            windows = folds + ([holdout] if holdout else [])
            per_window: dict = {}
            for w_start, _ in windows:
                as_of = days[max(0, days.index(w_start) - 1)]
                per_window[w_start] = resolve_universe(
                    session, "top", [], top_n, as_of=as_of,
                    rank_offset=rank_offset)
            union = sorted(set().union(*per_window.values()))
            print(f"PIT universe top:{top_n}"
                  f"{f'+{rank_offset}' if rank_offset else ''}: "
                  f"{len(windows)} windows, union {len(union)} symbols")
            bars = load_bars(session, union, start=start)
            divs = (load_dividends(session, union + ["SPY"])
                    if args.div_adjust else None)
        finally:
            session.close()
        universe_fn = per_window.get
        symbols = union
        planning_calendar = days
    else:
        planning_calendar = None
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.bars_from == "r2":
            from edgefinder.engine.data import (
                adjust_for_splits,
                load_bars_from_store,
                load_splits,
            )

            bars = load_bars_from_store(symbols + ["SPY"], start=start)
            # the R2 store mirrors raw as-traded bars; apply the same
            # split adjustment the DB loader applies
            session = get_session_factory(get_engine())()
            try:
                bars = adjust_for_splits(
                    bars, load_splits(session, list(bars)))
                divs = (load_dividends(session, symbols + ["SPY"])
                        if args.div_adjust else None)
            finally:
                session.close()
            spy = bars.get("SPY")
            bars = {s: df for s, df in bars.items() if s in symbols}
            if spy is None:
                raise SystemExit("SPY not in the R2 store — run barstore sync first")
        else:
            session = get_session_factory(get_engine())()
            try:
                bars = load_bars(session, symbols, start=start)
                spy = spy_series(session)
                if start is not None:
                    spy = spy[spy["date"] >= start].reset_index(drop=True)
                divs = (load_dividends(session, symbols + ["SPY"])
                        if args.div_adjust else None)
            finally:
                session.close()   # never hold a pooler connection through folds
        missing = [s for s in symbols if s not in bars]
        if missing:
            raise SystemExit(f"no bars for: {', '.join(missing)}")

    prices_label = "split-adjusted, dividend-unadjusted"
    div_coverage = None
    if args.div_adjust:
        # dividends are declared per share AS OF their ex-date; with bars on
        # the split-adjusted basis, cash amounts must be scaled by splits
        # executing after each ex-date or pre-split yields read N-times high
        from edgefinder.engine.data import adjust_dividends_for_splits, load_splits

        session = get_session_factory(get_engine())()
        try:
            divs = adjust_dividends_for_splits(
                divs or {}, load_splits(session, list((divs or {}).keys())))
        finally:
            session.close()
        # the one anti-conservative failure mode of TR adjustment is an
        # adjusted strategy scored against an unadjusted benchmark — refuse
        # to run if SPY's dividends are missing
        if not (divs or {}).get("SPY"):
            raise SystemExit("--div-adjust: no SPY dividends in the dividends "
                             "table — run dividends_backfill first")
        div_coverage = {
            "symbols_with_dividends": sum(1 for s in symbols if (divs or {}).get(s)),
            "universe_size": len(symbols),
        }
        # adjust bars and benchmark in separate calls — the traded universe
        # may itself contain SPY, and a merged dict would clobber it
        bars = adjust_for_dividends(bars, divs or {})
        spy = adjust_for_dividends({"SPY": spy}, divs or {})["SPY"]
        prices_label = "split+dividend-adjusted (total return)"

    cost_model = None
    if args.costed:
        from edgefinder.backtest.costs import CostModel

        cost_model = CostModel()   # FIXED params — never optimized

    scorecard = run_walkforward(
        bars, factory,
        spy_bars=spy,
        is_days=args.is_days, oos_days=args.oos_days, step_days=args.step_days,
        holdout_days=args.holdout_days,
        holdout_start=holdout_start,
        holdout_eval=args.burn_holdout,
        warmup_days=args.warmup_days, start_cash=args.start_cash,
        schedule=args.schedule, cost_bps=args.cost_bps,
        risk_adjusted=not args.total_return,
        universe_fn=universe_fn,
        cost_model=cost_model,
        prices_label=prices_label,
        calendar=planning_calendar,
    )
    if div_coverage is not None:
        scorecard["config"]["dividend_coverage"] = div_coverage
    if args.universe:
        scorecard["config"]["universe_sizes"] = {
            str(k): len(v) for k, v in per_window.items()}
    print(json.dumps(scorecard, indent=2, default=str))

    if args.record:
        universe = args.universe_label or (
            f"{args.universe}@pit+v2" if args.universe
            else f"{len(symbols)}syms+v2")
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

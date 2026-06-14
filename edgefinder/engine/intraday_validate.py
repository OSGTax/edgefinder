"""CLI: walk-forward validation on the INTRADAY (minute-bar) engine.

The committed, reproducible runner for intraday strategy ideas — mirrors
engine/validate.py. Minute bars come from the verified R2 MinuteStore
(``--bars-from r2``); SPY is the benchmark. The sealed-holdout discipline is
the same as the daily lane: pin ONE ``--holdout-start`` for a research round and
never evaluate it without ``--burn-holdout``.

Examples:

    # real-data smoke (mean reversion on a few liquid names, costed)
    python -m edgefinder.engine.intraday_validate \
        --strategy mean_rev:AAPL --symbols AAPL --bars-from r2 \
        --start 2024-01-01 --costed --holdout-start 2026-06-11

    # the whole pilot menu, flat null control (must NOT pass)
    python -m edgefinder.engine.intraday_validate --strategy flat \
        --menu intraday/menu.json --bars-from r2 --start 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date

logger = logging.getLogger(__name__)


def make_intraday_factory(spec: str):
    """Intraday strategy spec -> fresh-instance factory.

    Specs: ``flat`` | ``buy_hold_open:SYM`` | ``mean_rev:SYM[:lookback:z]``.
    """
    from edgefinder.engine.intraday_strategy import (
        BuyHoldFromOpen,
        IntradayFlat,
        IntradayMeanReversion,
    )

    if spec == "flat":
        return IntradayFlat
    if spec.startswith("buy_hold_open:"):
        sym = spec.split(":", 1)[1].upper()
        return lambda: BuyHoldFromOpen(sym)
    if spec.startswith("mean_rev:"):
        parts = spec.split(":")
        sym = parts[1].upper()
        lookback = int(parts[2]) if len(parts) > 2 else 20
        z = float(parts[3]) if len(parts) > 3 else 1.0
        return lambda: IntradayMeanReversion(sym, lookback, z)

    from edgefinder.engine.intraday_roster import INTRADAY_R1_SPECS

    if spec in INTRADAY_R1_SPECS:
        return INTRADAY_R1_SPECS[spec]
    raise ValueError(
        f"unknown intraday strategy spec {spec!r} (use flat, buy_hold_open:SYM, "
        "mean_rev:SYM[:lookback:z], or a pre-registered roster spec from "
        f"intraday_roster.INTRADAY_R1_SPECS: {', '.join(sorted(INTRADAY_R1_SPECS))})")


def _menu_symbols(path: str) -> list[str]:
    with open(path) as f:
        menu = json.load(f)
    syms = list(menu.get("symbols") or [])
    if not syms:
        raise SystemExit(f"{path}: menu has no symbols yet (resolve it first)")
    return syms


def _split_frames(frames: dict, symbols: list[str]) -> tuple[dict, "object"]:
    """Split loaded minute frames into (tradable bars, SPY benchmark frame).

    SPY is always loaded for the benchmark, but it is ALSO a legitimate
    tradable symbol — so we must NOT remove it from the tradable set when
    splitting. (Popping SPY out for the benchmark produced a false
    "no minute bars for SPY" whenever SPY itself was traded — caught by the
    first real-data smoke, 2026-06-14.) Returns the bars dict restricted to
    the requested symbols and the SPY frame (None if absent)."""
    want = set(symbols)
    spy = frames.get("SPY")
    bars = {s: df for s, df in frames.items() if s in want}
    return bars, spy


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True,
                   help="flat | buy_hold_open:SYM | mean_rev:SYM[:lookback:z]")
    p.add_argument("--symbols", default=None,
                   help="comma-separated FIXED universe")
    p.add_argument("--menu", default=None,
                   help="path to intraday menu JSON (e.g. intraday/menu.json)")
    p.add_argument("--start", default=None, help="clip to dates >= this (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="clip to dates <= this (YYYY-MM-DD)")
    p.add_argument("--bars-from", default="r2", choices=["r2"],
                   help="minute-bar source (R2 MinuteStore — the only path)")
    p.add_argument("--costed", action="store_true",
                   help="the FIXED intraday cost model (spread + impact + caps)")
    p.add_argument("--cost-bps", type=float, default=2.0)
    fc = p.add_mutually_exclusive_group()
    fc.add_argument("--flatten-at-close", dest="flatten", action="store_true",
                    default=True, help="(default) flatten all positions at the close")
    fc.add_argument("--hold-overnight", dest="flatten", action="store_false",
                    help="carry positions across sessions (overnight gap marked in)")
    p.add_argument("--decision-interval", type=int, default=1,
                   help="decide every Nth bar (1 = every bar)")
    p.add_argument("--bar-seconds", type=int, default=60)
    p.add_argument("--is-days", type=int, default=60)
    p.add_argument("--oos-days", type=int, default=21)
    p.add_argument("--step-days", type=int, default=21)
    p.add_argument("--holdout-days", type=int, default=0)
    p.add_argument("--holdout-start", default=None,
                   help="pin the sealed holdout to a fixed DATE (YYYY-MM-DD)")
    p.add_argument("--burn-holdout", action="store_true",
                   help="EVALUATE the sealed holdout (spends the one look)")
    p.add_argument("--warmup-days", type=int, default=5)
    p.add_argument("--start-cash", type=float, default=1_000_000.0)
    p.add_argument("--total-return", action="store_true",
                   help="score on the total-return bar instead of risk-adjusted")
    p.add_argument("--record", action="store_true",
                   help="persist the scorecard to validation_runs")
    p.add_argument("--universe-label", default=None,
                   help="label stored with the run (default: intraday:<n>syms)")
    args = p.parse_args(argv)

    if bool(args.symbols) == bool(args.menu):
        raise SystemExit("exactly one of --symbols or --menu is required")
    try:
        factory = make_intraday_factory(args.strategy)
    except ValueError as e:
        raise SystemExit(str(e)) from None

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else _menu_symbols(args.menu))
    start = date.fromisoformat(args.start) if args.start else date(2000, 1, 1)
    end = date.fromisoformat(args.end) if args.end else date.today()
    holdout_start = (date.fromisoformat(args.holdout_start)
                     if args.holdout_start else None)

    from edgefinder.data.minutestore import MinuteStore

    store = MinuteStore()
    load = sorted(set(symbols) | {"SPY"})
    frames = store.load_minute_bars(load, start, end)
    bars, spy = _split_frames(frames, symbols)
    if spy is None:
        raise SystemExit("SPY not in the minute store — backfill it first")
    missing = [s for s in symbols if s not in bars]
    if missing:
        raise SystemExit(f"no minute bars for: {', '.join(missing)}")

    cost_model = None
    if args.costed:
        from edgefinder.backtest.costs import CostModel

        cost_model = CostModel()   # FIXED params — never optimized

    from edgefinder.engine.intraday_walkforward import run_intraday_walkforward

    scorecard = run_intraday_walkforward(
        bars, factory,
        spy_bars=spy,
        is_days=args.is_days, oos_days=args.oos_days, step_days=args.step_days,
        holdout_days=args.holdout_days, holdout_start=holdout_start,
        holdout_eval=args.burn_holdout, warmup_days=args.warmup_days,
        start_cash=args.start_cash, cost_model=cost_model, cost_bps=args.cost_bps,
        flatten_at_close=args.flatten, decision_interval=args.decision_interval,
        bar_seconds=args.bar_seconds, risk_adjusted=not args.total_return)
    print(json.dumps(scorecard, indent=2, default=str))

    if args.record:
        from edgefinder.db.engine import get_engine, get_session_factory
        from edgefinder.engine.record import current_git_sha, record_validation_run

        universe = args.universe_label or f"intraday:{len(symbols)}syms"
        for attempt in range(1, 5):
            try:
                session = get_session_factory(get_engine())()
                try:
                    run_id = record_validation_run(
                        session, scorecard, universe=universe,
                        git_sha=current_git_sha())
                finally:
                    session.close()
                print(f"\nrecorded validation_runs id={run_id} universe={universe}")
                break
            except Exception:
                logger.exception(
                    "failed to record validation run (attempt %d/4)", attempt)
                if attempt == 4:
                    raise SystemExit(3)
                time.sleep(20 * attempt)
    return scorecard


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""CLI: cheap strategy screen — fixed params, dev window, no optimization.

Stage 1 of the research protocol: before any walk-forward optimization, a
candidate runs once with default (or explicitly given) params over the
development window — everything EXCEPT the sealed final ``--holdout-days``
trading days, which stay untouched for the lab's final test. Dead ideas die
here in ~a minute instead of burning a 12-minute walk-forward (or worse,
a look at the holdout).

Examples:
    python -m edgefinder.backtest.screen --strategy pullback --top-n 300
    python -m edgefinder.backtest.screen --strategy breakout --params '{"channel_days": 20}'
"""

from __future__ import annotations

import argparse
import json
import logging

from edgefinder.backtest.daily_backtest import run_daily_backtest
from edgefinder.backtest.jobs import _load_bars, resolve_universe
from edgefinder.backtest.validate import _spy_bars
from edgefinder.backtest.walkforward import _all_dates, _benchmark_window, _slice
from edgefinder.core.logging_config import configure_logging
from edgefinder.db.engine import get_engine, get_session_factory

logger = logging.getLogger(__name__)


def screen(strategy: str, *, mode: str = "top", top_n: int = 300,
           holdout_days: int = 126, params: dict | None = None,
           starting_cash: float = 10_000.0, universe_as_of=None) -> dict:
    """One fixed-config backtest over the dev window. Returns the stats dict."""
    db = get_session_factory(get_engine())()
    try:
        universe = resolve_universe(db, mode, [], top_n, as_of=universe_as_of)
        bars, _, _ = _load_bars(db, universe, None, None)
        if not bars:
            raise SystemExit("no daily_bars for that universe — run the backfill first")
        spy = _spy_bars(db)
    finally:
        db.close()

    days = _all_dates(bars)
    if len(days) <= holdout_days + 60:
        raise SystemExit("not enough history for a dev window")
    dev_end = days[len(days) - holdout_days - 1]
    dev_start = days[0]
    dev_bars = _slice(bars, dev_start, dev_end)
    benchmark = _benchmark_window(spy, dev_start, dev_end)

    logger.info(
        "Screening %s over %d symbols, dev window %s..%s (sealed holdout untouched)",
        strategy, len(dev_bars), dev_start, dev_end,
    )
    res = run_daily_backtest(
        strategy, dev_bars, starting_cash=starting_cash,
        benchmark=benchmark, params=params or {}, spy_bars=spy,
    )
    return res["stats"] | {
        "strategy": strategy,
        "dev_window": f"{dev_start}..{dev_end}",
        "params": params or {},
        "universe_as_of": str(universe_as_of) if universe_as_of else None,
    }


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description="Cheap fixed-config strategy screen")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--mode", default="top", choices=["top", "full"])
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--holdout-days", type=int, default=126,
                    help="sealed trading days excluded from the dev window")
    ap.add_argument("--params", default="", help="JSON param overrides")
    ap.add_argument("--universe-as-of", default=None,
                    help="YYYY-MM-DD: point-in-time universe ranking cut")
    args = ap.parse_args()

    params = json.loads(args.params) if args.params else None
    from datetime import date as _date
    as_of = _date.fromisoformat(args.universe_as_of) if args.universe_as_of else None
    stats = screen(args.strategy, mode=args.mode, top_n=args.top_n,
                   holdout_days=args.holdout_days, params=params,
                   universe_as_of=as_of)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()

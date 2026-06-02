"""CLI: walk-forward out-of-sample validation against the daily_bars table.

Examples:
    python -m edgefinder.backtest.validate --strategy coward --mode top --top-n 100
    python -m edgefinder.backtest.validate --all --mode top --top-n 200 --write

Loads bars from the persisted ``daily_bars`` table (reusing the universe/bar
loaders that power the backtest jobs), runs the walk-forward harness, prints
the OOS scorecard, and optionally writes a ``reviews/validation-<strategy>-
<date>.md`` report (mirroring the coach's review artifacts).

Needs DATABASE_URL (Supabase pooler). Designed to run from a dev box, a
Codespace, or a GitHub Actions job — not from the idle web service.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date

import pandas as pd

from edgefinder.backtest.jobs import _load_bars, resolve_universe
from edgefinder.backtest.walkforward import run_walkforward
from edgefinder.core.logging_config import configure_logging
from edgefinder.db.engine import get_engine, get_session_factory
from edgefinder.db.models import DailyBar

logger = logging.getLogger(__name__)

ALL_STRATEGIES = ["coward", "gambler", "degenerate"]


def _spy_bars(db) -> pd.DataFrame:
    rows = (
        db.query(DailyBar.date, DailyBar.open, DailyBar.high,
                 DailyBar.low, DailyBar.close, DailyBar.volume)
        .filter(DailyBar.symbol == "SPY").order_by(DailyBar.date).all()
    )
    return pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume"]
    )


def _format_report(scorecard: dict) -> str:
    s = scorecard["oos"]
    cfg = scorecard["config"]
    lines = [
        f"# Walk-forward validation — {scorecard['strategy']}  ({date.today()})",
        "",
        f"**Verdict: {scorecard['verdict']}**  "
        f"(bar: positive OOS return, beats SPY on average and in a majority of folds)",
        "",
        f"- Folds: {cfg['num_folds']}  (IS {cfg['is_days']}d / OOS {cfg['oos_days']}d, "
        f"step {cfg['step_days']}d)",
        f"- OOS compounded return: {s['total_return_pct']}%",
        f"- Mean OOS Sharpe: {s['mean_sharpe']}",
        f"- Mean excess vs SPY: {s['mean_excess_vs_spy_pct']}%  "
        f"(folds beating SPY: {s['folds_beating_spy']})",
        f"- Total OOS trades: {s['total_trades']}  | mean win rate: {s['mean_win_rate']}%",
        f"- Worst fold max drawdown: {s['worst_max_drawdown_pct']}%",
        "",
        "## By regime (OOS avg return %)",
    ]
    for regime, info in scorecard["by_regime"].items():
        lines.append(f"- {regime}: {info['avg_return_pct']}% over {info['folds']} fold(s)")
    lines += ["", "## Per-fold", ""]
    for f in scorecard["folds"]:
        st = f["stats"]
        lines.append(
            f"- {f['oos']} [{f['regime']}]: return {st.get('return_pct')}%, "
            f"sharpe {st.get('sharpe')}, trades {st.get('num_closed_trades')}, "
            f"params {json.dumps(f['params'])}"
        )
    return "\n".join(lines) + "\n"


def run(strategy: str, *, mode: str, top_n: int, symbols: list[str],
        search_iters: int, write: bool) -> dict:
    engine = get_engine()
    session_factory = get_session_factory(engine)
    db = session_factory()
    try:
        universe = resolve_universe(db, mode, symbols, top_n)
        bars, _, _ = _load_bars(db, universe, None, None)
        if not bars:
            raise SystemExit("no daily_bars for that universe — run the backfill first")
        spy = _spy_bars(db)
    finally:
        db.close()

    logger.info("Validating %s over %d symbols...", strategy, len(bars))
    scorecard = run_walkforward(
        strategy, bars, spy_bars=spy, search_iters=search_iters,
        progress_cb=lambda i: logger.info("fold %s %s", i.get("fold"), i.get("oos")),
    )
    print(json.dumps(scorecard["oos"] | {"verdict": scorecard["verdict"]}, indent=2))

    if write:
        import os
        os.makedirs("reviews", exist_ok=True)
        path = f"reviews/validation-{strategy}-{date.today()}.md"
        with open(path, "w") as fh:
            fh.write(_format_report(scorecard))
        logger.info("Wrote %s", path)
    return scorecard


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description="Walk-forward OOS validation")
    ap.add_argument("--strategy", choices=ALL_STRATEGIES)
    ap.add_argument("--all", action="store_true", help="validate every strategy")
    ap.add_argument("--mode", default="top", choices=["symbols", "top", "full"])
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--symbols", default="", help="comma-separated (mode=symbols)")
    ap.add_argument("--search-iters", type=int, default=40)
    ap.add_argument("--write", action="store_true", help="write reviews/ report")
    args = ap.parse_args()

    if not args.all and not args.strategy:
        ap.error("pass --strategy NAME or --all")

    targets = ALL_STRATEGIES if args.all else [args.strategy]
    symbols = [s for s in args.symbols.split(",") if s.strip()]
    for strat in targets:
        run(strat, mode=args.mode, top_n=args.top_n, symbols=symbols,
            search_iters=args.search_iters, write=args.write)


if __name__ == "__main__":
    main()

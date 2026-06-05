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
from edgefinder.db.models import DailyBar, ValidationRun

logger = logging.getLogger(__name__)

# All registered strategies (live + research candidates) — the lab validates
# anything in the registry; live trading is gated separately by
# settings.live_strategies.
import edgefinder.strategies  # noqa: F401,E402 — import populates the registry
from edgefinder.strategies.base import StrategyRegistry  # noqa: E402

ALL_STRATEGIES = sorted(StrategyRegistry.get_all().keys())


def record_validation_run(
    session, scorecard: dict, *, universe: str, git_sha: str | None = None
) -> int:
    """Persist a walk-forward scorecard to validation_runs (offline evidence).

    The dashboard reads the latest row per strategy to show the offline
    verdict beside the live scorecard.
    """
    row = ValidationRun(
        strategy_name=scorecard["strategy"],
        git_sha=git_sha,
        universe=universe,
        config=scorecard.get("config"),
        oos=scorecard.get("oos"),
        criteria=scorecard.get("criteria"),
        holdout=scorecard.get("holdout"),
        verdict=scorecard.get("verdict", "FAIL"),
    )
    session.add(row)
    session.commit()
    return row.id


def _current_git_sha() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def _spy_bars(db) -> pd.DataFrame:
    """Longest available SPY close series for the benchmark/regime tagging.

    Unions daily_bars with index_daily (daily_bars wins on overlap) because
    SPY coverage is currently split across both tables. Only date+close are
    needed downstream, so OHLC are filled with the close.
    """
    from edgefinder.db.models import IndexDaily

    def _to_date(x):
        return x.date() if hasattr(x, "date") else x

    by_date: dict = {}
    for d, c in (db.query(IndexDaily.date, IndexDaily.close)
                 .filter(IndexDaily.symbol == "SPY").all()):
        if c:
            by_date[_to_date(d)] = float(c)
    for d, c in (db.query(DailyBar.date, DailyBar.close)
                 .filter(DailyBar.symbol == "SPY").all()):
        if c:
            by_date[_to_date(d)] = float(c)  # daily_bars wins on overlap

    cols = ["date", "open", "high", "low", "close", "volume"]
    if not by_date:
        return pd.DataFrame(columns=cols)
    items = sorted(by_date.items())
    closes = [c for _, c in items]
    return pd.DataFrame({
        "date": [d for d, _ in items],
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [0.0] * len(items),
    })


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
    ]
    c = scorecard.get("criteria")
    if c:
        lines += [
            "",
            "## Bar: positive OOS Sharpe AND beats SPY in a majority of folds "
            f"AND >= {c['min_trades_threshold']} trades",
            f"- OOS Sharpe > 0: {c['oos_sharpe_positive']}",
            f"- Beats SPY in majority of folds: {c['beats_spy_majority_folds']}",
            f"- >= {c['min_trades_threshold']} trades: {c['min_trades_met']}",
            f"- **ALL MET: {c['all_met']}**",
        ]
    h = scorecard.get("holdout")
    if h:
        lines += [
            "",
            f"## Sealed holdout ({h['window']}, {h['regime']})",
            f"- return {h['return_pct']}%, sharpe {h['sharpe']}, "
            f"excess vs SPY {h['excess_vs_spy_pct']}%, trades {h['trades']}",
            f"- **holdout passes (Sharpe>0 & beats SPY & enough trades): {h['passes']}**",
            f"- config: {json.dumps(h['params'])}",
        ]
    lines += ["", "## By regime (OOS avg return %)"]
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
        search_iters: int, write: bool, is_days: int, oos_days: int,
        step_days: int, holdout_days: int, holdout_is_days: int,
        pass_min_trades: int, holdout_eval: bool = True) -> dict:
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
        is_days=is_days, oos_days=oos_days, step_days=step_days,
        holdout_days=holdout_days, holdout_is_days=holdout_is_days,
        holdout_eval=holdout_eval, pass_min_trades=pass_min_trades,
        progress_cb=lambda i: logger.info(
            "fold %s %s %s", i.get("fold"), i.get("oos"), i.get("holdout") or ""),
    )
    summary = {
        "strategy": strategy,
        "config": scorecard["config"],
        "oos": scorecard["oos"],
        "criteria": scorecard["criteria"],
        "holdout": scorecard["holdout"],
        "verdict": scorecard["verdict"],
    }
    print(json.dumps(summary, indent=2, default=str))

    # Persist the scorecard as the offline evidence record (best-effort —
    # a DB hiccup must never void a completed validation run).
    try:
        db = session_factory()
        try:
            record_validation_run(
                db, scorecard,
                universe=f"{mode}-{top_n}" if mode == "top" else mode,
                git_sha=_current_git_sha(),
            )
            logger.info("Recorded validation run for %s", strategy)
        finally:
            db.close()
    except Exception:
        logger.exception("Failed to record validation run for %s", strategy)

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
    ap.add_argument("--is-days", type=int, default=378, help="in-sample window (trading days)")
    ap.add_argument("--oos-days", type=int, default=126, help="out-of-sample window")
    ap.add_argument("--step-days", type=int, default=126, help="fold step")
    ap.add_argument("--holdout-days", type=int, default=0,
                    help="reserve a final sealed holdout of N trading days (0 = none)")
    ap.add_argument("--holdout-is-days", type=int, default=0,
                    help="fit the holdout config on the last N pre-holdout days "
                         "(0 = use all pre-holdout)")
    ap.add_argument("--pass-min-trades", type=int, default=30,
                    help="min OOS trades for the criteria to pass")
    ap.add_argument("--no-holdout-eval", action="store_true",
                    help="reserve the holdout region but do NOT evaluate it "
                         "(research stages; only the finalist burns the holdout)")
    ap.add_argument("--write", action="store_true", help="write reviews/ report")
    args = ap.parse_args()

    if not args.all and not args.strategy:
        ap.error("pass --strategy NAME or --all")

    targets = ALL_STRATEGIES if args.all else [args.strategy]
    symbols = [s for s in args.symbols.split(",") if s.strip()]
    for strat in targets:
        run(strat, mode=args.mode, top_n=args.top_n, symbols=symbols,
            search_iters=args.search_iters, write=args.write,
            is_days=args.is_days, oos_days=args.oos_days, step_days=args.step_days,
            holdout_days=args.holdout_days, holdout_is_days=args.holdout_is_days,
            pass_min_trades=args.pass_min_trades,
            holdout_eval=not args.no_holdout_eval)


if __name__ == "__main__":
    main()

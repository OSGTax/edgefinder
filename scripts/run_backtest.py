"""Run a minute-bar backtest over Massive flat-files data.

Loads minute aggregates for the requested symbols/date range from the
flat-files bucket and runs the SMA-crossover example strategy.

Example:
    python scripts/run_backtest.py --start 2026-05-20 --end 2026-05-26 \
        --symbols NVDA,AAPL --capital 10000 --fast 10 --slow 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

sys.path.insert(0, ".")

from edgefinder.backtest.engine import BacktestEngine, load_minute_bars
from edgefinder.backtest.examples import SmaCrossStrategy
from edgefinder.core.logging_config import configure_logging
from edgefinder.data.flatfiles import FlatFilesClient

logger = logging.getLogger(__name__)


def _parse_day(s: str):
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    configure_logging(level=logging.INFO)
    p = argparse.ArgumentParser(description="Minute-bar backtest on flat files")
    p.add_argument("--start", required=True, type=_parse_day)
    p.add_argument("--end", required=True, type=_parse_day)
    p.add_argument("--symbols", required=True, help="comma-separated symbols")
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--fast", type=int, default=10)
    p.add_argument("--slow", type=int, default=30)
    p.add_argument("--target-dollars", type=float, default=2_000.0)
    p.add_argument("--equity-out", default=None, help="optional path to write equity-curve CSV")
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    client = FlatFilesClient()

    days = client.available_days("minute_aggs", args.start, args.end)
    logger.info("Loading minute bars: %d days x %d symbols", len(days), len(symbols))
    bars = load_minute_bars(client, days, symbols)
    logger.info("Loaded %d bars", len(bars))
    if bars.empty:
        logger.error("No bars loaded — nothing to backtest")
        return 1

    strategy = SmaCrossStrategy(fast=args.fast, slow=args.slow, target_dollars=args.target_dollars)
    engine = BacktestEngine(starting_cash=args.capital)
    result = engine.run(bars, strategy)

    logger.info("─" * 48)
    logger.info("Backtest %s..%s  symbols=%s", args.start, args.end, ",".join(symbols))
    logger.info("Starting cash : $%s", f"{result.starting_cash:,.2f}")
    logger.info("Final equity  : $%s", f"{result.final_equity:,.2f}")
    logger.info("Return        : %+.2f%%", result.return_pct)
    logger.info("Realized P&L  : $%s", f"{result.realized_pnl:,.2f}")
    logger.info("Fills         : %d", result.num_fills)

    if args.equity_out:
        result.equity_frame().to_csv(args.equity_out, index=False)
        logger.info("Equity curve written to %s", args.equity_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

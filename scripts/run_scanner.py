"""Run the unified multi-strategy scanner (CLI).

Usage:
    python scripts/run_scanner.py                  # Full universe scan
    python scripts/run_scanner.py --tickers AAPL MSFT GOOG  # Specific tickers
    python scripts/run_scanner.py --quick           # 20 popular tickers
"""

import argparse
import sys
sys.path.insert(0, ".")

from rich.console import Console
from rich.table import Table

from edgefinder.data.polygon import PolygonDataProvider
from edgefinder.data.cache import DataCache
from edgefinder.data.provider import CachedDataProvider
from edgefinder.db.engine import Base, get_engine, get_session_factory
from edgefinder.db import models  # noqa: F401
from edgefinder.db.models import Fundamental, Ticker, TickerStrategyQualification
from edgefinder.scanner.unified_scanner import UnifiedScanner
from edgefinder.strategies import (  # noqa: F401  — populates the registry
    alpha, bravo, charlie, degenerate, echo,
)
from edgefinder.strategies.base import StrategyRegistry

console = Console()

QUICK_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
    "CROX", "DKNG", "DUOL", "FIVE", "GRMN",
    "LULU", "NCLH", "OLED", "PAYC", "PSTG",
    "SAIA", "DECK", "TMDX", "WSM", "ZWS",
]


def _render_strategy_table(session, strategy_name: str, qualified_count: int, total: int) -> None:
    """Render top-20 qualifications for a strategy from the DB."""
    rows = (
        session.query(TickerStrategyQualification, Fundamental)
        .join(Fundamental, Fundamental.ticker_id == TickerStrategyQualification.ticker_id)
        .filter(
            TickerStrategyQualification.strategy_name == strategy_name,
            TickerStrategyQualification.qualified == True,  # noqa: E712
        )
        .order_by(TickerStrategyQualification.score.desc().nulls_last())
        .limit(20)
        .all()
    )

    table = Table(title=f"{strategy_name.upper()} — {qualified_count} qualified of {total}")
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("EG%", justify="right")
    table.add_column("RG%", justify="right")
    table.add_column("D/E", justify="right")
    table.add_column("P/E", justify="right")
    table.add_column("ROE%", justify="right")
    table.add_column("SI%", justify="right")

    pct = lambda v: f"{v*100:.1f}%" if v is not None else "—"
    fmt = lambda v: f"{v:.1f}" if v is not None else "—"

    for qual, fund in rows:
        table.add_row(
            qual.symbol,
            f"{qual.score:.0f}" if qual.score is not None else "—",
            pct(fund.earnings_growth),
            pct(fund.revenue_growth),
            fmt(fund.debt_to_equity),
            fmt(fund.price_to_earnings),
            pct(fund.return_on_equity),
            pct(fund.short_interest),
        )

    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="EdgeFinder Unified Scanner")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    parser.add_argument("--quick", action="store_true", help="Scan 20 popular tickers")
    args = parser.parse_args()

    console.print("[bold]EdgeFinder — Unified Scanner[/bold]\n")

    engine = get_engine()
    Base.metadata.create_all(engine)
    session_factory = get_session_factory(engine)

    try:
        provider = PolygonDataProvider()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    cached = CachedDataProvider(provider, DataCache())
    tickers = args.tickers or (QUICK_TICKERS if args.quick else None)
    if not tickers:
        tickers = sorted(cached.get_ticker_universe())
    console.print(f"Scanning {len(tickers)} tickers across all registered strategies...\n")

    strategies = list(StrategyRegistry.get_instances())
    scanner = UnifiedScanner(strategies, cached, session_factory)
    summary = scanner.run(tickers)

    session = session_factory()
    try:
        for strategy in strategies:
            _render_strategy_table(
                session, strategy.name, summary.get(strategy.name, 0), len(tickers),
            )
    finally:
        session.close()

    engine.dispose()


if __name__ == "__main__":
    main()

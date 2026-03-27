"""Run the fundamental scanner (CLI).

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
from edgefinder.scanner.scanner import FundamentalScanner

console = Console()

QUICK_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "JNJ", "WMT", "PG", "MA", "HD", "BAC", "XOM", "PFE", "COST",
    "DIS", "NFLX",
]


def main():
    parser = argparse.ArgumentParser(description="EdgeFinder Fundamental Scanner")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    parser.add_argument("--quick", action="store_true", help="Scan 20 popular tickers")
    args = parser.parse_args()

    console.print("[bold]EdgeFinder v2 — Fundamental Scanner[/bold]\n")

    # Setup
    engine = get_engine()
    Base.metadata.create_all(engine)
    session = get_session_factory(engine)()

    try:
        provider = PolygonDataProvider()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Set EDGEFINDER_POLYGON_API_KEY in .env")
        return

    cached = CachedDataProvider(provider, DataCache())

    scanner = FundamentalScanner(cached, session)

    tickers = args.tickers or (QUICK_TICKERS if args.quick else None)
    console.print(f"Scanning {'specific tickers' if tickers else 'full universe'}...\n")

    results = scanner.run(tickers=tickers)

    if not results:
        console.print("[yellow]No stocks passed screening.[/yellow]")
        return

    # Display results
    table = Table(title=f"Scan Results ({len(results)} stocks)")
    table.add_column("Symbol", style="cyan")
    table.add_column("Company", max_width=25)
    table.add_column("Sector")
    table.add_column("Lynch", justify="right")
    table.add_column("Burry", justify="right")
    table.add_column("Composite", justify="right")
    table.add_column("Category")
    table.add_column("Strategies")

    for stock in sorted(results, key=lambda s: s.composite_score, reverse=True):
        strats = ", ".join(stock.qualifying_strategies) or "—"
        table.add_row(
            stock.symbol,
            stock.fundamentals.company_name or "—",
            stock.fundamentals.sector or "—",
            f"{stock.lynch_score:.1f}",
            f"{stock.burry_score:.1f}",
            f"{stock.composite_score:.1f}",
            stock.lynch_category,
            strats,
        )

    console.print(table)
    qualified = sum(1 for s in results if s.qualifying_strategies)
    console.print(f"\n[green]{qualified}[/green] stocks qualified by at least one strategy.")

    session.close()
    engine.dispose()


if __name__ == "__main__":
    main()

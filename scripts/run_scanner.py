"""Run the per-strategy scanner (CLI).

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
from edgefinder.scanner.strategy_scanner import StrategyScanner
from edgefinder.strategies.base import StrategyRegistry

console = Console()

QUICK_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
    "CROX", "DKNG", "DUOL", "FIVE", "GRMN",
    "LULU", "NCLH", "OLED", "PAYC", "PSTG",
    "SAIA", "DECK", "TMDX", "WSM", "ZWS",
]


def main():
    parser = argparse.ArgumentParser(description="EdgeFinder Per-Strategy Scanner")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    parser.add_argument("--quick", action="store_true", help="Scan 20 popular tickers")
    args = parser.parse_args()

    console.print("[bold]EdgeFinder v4 — Per-Strategy Scanner[/bold]\n")

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
    console.print(f"Scanning {len(tickers)} tickers...\n")

    # Import strategies
    from edgefinder.strategies import alpha, bravo, charlie  # noqa: F401

    for strategy in StrategyRegistry.get_instances():
        session = session_factory()
        try:
            scanner = StrategyScanner(strategy, cached, session)
            results = scanner.run(tickers=tickers)
            qualified = sum(1 for r in results if r.qualified)

            table = Table(title=f"{strategy.name.upper()} — {qualified} qualified of {len(results)}")
            table.add_column("Symbol", style="cyan")
            table.add_column("Company", max_width=25)
            table.add_column("Score", justify="right")
            table.add_column("EG%", justify="right")
            table.add_column("RG%", justify="right")
            table.add_column("D/E", justify="right")
            table.add_column("P/E", justify="right")
            table.add_column("ROE%", justify="right")
            table.add_column("SI%", justify="right")

            pct = lambda v: f"{v*100:.1f}%" if v is not None else "—"
            fmt = lambda v: f"{v:.1f}" if v is not None else "—"

            for r in sorted(results, key=lambda x: x.score, reverse=True)[:20]:
                if not r.qualified:
                    continue
                f = r.profile.fundamentals
                if not f:
                    continue
                table.add_row(
                    r.symbol,
                    f.company_name or "—",
                    f"{r.score:.0f}",
                    pct(f.earnings_growth),
                    pct(f.revenue_growth),
                    fmt(f.debt_to_equity),
                    fmt(f.price_to_earnings),
                    pct(f.return_on_equity),
                    pct(f.short_interest),
                )

            console.print(table)
            console.print()
        except Exception as e:
            console.print(f"[red]{strategy.name} scan failed: {e}[/red]")
        finally:
            session.close()

    engine.dispose()


if __name__ == "__main__":
    main()

"""Run the nightly data scanner (CLI).

Usage:
    python scripts/run_scanner.py                  # Full universe scan
    python scripts/run_scanner.py --tickers AAPL MSFT GOOG  # Specific tickers
    python scripts/run_scanner.py --quick           # 20 popular tickers

The scanner is a pure data collector (per-strategy qualification was retired
with the old arena): it fetches fundamentals for each ticker and persists
tickers + fundamentals rows, then prints a summary table.
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
from edgefinder.db.models import Fundamental, Ticker
from edgefinder.scanner.unified_scanner import UnifiedScanner

console = Console()

QUICK_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
    "CROX", "DKNG", "DUOL", "FIVE", "GRMN",
    "LULU", "NCLH", "OLED", "PAYC", "PSTG",
    "SAIA", "DECK", "TMDX", "WSM", "ZWS",
]


def main():
    parser = argparse.ArgumentParser(description="EdgeFinder Nightly Data Scanner")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    parser.add_argument("--quick", action="store_true", help="Scan 20 popular tickers")
    args = parser.parse_args()

    console.print("[bold]EdgeFinder — Nightly Data Scanner[/bold]\n")

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

    scanner = UnifiedScanner(cached, session_factory)
    summary = scanner.run(tickers)
    console.print(f"Scan summary: {summary}\n")

    # Show the freshest fundamentals for the scanned set
    session = session_factory()
    try:
        rows = (
            session.query(Ticker, Fundamental)
            .join(Fundamental, Ticker.id == Fundamental.ticker_id)
            .filter(Ticker.symbol.in_(tickers))
            .order_by(Ticker.market_cap.desc().nullslast())
            .limit(20)
            .all()
        )
        table = Table(title=f"Scanned fundamentals (top {len(rows)} by market cap)")
        table.add_column("Symbol", style="cyan")
        table.add_column("Company", max_width=25)
        table.add_column("Mkt cap", justify="right")
        table.add_column("EG%", justify="right")
        table.add_column("RG%", justify="right")
        table.add_column("D/E", justify="right")
        table.add_column("P/E", justify="right")
        table.add_column("ROE%", justify="right")

        pct = lambda v: f"{v*100:.1f}%" if v is not None else "—"
        fmt = lambda v: f"{v:.1f}" if v is not None else "—"
        cap = lambda v: f"${v/1e9:.1f}B" if v else "—"

        for t, f in rows:
            table.add_row(
                t.symbol,
                t.company_name or "—",
                cap(t.market_cap),
                pct(f.earnings_growth),
                pct(f.revenue_growth),
                fmt(f.debt_to_equity),
                fmt(f.price_to_earnings),
                pct(f.return_on_equity),
            )
        console.print(table)
    finally:
        session.close()

    engine.dispose()


if __name__ == "__main__":
    main()

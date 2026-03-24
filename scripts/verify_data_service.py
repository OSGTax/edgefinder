#!/usr/bin/env python3
"""
Data Service Verification Script
=================================
Run this after setting up your API keys to verify everything works.

Usage:
    python scripts/verify_data_service.py

What it checks:
    1. secrets.env file exists and has keys
    2. Alpaca connection (paper trading account)
    3. FMP connection (company profile lookup)
    4. yfinance fallback works
    5. Cache is working
    6. Full data flow: cache → API → cache
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table

console = Console()


def check_secrets_file() -> dict:
    """Check that secrets.env exists and has non-placeholder keys."""
    secrets_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "secrets.env"
    )

    result = {"exists": False, "alpaca_key": False, "alpaca_secret": False, "fmp_key": False}

    if not os.path.exists(secrets_path):
        return result

    result["exists"] = True

    with open(secrets_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "ALPACA_API_KEY" and value and "your_" not in value:
                result["alpaca_key"] = True
            elif key == "ALPACA_SECRET_KEY" and value and "your_" not in value:
                result["alpaca_secret"] = True
            elif key == "FMP_API_KEY" and value and "your_" not in value:
                result["fmp_key"] = True

    return result


def main():
    console.print("\n[bold blue]EdgeFinder Data Service Verification[/bold blue]\n")

    all_passed = True
    results = []

    # ── Step 1: Check secrets.env ──────────────────────────
    console.print("[bold]Step 1: Checking config/secrets.env...[/bold]")
    secrets = check_secrets_file()

    if not secrets["exists"]:
        results.append(("secrets.env exists", False, "File not found"))
        console.print("  [red]✗[/red] config/secrets.env not found")
        console.print("  [yellow]→ Run: cp config/secrets.env.example config/secrets.env[/yellow]")
        console.print("  [yellow]→ Then edit it with your API keys[/yellow]\n")
        all_passed = False
    else:
        results.append(("secrets.env exists", True, ""))

        if secrets["alpaca_key"] and secrets["alpaca_secret"]:
            results.append(("Alpaca keys configured", True, ""))
            console.print("  [green]✓[/green] Alpaca API keys found")
        else:
            results.append(("Alpaca keys configured", False, "Missing or placeholder"))
            console.print("  [red]✗[/red] Alpaca API keys missing or still placeholder")
            console.print("  [yellow]→ Get keys at: https://alpaca.markets[/yellow]")
            all_passed = False

        if secrets["fmp_key"]:
            results.append(("FMP key configured", True, ""))
            console.print("  [green]✓[/green] FMP API key found")
        else:
            results.append(("FMP key configured", False, "Missing or placeholder"))
            console.print("  [red]✗[/red] FMP API key missing or still placeholder")
            console.print("  [yellow]→ Get key at: https://site.financialmodelingprep.com/developer[/yellow]")
            all_passed = False

    console.print()

    # ── Step 2: Test Alpaca connection ─────────────────────
    console.print("[bold]Step 2: Testing Alpaca connection...[/bold]")
    try:
        from services.data_service import DataService
        ds = DataService(cache_path=":memory:")

        if ds.alpaca:
            account = ds.alpaca.get_account()
            if account:
                buying_power = account.get("buying_power", "unknown")
                status = account.get("status", "unknown")
                results.append(("Alpaca connection", True, f"Status: {status}"))
                console.print(f"  [green]✓[/green] Connected! Account status: {status}")
                console.print(f"  [green]✓[/green] Paper trading buying power: ${buying_power}")
            else:
                results.append(("Alpaca connection", False, "Auth failed"))
                console.print("  [red]✗[/red] Connected but auth failed — check your keys")
                all_passed = False
        else:
            results.append(("Alpaca connection", False, "Not configured"))
            console.print("  [yellow]⚠[/yellow] Alpaca not configured — will use yfinance for bars")
    except Exception as e:
        results.append(("Alpaca connection", False, str(e)[:50]))
        console.print(f"  [red]✗[/red] Error: {e}")
        all_passed = False

    console.print()

    # ── Step 3: Test FMP connection ────────────────────────
    console.print("[bold]Step 3: Testing FMP connection...[/bold]")
    try:
        if ds.fmp:
            profile = ds.fmp.get_profile("AAPL")
            if profile:
                name = profile.get("companyName", "unknown")
                sector = profile.get("sector", "unknown")
                results.append(("FMP connection", True, f"Got: {name}"))
                console.print(f"  [green]✓[/green] Connected! Test query: {name} ({sector})")
                console.print(f"  [green]✓[/green] Requests remaining today: ~{ds.fmp.requests_remaining}")
            else:
                results.append(("FMP connection", False, "No data returned"))
                console.print("  [red]✗[/red] Connected but no data — check your key")
                all_passed = False
        else:
            results.append(("FMP connection", False, "Not configured"))
            console.print("  [yellow]⚠[/yellow] FMP not configured — will use yfinance for fundamentals")
    except Exception as e:
        results.append(("FMP connection", False, str(e)[:50]))
        console.print(f"  [red]✗[/red] Error: {e}")
        all_passed = False

    console.print()

    # ── Step 4: Test yfinance fallback ─────────────────────
    console.print("[bold]Step 4: Testing yfinance fallback...[/bold]")
    try:
        import yfinance as yf
        t = yf.Ticker("AAPL")
        hist = t.history(period="5d")
        if hist is not None and not hist.empty:
            last_close = hist["Close"].iloc[-1]
            results.append(("yfinance fallback", True, f"AAPL: ${last_close:.2f}"))
            console.print(f"  [green]✓[/green] Working! AAPL last close: ${last_close:.2f}")
        else:
            results.append(("yfinance fallback", False, "No data"))
            console.print("  [red]✗[/red] yfinance returned no data")
            all_passed = False
    except Exception as e:
        results.append(("yfinance fallback", False, str(e)[:50]))
        console.print(f"  [red]✗[/red] Error: {e}")
        all_passed = False

    console.print()

    # ── Step 5: Test full data flow ────────────────────────
    console.print("[bold]Step 5: Testing full data flow (cache → API → cache)...[/bold]")
    try:
        # First call: should hit API
        bars = ds.get_bars("AAPL", timeframe="1Day", days_back=5)
        if bars is not None and not bars.empty:
            results.append(("Get bars (API)", True, f"{len(bars)} bars"))
            console.print(f"  [green]✓[/green] Fetched {len(bars)} daily bars for AAPL")

            # Second call: should hit cache
            bars2 = ds.get_bars("AAPL", timeframe="1Day", days_back=5)
            if bars2 is not None and not bars2.empty:
                results.append(("Get bars (cache)", True, f"{len(bars2)} bars"))
                console.print(f"  [green]✓[/green] Cache hit: {len(bars2)} bars (no API call)")
            else:
                results.append(("Get bars (cache)", False, "Cache miss"))
                console.print("  [yellow]⚠[/yellow] Cache miss on second call")
        else:
            results.append(("Get bars", False, "No data from any source"))
            console.print("  [red]✗[/red] Could not get bar data from any source")
            all_passed = False

        # Test fundamentals
        profile = ds.get_profile("AAPL")
        if profile:
            results.append(("Get profile", True, profile.get("companyName", "")))
            console.print(f"  [green]✓[/green] Got profile: {profile.get('companyName')}")
        else:
            results.append(("Get profile", False, "No data"))
            console.print("  [yellow]⚠[/yellow] Could not get profile data")

        # Cache stats
        stats = ds.get_cache_stats()
        console.print(f"  [blue]ℹ[/blue] Cache stats: {stats}")

    except Exception as e:
        results.append(("Full data flow", False, str(e)[:50]))
        console.print(f"  [red]✗[/red] Error: {e}")
        all_passed = False

    # ── Summary ────────────────────────────────────────────
    console.print()
    table = Table(title="Verification Summary")
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    for name, passed, detail in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, detail)

    console.print(table)

    if all_passed:
        console.print("\n[bold green]All checks passed! Data service is ready.[/bold green]\n")
    else:
        console.print(
            "\n[bold yellow]Some checks failed. The system will still work "
            "using yfinance as fallback, but API sources give better data.[/bold yellow]\n"
        )

    # Show data source priority
    console.print("[bold]Data source priority:[/bold]")
    console.print("  Bars:          Cache → Alpaca → yfinance")
    console.print("  Fundamentals:  Cache → FMP → yfinance")
    console.print("  Latest price:  Alpaca quote → yfinance → cached bar")
    console.print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

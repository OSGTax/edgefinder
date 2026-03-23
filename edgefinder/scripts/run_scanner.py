#!/usr/bin/env python3
"""
EdgeFinder Scanner Runner
=========================
Runs the fundamental scan pipeline and displays results.

Usage:
  python scripts/run_scanner.py              # Full scan (~1500 tickers)
  python scripts/run_scanner.py --quick      # Quick scan (50 popular tickers)
  python scripts/run_scanner.py --tickers AAPL MSFT GOOGL   # Specific tickers
  python scripts/run_scanner.py --no-save    # Don't save to database
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from modules.scanner import run_scan, get_active_watchlist
from modules.database import init_db


# Popular tickers for quick mode
QUICK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "NFLX",
    "ADBE", "CRM", "INTC", "AMD", "QCOM", "TXN", "AVGO", "COST", "PEP",
    "KO", "MRK", "ABT", "TMO", "DHR", "LLY", "ABBV", "BMY", "GILD",
    "CVX", "XOM", "COP", "SLB", "CAT", "DE", "GE", "BA", "RTX",
    "WMT", "TGT", "LOW", "SBUX", "NKE",
]


def main():
    parser = argparse.ArgumentParser(description="EdgeFinder Fundamental Scanner")
    parser.add_argument("--quick", action="store_true", help="Quick scan (50 popular tickers)")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to database")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=settings.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.LOG_FILE, mode="a"),
        ],
    )
    logger = logging.getLogger("edgefinder")

    # Ensure database exists
    os.makedirs("data", exist_ok=True)
    init_db(settings.DATABASE_PATH)

    # Determine tickers
    tickers = None
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        print(f"\nScanning {len(tickers)} specified tickers...")
    elif args.quick:
        tickers = QUICK_TICKERS
        print(f"\nQuick scan: {len(tickers)} popular tickers...")
    else:
        print("\nFull scan: fetching S&P 500/400/600 universe...")
        print("(This will take 15-30 minutes for ~1500 tickers)")
        # ============================================================
        # HUMAN_ACTION_REQUIRED
        # What: Wait for the full scan to complete
        # Why: Fetching 1500+ tickers from yfinance takes time
        # How: Just wait. Progress is logged every 50 tickers.
        #      If it stalls, yfinance may be rate-limiting.
        #      Try --quick mode first to verify everything works.
        # ============================================================

    # Run scan
    start_time = datetime.now()
    results = run_scan(
        tickers=tickers,
        save_to_db=not args.no_save,
        verbose=args.verbose,
    )
    elapsed = (datetime.now() - start_time).total_seconds()

    # Display results
    print()
    print("=" * 70)
    print(f"  SCAN COMPLETE — {len(results)} stocks made the watchlist")
    print(f"  Time elapsed: {elapsed:.1f} seconds")
    print("=" * 70)

    if results:
        print()
        print(f"  {'#':>3}  {'Ticker':<7} {'Company':<25} {'Composite':>9} "
              f"{'Lynch':>6} {'Burry':>6} {'Category':<13} {'Price':>8}")
        print("  " + "-" * 90)

        for i, s in enumerate(results[:25], 1):
            name = s.data.company_name[:24] if s.data.company_name else "N/A"
            print(f"  {i:>3}. {s.data.ticker:<7} {name:<25} {s.composite_score:>8.1f} "
                  f"{s.lynch_score:>6.1f} {s.burry_score:>6.1f} "
                  f"{s.lynch_category:<13} ${s.data.price:>7.2f}")

        if len(results) > 25:
            print(f"\n  ... and {len(results) - 25} more. "
                  f"View full list in the dashboard or database.")

        if not args.no_save:
            print(f"\n  ✓ Saved to database: {settings.DATABASE_PATH}")
    else:
        print("\n  No stocks met the minimum composite score threshold.")
        print(f"  Current threshold: {settings.WATCHLIST_MIN_COMPOSITE_SCORE}")
        print("  Try lowering WATCHLIST_MIN_COMPOSITE_SCORE in config/settings.py")

    # ============================================================
    # HUMAN_ACTION_REQUIRED
    # What: Review the watchlist output
    # Why: Sanity check — do these stocks look reasonable?
    # How: 1. Do you recognize the top tickers? They should be
    #         quality companies, not random junk.
    #      2. Are the scores distributed well (not all 60 or all 99)?
    #      3. If the list is empty, lower WATCHLIST_MIN_COMPOSITE_SCORE
    #         in config/settings.py and re-run.
    #      4. If it's full of garbage, the scoring weights need tuning.
    #
    # Once you're satisfied the scanner is working:
    #   → Proceed to build Module 2 (Technical Signal Engine)
    # ============================================================

    print()
    print("  Next steps:")
    print("    • Review the watchlist above for quality")
    print("    • Adjust thresholds in config/settings.py if needed")
    print("    • Proceed to Module 2: Technical Signal Engine")
    print()


if __name__ == "__main__":
    main()

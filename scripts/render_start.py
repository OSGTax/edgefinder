"""
Render Startup Script
=====================
Initializes the database, runs an initial scan if the watchlist is empty,
then starts the FastAPI dashboard via uvicorn.

This runs on every Render deploy and on cold starts.
"""

import logging
import os
import subprocess
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("render_start")


def main():
    from modules.database import init_db, get_session
    from modules.database import WatchlistStock

    # Step 1: Initialize database
    logger.info("Initializing database...")
    init_db()

    # Step 2: Check if watchlist has data; if not, run a scan
    session = get_session()
    try:
        active_count = session.query(WatchlistStock).filter(
            WatchlistStock.is_active == True  # noqa: E712
        ).count()
    finally:
        session.close()

    if active_count == 0:
        logger.info("No active watchlist found — running initial scan...")
        _run_initial_scan()
    else:
        logger.info(f"Watchlist has {active_count} active stocks — skipping scan")

    # Step 3: Start uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting dashboard on port {port}...")
    os.execvp("uvicorn", [
        "uvicorn",
        "dashboard.app:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ])


def _run_initial_scan():
    """Run a scan with a curated mid-cap ticker list."""
    try:
        from modules.scanner import run_scan

        # Curated universe — mid/small caps that fit EdgeFinder's filters
        # ($300M-$200B market cap, $5-$500 price)
        tickers = [
            # Tech / Software
            "PYPL", "SNAP", "PLTR", "ROKU", "SQ", "PATH", "BILL",
            "CFLT", "MDB", "DDOG", "NET", "ZS", "CRWD", "DUOL",
            "DOCS", "HIMS", "APP", "TOST",
            # Semiconductors
            "ON", "SWKS", "QRVO", "MRVL", "SMCI", "ENPH", "SEDG", "FSLR",
            # Consumer
            "ETSY", "W", "CHWY", "CELH", "ELF", "RVLV", "DECK",
            "CAVA", "BROS", "SHAK", "LULU",
            # Energy
            "DVN", "MRO", "OVV", "CTRA", "FANG",
            # Materials / Industrials
            "CLF", "AA", "X", "NUE", "STLD", "AXON",
            # Transport
            "DAL", "UAL", "LUV", "JBLU", "UBER", "LYFT",
            # Gaming / Entertainment
            "MGM", "WYNN", "CZR", "DKNG",
            # Fintech / Crypto-adjacent
            "SOFI", "COIN", "HOOD",
            # Travel
            "ABNB", "DASH",
            # Space / Quantum
            "RKLB", "IONQ", "RIVN",
            # Pharma / Health
            "PFE", "HIMS", "DOCS",
        ]
        # Deduplicate
        tickers = sorted(set(tickers))

        results = run_scan(tickers=tickers, save_to_db=True)
        logger.info(f"Initial scan complete: {len(results)} stocks on watchlist")
    except Exception as e:
        logger.error(f"Initial scan failed (non-fatal): {e}")


if __name__ == "__main__":
    main()

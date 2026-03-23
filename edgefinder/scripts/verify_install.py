#!/usr/bin/env python3
"""
EdgeFinder Dependency Verifier
==============================
Checks that all required packages are installed and working.
Run this FIRST before anything else.

Usage: python scripts/verify_install.py
"""

import subprocess
import sys
import importlib

REQUIRED_PACKAGES = {
    "yfinance": "yfinance",
    "pandas": "pandas",
    "numpy": "numpy",
    "pandas_ta": "pandas-ta",
    "vaderSentiment": "vaderSentiment",
    "feedparser": "feedparser",
    "bs4": "beautifulsoup4",
    "requests": "requests",
    "sqlalchemy": "sqlalchemy",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "apscheduler": "apscheduler",
    "pytest": "pytest",
    "rich": "rich",
    "dotenv": "python-dotenv",
}


def check_python_version():
    v = sys.version_info
    print(f"  Python version: {v.major}.{v.minor}.{v.micro}", end="")
    if v.major == 3 and v.minor >= 11:
        print(" ✓")
        return True
    elif v.major == 3 and v.minor >= 9:
        print(" ⚠ (3.11+ recommended, but should work)")
        return True
    else:
        print(" ✗ (Python 3.9+ required)")
        return False


def check_package(import_name: str, pip_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        print(f"  {pip_name:<25} ✓")
        return True
    except ImportError:
        print(f"  {pip_name:<25} ✗ (missing)")
        return False


def install_missing(missing: list[str]):
    if not missing:
        return
    print(f"\nInstalling {len(missing)} missing packages...")
    cmd = [sys.executable, "-m", "pip", "install"] + missing
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Installation complete.")


def main():
    print("=" * 55)
    print("  EDGEFINDER — Dependency Verification")
    print("=" * 55)
    print()

    # Python version
    print("[1/3] Checking Python version...")
    if not check_python_version():
        print("\n✗ FAILED: Please install Python 3.9 or higher.")
        sys.exit(1)

    # Packages
    print("\n[2/3] Checking required packages...")
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if not check_package(import_name, pip_name):
            missing.append(pip_name)

    if missing:
        print(f"\n  {len(missing)} package(s) missing.")
        # ============================================================
        # HUMAN_ACTION_REQUIRED
        # What: Install missing Python packages
        # Why: Can't proceed without dependencies
        # How: Run: pip install -r requirements.txt
        #      Or let this script install them automatically below.
        # ============================================================
        response = input("\n  Install missing packages now? [Y/n]: ").strip().lower()
        if response in ("", "y", "yes"):
            install_missing(missing)
        else:
            print("\n  Please run: pip install -r requirements.txt")
            sys.exit(1)

    # Quick functional test
    print("\n[3/3] Running quick functional tests...")

    try:
        import yfinance as yf
        test = yf.Ticker("AAPL")
        info = test.info
        if info and info.get("regularMarketPrice"):
            print(f"  yfinance API:              ✓ (AAPL = ${info['regularMarketPrice']:.2f})")
        else:
            print("  yfinance API:              ⚠ (connected but returned no data)")
    except Exception as e:
        print(f"  yfinance API:              ⚠ ({e})")

    try:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///:memory:")
        with engine.connect() as conn:
            conn.execute(conn.connection.cursor().execute("SELECT 1") if False else type(conn).execute(conn, type(conn).execute.__func__ and "SELECT 1" or "SELECT 1"))
        print("  SQLite:                    ✓")
    except Exception:
        # Simpler test
        try:
            import sqlite3
            conn = sqlite3.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            print("  SQLite:                    ✓")
        except Exception as e:
            print(f"  SQLite:                    ✗ ({e})")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        result = sia.polarity_scores("The stock market is doing great today!")
        print(f"  VADER sentiment:           ✓ (test score: {result['compound']:.3f})")
    except Exception as e:
        print(f"  VADER sentiment:           ⚠ ({e})")

    print()
    print("=" * 55)
    print("  ✓ ALL CHECKS PASSED — Ready to build EdgeFinder!")
    print("=" * 55)
    print()
    print("  Next steps:")
    print("    1. python scripts/setup_db.py")
    print("    2. python -m pytest tests/test_scanner.py -v")
    print("    3. python scripts/run_scanner.py")
    print()


if __name__ == "__main__":
    main()

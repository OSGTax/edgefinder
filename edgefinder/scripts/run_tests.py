#!/usr/bin/env python3
"""
EdgeFinder Test Runner
======================
Runs all tests with a clear summary.

Usage:
  python scripts/run_tests.py                    # Run all unit tests
  python scripts/run_tests.py --integration      # Include integration tests
  python scripts/run_tests.py --module scanner   # Run one module's tests
  python scripts/run_tests.py --coverage         # With coverage report
"""

import subprocess
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULE_MAP = {
    "scanner":   "tests/test_scanner.py",
    "signals":   "tests/test_signals.py",
    "sentiment": "tests/test_sentiment.py",
    "trader":    "tests/test_trader.py",
    "journal":   "tests/test_journal.py",
    "optimizer": "tests/test_optimizer.py",
}


def main():
    parser = argparse.ArgumentParser(description="EdgeFinder Test Runner")
    parser.add_argument("--integration", action="store_true",
                        help="Include integration tests (hits real APIs)")
    parser.add_argument("--module", choices=MODULE_MAP.keys(),
                        help="Run tests for a specific module only")
    parser.add_argument("--coverage", action="store_true",
                        help="Generate coverage report")
    args = parser.parse_args()

    print("=" * 55)
    print("  EDGEFINDER — Test Suite")
    print("=" * 55)
    print()

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]

    if args.module:
        test_file = MODULE_MAP[args.module]
        if not os.path.exists(test_file):
            print(f"  ✗ Test file not found: {test_file}")
            print(f"    Module '{args.module}' tests haven't been built yet.")
            sys.exit(1)
        cmd.append(test_file)
    else:
        cmd.append("tests/")

    cmd.extend(["-v", "--tb=short"])

    if not args.integration:
        cmd.extend(["-m", "not integration"])

    if args.coverage:
        cmd.extend(["--cov=modules", "--cov-report=term-missing"])

    # Run
    print(f"  Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Summary
    print()
    print("=" * 55)
    if result.returncode == 0:
        print("  ✓ ALL TESTS PASSED")
        if args.module:
            next_modules = list(MODULE_MAP.keys())
            idx = next_modules.index(args.module)
            if idx + 1 < len(next_modules):
                next_mod = next_modules[idx + 1]
                print(f"\n  Ready to build Module: {next_mod}")
                print(f"    1. Implement modules/{next_mod}.py")
                print(f"    2. Write tests in tests/test_{next_mod}.py")
                print(f"    3. Run: python scripts/run_tests.py --module {next_mod}")
    else:
        print("  ✗ SOME TESTS FAILED")
        print("    Fix failing tests before proceeding to the next module.")
        # ============================================================
        # HUMAN_ACTION_REQUIRED
        # What: Review and fix failing tests
        # Why: Each module must pass ALL tests before building the next
        # How: 1. Read the error messages above
        #      2. Common issues:
        #         - yfinance returning None (add None handling)
        #         - Score out of range (check clamping logic)
        #         - Database errors (run setup_db.py first)
        #      3. Fix the module code, not the tests
        #         (unless the test expectations are wrong)
        # ============================================================
    print("=" * 55)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

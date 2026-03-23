#!/usr/bin/env python3
"""
EdgeFinder Database Setup
=========================
Creates the SQLite database and all tables.

Usage: python scripts/setup_db.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.database import init_db, Base
from config import settings


def main():
    print("=" * 50)
    print("  EDGEFINDER — Database Setup")
    print("=" * 50)
    print()

    db_path = settings.DATABASE_PATH
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    if os.path.exists(db_path):
        print(f"  Database already exists at: {db_path}")
        response = input("  Recreate from scratch? This deletes all data. [y/N]: ").strip().lower()
        if response in ("y", "yes"):
            os.remove(db_path)
            print(f"  Deleted old database.")
        else:
            print(f"  Keeping existing database.")
            return

    engine = init_db(db_path, echo=False)

    # Verify tables
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print(f"\n  Database created at: {db_path}")
    print(f"  Tables created: {len(tables)}")
    for t in tables:
        cols = inspector.get_columns(t)
        print(f"    • {t} ({len(cols)} columns)")

    print()
    print("  ✓ Database ready!")
    print()
    print("  Next: python -m pytest tests/test_scanner.py -v")
    print()


if __name__ == "__main__":
    main()

"""Resolve the FROZEN intraday pilot menu from the daily store/DB.

The menu (intraday/menu.json) is pre-registration: the criteria string was
committed BEFORE the first symbol was resolved. This script computes the
symbol list — top-N common stocks by trailing-``rank_window``-trading-day
mean dollar volume as of ``as_of``, using the daily store's exact ranking
semantics (engine/data.resolve_universe + trailing_rank_start) — unions in
the protected ETF menu, and writes the file. It also prints the JSON, so a
workflow run (which cannot push) still leaves the result in its log for the
orchestrator to commit.

Examples:
    python scripts/resolve_intraday_menu.py                       # defaults
    python scripts/resolve_intraday_menu.py --as-of 2026-06-11 \
        --top 50 --rank-window 126 --out intraday/menu.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime

sys.path.insert(0, ".")

from edgefinder.data.barstore import DB_PROTECTED_ETFS
from edgefinder.db.models import DailyBar
from edgefinder.engine.data import resolve_universe, trailing_rank_start

logger = logging.getLogger(__name__)


def resolve_menu(session, as_of: date, top_n: int = 50,
                 rank_window: int = 126,
                 etfs: tuple[str, ...] = DB_PROTECTED_ETFS) -> dict:
    """Compute the menu dict (sorted, deduped symbols) from daily_bars.

    The trading calendar comes from SPY (a protected ETF — full history in
    the DB), falling back to the table's distinct dates; ``rank_start`` is
    the validator's exact trailing-window arithmetic.
    """
    days = [d for (d,) in (session.query(DailyBar.date)
                           .filter(DailyBar.symbol == "SPY")
                           .order_by(DailyBar.date).all())]
    if not days:
        days = sorted(d for (d,) in session.query(DailyBar.date).distinct())
    if not days:
        raise SystemExit("daily_bars is empty — cannot rank a menu")

    rank_start = trailing_rank_start(days, as_of, rank_window)
    top = resolve_universe(session, "top", [], top_n,
                           as_of=as_of, rank_start=rank_start)
    if not top:
        raise SystemExit(f"ranking returned no symbols as of {as_of} — "
                         "is the daily store populated?")
    etfs_up = [e.strip().upper() for e in etfs]
    symbols = sorted({s.upper() for s in top} | set(etfs_up))
    return {
        "frozen_at": "2026-06-12",
        "criteria": (f"top-{top_n} by trailing-{rank_window}-trading-day "
                     f"mean dollar volume as of {as_of} (the daily store's "
                     "ranking, resolve_universe semantics) + the protected "
                     "ETF menu"),
        "as_of": str(as_of),
        "top_n": top_n,
        "rank_window": rank_window,
        "etfs": etfs_up,
        "symbols": symbols,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default="2026-06-11",
                   help="ranking as-of date (default: the pre-registered "
                        "2026-06-11)")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--rank-window", type=int, default=126)
    p.add_argument("--out", default="intraday/menu.json")
    p.add_argument("--db-url", default=None,
                   help="override DATABASE_URL/settings for the daily store")
    args = p.parse_args(argv)

    from edgefinder.db.engine import get_engine, get_session_factory

    engine = get_engine(url=args.db_url) if args.db_url else get_engine()
    session = get_session_factory(engine)()
    try:
        menu = resolve_menu(session,
                            datetime.strptime(args.as_of, "%Y-%m-%d").date(),
                            top_n=args.top, rank_window=args.rank_window)
    finally:
        session.close()

    payload = json.dumps(menu, indent=2) + "\n"
    with open(args.out, "w") as f:
        f.write(payload)
    print(payload)
    print(f"wrote {len(menu['symbols'])} symbols to {args.out} "
          "(committing it is the orchestrator's job)", file=sys.stderr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

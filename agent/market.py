"""Market-data CLI the agent calls to observe before it decides.

Thin JSON front-end over ``agent.data`` (which delegates to the kept
EdgeFinder data layer). No external API calls — everything reads the local
bar history (R2 archive / Postgres hot set), so it is fast and PIT-safe.

CLI:
  python -m agent.market regime
  python -m agent.market quote --symbols NVDA,AAPL,MSFT
  python -m agent.market history --symbol NVDA --days 120
  python -m agent.market news --symbol NVDA --limit 8
  python -m agent.market universe --top 200
"""

from __future__ import annotations

import json
from datetime import date

from agent import data


def _parse_date(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    rg = sub.add_parser("regime")
    rg.add_argument("--as-of", default=None)

    qt = sub.add_parser("quote")
    qt.add_argument("--symbols", required=True)
    qt.add_argument("--as-of", default=None)
    qt.add_argument("--source", default="auto", choices=["auto", "r2", "db"])

    hi = sub.add_parser("history")
    hi.add_argument("--symbol", required=True)
    hi.add_argument("--days", type=int, default=120)
    hi.add_argument("--source", default="auto", choices=["auto", "r2", "db"])

    nw = sub.add_parser("news")
    nw.add_argument("--symbol", required=True)
    nw.add_argument("--limit", type=int, default=8)

    un = sub.add_parser("universe")
    un.add_argument("--top", type=int, default=200)
    un.add_argument("--as-of", default=None)

    args = p.parse_args(argv)

    if args.cmd == "regime":
        out = data.regime(as_of=_parse_date(args.as_of))
    elif args.cmd == "quote":
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        out = data.latest_indicators(symbols, as_of=_parse_date(args.as_of),
                                     source=args.source)
    elif args.cmd == "history":
        out = {"symbol": args.symbol.upper(),
               "bars": data.history(args.symbol.upper(), days=args.days,
                                    source=args.source)}
    elif args.cmd == "news":
        out = {"symbol": args.symbol.upper(),
               "news": data.news(args.symbol.upper(), limit=args.limit)}
    elif args.cmd == "universe":
        out = {"top": args.top, "symbols": data.universe(args.top,
                                                         as_of=_parse_date(args.as_of))}
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

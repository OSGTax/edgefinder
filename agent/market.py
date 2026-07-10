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
  python -m agent.market brief-build --top 40   # nightly, after the ingest
  python -m agent.market brief                  # the hourly cycle's first read
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from agent import data

ACCOUNT = "agent"
BRIEF_STALE_HOURS = 36  # older than this → the cycle should scan manually


def _parse_date(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


# ── the nightly research pack (desk_briefs) ─────────────────
#
# Built once per night by the data-refresh routine, right after the ingest,
# while the whole-market picture is fresh. The hourly trading cycle reads ONE
# dense payload instead of re-deriving regime/universe/movers/news with a
# dozen scans — its context goes to deciding, not gathering.


def _movers(store, *, top_k: int = 8, min_coverage: int = 300) -> dict:
    """Gainers/losers/most-active across the last two WELL-COVERED sessions.

    A session only counts when it has >= ``min_coverage`` bars — today's
    partial intraday top-up (a handful of held names) must never be one side
    of a market-wide comparison."""
    lo = date.today() - timedelta(days=7)
    rows = store.select("daily_bars", columns="symbol,close,volume,date",
                        filters={"date": ("gte", lo)})
    by_date: dict[str, dict[str, tuple[float, float]]] = {}
    for r in rows:
        c = r.get("close")
        if c is None:
            continue
        d = str(r["date"])[:10]
        by_date.setdefault(d, {})[r["symbol"]] = (float(c), float(r.get("volume") or 0))
    fat = sorted((d for d, m in by_date.items() if len(m) >= min_coverage),
                 reverse=True)
    if len(fat) < 2:
        return {"as_of": None, "prior": None, "gainers": [], "losers": [],
                "most_active": [],
                "note": f"fewer than two sessions with >={min_coverage} "
                        "symbols in the last week — is the nightly ingest ok?"}
    cur_d, prev_d = fat[0], fat[1]
    prev = by_date[prev_d]
    changed = []
    for sym, (c, v) in by_date[cur_d].items():
        if c < 1.0 or any(ch in sym for ch in (".", "/", "=")):
            continue
        p = prev.get(sym)
        if p and p[0] > 0:
            changed.append({"symbol": sym, "close": round(c, 2),
                            "change_pct": round((c - p[0]) / p[0] * 100, 2),
                            "dollar_volume": round(c * v)})
    return {"as_of": cur_d, "prior": prev_d,
            "gainers": sorted(changed, key=lambda r: -r["change_pct"])[:top_k],
            "losers": sorted(changed, key=lambda r: r["change_pct"])[:top_k],
            "most_active": sorted(changed,
                                  key=lambda r: -r["dollar_volume"])[:top_k]}


def build_brief(*, top: int = 40) -> dict:
    """Assemble tonight's research pack and upsert it (one row per date)."""
    from agent.store import get_store

    store = get_store()
    today = date.today()

    regime = data.regime()
    coverage = data.universe_coverage()
    top_syms = data.universe(top)
    movers = _movers(store)

    roster = data.latest_indicators(top_syms) if top_syms else {}
    trend = []
    for sym in top_syms:
        info = roster.get(sym)
        if not info:
            continue
        ind = info.get("indicators", {})
        close = info.get("close")
        sma50, sma200 = ind.get("ema_50"), ind.get("ema_200")
        trend.append({
            "symbol": sym, "close": close, "date": str(info.get("date"))[:10],
            "ret_1m": info.get("ret_1m"), "ret_3m": info.get("ret_3m"),
            "ret_6m": info.get("ret_6m"), "rsi": ind.get("rsi"),
            "above_50": (close > sma50) if (close and sma50) else None,
            "above_200": (close > sma200) if (close and sma200) else None,
        })

    news_syms = list(dict.fromkeys(
        [*top_syms[:10],
         *[m["symbol"] for m in movers.get("gainers", [])[:5]],
         *[m["symbol"] for m in movers.get("losers", [])[:5]]]))[:15]
    headlines = {}
    for sym in news_syms:
        items = data.news(sym, limit=3)
        if items:
            headlines[sym] = [{"title": i.get("title"),
                               "published": str(i.get("published_utc"))[:16]}
                              for i in items]

    payload = {"as_of": str(today), "regime": regime, "coverage": coverage,
               "universe_top": top_syms, "movers": movers,
               "trend_roster": trend, "headlines": headlines}
    built_at = datetime.now(timezone.utc).replace(tzinfo=None)
    existing = store.select("desk_briefs",
                            filters={"account": ACCOUNT, "brief_date": today},
                            limit=1)
    if existing:
        store.update("desk_briefs", {"id": existing[0]["id"]},
                     {"payload": payload, "built_at": built_at},
                     returning=False)
    else:
        store.insert("desk_briefs",
                     {"account": ACCOUNT, "brief_date": today,
                      "built_at": built_at, "payload": payload},
                     returning=False)
    return {"ok": True, "brief_date": str(today),
            "universe": len(top_syms), "trend_roster": len(trend),
            "movers_as_of": movers.get("as_of"),
            "coverage_status": coverage.get("status"),
            "headline_symbols": len(headlines)}


def get_brief() -> dict:
    """The latest research pack, with an honest staleness flag."""
    from agent.store import get_store

    rows = get_store().select("desk_briefs", filters={"account": ACCOUNT},
                              order=[("brief_date", "desc")], limit=1)
    if not rows:
        return {"exists": False,
                "note": "no brief built yet — scan manually this cycle"}
    r = rows[0]
    stale = None
    try:
        built = r["built_at"]
        if isinstance(built, str):
            built = datetime.fromisoformat(built.replace("Z", "+00:00"))
        if built.tzinfo is not None:
            built = built.astimezone(timezone.utc).replace(tzinfo=None)
        age_h = (datetime.now(timezone.utc).replace(tzinfo=None)
                 - built).total_seconds() / 3600
        stale = age_h > BRIEF_STALE_HOURS
    except (TypeError, ValueError):
        pass
    return {"exists": True, "brief_date": str(r["brief_date"])[:10],
            "built_at": str(r["built_at"]), "stale": stale,
            "payload": r["payload"]}


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

    bb = sub.add_parser("brief-build")
    bb.add_argument("--top", type=int, default=40)

    sub.add_parser("brief")

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
    elif args.cmd == "brief-build":
        out = build_brief(top=args.top)
    elif args.cmd == "brief":
        out = get_brief()
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

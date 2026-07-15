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
from zoneinfo import ZoneInfo

from agent import data

ACCOUNT = "agent"
BRIEF_STALE_HOURS = 36  # wall-clock cap; brief_date recency is checked too
ET = ZoneInfo("America/New_York")


def _et_today() -> date:
    """Trading dates are ET dates — a 21:00 ET nightly build is 01:00 UTC
    'tomorrow', and stamping UTC would mislabel the brief and split the
    per-night upsert across two rows."""
    return datetime.now(ET).date()


def _parse_date(v: str | None) -> date | None:
    return date.fromisoformat(v) if v else None


# ── the nightly research pack (desk_briefs) ─────────────────
#
# Built once per night by the data-refresh routine, right after the ingest,
# while the whole-market picture is fresh. The hourly trading cycle reads ONE
# dense payload instead of re-deriving regime/universe/movers/news with a
# dozen scans — its context goes to deciding, not gathering.


def _movers(store, *, top_k: int = 8, min_coverage: int | None = None) -> dict:
    """Gainers/losers/most-active across the last two WELL-COVERED sessions.

    A session only counts when it has >= ``min_coverage`` bars (default: the
    same FULL_COVERAGE_MIN the coverage verdict uses — one payload, one
    definition of trustworthy). Today's partial intraday top-up, or a
    partially-failed nightly, must never be one side of a market-wide
    comparison. Symbols with a split between the two sessions are excluded:
    daily_bars stores RAW closes, so a 10:1 split would otherwise fabricate
    a -90% 'loser'."""
    min_coverage = data.FULL_COVERAGE_MIN if min_coverage is None else min_coverage
    lo = _et_today() - timedelta(days=7)
    # Explicit total order — the REST lane pages with limit/offset, and
    # unordered offset pagination can skip/duplicate rows between pages.
    rows = store.select("daily_bars", columns="symbol,close,volume,date",
                        filters={"date": ("gte", lo)},
                        order=[("date", "asc"), ("symbol", "asc")])
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
    split_syms: set[str] = set()
    try:
        srows = store.select("ticker_splits", columns="symbol,execution_date",
                             filters={"execution_date": ("gt", prev_d)})
        split_syms = {r["symbol"] for r in srows
                      if str(r.get("execution_date") or "")[:10] <= cur_d}
    except Exception:  # noqa: BLE001 — split guard is best-effort
        pass
    prev = by_date[prev_d]
    changed = []
    for sym, (c, v) in by_date[cur_d].items():
        if c < 1.0 or sym in split_syms or any(ch in sym for ch in (".", "/", "=")):
            continue
        p = prev.get(sym)
        if p and p[0] > 0:
            changed.append({"symbol": sym, "close": round(c, 2),
                            "change_pct": round((c - p[0]) / p[0] * 100, 2),
                            "dollar_volume": round(c * v)})
    span = (date.fromisoformat(cur_d) - date.fromisoformat(prev_d)).days
    out = {"as_of": cur_d, "prior": prev_d, "span_days": span,
           "gainers": sorted(changed, key=lambda r: -r["change_pct"])[:top_k],
           "losers": sorted(changed, key=lambda r: r["change_pct"])[:top_k],
           "most_active": sorted(changed,
                                 key=lambda r: -r["dollar_volume"])[:top_k]}
    if span > 4:
        out["note"] = (f"the two covered sessions are {span} days apart — "
                       "these are multi-session moves, not one day")
    if split_syms:
        out["splits_excluded"] = sorted(split_syms)
    return out


def compute_screens(rows: list[dict], *, top_exclude: int = 40,
                    pool_max_rank: int = 1000) -> dict:
    """Full-market discovery screens (pure logic over bar rows).

    The trader's research funnel starts from the top-40 dollar-volume list,
    which is megacaps by construction — after a week it had traded 9 famous
    names. These screens surface what that funnel structurally misses:
    leaders among dollar-volume ranks 41..pool_max_rank. 3-MONTH structure
    (the window the nightly fresh set affordably supports over ~1,000 names):
    - beyond_megacaps: top 15 by 3-month return, price > $5, above its
      50-session average (uptrend gate at this window).
    - new_highs: within 2% of a fresh 70-session high with positive 3m return.
    ``rows``: dicts with symbol/date/close/volume, any order.
    """
    series: dict[str, list[tuple[str, float, float]]] = {}
    for r in rows:
        c = r.get("close")
        if c is None:
            continue
        series.setdefault(r["symbol"], []).append(
            (str(r["date"])[:10], float(c), float(r.get("volume") or 0.0)))
    for sym in series:
        series[sym].sort()
    # Rank by median dollar volume over each name's last 5 sessions.
    dv = {sym: sorted(c * v for _, c, v in s[-5:])[len(s[-5:]) // 2]
          for sym, s in series.items() if len(s) >= 5}
    ranked = sorted(dv, key=lambda s: -dv[s])
    rank = {sym: i + 1 for i, sym in enumerate(ranked)}

    pool = []
    for sym, s in series.items():
        rk = rank.get(sym)
        if rk is None or rk <= top_exclude or rk > pool_max_rank:
            continue
        closes = [c for _, c, _ in s]
        if len(closes) < 64 or closes[-1] < 5.0 \
                or any(ch in sym for ch in (".", "/", "=")):
            continue
        ret3m = closes[-1] / closes[-64] - 1.0
        avg50 = sum(closes[-50:]) / 50
        hi70 = max(closes[-70:])
        pool.append({"symbol": sym, "rank": rk, "close": round(closes[-1], 2),
                     "ret_3m_pct": round(ret3m * 100, 1),
                     "above_50d": closes[-1] > avg50,
                     "high_prox": round(closes[-1] / hi70, 4) if hi70 else 0})
    beyond = sorted((p for p in pool if p["ret_3m_pct"] > 0 and p["above_50d"]),
                    key=lambda p: -p["ret_3m_pct"])[:15]
    highs = sorted((p for p in pool if p["high_prox"] >= 0.98
                    and p["ret_3m_pct"] > 0),
                   key=lambda p: -p["high_prox"])[:10]
    strip = ("above_50d", "high_prox")
    return {"pool_size": len(pool),
            "note": "3-month structure over dollar-volume ranks "
                    f"{top_exclude + 1}-{pool_max_rank} of the fresh set — "
                    "the leaders the top-40 funnel structurally misses",
            "beyond_megacaps": [{k: v for k, v in p.items() if k not in strip}
                                for p in beyond],
            "new_highs": [{k: v for k, v in p.items() if k not in strip}
                          for p in highs]}


def _screens(store) -> dict:
    """Gather ~100 trading days of the fresh set and compute the screens."""
    lo = _et_today() - timedelta(days=150)
    rows = store.select("daily_bars", columns="symbol,date,close,volume",
                        filters={"date": ("gte", lo)},
                        order=[("date", "asc"), ("symbol", "asc")])
    return compute_screens(rows)


def build_brief(*, top: int = 40) -> dict:
    """Assemble tonight's research pack and upsert it (one row per ET date).

    Every section is independently fault-tolerant: on the REST lane this is
    ~70 sequential network calls, and one transient failure must degrade ONE
    section (recorded in ``payload.errors``), not abort the night's brief.
    """
    from agent.store import get_store

    store = get_store()
    today = _et_today()
    errors: list[str] = []

    def _safe(name, fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — degrade the section, keep the brief
            errors.append(f"{name}: {type(exc).__name__}: {exc}"[:200])
            return default

    regime = _safe("regime", data.regime, {})
    coverage = _safe("coverage", data.universe_coverage, {})
    top_syms = _safe("universe", lambda: data.universe(top), [])
    movers = _safe("movers", lambda: _movers(store),
                   {"as_of": None, "gainers": [], "losers": [],
                    "most_active": []})

    roster = (_safe("trend_roster", lambda: data.latest_indicators(top_syms), {})
              if top_syms else {})
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
        items = _safe(f"news:{sym}", lambda s=sym: data.news(s, limit=3), [])
        if items:
            headlines[sym] = [{"title": i.get("title"),
                               "published": str(i.get("published_utc"))[:16]}
                              for i in items]

    # Strategy Lab leaderboard: what the nightly sweeps currently rate as the
    # most robust rules (split-sample qualified, worst-half ranked) — the
    # trading cycle's grounding evidence, precomputed.
    def _lab():
        from agent import lab

        return lab.leaderboard(top=8)

    lab_board = _safe("lab", _lab, {})
    screens = _safe("screens", lambda: _screens(store), {})

    def _fundamentals():
        from agent import edgar

        return edgar.coverage(store)

    fundamentals = _safe("fundamentals", _fundamentals, {})

    payload = {"as_of": str(today), "regime": regime, "coverage": coverage,
               "universe_top": top_syms, "movers": movers,
               "trend_roster": trend, "headlines": headlines,
               "lab_leaderboard": lab_board, "screens": screens,
               "fundamentals": fundamentals,
               "errors": errors}
    built_at = datetime.now(timezone.utc).replace(tzinfo=None)
    values = {"payload": payload, "built_at": built_at}
    existing = store.select("desk_briefs",
                            filters={"account": ACCOUNT, "brief_date": today},
                            limit=1)
    if existing:
        store.update("desk_briefs", {"id": existing[0]["id"]}, values,
                     returning=False)
    else:
        try:
            store.insert("desk_briefs",
                         {"account": ACCOUNT, "brief_date": today, **values},
                         returning=False)
        except Exception:  # noqa: BLE001 — lost the insert race: update instead
            rows = store.select("desk_briefs",
                                filters={"account": ACCOUNT,
                                         "brief_date": today}, limit=1)
            if not rows:
                raise
            store.update("desk_briefs", {"id": rows[0]["id"]}, values,
                         returning=False)
    return {"ok": True, "brief_date": str(today),
            "universe": len(top_syms), "trend_roster": len(trend),
            "movers_as_of": movers.get("as_of"),
            "coverage_status": coverage.get("status"),
            "headline_symbols": len(headlines),
            "errors": errors}


def get_brief() -> dict:
    """The latest research pack, with an honest staleness flag.

    ``stale`` is ALWAYS a bool and defaults to True — a brief is fresh only
    when proven fresh on both clocks: built recently (wall-clock hours) AND
    built for today-or-yesterday in ET (session recency). Pure hours alone
    let a brief that predates an entire completed session read fresh after
    one missed nightly; an unparseable/NULL built_at must degrade to stale,
    never crash or read as fresh."""
    from agent.store import get_store

    rows = get_store().select("desk_briefs", filters={"account": ACCOUNT},
                              order=[("brief_date", "desc")], limit=1)
    if not rows:
        return {"exists": False,
                "note": "no brief built yet — scan manually this cycle"}
    r = rows[0]
    stale = True
    try:
        built = r["built_at"]
        if isinstance(built, str):
            built = datetime.fromisoformat(built.replace("Z", "+00:00"))
        if built.tzinfo is not None:
            built = built.astimezone(timezone.utc).replace(tzinfo=None)
        age_ok = ((datetime.now(timezone.utc).replace(tzinfo=None) - built)
                  .total_seconds() / 3600 <= BRIEF_STALE_HOURS)
        bdate = date.fromisoformat(str(r["brief_date"])[:10])
        session_ok = bdate >= _et_today() - timedelta(days=1)
        stale = not (age_ok and session_ok)
    except Exception:  # noqa: BLE001 — not provably fresh ⇒ stale
        stale = True
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

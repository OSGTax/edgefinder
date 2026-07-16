"""Trading-desk API — projects the agent's desk_* tables onto the page.

Read-only endpoints over the autonomous agent's own schema: the paper book +
equity curve, the latest decision (chart-forward picks with why-now /
rationale / news), the live thinking feed, the backtest evidence, the strategy
state, and the journal of pivots. All times are ISO UTC; the page normalizes.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import desc
from sqlalchemy.orm import Session

from agent.models import (
    ACCOUNT,
    DeskBacktest,
    DeskChangelog,
    DeskDecision,
    DeskEquity,
    DeskJournal,
    DeskOutcome,
    DeskPosition,
    DeskStrategyState,
    DeskThinking,
    DeskTrade,
    DeskWiki,
)
from dashboard.dependencies import get_db

router = APIRouter()

# An entry is "new" (lights the badge) for this many days after it ships.
# 7 aligns with the weekly UI-evolution routine's cadence — Monday visitors
# still see Friday's changes badged, Fridays are always clean of last week's.
WHATSNEW_SPOTLIGHT_DAYS = 7


def _iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _iso_any(v):
    """ISO string from a datetime OR an already-string timestamp (the two DB
    transports disagree on what a TIMESTAMP round-trips as)."""
    if v is None or isinstance(v, str):
        return v
    return _iso(v)


def _parse_meta(meta):
    """A mark_meta value as a dict (transports may hand JSON back as text)."""
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = None
    return meta if isinstance(meta, dict) else None


# /portfolio answers via ledger.state(), which recomputes cash by scanning
# the ENTIRE append-only desk_trades ledger — a table that grows forever —
# and the endpoint is polled once per open viewer. A short TTL cache bounds
# the full-scan frequency to once per _PORTFOLIO_TTL regardless of viewer
# count (same reset pattern as the other module caches below). Phase-later:
# aggregate cash server-side instead of rescanning.
_PORTFOLIO_TTL = 10.0
_portfolio_cache: tuple[float, dict] | None = None


@router.get("/portfolio")
def portfolio():
    """Cash, positions (marked), equity, and P&L — the book right now.

    ONE implementation of the account math (Phase E): this endpoint calls
    ``agent.ledger.state()`` — the exact cash-from-ledger recompute, option
    multiplier, and weight math the agent trades against — instead of the
    old parallel re-derivation, which had already drifted on rounding.
    ``vs_spy`` reuses the ledger's own TOTAL-RETURN SPY helpers
    (``_spy_closes`` / ``_spy_window_pct``): dividend back-adjusted closes,
    baseline = last close STRICTLY BEFORE the ET inception date, and None
    means too-young-to-benchmark, never zero. ``mark_meta`` (the latest
    snapshot's mark provenance) rides along additively. The response body is
    cached ~10s (the underlying cash recompute scans the whole trade ledger).
    """
    global _portfolio_cache
    now = time.time()
    if _portfolio_cache is not None and now - _portfolio_cache[0] < _PORTFOLIO_TTL:
        return _portfolio_cache[1]

    from agent import ledger
    from agent.store import get_store

    # Transport expectation: pg on Render (DATABASE_URL is set there); `auto`
    # may pick the rest lane when SUPABASE_* env is present — an intentional
    # fallback, same tables either way.
    store = get_store()
    st = ledger.state(store)

    # opened_at/marked_at are page-only fields state() doesn't emit — merge
    # them from the same positions projection state() just read.
    meta_by_sym: dict = {}
    try:
        for p in store.select("desk_positions",
                              columns="symbol,opened_at,marked_at",
                              filters={"account": ACCOUNT}):
            meta_by_sym[p["symbol"]] = p
    except Exception:  # noqa: BLE001 — cosmetic fields, never a 500
        meta_by_sym = {}
    positions = []
    for r in st["positions"]:
        m = meta_by_sym.get(r["symbol"]) or {}
        positions.append({**r, "opened_at": _iso_any(m.get("opened_at")),
                          "marked_at": _iso_any(m.get("marked_at"))})

    total_return_pct = round(st["total_return_pct"], 2)
    vs_spy = None
    first = store.select("desk_trades", columns="ts",
                         filters={"account": ACCOUNT},
                         order=[("ts", "asc")], limit=1)
    inception = ledger._et_date(first[0]["ts"]) if first and first[0].get("ts") else None
    if inception:
        spy = ledger._spy_closes(store, since=inception)
        spy_pct = ledger._spy_window_pct(spy, inception)
        if spy_pct is not None:
            vs_spy = {"inception": inception, "spy_as_of": spy[-1][0],
                      "spy_return_pct": spy_pct,
                      "alpha_pct": round(total_return_pct - spy_pct, 2)}

    out = {
        "account": st["account"], "cash": st["cash"],
        "positions_value": st["positions_value"], "equity": st["equity"],
        "starting_capital": st["starting_capital"],
        "total_pnl": st["total_pnl"],
        "total_return_pct": total_return_pct,
        "vs_spy": vs_spy,
        "mark_meta": st.get("mark_meta"),
        "positions": positions,
    }
    _portfolio_cache = (now, out)
    return out


@router.get("/equity")
def equity(db: Session = Depends(get_db), limit: int = Query(2000, le=10000),
           with_spy: int = Query(0, ge=0, le=1)):
    """Equity-curve series (oldest→newest) for the chart.

    Additive honesty (Phase E): a point whose stored ``mark_meta`` flagged
    the snapshot carries ``degraded: true`` (+ ``cost_marked_value_pct`` /
    ``cost_marked``) — cost-basis marks are fake-flat P&L and the chart must
    show them, not bury them. Plain calls keep the original bare-list shape.

    With ``with_spy=1`` the response becomes ``{"points": [...], "spy":
    [{date, pct}]}`` — a TOTAL-RETURN SPY series over the same window,
    rebased to inception = 0, computed with the ledger's own helpers (the
    exact convention behind /portfolio's vs_spy) so the chart can overlay
    the benchmark.
    """
    rows = (db.query(DeskEquity)
            .filter(DeskEquity.account == ACCOUNT)
            .order_by(desc(DeskEquity.ts)).limit(limit).all())
    rows.reverse()
    points = []
    for r in rows:
        p = {"t": _iso(r.ts), "equity": r.equity, "cash": r.cash,
             "positions_value": r.positions_value, "return_pct": r.return_pct}
        meta = _parse_meta(r.mark_meta)
        if meta and meta.get("degraded"):
            p["degraded"] = True
            if meta.get("cost_marked_value_pct") is not None:
                p["cost_marked_value_pct"] = meta["cost_marked_value_pct"]
            if meta.get("cost_marked"):
                p["cost_marked"] = meta["cost_marked"]
        points.append(p)
    if not with_spy:
        return points

    from agent import ledger
    from agent.store import get_store

    store = get_store()
    spy_series: list[dict] = []
    first = store.select("desk_trades", columns="ts",
                         filters={"account": ACCOUNT},
                         order=[("ts", "asc")], limit=1)
    inception = ledger._et_date(first[0]["ts"]) if first and first[0].get("ts") else None
    if inception:
        closes = ledger._spy_closes(store, since=inception)
        base = None
        for d, c in closes:
            if d < inception:
                base = c
            else:
                break
        if base:
            spy_series = [{"date": d, "pct": round((c - base) / base * 100, 2)}
                          for d, c in closes if d >= inception]
    return {"points": points, "spy": spy_series, "spy_inception": inception,
            "spy_basis": "total_return"}


@router.get("/decision/latest")
def latest_decision(db: Session = Depends(get_db)):
    """The most recent decision dossier: regime, picks, target book, watchlist."""
    d = (db.query(DeskDecision)
         .filter(DeskDecision.account == ACCOUNT)
         .order_by(desc(DeskDecision.ts)).first())
    if not d:
        return {"exists": False}
    return {
        "exists": True, "run_id": d.run_id, "ts": _iso(d.ts),
        "decision_date": str(d.decision_date) if d.decision_date else None,
        "regime": d.regime, "summary": d.summary,
        "target_weights": d.target_weights or {}, "picks": d.picks or [],
        "watchlist": d.watchlist or [], "rejected": d.rejected or [],
        "strategy_version": d.strategy_version,
    }


@router.get("/decisions")
def decisions_archive(db: Session = Depends(get_db),
                      limit: int = Query(10, le=50),
                      before: str | None = None):
    """The decision archive, newest first — the full dossier per row (same
    shape as /decision/latest, plus ``id``). Page with ``before=<row id>``
    (or an ISO timestamp): returns rows strictly older than it;
    ``next_before`` is ready to pass back when more rows may exist."""
    q = db.query(DeskDecision).filter(DeskDecision.account == ACCOUNT)
    if before:
        b = before.strip()
        if b.isdigit():
            # keyset pagination on (ts, id) — the sort key. Filtering on raw
            # id would silently skip rows whenever id order and ts order
            # disagree (backfills, imported history).
            from sqlalchemy import and_, or_

            anchor = (db.query(DeskDecision)
                      .filter(DeskDecision.account == ACCOUNT,
                              DeskDecision.id == int(b)).first())
            if anchor is None:
                raise HTTPException(status_code=404,
                                    detail="before row id not found")
            q = q.filter(or_(DeskDecision.ts < anchor.ts,
                             and_(DeskDecision.ts == anchor.ts,
                                  DeskDecision.id < anchor.id)))
        else:
            from datetime import datetime as _dt
            try:
                ts = _dt.fromisoformat(b.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail="before must be a decision row id or an ISO timestamp")
            if ts.tzinfo is not None:  # desk timestamps are naive UTC
                ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
            q = q.filter(DeskDecision.ts < ts)
    rows = (q.order_by(desc(DeskDecision.ts), desc(DeskDecision.id))
            .limit(limit).all())
    out = [{
        "id": d.id, "run_id": d.run_id, "ts": _iso(d.ts),
        "decision_date": str(d.decision_date) if d.decision_date else None,
        "regime": d.regime, "summary": d.summary,
        "target_weights": d.target_weights or {}, "picks": d.picks or [],
        "watchlist": d.watchlist or [], "rejected": d.rejected or [],
        "strategy_version": d.strategy_version,
    } for d in rows]
    return {"decisions": out,
            "next_before": out[-1]["id"] if out and len(out) == limit else None}


@router.get("/outcomes")
def outcomes_scoreboard(db: Session = Depends(get_db),
                        status: str = Query("all"),
                        limit: int = Query(100, le=200)):
    """The predictions scoreboard — machine-graded pick facts
    (``desk_outcomes``, written by ``agent.ledger grade``) joined with each
    pick's own words (prediction / horizon / kill free text from the
    decision row) so the page shows what was SAID next to what HAPPENED.

    Open rows come first (newest decision first), then recent closed rows.
    ``sessions_elapsed`` counts stored SPY closes on/after the decision's ET
    date (the ledger's session convention) for horizon countdowns.
    ``summary`` carries whole-table counts by status and verdict plus the
    hit rate over closed, reflection-graded rows (TRUE vs FALSE)."""
    from bisect import bisect_left

    from sqlalchemy import func as safunc

    from agent import occ
    from agent.ledger import _et_date
    from edgefinder.db.models import DailyBar

    def fetch(st: str, lim: int):
        if lim <= 0:
            return []
        return (db.query(DeskOutcome)
                .filter(DeskOutcome.account == ACCOUNT,
                        DeskOutcome.status == st)
                .order_by(desc(DeskOutcome.id)).limit(lim).all())

    if status in ("open", "closed"):
        rows = fetch(status, limit)
    else:
        rows = fetch("open", limit)
        rows += fetch("closed", limit - len(rows))

    # Pick context: one decisions query covering every run_id involved.
    run_ids = sorted({r.run_id for r in rows})
    decisions: dict[str, DeskDecision] = {}
    if run_ids:
        for d in (db.query(DeskDecision)
                  .filter(DeskDecision.account == ACCOUNT,
                          DeskDecision.run_id.in_(run_ids)).all()):
            decisions[d.run_id] = d
    run_dates = {rid: (_et_date(d.ts) if d.ts is not None else None)
                 for rid, d in decisions.items()}

    # Completed SPY sessions since each decision — one bounded date query.
    spy_dates: list[str] = []
    dated = [v for v in run_dates.values() if v]
    if dated:
        spy_dates = [str(x[0])[:10] for x in
                     (db.query(DailyBar.date)
                      .filter(DailyBar.symbol == "SPY",
                              DailyBar.date >= min(dated))
                      .order_by(DailyBar.date).all())]

    out_rows = []
    for r in rows:
        d = decisions.get(r.run_id)
        pick: dict = {}
        if d:
            for p in (d.picks or []):
                if isinstance(p, dict) \
                        and str(p.get("symbol") or "").upper() == r.symbol:
                    pick = p
                    break
        rd = run_dates.get(r.run_id)
        sessions = (len(spy_dates) - bisect_left(spy_dates, rd)
                    if rd and spy_dates else None)
        out_rows.append({
            "id": r.id, "run_id": r.run_id, "symbol": r.symbol,
            "is_option": occ.is_option(r.symbol),
            "status": r.status, "decision_ts": _iso(d.ts) if d else None,
            "action": pick.get("action"), "prediction": pick.get("prediction"),
            "kill": pick.get("kill"),
            "horizon_days": r.horizon_days,
            "horizon_elapsed": r.horizon_elapsed,
            "sessions_elapsed": sessions,
            "entry_avg_px": r.entry_avg_px, "mark_px": r.mark_px,
            "mark_basis": r.mark_basis, "since_pct": r.since_pct,
            "spy_pct": r.spy_pct, "alpha_pct": r.alpha_pct,
            "exit_kind": r.exit_kind, "exit_avg_px": r.exit_avg_px,
            "realized_pnl": r.realized_pnl,
            "kill_level": r.kill_level, "kill_breached": r.kill_breached,
            "degraded": bool(r.degraded),
            "verdict": r.verdict, "verdict_note": r.verdict_note,
            "grade_date": str(r.grade_date) if r.grade_date else None,
            "graded_at": _iso(r.graded_at),
        })
    opens = [x for x in out_rows if x["status"] == "open"]
    closed = [x for x in out_rows if x["status"] != "open"]
    key = lambda x: (x["decision_ts"] or "", x["id"])  # noqa: E731
    opens.sort(key=key, reverse=True)
    closed.sort(key=key, reverse=True)

    status_counts = {s: int(n) for s, n in
                     (db.query(DeskOutcome.status, safunc.count())
                      .filter(DeskOutcome.account == ACCOUNT)
                      .group_by(DeskOutcome.status).all())}
    verdict_counts: dict[str, int] = {}
    for v, n in (db.query(DeskOutcome.verdict, safunc.count())
                 .filter(DeskOutcome.account == ACCOUNT)
                 .group_by(DeskOutcome.verdict).all()):
        verdict_counts[v or "ungraded"] = int(n)

    def _closed_verdicts(v: str) -> int:
        return int(db.query(safunc.count())
                   .filter(DeskOutcome.account == ACCOUNT,
                           DeskOutcome.status == "closed",
                           DeskOutcome.verdict == v).scalar() or 0)

    hits, misses = _closed_verdicts("TRUE"), _closed_verdicts("FALSE")
    return {
        "summary": {
            "open": status_counts.get("open", 0),
            "closed": status_counts.get("closed", 0),
            "verdicts": verdict_counts,
            "closed_graded": hits + misses,
            "hit_rate_pct": (round(hits / (hits + misses) * 100, 1)
                             if hits + misses else None),
        },
        "rows": opens + closed,
    }


@router.get("/thinking")
def thinking(db: Session = Depends(get_db), limit: int = Query(60, le=500),
             run_id: str | None = None):
    """Recent thinking-feed lines (newest first). Defaults to the latest run."""
    if run_id is None:
        last = (db.query(DeskThinking.run_id)
                .filter(DeskThinking.account == ACCOUNT)
                .order_by(desc(DeskThinking.ts)).first())
        run_id = last[0] if last else None
    q = db.query(DeskThinking).filter(DeskThinking.account == ACCOUNT)
    if run_id is not None:
        q = q.filter(DeskThinking.run_id == run_id)
    rows = q.order_by(desc(DeskThinking.ts)).limit(limit).all()
    return {"run_id": run_id,
            "lines": [{"t": _iso(r.ts), "phase": r.phase, "text": r.text}
                      for r in rows]}


@router.get("/backtests")
def backtests(db: Session = Depends(get_db), limit: int = Query(20, le=100)):
    """Recent backtests the agent ran as grounding evidence."""
    rows = (db.query(DeskBacktest)
            .filter(DeskBacktest.account == ACCOUNT)
            .order_by(desc(DeskBacktest.ts)).limit(limit).all())
    return [{"t": _iso(r.ts), "label": r.label, "spec": r.spec or {},
             "result": r.result or {}} for r in rows]


@router.get("/strategy")
def strategy(db: Session = Depends(get_db)):
    """The agent's current strategy + its journal of pivots/tweaks."""
    cur = (db.query(DeskStrategyState)
           .filter(DeskStrategyState.account == ACCOUNT)
           .order_by(desc(DeskStrategyState.version), desc(DeskStrategyState.id))
           .first())
    journal = (db.query(DeskJournal)
               .filter(DeskJournal.account == ACCOUNT)
               .order_by(desc(DeskJournal.ts)).limit(30).all())
    return {
        "current": None if not cur else {
            "version": cur.version, "name": cur.name, "thesis": cur.thesis,
            "rules": cur.rules or {}, "params": cur.params or {},
            "updated_at": _iso(cur.updated_at)},
        "journal": [{"t": _iso(j.ts), "kind": j.kind, "title": j.title,
                     "body": j.body, "version_from": j.version_from,
                     "version_to": j.version_to} for j in journal],
    }


@router.get("/wiki")
def wiki(db: Session = Depends(get_db)):
    """The agent's lessons wiki — curated pages of what it has learned from
    its own measured wins and losses (Karpathy-style system-prompt learning).
    Served in canonical page order for the "What the AI has learned" card."""
    rows = db.query(DeskWiki).filter(DeskWiki.account == ACCOUNT).all()
    # Mirrors agent.brain.WIKI_SLUGS — the canonical page order.
    order = {"playbook": 0, "setups": 1, "lessons": 2, "mistakes": 3,
             "postmortems": 4, "market-notes": 5}
    rows.sort(key=lambda r: order.get(r.slug, 9))
    return {"pages": [{"slug": r.slug, "title": r.title, "body": r.body,
                       "revision": r.revision,
                       "updated_at": _iso(r.updated_at)} for r in rows]}


@router.get("/regime")
def regime():
    """A compact market-regime read (SPY/QQQ/IWM trend) for the header chip.

    Computed from the kept bar layer; returns a neutral stub if the data
    layer is unreachable so the page never hard-fails on the header.
    """
    try:
        from agent import data
        return data.regime()
    except Exception as exc:  # noqa: BLE001 — header must degrade gracefully
        return {"tag": "neutral", "error": f"{type(exc).__name__}: {exc}", "indices": {}}


@router.get("/movers")
def movers(db: Session = Depends(get_db), top: int = Query(5, ge=1, le=15)):
    """Top gainers / losers / most-active as of the last completed session.

    Computed read-only from the fresh daily-bar hot set (the ~500-name universe
    the refresh keeps current) — biggest close-to-close moves and largest dollar
    volume. No external calls; this is last-close data, not a live intraday tape
    (the live tape is the SSE ``/stream``).
    """
    from sqlalchemy import func as safunc

    from edgefinder.db.models import DailyBar

    latest = db.query(safunc.max(DailyBar.date)).scalar()
    if latest is None:
        return {"as_of": None, "prior": None,
                "gainers": [], "losers": [], "most_active": []}
    prior = (db.query(safunc.max(DailyBar.date))
             .filter(DailyBar.date < latest).scalar())
    cur = (db.query(DailyBar.symbol, DailyBar.close, DailyBar.volume)
           .filter(DailyBar.date == latest).all())
    prev = {} if prior is None else {
        s: c for s, c in db.query(DailyBar.symbol, DailyBar.close)
        .filter(DailyBar.date == prior).all()}
    rows = []
    for sym, close, vol in cur:
        if close is None or close < 1.0 or any(ch in sym for ch in (".", "/", "=")):
            continue
        pc = prev.get(sym)
        chg = ((close - pc) / pc * 100.0) if pc else None
        rows.append({"symbol": sym, "close": round(close, 2),
                     "change_pct": round(chg, 2) if chg is not None else None,
                     "dollar_volume": round(close * (vol or 0.0))})
    with_chg = [r for r in rows if r["change_pct"] is not None]
    return {
        "as_of": str(latest), "prior": (str(prior) if prior else None),
        "gainers": sorted(with_chg, key=lambda r: -r["change_pct"])[:top],
        "losers": sorted(with_chg, key=lambda r: r["change_pct"])[:top],
        "most_active": sorted(rows, key=lambda r: -r["dollar_volume"])[:top],
    }


@router.get("/holding-stats")
def holding_stats(db: Session = Depends(get_db),
                  spark_days: int = Query(30, ge=5, le=120)):
    """Per-held-name enrichment from the daily-bar hot set: last-session day
    change, 52-week high/low, and a short close series for a sparkline. Read-only
    (no external calls); options legs are skipped (not in daily_bars)."""
    from datetime import timedelta

    from sqlalchemy import func as safunc

    from agent import occ
    from edgefinder.db.models import DailyBar

    held = [s for (s,) in db.query(DeskPosition.symbol)
            .filter(DeskPosition.account == ACCOUNT).all() if not occ.is_option(s)]
    if not held:
        return {"as_of": None, "symbols": {}}
    latest = (db.query(safunc.max(DailyBar.date))
              .filter(DailyBar.symbol.in_(held)).scalar())
    if latest is None:
        return {"as_of": None, "symbols": {}}
    lo = latest - timedelta(days=400)  # ~252 trading days of headroom
    rows = (db.query(DailyBar.symbol, DailyBar.close)
            .filter(DailyBar.symbol.in_(held), DailyBar.date >= lo)
            .order_by(DailyBar.symbol, DailyBar.date).all())
    series: dict[str, list[float]] = {}
    for sym, close in rows:
        if close is not None:
            series.setdefault(sym, []).append(float(close))
    out = {}
    for sym, closes in series.items():
        if len(closes) < 2:
            continue
        last, prev = closes[-1], closes[-2]
        wk = closes[-252:]
        out[sym] = {
            "last": round(last, 2), "prev": round(prev, 2),
            "day_change_pct": round((last - prev) / prev * 100, 2) if prev else None,
            "wk52_high": round(max(wk), 2), "wk52_low": round(min(wk), 2),
            "spark": [round(c, 2) for c in closes[-spark_days:]],
        }
    return {"as_of": str(latest), "symbols": out}


@router.get("/dividends")
def holdings_dividends(db: Session = Depends(get_db)):
    """Per-holding dividend calendar from the ``dividends`` table (fed by the
    refresh's Alpaca corporate-actions ingest): the most recent ex-dividend and
    the next upcoming one, plus a trailing-4 annual estimate. Read-only."""
    from datetime import date

    from agent import occ
    from edgefinder.db.models import DividendRecord

    held = [s for (s,) in db.query(DeskPosition.symbol)
            .filter(DeskPosition.account == ACCOUNT).all() if not occ.is_option(s)]
    today = str(date.today())
    out = []
    for sym in held:
        rows = (db.query(DividendRecord).filter(DividendRecord.symbol == sym)
                .order_by(desc(DividendRecord.ex_date)).limit(8).all())
        if not rows:
            out.append({"symbol": sym, "has_dividend": False})
            continue
        past = [r for r in rows if str(r.ex_date) <= today]
        upcoming = sorted((r for r in rows if str(r.ex_date) > today),
                          key=lambda r: str(r.ex_date))
        last = past[0] if past else None
        nxt = upcoming[0] if upcoming else None
        ttm = round(sum(r.cash_amount or 0 for r in rows[:4]), 4)
        out.append({
            "symbol": sym, "has_dividend": True,
            "last_ex_date": str(last.ex_date) if last else None,
            "last_amount": round(last.cash_amount, 4) if last and last.cash_amount else None,
            "next_ex_date": str(nxt.ex_date) if nxt else None,
            "next_amount": round(nxt.cash_amount, 4) if nxt and nxt.cash_amount else None,
            "ttm_amount": ttm,
        })
    return {"as_of": today, "holdings": out}


@router.get("/quotes")
def live_quotes():
    """Point-in-time snapshot of the live SIP quote cache (the tools read this).

    Each entry: bid/ask/mid/last + age_secs + stale. ``connected`` tells you if
    the WebSocket is currently up; a warmed-but-disconnected cache still serves
    (clearly-aged) quotes."""
    from agent.streamer import cache
    return cache.snapshot()


# The equity market session ('regular' | 'extended' | 'closed'), 60s-cached
# so the 1 Hz SSE loop never talks to Alpaca more than once a minute. None
# when keys/clock are unavailable — the page then falls back to
# freshness-only pill logic (never a fake "open").
_SESSION_TTL = 60.0
_SESSION_FETCH_TIMEOUT = 5.0   # alpaca-py sets NO HTTP timeout — bound it here
_SESSION_ERROR_BACKOFF = 30.0  # after a timeout/failure, don't retry at once
_session_cache: tuple[float, str | None] = (0.0, None)
# Single-flight latch: at most one refresh in flight. A plain bool (not an
# asyncio.Lock) on purpose — waiters must serve stale IMMEDIATELY rather than
# queue behind the refresher, the event loop is single-threaded so flag flips
# never race, and a module-level Lock would bind to whichever event loop
# touched it first (breaking under test clients / server restarts).
_session_refreshing = False


def _fetch_market_session() -> str | None:
    """One blocking broker clock round-trip (runs on a worker thread)."""
    try:
        from agent import broker

        if broker.enabled():
            return broker.Broker().session()
    except Exception:  # noqa: BLE001 — unknown session, never a dead stream
        pass
    return None


async def _market_session() -> str | None:
    """The cached market session for SSE frames — single-flight refresh.

    On TTL expiry exactly ONE frame (across ALL open SSE connections)
    performs the refresh; every other frame serves the stale value
    immediately instead of stacking N simultaneous broker calls. The refresh
    is bounded by ``asyncio.wait_for`` because alpaca-py sets no HTTP
    timeout — unbounded, one hung socket per frame would progressively pin
    every default-executor thread and stall ALL streams. On timeout/error
    the stale/null value keeps serving and the cache timestamp moves forward
    so the next attempt backs off ~30s instead of hammering a dead socket.
    A timed-out ``to_thread`` worker may linger until its socket dies, but
    single-flight bounds that leak to one thread at a time.
    """
    global _session_cache, _session_refreshing
    now = time.time()
    ts, val = _session_cache
    if now - ts < _SESSION_TTL or _session_refreshing:
        return val
    _session_refreshing = True
    try:
        val = await asyncio.wait_for(asyncio.to_thread(_fetch_market_session),
                                     timeout=_SESSION_FETCH_TIMEOUT)
        _session_cache = (now, val)
    except Exception:  # noqa: BLE001 — timeout/cancel: keep stale, back off
        _session_cache = (now - _SESSION_TTL + _SESSION_ERROR_BACKOFF, val)
    finally:
        _session_refreshing = False
    return val


@router.get("/stream")
async def stream():
    """Server-Sent Events: the live tape for the /desk page.

    Emits an ``event: quotes`` frame with the full cache snapshot every second
    (the universe is small, so full snapshots beat diff bookkeeping). Each
    frame also carries ``session`` (regular|extended|closed|null) so the page
    can tell a quiet-but-open tape from a closed market — the LIVE pill's
    honesty input. The browser's EventSource auto-reconnects; frames double
    as heartbeats."""
    import json as _json

    from fastapi.responses import StreamingResponse

    from agent.streamer import cache

    async def gen():
        while True:
            snap = cache.snapshot()
            # the cached path returns instantly; the once-a-minute refresh is
            # single-flight and time-bounded (see _market_session), so no
            # frame — on any connection — ever stalls on a network call.
            snap["session"] = await _market_session()
            payload = _json.dumps(snap, default=str)
            yield f"event: quotes\ndata: {payload}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ── options endpoint guards (E4): allowlist + a tiny rate limit ──
#
# /options/{symbol} fans out to LIVE paid Alpaca calls (quote + chain) for
# whatever string is in the URL — a public endpoint must not be an open
# proxy to a metered API. The allowlist is every symbol the desk actually
# has a reason to show: held positions (options mapped to their underlying),
# armed/tripped tripwires, the latest decision's picks + watchlist, and the
# streamer's seed universe. One cheap cached query, 60s TTL.

_OPTIONS_ALLOW_TTL = 60.0
_options_allow: tuple[float, frozenset] | None = None


def _options_allowlist(db: Session) -> frozenset[str]:
    global _options_allow
    now = time.time()
    if _options_allow is not None and now - _options_allow[0] < _OPTIONS_ALLOW_TTL:
        return _options_allow[1]

    from agent import occ
    from agent.models import DeskWatch
    from config.settings import settings

    syms: set[str] = set()

    def add(s) -> None:
        s = str(s or "").upper().strip()
        if not s or s == "BOOK":
            return
        if occ.is_option(s):
            try:
                s = occ.parse(s)["underlying"]
            except Exception:  # noqa: BLE001 — a garbled OCC symbol adds nothing
                return
        syms.add(s)

    for (s,) in (db.query(DeskPosition.symbol)
                 .filter(DeskPosition.account == ACCOUNT).all()):
        add(s)
    for (s,) in (db.query(DeskWatch.symbol)
                 .filter(DeskWatch.account == ACCOUNT,
                         DeskWatch.status.in_(("armed", "tripped"))).all()):
        add(s)
    d = (db.query(DeskDecision).filter(DeskDecision.account == ACCOUNT)
         .order_by(desc(DeskDecision.ts)).first())
    if d:
        for p in (d.picks or []):
            add(p.get("symbol") if isinstance(p, dict) else p)
        for w in (d.watchlist or []):
            add(w.get("symbol") if isinstance(w, dict) else w)
    for s in str(settings.stream_symbols or "").split(","):
        add(s)

    allow = frozenset(syms)
    _options_allow = (now, allow)
    return allow


class _TokenBucket:
    """Tiny per-key token bucket — dependency-free, in-process. Refills
    continuously; a full-again bucket is pruned so the dict stays bounded."""

    def __init__(self, capacity: float = 30.0, refill_per_sec: float = 0.5):
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, ts)

    def allow(self, key: str) -> bool:
        now = time.time()
        tokens, ts = self._buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - ts) * self.refill)
        ok = tokens >= 1.0
        self._buckets[key] = (tokens - 1.0 if ok else tokens, now)
        if len(self._buckets) > 2048:  # bound memory under key churn
            self._buckets = {k: v for k, v in self._buckets.items()
                             if v[0] < self.capacity - 1.0}
        return ok

    def reset(self) -> None:
        self._buckets.clear()


_options_bucket = _TokenBucket()


def _client_key(request: Request) -> str:
    # Prefer the direct peer; consult X-Forwarded-For only when a proxy set
    # it, and then take the LAST hop, not the first: the head of XFF is
    # client-supplied (an attacker can pre-populate it and rotate a fake
    # first hop per request to mint fresh buckets), while the last entry is
    # the one appended by the nearest trusted proxy (Render's) — the real
    # peer that proxy saw.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        last = fwd.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"


@router.get("/options/{symbol}")
def options_summary(symbol: str, request: Request, db: Session = Depends(get_db)):
    """Live options intelligence for an underlying: spot, focus expiry, ATM IV,
    straddle-implied expected move, 25-delta skew, and a strikes table around
    the money. 60s-cached; degrades to {"available": false} without keys.
    Allowlisted (held ∪ watched ∪ latest picks/watchlist ∪ streamed seeds)
    and rate-limited — this endpoint triggers metered external calls."""
    from agent import options_data

    sym = symbol.upper().strip()
    if not _options_bucket.allow(_client_key(request)):
        raise HTTPException(status_code=429,
                            detail="too many options requests — slow down")
    if sym not in _options_allowlist(db):
        raise HTTPException(status_code=404,
                            detail=f"{sym} is not on the desk's radar")
    return options_data.get_summary(sym)


@router.get("/options/{symbol}/history")
def options_history(symbol: str, db: Session = Depends(get_db),
                    limit: int = Query(250, le=1000)):
    """The IV data bank series (one snapshot/day, accumulated by the agent's
    refresh) — powers the IV/expected-move history charts. DB-only (no
    external calls), so the allowlist alone is enough."""
    from agent import options_data
    from agent.store import get_store

    sym = symbol.upper().strip()
    if sym not in _options_allowlist(db):
        raise HTTPException(status_code=404,
                            detail=f"{sym} is not on the desk's radar")
    return {"symbol": sym,
            "series": options_data.history(get_store(), sym, limit=limit)}


@router.get("/broker-health")
def broker_health():
    """Preflight diagnostic: are the Alpaca keys on this host valid + SIP-entitled?

    Exposes NO secrets and no dollar amounts — just reachability, account
    status, and one SPY quote timestamp proving the data entitlement works.
    Safe to leave up; the streamer build replaces it as the health source.
    """
    from agent import broker

    out: dict = {"keys_present": broker.enabled(), "paper": None,
                 "account_status": None, "feed": None, "quote": None, "error": None}
    if not out["keys_present"]:
        out["error"] = "no EDGEFINDER_ALPACA_* keys in this environment"
        return out
    try:
        b = broker.Broker()
        acct = b.account()
        out["paper"] = acct.get("paper")
        out["account_status"] = str(acct.get("status"))
        out["feed"] = broker.resolve_creds()["feed"]
        q = b.quotes(["SPY"]).get("SPY") or {}
        out["quote"] = {"symbol": "SPY", "bid": q.get("bid"), "ask": q.get("ask"),
                        "t": q.get("t")}
    except Exception as exc:  # noqa: BLE001 — diagnostic must report, not raise
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


@router.get("/data-health")
def data_health(db: Session = Depends(get_db)):
    """Freshness of the market-data asset behind research (not the live tape).

    Bar age alone can't detect a dead nightly ingest — the hourly top-up keeps
    a handful of held names current while the other ~2,000 symbols go stale.
    This counts bar rows per recent date and reports sessions since the last
    full-coverage ingest (one definition, shared with agent.preflight).

    ``marks`` (additive, Phase E) surfaces the latest equity snapshot's mark
    provenance — degraded flag + which symbols were marked at cost basis —
    so the desk pill can show a fake-flat book for what it is.
    """
    from datetime import timedelta

    from sqlalchemy import func as safunc

    from agent.data import coverage_verdict
    from edgefinder.db.models import DailyBar

    latest = db.query(safunc.max(DailyBar.date)).scalar()
    if latest is None:
        out = coverage_verdict([])
    else:
        lo = latest - timedelta(days=21)
        rows = (db.query(DailyBar.date, safunc.count(DailyBar.symbol))
                .filter(DailyBar.date >= lo).group_by(DailyBar.date).all())
        out = coverage_verdict(rows)
    meta = _parse_meta(
        (db.query(DeskEquity.mark_meta)
         .filter(DeskEquity.account == ACCOUNT)
         .order_by(desc(DeskEquity.ts), desc(DeskEquity.id))
         .first() or [None])[0])
    out["marks"] = None if meta is None else {
        "degraded": bool(meta.get("degraded")),
        "cost_marked": meta.get("cost_marked") or [],
        "cost_marked_value_pct": meta.get("cost_marked_value_pct"),
    }
    return out


@router.get("/watch")
def attention(db: Session = Depends(get_db)):
    """The attention system, visible: armed/tripped tripwires and the
    trader's planned self-scheduled check-ins (with reasons). Read-only."""
    from datetime import datetime, timedelta

    from agent.models import DeskWake, DeskWatch

    wires = (db.query(DeskWatch).filter(DeskWatch.account == ACCOUNT)
             .order_by(desc(DeskWatch.id)).limit(40).all())
    horizon = datetime.utcnow() - timedelta(hours=24)
    wakes = (db.query(DeskWake)
             .filter(DeskWake.account == ACCOUNT, DeskWake.at >= horizon)
             .order_by(DeskWake.at).limit(40).all())
    return {
        "watches": [{
            "id": w.id, "symbol": w.symbol, "kind": w.kind, "level": w.level,
            "reason": w.reason, "status": w.status, "armed_at": _iso(w.armed_at),
            "until": _iso(w.until), "tripped_at": _iso(w.tripped_at),
            "tripped_price": w.tripped_price, "run_id": w.run_id,
        } for w in wires],
        "wakes": [{
            "id": k.id, "at": _iso(k.at), "reason": k.reason,
            "run_id": k.run_id, "created_at": _iso(k.created_at),
            "honored_run_id": k.honored_run_id,
        } for k in wakes],
    }


@router.get("/lab")
def lab_leaderboard():
    """The Strategy Lab's current leaderboard — split-sample qualified rules
    ranked by their WORST half's excess vs SPY, always with the tested count
    (multiple-comparisons honesty). Read-only; same source the brief carries."""
    from agent import lab

    try:
        return lab.leaderboard(top=10)
    except Exception as exc:  # noqa: BLE001 — panel must degrade, not 500
        return {"error": f"{type(exc).__name__}: {exc}", "top": [],
                "combos_tested": 0, "qualified": 0}


@router.get("/brief")
def research_brief(db: Session = Depends(get_db)):
    """The nightly research pack the agent reads first each cycle — surfaced
    so the owner can inspect exactly what the trader saw. Read-only."""
    from agent.models import DeskBrief

    r = (db.query(DeskBrief).filter(DeskBrief.account == ACCOUNT)
         .order_by(desc(DeskBrief.brief_date)).first())
    if not r:
        return {"exists": False}
    return {"exists": True, "brief_date": str(r.brief_date),
            "built_at": _iso(r.built_at), "payload": r.payload}


@router.get("/whatsnew")
def whatsnew(db: Session = Depends(get_db), limit: int = Query(25, le=100)):
    """The "What's New" feed — dashboard improvements the agent shipped.

    Each entry carries a plain-English explanation of the feature. ``new_count``
    is how many landed inside the spotlight window (drives the header badge);
    ``latest`` is the single newest entry (the attention banner reads it)."""
    from datetime import datetime, timedelta

    rows = (db.query(DeskChangelog)
            .filter(DeskChangelog.account == ACCOUNT)
            .order_by(desc(DeskChangelog.ts)).limit(limit).all())
    entries = [{"id": r.id, "t": _iso(r.ts), "kind": r.kind, "title": r.title,
                "detail": r.detail, "version": r.version} for r in rows]
    cutoff = datetime.now(timezone.utc) - timedelta(days=WHATSNEW_SPOTLIGHT_DAYS)
    new_count = sum(
        1 for r in rows
        if r.ts and (r.ts if r.ts.tzinfo else r.ts.replace(tzinfo=timezone.utc)) >= cutoff)
    return {"entries": entries, "new_count": new_count,
            "spotlight_days": WHATSNEW_SPOTLIGHT_DAYS,
            "latest": entries[0] if entries else None}


@router.get("/trades")
def trades(db: Session = Depends(get_db), limit: int = Query(100, le=1000)):
    """Recent executed fills (newest first)."""
    rows = (db.query(DeskTrade)
            .filter(DeskTrade.account == ACCOUNT)
            .order_by(desc(DeskTrade.ts)).limit(limit).all())
    return [{"t": _iso(r.ts), "symbol": r.symbol, "side": r.side,
             "shares": r.shares, "price": r.price, "dollars": r.dollars,
             "rationale": r.rationale, "run_id": r.run_id,
             "fill_quote": r.fill_quote or None} for r in rows]

"""Trading-desk API — projects the desk's book + knowledge onto the page.

Read-only endpoints over two sources (REBUILD-V4):

- **The Alpaca paper account** is the book of record: the account hero and
  positions come live from ``agent.trade`` (10s TTL), the equity curve from
  the nightly ``desk_portfolio_history`` snapshots, fills from the
  ``desk_orders`` mirror, and resting protection from the broker's own open
  orders. The frozen pre-migration book (Era 1) is read from the renamed
  ``era1_*`` archive tables when they exist — every read degrades to
  "no era-1 rows" before the cutover rename.
- **The knowledge store** (decisions, thinking, outcomes, wiki, claims,
  proposals) renders exactly as before — the decision-side registry is the
  point of the desk and survived the migration unchanged.

All times are ISO UTC; the page normalizes.
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
    DeskJournal,
    DeskOutcome,
    DeskStrategyState,
    DeskThinking,
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


def _stamp_utc(s: str) -> str:
    """A zone-less date-TIME string stamped UTC; anything else untouched.

    Date-only text ("2026-05-09") names a calendar day, not an instant, and
    malformed text is not ours to guess at — both pass through.
    """
    from datetime import datetime as _dt

    if len(s) <= 10:
        return s
    try:
        dt = _dt.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    return s if dt.tzinfo else dt.replace(tzinfo=timezone.utc).isoformat()


def _iso_any(v):
    """ISO string from a datetime OR an already-string timestamp (the two DB
    transports disagree on what a TIMESTAMP round-trips as).

    Every returned timestamp carries an explicit offset. The browser reads an
    offset-less date-time as LOCAL time (ES spec), so a naked string would
    render a 19:25 UTC fill as 19:25 on the viewer's clock — hours off the
    broker's own stamp, on the very surfaces built to check it. The pg
    transport hands back datetimes (``_iso`` stamps those); PostgREST hands
    back text, which is where a naked one can come from.
    """
    if v is None:
        return v
    if isinstance(v, str):
        return _stamp_utc(v)
    return _iso(v)


def _json_dict(v):
    """A JSON column value as a dict (transports may hand JSON back as text)."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            v = None
    return v if isinstance(v, dict) else {}


# ── Era 1: the frozen pre-migration book ─────────────────────────────────
#
# At cutover the old ledger tables are RENAMED desk_trades→era1_trades /
# desk_equity→era1_equity (REBUILD-V4 runbook). The archive is read-only and
# may simply not exist yet (pre-cutover), so every read must degrade to []
# rather than 500. The pg transport resolves table names through the shared
# SQLAlchemy metadata, so the era1_* shapes are registered here — the raw
# prod DDL never creates them (only the rename does); registration is not
# existence.


def _register_era1_tables() -> None:
    from sqlalchemy import Column, DateTime, Float, Integer, String, Table, Text
    from sqlalchemy.dialects.sqlite import JSON

    from edgefinder.db.engine import Base

    if "era1_trades" not in Base.metadata.tables:
        Table("era1_trades", Base.metadata,
              Column("id", Integer, primary_key=True),
              Column("account", String(30)),
              Column("ts", DateTime),
              Column("run_id", String(40)),
              Column("symbol", String(24)),
              Column("side", String(4)),
              Column("shares", Float),
              Column("price", Float),
              Column("dollars", Float),
              Column("rationale", Text),
              Column("fill_quote", JSON))
    if "era1_equity" not in Base.metadata.tables:
        Table("era1_equity", Base.metadata,
              Column("id", Integer, primary_key=True),
              Column("account", String(30)),
              Column("ts", DateTime),
              Column("cash", Float),
              Column("positions_value", Float),
              Column("equity", Float),
              Column("return_pct", Float),
              Column("mark_meta", JSON))


_register_era1_tables()


def _era1_select(store, table: str, **kw) -> list[dict]:
    """Rows from a frozen Era-1 archive table — [] when the table does not
    exist yet (pre-cutover both-states rule), other errors re-raise."""
    from agent.store import is_missing_table_error

    try:
        return store.select(table, **kw)
    except KeyError:
        return []  # pg metadata miss — table shape not registered
    except Exception as exc:  # noqa: BLE001 — classify, then re-raise
        if is_missing_table_error(exc):
            return []
        raise


# ── the Alpaca paper account, 10s-cached ─────────────────────────────────
#
# /portfolio answers via agent.trade.Trade().state() — a live broker
# round-trip — and the endpoint is polled once per open viewer. A short TTL
# cache bounds the call frequency to once per _PORTFOLIO_TTL regardless of
# viewer count, and every sibling endpoint that needs "what does the desk
# hold" (dividends, holding-stats, the options allowlist, grade overlays)
# reads THE SAME cached body instead of dialing the broker again.

_PORTFOLIO_TTL = 10.0
_portfolio_cache: tuple[float, dict] | None = None


def _build_portfolio() -> dict:
    from agent import grade
    from agent import trade as trade_mod
    from agent.store import get_store

    st = None
    note = None
    try:
        st = trade_mod.Trade().state()
    except Exception as exc:  # noqa: BLE001 — no creds / dead broker degrades
        note = f"{type(exc).__name__}: {exc}"
    if st is None:
        return {
            "available": False,
            "note": "paper account unreachable — " + (note or "unknown"),
            "account": "agent", "paper": True,
            "cash": None, "equity": None, "buying_power": None,
            "positions_value": None, "starting_capital": None,
            "total_pnl": None, "total_return_pct": None,
            "vs_spy": None, "positions": [],
        }

    total_return_pct = (round(st["total_return_pct"], 2)
                        if st.get("total_return_pct") is not None else None)
    vs_spy = None
    try:
        store = get_store()
        inception = _alltime_inception(store)
        if inception and total_return_pct is not None:
            # SYMMETRIC WINDOW (charter): the benchmark is measured from the
            # all-time inception, so our side must cover the same span. With
            # a frozen era-1 archive the era-2 total return covers a shorter
            # window — re-anchor at the first era-1 equity mark (the same
            # point the equity chart plots first). Without the archive the
            # windows coincide and the era-2 return IS the all-time return.
            port_pct = total_return_pct
            base_equity = None
            e1 = _era1_select(store, "era1_equity",
                              filters={"account": ACCOUNT},
                              order=[("ts", "asc")], limit=1)
            try:
                base_equity = float(e1[0]["equity"]) if e1 else None
            except (TypeError, ValueError, KeyError):
                base_equity = None
            if base_equity and base_equity > 0 and st.get("equity") is not None:
                port_pct = round((float(st["equity"]) / base_equity - 1) * 100, 2)
            else:
                base_equity = None
            spy = grade.spy_price_closes(store, since=inception)
            spy_pct = grade._spy_window_pct(spy, inception)
            if spy_pct is not None:
                vs_spy = {"inception": inception, "spy_as_of": spy[-1][0],
                          "spy_return_pct": spy_pct,
                          "portfolio_return_pct": port_pct,
                          "alltime_base_equity": base_equity,
                          "alpha_pct": round(port_pct - spy_pct, 2),
                          "basis": "price_return"}
    except Exception:  # noqa: BLE001 — the benchmark is additive, never a 500
        vs_spy = None

    positions = sorted(st.get("positions") or [],
                       key=lambda p: -(p.get("market_value") or 0.0))
    return {
        "available": True, "account": st.get("account") or "agent",
        "paper": True,
        "cash": st.get("cash"), "equity": st.get("equity"),
        "buying_power": st.get("buying_power"),
        "positions_value": st.get("positions_value"),
        "starting_capital": st.get("starting_capital"),
        "total_pnl": st.get("total_pnl"),
        "total_return_pct": total_return_pct,
        "vs_spy": vs_spy,
        "positions": positions,
    }


def _cached_portfolio() -> dict:
    global _portfolio_cache
    now = time.time()
    if _portfolio_cache is not None and now - _portfolio_cache[0] < _PORTFOLIO_TTL:
        return _portfolio_cache[1]
    out = _build_portfolio()
    _portfolio_cache = (now, out)
    return out


def _cached_positions() -> list[dict]:
    """The held book from the cached account read; [] when degraded —
    callers must treat that as 'holdings unknown', never as proof of flat."""
    pf = _cached_portfolio()
    return list(pf.get("positions") or []) if pf.get("available") else []


def _era2_inception(store) -> str | None:
    """The ET date of the first desk_orders fill — computed, never
    hard-coded (spec: era-2 begins at the first mirrored fill)."""
    from agent import grade

    try:
        fills = grade.fills_from_orders(store)
    except Exception:  # noqa: BLE001 — mirror table not migrated yet
        return None
    return grade._et_date(fills[0]["ts"]) if fills else None


def _alltime_inception(store) -> str | None:
    """All-time inception for the vs-SPY window: the era-1 archive's first
    fill when the frozen book exists, else the era-2 inception."""
    from agent import grade

    rows = _era1_select(store, "era1_trades", columns="ts",
                        filters={"account": ACCOUNT},
                        order=[("ts", "asc")], limit=1)
    if rows and rows[0].get("ts"):
        return grade._et_date(rows[0]["ts"])
    return _era2_inception(store)


@router.get("/portfolio")
def portfolio():
    """Cash, positions (marked), equity, and P&L — the book right now,
    straight from the Alpaca PAPER account (``agent.trade.Trade().state()``,
    ~10s TTL). Degrades to ``{"available": false, ...}`` with an empty
    positions list when the trade keys are absent or the broker is down —
    the page renders the gap, never a fake book.

    ``vs_spy`` is **symmetric price return** (charter V4): the paper broker
    credits no dividends into the book, so the benchmark must not carry its
    dividends either — computed with ``agent.grade``'s SPY helpers from the
    ALL-TIME inception (era-1 first fill when the archive exists, else the
    first mirrored era-2 fill). Our side covers the SAME window: with a
    frozen era-1 archive it re-anchors at the first era-1 equity mark
    (``portfolio_return_pct``, with ``alltime_base_equity`` on the wire so
    the live page can re-derive it from ticks); ``total_return_pct`` stays
    the era-2-only number. ``basis`` says so on the wire."""
    return _cached_portfolio()


@router.get("/equity")
def equity(limit: int = Query(2000, le=10000),
           with_spy: int = Query(0, ge=0, le=1)):
    """The stitched equity curve: Era-1 points from the frozen archive
    (``era1_equity``, absent pre-cutover), Era-2 points from the nightly
    ``desk_portfolio_history`` snapshots, plus a live tip from the cached
    account read. Each point ``{ts, equity, era}``; ``era2_inception`` lets
    the chart draw the cutover marker. ``with_spy=1`` adds a PRICE-RETURN
    SPY series rebased to the all-time inception."""
    from datetime import datetime as _dt

    from agent import grade
    from agent.store import get_store

    store = get_store()
    points: list[dict] = []
    for r in _era1_select(store, "era1_equity", columns="ts,equity",
                          filters={"account": ACCOUNT},
                          order=[("ts", "asc")]):
        if r.get("equity") is None:
            continue
        points.append({"ts": _iso_any(r.get("ts")), "equity": r["equity"],
                       "era": 1})
    try:
        hist = store.select("desk_portfolio_history",
                            columns="snap_date,equity",
                            filters={"account": ACCOUNT},
                            order=[("snap_date", "asc")])
    except Exception as exc:  # noqa: BLE001 — pre-deploy schema grace
        from agent.store import is_missing_table_error

        if not is_missing_table_error(exc):
            raise
        hist = []
    for r in hist:
        if r.get("equity") is None:
            continue
        points.append({"ts": str(r["snap_date"]), "equity": r["equity"],
                       "era": 2})
    pf = _cached_portfolio()
    if pf.get("available") and pf.get("equity") is not None:
        points.append({"ts": _dt.now(timezone.utc).isoformat(),
                       "equity": pf["equity"], "era": 2, "live": True})

    out = {"points": points[-limit:], "era2_inception": _era2_inception(store)}
    if not with_spy:
        return out

    inception = _alltime_inception(store)
    spy_series: list[dict] = []
    if inception:
        closes = grade.spy_price_closes(store, since=inception)
        base = None
        for d, c in closes:
            if d < inception:
                base = c
            else:
                break
        if base:
            spy_series = [{"date": d, "pct": round((c - base) / base * 100, 2)}
                          for d, c in closes if d >= inception]
    out.update({"spy": spy_series, "spy_inception": inception,
                "spy_basis": "price_return"})
    return out


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


# The live-outcomes overlay: agent.grade.outcomes over the desk_orders
# mirror + the cached Alpaca positions, so OPEN scoreboard rows show fresh
# marks between grade runs. 30s TTL bounds the replay cost; any failure
# degrades to {} and the stored grade facts serve alone.
_OUTCOMES_LIVE_TTL = 30.0
_OUTCOMES_LIVE_DAYS = 90
_outcomes_live_cache: tuple[float, dict] | None = None


def _live_outcome_picks() -> dict:
    global _outcomes_live_cache
    now = time.time()
    if _outcomes_live_cache is not None \
            and now - _outcomes_live_cache[0] < _OUTCOMES_LIVE_TTL:
        return _outcomes_live_cache[1]
    picks: dict = {}
    try:
        from agent import grade
        from agent.store import get_store

        live = grade.outcomes(get_store(), days=_OUTCOMES_LIVE_DAYS,
                              positions=_cached_positions())
        for run in live.get("runs") or []:
            for p in run.get("picks") or []:
                picks[(run["run_id"], p["symbol"])] = p
    except Exception:  # noqa: BLE001 — overlay only, stored facts still serve
        picks = {}
    _outcomes_live_cache = (now, picks)
    return picks


@router.get("/outcomes")
def outcomes_scoreboard(db: Session = Depends(get_db),
                        status: str = Query("all"),
                        limit: int = Query(100, le=200)):
    """The predictions scoreboard — machine-graded pick facts
    (``desk_outcomes``, written by ``agent.grade run``) joined with each
    pick's own words (prediction / horizon / kill free text from the
    decision row) so the page shows what was SAID next to what HAPPENED.

    OPEN rows are additionally overlaid with fresh facts from
    ``agent.grade.outcomes`` computed against the desk_orders mirror and the
    cached Alpaca positions (30s TTL) — the mark moves between grade runs,
    the verdicts never do. ``degraded`` is grade's own bool (Alpaca returned
    no price for the mark) and passes straight through.

    Open rows come first (newest decision first), then recent closed rows.
    ``sessions_elapsed`` counts stored SPY closes on/after the decision's ET
    date for horizon countdowns. ``summary`` carries whole-table counts by
    status and verdict plus the hit rate over closed, reflection-graded rows
    (TRUE vs FALSE)."""
    from bisect import bisect_left

    from sqlalchemy import func as safunc

    from agent import occ
    from agent.grade import _et_date
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

    live_picks = _live_outcome_picks()

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
        since_pct, spy_pct = r.since_pct, r.spy_pct
        alpha_pct, mark_px = r.alpha_pct, r.mark_px
        if r.status == "open":
            live = live_picks.get((r.run_id, r.symbol))
            if live and live.get("since_this_run_pct") is not None:
                since_pct = live["since_this_run_pct"]
                if live.get("spy_same_window_pct") is not None:
                    spy_pct = live["spy_same_window_pct"]
                if live.get("alpha_pct") is not None:
                    alpha_pct = live["alpha_pct"]
                live_mark = (live.get("open_now") or {}).get("last_price")
                if live_mark is not None:
                    mark_px = live_mark
        out_rows.append({
            "id": r.id, "run_id": r.run_id, "symbol": r.symbol,
            "is_option": occ.is_option(r.symbol),
            "status": r.status, "decision_ts": _iso(d.ts) if d else None,
            "action": pick.get("action"), "prediction": pick.get("prediction"),
            "kill": pick.get("kill"),
            "horizon_days": r.horizon_days,
            "horizon_elapsed": r.horizon_elapsed,
            "sessions_elapsed": sessions,
            "entry_avg_px": r.entry_avg_px, "mark_px": mark_px,
            "mark_basis": r.mark_basis, "since_pct": since_pct,
            "spy_pct": spy_pct, "alpha_pct": alpha_pct,
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


@router.get("/claims")
def claims(db: Session = Depends(get_db),
           include_inactive: bool = Query(False)):
    """The structured claims registry behind the wiki (v9.18 knowledge layer)
    — every behavior-influencing fact with its tier, class, recorded sample
    sizes (never a confidence score; none exist), scope, decay, and status.
    Only ``established`` and ``experimental``-flagged claims may justify a
    pick; candidates/observations are watch-only — the panel shows which is
    which. Read-only projection of ``desk_claims``."""
    from agent.models import DeskClaim

    q = db.query(DeskClaim).filter(DeskClaim.account == ACCOUNT)
    rows = q.all()
    if not include_inactive:
        rows = [r for r in rows if r.status == "active"]
    tier_order = {"established": 0, "candidate": 1, "observation": 2,
                  "digest": 3}
    rows.sort(key=lambda r: (0 if r.status == "active" else 1,
                             tier_order.get(r.tier, 9), r.id))
    by_tier: dict = {}
    by_class: dict = {}
    for r in rows:
        if r.status == "active":
            by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
            by_class[r.kclass] = by_class.get(r.kclass, 0) + 1
    return {
        "claims": [{
            "id": r.id, "cite": f"[C-{r.id}]", "kclass": r.kclass,
            "tier": r.tier, "experimental": bool(r.experimental),
            "status": r.status, "statement": r.statement,
            "regimes": (r.scope or {}).get("regimes"),
            "stats": r.stats or {}, "evidence_count": len(r.evidence or []),
            "decay_class": r.decay_class,
            "expires_at": str(r.expires_at) if r.expires_at else None,
            "superseded_by": r.superseded_by,
            "updated_at": _iso(r.updated_at)} for r in rows],
        "summary": {"active": sum(1 for r in rows if r.status == "active"),
                    "by_tier": by_tier, "by_class": by_class,
                    "experimental": sum(1 for r in rows
                                        if r.status == "active"
                                        and r.experimental)},
    }


@router.get("/proposals")
def proposals(db: Session = Depends(get_db)):
    """The owner-approval queue — learned-behavior changes (pivots, cap
    raises) the agent has proposed and their decisions. Pending first; the
    desk shows these so the owner sees what's waiting without opening
    GitHub. Read-only; approvals happen on the PROPOSAL-<id> issue or the
    owner CLI, never from the web."""
    from agent.models import DeskProposal

    rows = (db.query(DeskProposal)
            .filter(DeskProposal.account == ACCOUNT)
            .order_by(desc(DeskProposal.id)).limit(50).all())
    rows.sort(key=lambda r: (0 if r.status == "pending" else 1, -r.id))
    return {
        "proposals": [{
            "id": r.id, "ref": f"PROPOSAL-{r.id}", "title": r.title,
            "change_kind": r.change_kind, "status": r.status,
            "claim_ids": r.claim_ids or [], "created_at": _iso(r.created_at),
            "decided_at": _iso(r.decided_at) if r.decided_at else None,
            "decided_via": r.decided_via,
            "expires_at": str(r.expires_at) if r.expires_at else None}
            for r in rows],
        "pending": sum(1 for r in rows if r.status == "pending"),
    }


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
    """Top gainers / losers / most-active across the last two WELL-COVERED
    sessions.

    Computed read-only from the fresh daily-bar hot set — biggest
    close-to-close moves and largest dollar volume. Same guards as the
    nightly brief's movers (``agent.market._movers``): a session only
    counts when it has a full-coverage bar set (today's partial intraday
    top-up must never be one side of a market-wide comparison), and symbols
    with a split between the two sessions are excluded — ``daily_bars``
    stores RAW closes, so a 10:1 split would fabricate a -90% 'loser'.
    No external calls; the live tape is the SSE ``/stream``.
    """
    from datetime import timedelta as _td

    from sqlalchemy import func as safunc

    from agent import data as agent_data
    from edgefinder.db.models import DailyBar, TickerSplit

    empty = {"as_of": None, "prior": None,
             "gainers": [], "losers": [], "most_active": []}
    latest_any = db.query(safunc.max(DailyBar.date)).scalar()
    if latest_any is None:
        return empty
    lo = latest_any - _td(days=7)
    counts = (db.query(DailyBar.date, safunc.count(DailyBar.id))
              .filter(DailyBar.date >= lo).group_by(DailyBar.date).all())
    fat = sorted((d for d, n in counts if n >= agent_data.FULL_COVERAGE_MIN),
                 reverse=True)
    if len(fat) < 2:
        return {**empty, "note":
                "fewer than two full-coverage sessions in the last week — "
                "movers need two comparable sessions"}
    latest, prior = fat[0], fat[1]
    split_syms: set[str] = set()
    try:
        srows = db.query(TickerSplit.symbol, TickerSplit.execution_date).all()
        split_syms = {s for s, ed in srows
                      if str(prior) < str(ed or "")[:10] <= str(latest)}
    except Exception:  # noqa: BLE001 — split guard is best-effort
        split_syms = set()
    cur = (db.query(DailyBar.symbol, DailyBar.close, DailyBar.volume)
           .filter(DailyBar.date == latest).all())
    prev = {s: c for s, c in db.query(DailyBar.symbol, DailyBar.close)
            .filter(DailyBar.date == prior).all()}
    rows = []
    for sym, close, vol in cur:
        if (close is None or close < 1.0 or sym in split_syms
                or any(ch in sym for ch in (".", "/", "="))):
            continue
        pc = prev.get(sym)
        chg = ((close - pc) / pc * 100.0) if pc else None
        rows.append({"symbol": sym, "close": round(close, 2),
                     "change_pct": round(chg, 2) if chg is not None else None,
                     "dollar_volume": round(close * (vol or 0.0))})
    with_chg = [r for r in rows if r["change_pct"] is not None]
    out = {
        "as_of": str(latest), "prior": str(prior),
        "gainers": sorted(with_chg, key=lambda r: -r["change_pct"])[:top],
        "losers": sorted(with_chg, key=lambda r: r["change_pct"])[:top],
        "most_active": sorted(rows, key=lambda r: -r["dollar_volume"])[:top],
    }
    if split_syms:
        out["splits_excluded"] = sorted(split_syms)
    return out


def _held_equity_symbols() -> list[str]:
    """Plain-equity held names from the cached Alpaca positions — options
    map to nothing here (not in daily_bars) and crypto pairs are skipped.
    [] when the account read is degraded (holdings unknown ≠ flat)."""
    from agent import occ

    out = []
    for p in _cached_positions():
        s = str(p.get("symbol") or "").upper()
        if not s or occ.is_option(s) or "/" in s:
            continue
        out.append(s)
    return out


@router.get("/holding-stats")
def holding_stats(db: Session = Depends(get_db),
                  spark_days: int = Query(30, ge=5, le=120)):
    """Per-held-name enrichment from the daily-bar hot set: last-session day
    change, 52-week high/low, and a short close series for a sparkline.
    Held names come from the cached Alpaca positions read. ``daily_bars``
    stores RAW closes, so bars BEFORE a split's execution date are rebased
    onto the current share basis first — otherwise a held name's day-change,
    52-week range, and sparkline all fabricate the split as a price move.
    Read-only (no external calls); options legs are skipped."""
    from datetime import timedelta

    from sqlalchemy import func as safunc

    from edgefinder.db.models import DailyBar, TickerSplit

    held = _held_equity_symbols()
    if not held:
        return {"as_of": None, "symbols": {}}
    latest = (db.query(safunc.max(DailyBar.date))
              .filter(DailyBar.symbol.in_(held)).scalar())
    if latest is None:
        return {"as_of": None, "symbols": {}}
    lo = latest - timedelta(days=400)  # ~252 trading days of headroom
    rows = (db.query(DailyBar.symbol, DailyBar.date, DailyBar.close)
            .filter(DailyBar.symbol.in_(held), DailyBar.date >= lo)
            .order_by(DailyBar.symbol, DailyBar.date).all())
    series: dict[str, list[tuple[str, float]]] = {}
    for sym, d, close in rows:
        if close is not None:
            series.setdefault(sym, []).append((str(d)[:10], float(close)))
    splits_by_sym: dict[str, list[tuple[str, float]]] = {}
    try:
        for s, ed, frm, to in db.query(
                TickerSplit.symbol, TickerSplit.execution_date,
                TickerSplit.split_from, TickerSplit.split_to).all():
            if s in series and frm and to and frm > 0 and to > 0:
                splits_by_sym.setdefault(s, []).append(
                    (str(ed or "")[:10], to / frm))
    except Exception:  # noqa: BLE001 — split rebase is best-effort
        splits_by_sym = {}
    out = {}
    for sym, pts in series.items():
        for ed, factor in splits_by_sym.get(sym, ()):
            if ed and factor > 0:
                pts = [(d, c / factor) if d < ed else (d, c) for d, c in pts]
        closes = [c for _, c in pts]
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
    the next upcoming one, plus a trailing-4 annual estimate. Held names come
    from the cached Alpaca positions read ([] when degraded).

    ``missed_dividends`` is the honesty counter (charter V4): the paper
    broker pays NO dividends into the book, so this replays the desk_orders
    fills to shares held STRICTLY BEFORE each ex-date since era-2 inception
    and prices the foregone cash — disclosed, never silently embedded."""
    from datetime import date

    from agent import grade
    from agent.store import get_store
    from edgefinder.db.models import DividendRecord

    held = _held_equity_symbols()
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
        # trailing means PAST ex-dates only — the newest-4 slice used to
        # count a future declared dividend and drop an actual paid one
        ttm = round(sum(r.cash_amount or 0 for r in past[:4]), 4)
        out.append({
            "symbol": sym, "has_dividend": True,
            "last_ex_date": str(last.ex_date) if last else None,
            "last_amount": round(last.cash_amount, 4) if last and last.cash_amount else None,
            "next_ex_date": str(nxt.ex_date) if nxt else None,
            "next_amount": round(nxt.cash_amount, 4) if nxt and nxt.cash_amount else None,
            "ttm_amount": ttm,
        })

    # Estimated missed dividends since era-2 inception (fills replay).
    missed_total = 0.0
    missed_by_symbol: dict[str, float] = {}
    try:
        fills = grade.fills_from_orders(get_store())
    except Exception:  # noqa: BLE001 — mirror not migrated yet
        fills = []
    if fills:
        from agent import occ

        inception = grade._et_date(fills[0]["ts"]) or today
        eq_fills = [f for f in fills if not occ.is_option(f["symbol"])
                    and "/" not in f["symbol"]]
        traded = sorted({f["symbol"] for f in eq_fills})
        if traded:
            div_rows = (db.query(DividendRecord.symbol,
                                 DividendRecord.ex_date,
                                 DividendRecord.cash_amount)
                        .filter(DividendRecord.symbol.in_(traded))
                        .all())
            for sym, ex_date, amount in div_rows:
                ex = str(ex_date)[:10]
                if not amount or ex <= inception or ex > today:
                    continue
                held_shares = 0.0
                for f in eq_fills:
                    if f["symbol"] != sym:
                        continue
                    d = grade._et_date(f.get("ts"))
                    if d is None or d >= ex:
                        continue  # strictly before ex-date earns the dividend
                    held_shares += (f["shares"] if f["side"] == "BUY"
                                    else -f["shares"])
                if held_shares > 0:
                    amt = held_shares * float(amount)
                    missed_by_symbol[sym] = round(
                        missed_by_symbol.get(sym, 0.0) + amt, 2)
                    missed_total += amt
    return {"as_of": today, "holdings": out,
            "missed_dividends": {
                "total": round(missed_total, 2),
                "by_symbol": missed_by_symbol,
                "note": "the paper broker pays no dividends — estimated "
                        "foregone amount"}}


# ── open orders & resting protection (Alpaca open orders, 10s TTL) ──────

_OPEN_ORDERS_TTL = 10.0
_open_orders_cache: tuple[float, dict] | None = None


def _build_open_orders() -> dict:
    from datetime import datetime as _dt

    from agent import trade as trade_mod

    try:
        orders = trade_mod.Trade().orders(status="open", limit=100)
    except Exception as exc:  # noqa: BLE001 — no creds / dead broker degrades
        return {"available": False,
                "note": f"paper account unreachable — {type(exc).__name__}: {exc}",
                "orders": []}
    now = _dt.now(timezone.utc)
    rows = []
    for o in orders:
        ot = (o.get("order_type") or "").lower()
        kind = ("stop" if ot in ("stop", "stop_limit", "trailing_stop")
                else "limit" if ot == "limit" else "other")
        age = None
        sub = o.get("submitted_at")
        if sub:
            try:
                sub_dt = _dt.fromisoformat(str(sub).replace("Z", "+00:00"))
                if sub_dt.tzinfo is None:
                    sub_dt = sub_dt.replace(tzinfo=timezone.utc)
                age = max(0, (now - sub_dt).days)
            except ValueError:
                age = None
        rows.append({"symbol": o.get("symbol"), "side": o.get("side"),
                     "order_type": ot or None, "tif": o.get("tif"),
                     "limit_price": o.get("limit_price"),
                     "stop_price": o.get("stop_price"),
                     "qty": o.get("qty"), "filled_qty": o.get("filled_qty"),
                     "submitted_at": o.get("submitted_at"),
                     "age_days": age, "kind": kind,
                     "alpaca_order_id": o.get("alpaca_order_id")})
    order_rank = {"stop": 0, "limit": 1, "other": 2}
    rows.sort(key=lambda r: (order_rank.get(r["kind"], 9),
                             str(r.get("symbol") or "")))
    return {"available": True, "orders": rows}


@router.get("/open-orders")
def open_orders():
    """Orders resting at the broker RIGHT NOW — protective GTC stops (with
    age: Alpaca auto-cancels GTC at 90 days) and working limits. Real orders
    on Alpaca's book, so stops fire even while the AI is between check-ins.
    10s TTL; degrades to ``{"available": false}`` without trade creds."""
    global _open_orders_cache
    now = time.time()
    if _open_orders_cache is not None \
            and now - _open_orders_cache[0] < _OPEN_ORDERS_TTL:
        return _open_orders_cache[1]
    out = _build_open_orders()
    _open_orders_cache = (now, out)
    return out


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
# has a reason to show: held positions (options mapped to their underlying,
# from the cached Alpaca account read), the latest decision's picks +
# watchlist, and the streamer's seed universe. 60s TTL.

_OPTIONS_ALLOW_TTL = 60.0
_options_allow: tuple[float, frozenset] | None = None


def _options_allowlist(db: Session) -> frozenset[str]:
    global _options_allow
    now = time.time()
    if _options_allow is not None and now - _options_allow[0] < _OPTIONS_ALLOW_TTL:
        return _options_allow[1]

    from agent import occ
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

    for p in _cached_positions():
        add(p.get("symbol"))
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
            # stored token counts are frozen at each key's last touch —
            # apply the refill before judging, or nothing ever prunes
            self._buckets = {
                k: (tk, ts) for k, (tk, ts) in self._buckets.items()
                if min(self.capacity, tk + (now - ts) * self.refill)
                < self.capacity - 1.0}
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
    Allowlisted (held ∪ latest picks/watchlist ∪ streamed seeds) and
    rate-limited — this endpoint triggers metered external calls."""
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
    """Paper-account health: is the Alpaca paper book reachable (status,
    equity, cash), is the market open per the data-side clock, and when the
    desk_orders mirror last reconciled. Exposes no secrets."""
    from agent import broker
    from agent import trade as trade_mod
    from agent.store import get_store

    out: dict = {"paper_account": None, "clock": None, "last_reconcile": None}
    try:
        acct = trade_mod.Trade().account()
        out["paper_account"] = {"available": True,
                                "status": acct.get("status"),
                                "equity": acct.get("equity"),
                                "cash": acct.get("cash")}
    except Exception as exc:  # noqa: BLE001 — diagnostic reports, never raises
        out["paper_account"] = {"available": False,
                                "error": f"{type(exc).__name__}: {exc}"}
    try:
        if broker.enabled():
            b = broker.Broker()
            out["clock"] = {"is_open": b.is_market_open(),
                            "session": b.session()}
    except Exception as exc:  # noqa: BLE001
        out["clock"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        rows = get_store().select("desk_orders", columns="updated_at",
                                  filters={"account": ACCOUNT},
                                  order=[("updated_at", "desc")], limit=1)
        if rows and rows[0].get("updated_at"):
            out["last_reconcile"] = _iso_any(rows[0]["updated_at"])
    except Exception:  # noqa: BLE001 — mirror not migrated yet
        pass
    return out


@router.get("/data-health")
def data_health(db: Session = Depends(get_db)):
    """Freshness of the market-data asset behind research (not the live tape).

    Bar age alone can't detect a dead nightly ingest — the hourly top-up keeps
    a handful of held names current while the other ~2,000 symbols go stale.
    This counts bar rows per recent date and reports sessions since the last
    full-coverage ingest (one definition, shared with agent.preflight).
    """
    from datetime import timedelta

    from sqlalchemy import func as safunc

    from agent.data import coverage_verdict
    from edgefinder.db.models import DailyBar

    latest = db.query(safunc.max(DailyBar.date)).scalar()
    if latest is None:
        return coverage_verdict([])
    lo = latest - timedelta(days=21)
    rows = (db.query(DailyBar.date, safunc.count(DailyBar.symbol))
            .filter(DailyBar.date >= lo).group_by(DailyBar.date).all())
    return coverage_verdict(rows)


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
def trades(limit: int = Query(100, le=1000)):
    """Recent executed fills (newest first), era-tagged.

    Era-2 rows come from the ``desk_orders`` mirror (Alpaca is the book of
    record): rows with actual executions, mleg PARENT shells skipped (their
    legs carry the fills). Rationale is not stamped per-fill on this era —
    ``run_id`` links the fill to its decision dossier instead. Era-1 rows
    come from the frozen ``era1_trades`` archive when it exists, keeping
    their original rationale + ``fill_quote`` receipt. Bare list, one shared
    key set across eras."""
    from agent import grade
    from agent.store import get_store

    store = get_store()
    rows: list[dict] = []
    try:
        orders = store.select("desk_orders", filters={"account": ACCOUNT})
    except Exception as exc:  # noqa: BLE001 — pre-deploy schema grace
        from agent.store import is_missing_table_error

        if not is_missing_table_error(exc):
            raise
        orders = []
    for r in orders:
        fq, px = r.get("filled_qty"), r.get("filled_avg_price")
        if not fq or not px or float(fq) <= 0:
            continue
        if (r.get("order_class") == "mleg") and not r.get("parent_order_id"):
            continue  # parent shell — the legs carry the fills
        side = str(r.get("side") or "").upper()
        if side not in ("BUY", "SELL"):
            continue
        sym = r["symbol"]
        rows.append({
            "id": r.get("id"), "era": 2,
            "t": _iso_any(r.get("filled_at") or r.get("submitted_at")),
            "symbol": sym, "side": side,
            "shares": float(fq), "price": float(px),
            "dollars": round(float(fq) * float(px) * grade._mult(sym), 2),
            "rationale": None, "run_id": r.get("run_id"),
            "kind": r.get("kind"), "order_class": r.get("order_class"),
            "fill_quote": None})
    for r in _era1_select(store, "era1_trades", filters={"account": ACCOUNT}):
        rows.append({
            "id": r.get("id"), "era": 1, "t": _iso_any(r.get("ts")),
            "symbol": r["symbol"], "side": r.get("side"),
            "shares": r.get("shares"), "price": r.get("price"),
            "dollars": r.get("dollars"), "rationale": r.get("rationale"),
            "run_id": r.get("run_id"), "kind": None, "order_class": None,
            "fill_quote": _json_dict(r.get("fill_quote")) or None})
    rows.sort(key=lambda r: (str(r.get("t") or ""), r.get("era") or 0,
                             r.get("id") or 0), reverse=True)
    return rows[:limit]


# ── /trades page history: per-fill realized P&L, era-tagged ─────────────

_ERA1_KIND = {"split_adjustment": "split", "dividend": "dividend",
              "expiry_settlement": "expiry"}
_EPS_UNITS = 1e-6


def _replay_realized(fills: list[dict]) -> dict[int, dict]:
    """Per-fill realized P&L keyed by list INDEX — the avg-cost replay that
    keeps /trades honest, mirroring ``agent.grade._realized_pnl`` semantics.

    ``fills`` MUST be one era's FULL ledger in replay order — average cost
    is path-dependent, so a truncated list silently mis-prices every row.
    Slice for display AFTER calling this, never before. Values are
    ``{"pnl", "closed_units"}`` with the option x100 multiplier already
    applied. A fill that closed nothing (an opening leg, a shares=0 era-1
    dividend row, a split adjustment, a duplicate equity exit on a flat
    book) is ABSENT — callers render that as null, never 0.00."""
    from agent import occ

    out: dict[int, dict] = {}
    book: dict[str, dict] = {}
    for i, t in enumerate(fills):
        sym = t["symbol"]
        b = book.setdefault(sym, {"units": 0.0, "cost": 0.0})
        qty = float(t.get("shares") or 0.0)
        signed = qty if t["side"] == "BUY" else -qty
        cur = b["units"]
        if t.get("src") == "split_adjustment":
            b["units"] = cur + signed  # unit shift, cost untouched — no P&L
            continue
        mult = 100 if occ.is_option(sym) else 1
        if abs(cur) <= _EPS_UNITS:
            if signed < 0 and not occ.is_option(sym):
                continue  # equity sell on a flat book — no lot, no P&L
            b["units"] = cur + signed
            b["cost"] += abs(signed) * float(t["price"])
            continue
        if (cur > 0) == (signed > 0):  # extending the same direction
            b["units"] = cur + signed
            b["cost"] += abs(signed) * float(t["price"])
            continue
        closing = min(abs(signed), abs(cur))
        avg = b["cost"] / abs(cur)
        sign = 1.0 if cur > 0 else -1.0
        pnl = closing * (float(t["price"]) - avg) * sign * mult
        if closing > _EPS_UNITS:
            out[i] = {"pnl": pnl, "closed_units": closing}
        b["cost"] -= closing * avg
        b["units"] = cur + signed
        if abs(b["units"]) <= _EPS_UNITS:
            b["units"], b["cost"] = 0.0, 0.0
        elif (b["units"] > 0) != (cur > 0):
            b["cost"] = abs(b["units"]) * float(t["price"])
        if not occ.is_option(sym) and b["units"] < 0:
            b["units"], b["cost"] = 0.0, 0.0
    return out


def _era2_history_rows(store) -> list[dict]:
    from agent import grade, occ

    try:
        fills = grade.fills_from_orders(store)
    except Exception as exc:  # noqa: BLE001 — pre-deploy schema grace
        from agent.store import is_missing_table_error

        if not is_missing_table_error(exc):
            raise
        fills = []
    pnl = _replay_realized(fills)
    rows = []
    for i, f in enumerate(fills):
        sym = f["symbol"]
        is_opt = occ.is_option(sym)
        r = pnl.get(i)
        rows.append({
            "id": f.get("id"), "era": 2,
            "date": grade._et_date(f.get("ts")), "t": _iso_any(f.get("ts")),
            "symbol": sym,
            "underlying": occ.parse(sym)["underlying"] if is_opt else sym,
            "label": occ.describe(sym) if is_opt else sym,
            "side": f["side"],
            "kind": "stop" if f.get("kind") == "stop" else "trade",
            "shares": f["shares"], "dollars": f["dollars"],
            "realized": None if r is None else round(r["pnl"], 2),
            "closed_units": None if r is None else r["closed_units"]})
    return rows


def _era1_history_rows(store) -> list[dict]:
    from agent import grade, occ

    raw = _era1_select(store, "era1_trades", filters={"account": ACCOUNT})
    raw.sort(key=lambda r: (str(r.get("ts")), r.get("id") or 0))
    fills = []
    for r in raw:
        fq = _json_dict(r.get("fill_quote"))
        src = fq.get("src")
        eff = None  # corp actions date by their own effective date
        if src == "split_adjustment":
            eff = str(fq.get("execution_date") or "")[:10] or None
        elif src == "dividend":
            eff = str(fq.get("ex_date") or "")[:10] or None
        fills.append({"id": r.get("id"), "ts": r.get("ts"),
                      "symbol": r["symbol"], "side": r.get("side"),
                      "shares": float(r.get("shares") or 0.0),
                      "price": float(r.get("price") or 0.0),
                      "dollars": float(r.get("dollars") or 0.0),
                      "src": src, "eff_date": eff})
    pnl = _replay_realized(fills)
    rows = []
    for i, f in enumerate(fills):
        sym = f["symbol"]
        is_opt = occ.is_option(sym)
        r = pnl.get(i)
        rows.append({
            "id": f["id"], "era": 1,
            "date": f["eff_date"] or grade._et_date(f.get("ts")),
            "t": _iso_any(f.get("ts")),
            "symbol": sym,
            "underlying": occ.parse(sym)["underlying"] if is_opt else sym,
            "label": occ.describe(sym) if is_opt else sym,
            "side": f["side"],
            "kind": _ERA1_KIND.get(f["src"], "trade"),
            "shares": f["shares"], "dollars": f["dollars"],
            "realized": None if r is None else round(r["pnl"], 2),
            "closed_units": None if r is None else r["closed_units"]})
    return rows


@router.get("/trade-history")
def trade_history(limit: int = Query(200, ge=1, le=1000)):
    """The simple human history behind /trades: every fill and the profit it
    realized, across BOTH eras.

    Era-2 fills come from the ``desk_orders`` mirror via
    ``agent.grade.fills_from_orders``; Era-1 rows from the frozen
    ``era1_trades`` archive when it exists (their corp-action rows keep the
    old conventions: effective dates, no fake $0.00 sales). Each era replays
    its OWN full ledger — average cost is path-dependent, so ``limit``
    slices the annotated display list and NEVER the replay input. ``date``
    is the ET session date (a 19:30 ET fill is next-day in UTC and would
    otherwise render a day late)."""
    from agent.store import get_store

    store = get_store()
    era2 = _era2_history_rows(store)
    era1 = _era1_history_rows(store)
    era2_realized = round(sum(r["realized"] for r in era2
                              if r["realized"] is not None), 2)
    era1_realized = round(sum(r["realized"] for r in era1
                              if r["realized"] is not None), 2)
    rows = era2 + era1
    # newest first by the DISPLAYED date, so a corp action booked late still
    # sorts next to the fills it modified
    rows.sort(key=lambda r: (r["date"] or "", r["t"] or "", r["id"] or 0),
              reverse=True)
    return {"rows": rows[:limit],
            "era1_realized": era1_realized,
            "era2_realized": era2_realized,
            "realized_pnl": round(era1_realized + era2_realized, 2),
            "closing_fills": sum(1 for r in era2 + era1
                                 if r["realized"] is not None),
            "total": len(rows)}

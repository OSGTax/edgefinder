"""Trading-desk API — projects the agent's desk_* tables onto the page.

Read-only endpoints over the autonomous agent's own schema: the paper book +
equity curve, the latest decision (chart-forward picks with why-now /
rationale / news), the live thinking feed, the backtest evidence, the strategy
state, and the journal of pivots. All times are ISO UTC; the page normalizes.
"""

from __future__ import annotations

from datetime import timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from agent.models import (
    ACCOUNT,
    STARTING_CAPITAL,
    DeskBacktest,
    DeskChangelog,
    DeskDecision,
    DeskEquity,
    DeskJournal,
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


@router.get("/portfolio")
def portfolio(db: Session = Depends(get_db)):
    """Cash, positions (marked), equity, and P&L — the book right now."""
    from sqlalchemy import func as safunc

    from agent import occ

    positions = (db.query(DeskPosition)
                 .filter(DeskPosition.account == ACCOUNT).all())
    # Cash = starting capital + Σ sell − Σ buy, computed from the trade ledger
    # (source of truth) on this read-only session — same formula as agent.ledger.
    by_side = {side: float(total) for side, total in (
        db.query(DeskTrade.side, safunc.coalesce(safunc.sum(DeskTrade.dollars), 0.0))
        .filter(DeskTrade.account == ACCOUNT).group_by(DeskTrade.side).all())}
    c = round(STARTING_CAPITAL + by_side.get("SELL", 0.0) - by_side.get("BUY", 0.0), 2)
    rows, pos_value = [], 0.0
    for p in positions:
        mark = p.last_price or p.avg_price
        m = 100 if occ.is_option(p.symbol) else 1  # OCC contracts are ×100
        mv = round(p.shares * mark * m, 2)
        pos_value += mv
        rows.append({
            "symbol": p.symbol, "shares": p.shares,
            "avg_price": round(p.avg_price, 4), "last_price": round(mark, 4),
            "market_value": mv, "cost_basis": round(p.shares * p.avg_price * m, 2),
            "unrealized_pnl": round(p.shares * (mark - p.avg_price) * m, 2),
            "opened_at": _iso(p.opened_at), "marked_at": _iso(p.marked_at),
        })
    equity = round(c + pos_value, 2)
    for r in rows:
        r["weight"] = round(r["market_value"] / equity, 4) if equity else 0.0
    rows.sort(key=lambda r: -r["market_value"])
    total_return_pct = round((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2)

    # vs SPY since inception — a long book's raw return is mostly market beta,
    # so the hero shows the difference. SPY closes come from daily_bars
    # (index_daily froze at the cutover), back-adjusted to TOTAL RETURN with
    # the SPY rows in ``dividends`` — the book credits dividend cash on held
    # names at the ex-date (settle books it), so a price-only SPY would
    # flatter the book by the index's own yield. Same pure adjustment and
    # baseline convention as agent.ledger._spy_closes/_spy_window_pct:
    # baseline = last close STRICTLY BEFORE the ET inception date (a close ON
    # it is 16:00, after the first fill — and on day one it would be the
    # endpoint itself, a confident fake 0.0); missing dividend rows degrade
    # to price return. One bounded range query per call — the endpoint is
    # polled every minute per client, and the book is young. (Short-lived
    # duplication of the ledger's query plumbing; Phase E unifies.)
    vs_spy = None
    first_trade = (db.query(safunc.min(DeskTrade.ts))
                   .filter(DeskTrade.account == ACCOUNT).scalar())
    if first_trade is not None:
        from datetime import date as _date, timedelta as _td

        from agent.ledger import _adjust_closes_for_dividends, _et_date
        from edgefinder.db.models import DailyBar, DividendRecord

        inception = _date.fromisoformat(_et_date(first_trade))
        lo = inception - _td(days=10)  # weekend/holiday baseline buffer
        spy_rows = (db.query(DailyBar.date, DailyBar.close)
                    .filter(DailyBar.symbol == "SPY", DailyBar.date >= lo)
                    .order_by(DailyBar.date).all())
        closes = [(str(d), float(c)) for d, c in spy_rows if c]
        divs = sorted((str(x), float(a)) for x, a in
                      db.query(DividendRecord.ex_date, DividendRecord.cash_amount)
                      .filter(DividendRecord.symbol == "SPY",
                              DividendRecord.ex_date >= lo).all() if a)
        closes = _adjust_closes_for_dividends(closes, divs)
        base = base_d = None
        for d, c in closes:
            if d < str(inception):
                base, base_d = c, d
            else:
                break
        if base and closes and closes[-1][0] != base_d:
            end_d, end = closes[-1]
            spy_pct = round((end - base) / base * 100, 2)
            vs_spy = {"inception": str(inception), "spy_as_of": end_d,
                      "spy_return_pct": spy_pct,
                      "alpha_pct": round(total_return_pct - spy_pct, 2)}

    return {
        "account": ACCOUNT, "cash": c, "positions_value": round(pos_value, 2),
        "equity": equity, "starting_capital": STARTING_CAPITAL,
        "total_pnl": round(equity - STARTING_CAPITAL, 2),
        "total_return_pct": total_return_pct,
        "vs_spy": vs_spy,
        "positions": rows,
    }


@router.get("/equity")
def equity(db: Session = Depends(get_db), limit: int = Query(2000, le=10000)):
    """Equity-curve series (oldest→newest) for the chart."""
    rows = (db.query(DeskEquity)
            .filter(DeskEquity.account == ACCOUNT)
            .order_by(desc(DeskEquity.ts)).limit(limit).all())
    rows.reverse()
    return [{"t": _iso(r.ts), "equity": r.equity, "cash": r.cash,
             "positions_value": r.positions_value, "return_pct": r.return_pct}
            for r in rows]


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
    order = {"playbook": 0, "lessons": 1, "mistakes": 2, "market-notes": 3}
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


@router.get("/stream")
async def stream():
    """Server-Sent Events: the live tape for the /desk page.

    Emits an ``event: quotes`` frame with the full cache snapshot every second
    (the universe is small, so full snapshots beat diff bookkeeping). The
    browser's EventSource auto-reconnects; frames double as heartbeats."""
    import asyncio as _asyncio
    import json as _json

    from fastapi.responses import StreamingResponse

    from agent.streamer import cache

    async def gen():
        while True:
            payload = _json.dumps(cache.snapshot(), default=str)
            yield f"event: quotes\ndata: {payload}\n\n"
            await _asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.get("/options/{symbol}")
def options_summary(symbol: str):
    """Live options intelligence for an underlying: spot, focus expiry, ATM IV,
    straddle-implied expected move, 25-delta skew, and a strikes table around
    the money. 60s-cached; degrades to {"available": false} without keys."""
    from agent import options_data

    return options_data.get_summary(symbol)


@router.get("/options/{symbol}/history")
def options_history(symbol: str, limit: int = Query(250, le=1000)):
    """The IV data bank series (one snapshot/day, accumulated by the agent's
    refresh) — powers the IV/expected-move history charts."""
    from agent import options_data
    from agent.store import get_store

    return {"symbol": symbol.upper(),
            "series": options_data.history(get_store(), symbol, limit=limit)}


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

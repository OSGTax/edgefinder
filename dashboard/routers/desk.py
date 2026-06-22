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
)
from dashboard.dependencies import get_db

router = APIRouter()

# An entry is "new" (lights the badge) for this many days after it ships.
WHATSNEW_SPOTLIGHT_DAYS = 14


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
        mv = round(p.shares * mark, 2)
        pos_value += mv
        rows.append({
            "symbol": p.symbol, "shares": p.shares,
            "avg_price": round(p.avg_price, 4), "last_price": round(mark, 4),
            "market_value": mv, "cost_basis": round(p.shares * p.avg_price, 2),
            "unrealized_pnl": round(p.shares * (mark - p.avg_price), 2),
            "opened_at": _iso(p.opened_at), "marked_at": _iso(p.marked_at),
        })
    equity = round(c + pos_value, 2)
    for r in rows:
        r["weight"] = round(r["market_value"] / equity, 4) if equity else 0.0
    rows.sort(key=lambda r: -r["market_value"])
    return {
        "account": ACCOUNT, "cash": c, "positions_value": round(pos_value, 2),
        "equity": equity, "starting_capital": STARTING_CAPITAL,
        "total_pnl": round(equity - STARTING_CAPITAL, 2),
        "total_return_pct": round((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2),
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
        "watchlist": d.watchlist or [], "strategy_version": d.strategy_version,
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
             "rationale": r.rationale, "run_id": r.run_id} for r in rows]

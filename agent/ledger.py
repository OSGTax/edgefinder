"""The paper book — record fills, rebuild positions, mark to market.

The fill ledger (``desk_trades``) is the source of truth. Cash is always
recomputed from it; ``desk_positions`` is a projection rebuilt from the same
ledger, so the account can never silently drift. Long-only, paper only,
whole shares, no leverage. A fill-sanity guard rejects a price that is wildly
off the latest known close, so a bad quote can't book a wrong cost basis.

CLI (the agent calls these via Bash; all output is JSON):
  python -m agent.ledger state
  python -m agent.ledger record --symbol NVDA --side BUY --shares 100 \
      --price 123.45 --rationale "momentum breakout" --run-id 2026-06-22T14:00
  python -m agent.ledger mark
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from agent.models import (
    ACCOUNT,
    STARTING_CAPITAL,
    DeskBacktest,
    DeskEquity,
    DeskPosition,
    DeskTrade,
)

# Reject a fill whose price is more than this fraction away from the latest
# known close — a stale/garbled quote, not a real move. Generous enough not
# to block legitimate gaps; tight enough to catch a dropped digit.
FILL_BAND = 0.25


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def cash(session: Session, account: str = ACCOUNT) -> float:
    """Cash = starting capital + Σ sell proceeds − Σ buy cost (from the ledger)."""
    rows = (session.query(DeskTrade.side, func.coalesce(func.sum(DeskTrade.dollars), 0.0))
            .filter(DeskTrade.account == account)
            .group_by(DeskTrade.side).all())
    by_side = {side: float(total) for side, total in rows}
    return round(STARTING_CAPITAL + by_side.get("SELL", 0.0) - by_side.get("BUY", 0.0), 2)


def rebuild_positions(session: Session, account: str = ACCOUNT) -> dict[str, DeskPosition]:
    """Recompute open lots from the trade ledger (average-cost basis).

    Idempotent: wipes and rewrites desk_positions for the account so the
    projection always equals the ledger. Returns the live position rows.
    """
    trades = (session.query(DeskTrade)
              .filter(DeskTrade.account == account)
              .order_by(DeskTrade.ts, DeskTrade.id).all())
    book: dict[str, dict] = {}
    for t in trades:
        b = book.setdefault(t.symbol, {"shares": 0, "cost": 0.0, "opened_at": t.ts})
        if t.side == "BUY":
            if b["shares"] == 0:
                b["opened_at"] = t.ts
            b["shares"] += t.shares
            b["cost"] += t.shares * t.price
        else:  # SELL — reduce average-cost basis proportionally
            if b["shares"] > 0:
                avg = b["cost"] / b["shares"]
                sold = min(t.shares, b["shares"])
                b["shares"] -= sold
                b["cost"] -= sold * avg
                if b["shares"] <= 0:
                    b["shares"] = 0
                    b["cost"] = 0.0

    session.query(DeskPosition).filter(DeskPosition.account == account).delete()
    out: dict[str, DeskPosition] = {}
    for sym, b in book.items():
        if b["shares"] <= 0:
            continue
        pos = DeskPosition(
            account=account, symbol=sym, shares=int(b["shares"]),
            avg_price=round(b["cost"] / b["shares"], 4), opened_at=b["opened_at"],
        )
        session.add(pos)
        out[sym] = pos
    session.flush()
    return out


def _held_shares(session: Session, symbol: str, account: str = ACCOUNT) -> int:
    pos = (session.query(DeskPosition)
           .filter(DeskPosition.account == account, DeskPosition.symbol == symbol)
           .one_or_none())
    return int(pos.shares) if pos else 0


def record_trade(session: Session, *, symbol: str, side: str, shares: int,
                 price: float, rationale: str | None = None,
                 run_id: str | None = None, latest_close: float | None = None,
                 account: str = ACCOUNT) -> dict:
    """Append one fill, then rebuild positions. Returns a result dict.

    Guards: whole positive shares; long-only (no naked shorts; a SELL is
    capped at the held quantity); fill-sanity band vs ``latest_close``;
    a BUY is refused if it would overdraw cash.
    """
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    if side not in ("BUY", "SELL"):
        return {"ok": False, "error": f"bad side {side!r}"}
    shares = int(shares)
    price = float(price)
    if shares <= 0 or price <= 0:
        return {"ok": False, "error": "shares and price must be positive"}

    # fill-sanity: reject a price far from the latest known close
    if latest_close is None:
        latest_close = _latest_close(symbol)
    if latest_close and latest_close > 0:
        dev = abs(price - latest_close) / latest_close
        if dev > FILL_BAND:
            return {"ok": False, "error": "fill rejected by sanity guard",
                    "price": price, "latest_close": round(latest_close, 4),
                    "deviation": round(dev, 4), "band": FILL_BAND}

    if side == "SELL":
        held = _held_shares(session, symbol, account)
        if held <= 0:
            return {"ok": False, "error": f"no open position in {symbol} to sell"}
        if shares > held:
            shares = held  # long-only: cap the sell at what's held

    gross = round(shares * price, 2)
    if side == "BUY" and gross > cash(session, account) + 1e-6:
        return {"ok": False, "error": "insufficient cash for buy",
                "needed": gross, "cash": cash(session, account)}

    session.add(DeskTrade(
        account=account, run_id=run_id, symbol=symbol, side=side,
        shares=shares, price=round(price, 4), dollars=gross,
        rationale=rationale, ts=_utcnow(),
    ))
    session.flush()
    rebuild_positions(session, account)
    session.commit()
    return {"ok": True, "symbol": symbol, "side": side, "shares": shares,
            "price": round(price, 4), "dollars": gross,
            "cash_after": cash(session, account)}


def _latest_close(symbol: str) -> float | None:
    try:
        from agent.data import latest_indicators
        info = latest_indicators([symbol])
        return info.get(symbol, {}).get("close")
    except Exception:
        return None


def _latest_closes(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    try:
        from agent.data import latest_indicators
        info = latest_indicators(symbols)
        return {s: d["close"] for s, d in info.items() if d.get("close")}
    except Exception:
        return {}


def mark(session: Session, *, prices: dict[str, float] | None = None,
         account: str = ACCOUNT) -> dict:
    """Mark open positions to the latest close and append an equity snapshot."""
    positions = rebuild_positions(session, account)
    if prices is None:
        prices = _latest_closes(list(positions))
    now = _utcnow()
    pos_value = 0.0
    for sym, pos in positions.items():
        px = prices.get(sym, pos.last_price or pos.avg_price)
        pos.last_price = round(float(px), 4)
        pos.marked_at = now
        pos_value += pos.shares * px
    c = cash(session, account)
    equity = round(c + pos_value, 2)
    snap = DeskEquity(
        account=account, ts=now, cash=c, positions_value=round(pos_value, 2),
        equity=equity, return_pct=round((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 4),
    )
    session.add(snap)
    session.commit()
    return state(session, account)


def state(session: Session, account: str = ACCOUNT) -> dict:
    """Full account snapshot: cash, positions (marked), equity, P&L."""
    positions = (session.query(DeskPosition)
                 .filter(DeskPosition.account == account).all())
    c = cash(session, account)
    pos_list = []
    pos_value = 0.0
    for p in positions:
        mark_px = p.last_price or p.avg_price
        mv = round(p.shares * mark_px, 2)
        pos_value += mv
        pos_list.append({
            "symbol": p.symbol, "shares": p.shares,
            "avg_price": round(p.avg_price, 4), "last_price": round(mark_px, 4),
            "market_value": mv, "cost_basis": round(p.shares * p.avg_price, 2),
            "unrealized_pnl": round(p.shares * (mark_px - p.avg_price), 2),
            "weight": None,
        })
    equity = round(c + pos_value, 2)
    for row in pos_list:
        row["weight"] = round(row["market_value"] / equity, 4) if equity else 0.0
    return {
        "account": account,
        "cash": c,
        "positions_value": round(pos_value, 2),
        "equity": equity,
        "starting_capital": STARTING_CAPITAL,
        "total_pnl": round(equity - STARTING_CAPITAL, 2),
        "total_return_pct": round((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 4),
        "positions": sorted(pos_list, key=lambda r: -r["market_value"]),
    }


def save_backtest(label: str, result: dict, *, run_id: str | None = None,
                  account: str = ACCOUNT) -> int:
    """Persist a backtest the agent ran (evidence panel reads desk_backtests)."""
    from agent.data import session_factory

    sess = session_factory()()
    try:
        row = DeskBacktest(
            account=account, run_id=run_id, label=label,
            spec={k: result.get(k) for k in ("rule", "symbols", "schedule", "start", "end")},
            result={k: result.get(k) for k in (
                "return_pct", "sharpe", "max_drawdown_pct", "benchmark_return_pct",
                "excess_return_pct", "num_trades", "days", "final_equity")},
            ts=_utcnow(),
        )
        sess.add(row)
        sess.commit()
        return row.id
    finally:
        sess.close()


def main(argv: list[str] | None = None) -> None:
    import argparse

    from agent.data import session_factory

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("state")
    sub.add_parser("mark")
    rec = sub.add_parser("record")
    rec.add_argument("--symbol", required=True)
    rec.add_argument("--side", required=True, choices=["BUY", "SELL"])
    rec.add_argument("--shares", required=True, type=int)
    rec.add_argument("--price", required=True, type=float)
    rec.add_argument("--rationale", default=None)
    rec.add_argument("--run-id", default=None)
    rec.add_argument("--latest-close", default=None, type=float)
    args = p.parse_args(argv)

    sess = session_factory()()
    try:
        if args.cmd == "state":
            print(json.dumps(state(sess), indent=2))
        elif args.cmd == "mark":
            print(json.dumps(mark(sess), indent=2))
        elif args.cmd == "record":
            print(json.dumps(record_trade(
                sess, symbol=args.symbol, side=args.side, shares=args.shares,
                price=args.price, rationale=args.rationale, run_id=args.run_id,
                latest_close=args.latest_close), indent=2))
    finally:
        sess.close()


if __name__ == "__main__":
    main()

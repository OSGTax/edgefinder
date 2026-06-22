"""The paper book — record fills, rebuild positions, mark to market.

The fill ledger (``desk_trades``) is the source of truth. Cash is always
recomputed from it; ``desk_positions`` is a projection rebuilt from the same
ledger, so the account can never silently drift. Long-only, paper only,
whole shares, no leverage. A fill-sanity guard rejects a price that is wildly
off the latest known close, so a bad quote can't book a wrong cost basis.

Persistence goes through ``agent.store`` (pg or rest transport) — the same
integrity logic runs whether the book lives behind raw Postgres or the
Supabase Data API.

CLI (the agent calls these via Bash; all output is JSON):
  python -m agent.ledger state
  python -m agent.ledger record --symbol NVDA --side BUY --shares 100 \
      --price 123.45 --rationale "momentum breakout" --run-id 2026-06-22T14:00
  python -m agent.ledger mark
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.models import ACCOUNT, STARTING_CAPITAL

# Reject a fill whose price is more than this fraction away from the latest
# known close — a stale/garbled quote, not a real move.
FILL_BAND = 0.25


def _utcnow() -> datetime:
    # naive UTC to match the existing desk_* rows (TIMESTAMP without tz)
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _store():
    from agent.store import get_store

    return get_store()


# ── reads / projections ─────────────────────────────────────


def _trades(store, account: str) -> list[dict]:
    rows = store.select("desk_trades", filters={"account": account})
    rows.sort(key=lambda r: (str(r.get("ts")), r.get("id") or 0))
    return rows


def cash(store, account: str = ACCOUNT) -> float:
    """Cash = starting capital + Σ sell proceeds − Σ buy cost (from the ledger)."""
    buy = sell = 0.0
    for t in _trades(store, account):
        if t["side"] == "SELL":
            sell += float(t["dollars"])
        else:
            buy += float(t["dollars"])
    return round(STARTING_CAPITAL + sell - buy, 2)


def _compute_book(trades: list[dict]) -> dict[str, dict]:
    """Average-cost open lots from the trade ledger."""
    book: dict[str, dict] = {}
    for t in trades:
        b = book.setdefault(t["symbol"], {"shares": 0, "cost": 0.0, "opened_at": t.get("ts")})
        if t["side"] == "BUY":
            if b["shares"] == 0:
                b["opened_at"] = t.get("ts")
            b["shares"] += int(t["shares"])
            b["cost"] += int(t["shares"]) * float(t["price"])
        else:  # SELL — reduce average-cost basis proportionally
            if b["shares"] > 0:
                avg = b["cost"] / b["shares"]
                sold = min(int(t["shares"]), b["shares"])
                b["shares"] -= sold
                b["cost"] -= sold * avg
                if b["shares"] <= 0:
                    b["shares"] = 0
                    b["cost"] = 0.0
    return book


def rebuild_positions(store, account: str = ACCOUNT) -> dict[str, dict]:
    """Recompute open lots from the ledger and rewrite desk_positions.

    Idempotent: wipes and rewrites the projection so it always equals the
    ledger. Returns ``{symbol: {shares, avg_price, opened_at}}``.
    """
    book = _compute_book(_trades(store, account))
    store.delete("desk_positions", {"account": account})
    rows, out = [], {}
    for sym, b in book.items():
        if b["shares"] <= 0:
            continue
        avg = round(b["cost"] / b["shares"], 4)
        rows.append({"account": account, "symbol": sym, "shares": int(b["shares"]),
                     "avg_price": avg, "last_price": None,
                     "opened_at": b["opened_at"], "marked_at": None})
        out[sym] = {"shares": int(b["shares"]), "avg_price": avg,
                    "opened_at": b["opened_at"]}
    if rows:
        store.insert("desk_positions", rows, returning=False)
    return out


def _held_shares(store, symbol: str, account: str = ACCOUNT) -> int:
    rows = store.select("desk_positions", filters={"account": account, "symbol": symbol})
    return int(rows[0]["shares"]) if rows else 0


# ── writes ──────────────────────────────────────────────────


def record_trade(store=None, *, symbol: str, side: str, shares: int, price: float,
                 rationale: str | None = None, run_id: str | None = None,
                 latest_close: float | None = None, account: str = ACCOUNT) -> dict:
    """Append one fill, then rebuild positions. Returns a result dict.

    Guards: whole positive shares; long-only (a SELL is capped at the held
    quantity); fill-sanity band vs ``latest_close``; a BUY is refused if it
    would overdraw cash.
    """
    store = store or _store()
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    if side not in ("BUY", "SELL"):
        return {"ok": False, "error": f"bad side {side!r}"}
    shares = int(shares)
    price = float(price)
    if shares <= 0 or price <= 0:
        return {"ok": False, "error": "shares and price must be positive"}

    if latest_close is None:
        latest_close = _latest_close(symbol)
    if latest_close and latest_close > 0:
        dev = abs(price - latest_close) / latest_close
        if dev > FILL_BAND:
            return {"ok": False, "error": "fill rejected by sanity guard",
                    "price": price, "latest_close": round(latest_close, 4),
                    "deviation": round(dev, 4), "band": FILL_BAND}

    if side == "SELL":
        held = _held_shares(store, symbol, account)
        if held <= 0:
            return {"ok": False, "error": f"no open position in {symbol} to sell"}
        if shares > held:
            shares = held  # long-only: cap the sell at what's held

    gross = round(shares * price, 2)
    if side == "BUY" and gross > cash(store, account) + 1e-6:
        return {"ok": False, "error": "insufficient cash for buy",
                "needed": gross, "cash": cash(store, account)}

    store.insert("desk_trades", {
        "account": account, "run_id": run_id, "symbol": symbol, "side": side,
        "shares": shares, "price": round(price, 4), "dollars": gross,
        "rationale": rationale, "ts": _utcnow()}, returning=False)
    rebuild_positions(store, account)
    return {"ok": True, "symbol": symbol, "side": side, "shares": shares,
            "price": round(price, 4), "dollars": gross,
            "cash_after": cash(store, account)}


def _latest_close(symbol: str) -> float | None:
    return _latest_closes([symbol]).get(symbol)


def _latest_closes(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    try:
        from agent.data import latest_indicators

        info = latest_indicators(symbols)
        return {s: d["close"] for s, d in info.items() if d.get("close")}
    except Exception:
        return {}


def mark(store=None, *, prices: dict[str, float] | None = None,
         account: str = ACCOUNT) -> dict:
    """Mark open positions to the latest close and append an equity snapshot."""
    store = store or _store()
    positions = rebuild_positions(store, account)
    if prices is None:
        prices = _latest_closes(list(positions))
    now = _utcnow()
    pos_value = 0.0
    for sym, p in positions.items():
        px = prices.get(sym) or p["avg_price"]
        px = round(float(px), 4)
        pos_value += p["shares"] * px
        store.update("desk_positions", {"account": account, "symbol": sym},
                     {"last_price": px, "marked_at": now}, returning=False)
    c = cash(store, account)
    equity = round(c + pos_value, 2)
    store.insert("desk_equity", {
        "account": account, "ts": now, "cash": c,
        "positions_value": round(pos_value, 2), "equity": equity,
        "return_pct": round((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 4),
    }, returning=False)
    return state(store, account)


def state(store=None, account: str = ACCOUNT) -> dict:
    """Full account snapshot: cash, positions (marked), equity, P&L."""
    store = store or _store()
    positions = store.select("desk_positions", filters={"account": account})
    c = cash(store, account)
    pos_list, pos_value = [], 0.0
    for p in positions:
        mark_px = p.get("last_price") or p["avg_price"]
        mv = round(p["shares"] * mark_px, 2)
        pos_value += mv
        pos_list.append({
            "symbol": p["symbol"], "shares": p["shares"],
            "avg_price": round(p["avg_price"], 4), "last_price": round(mark_px, 4),
            "market_value": mv, "cost_basis": round(p["shares"] * p["avg_price"], 2),
            "unrealized_pnl": round(p["shares"] * (mark_px - p["avg_price"]), 2),
            "weight": None,
        })
    equity = round(c + pos_value, 2)
    for row in pos_list:
        row["weight"] = round(row["market_value"] / equity, 4) if equity else 0.0
    return {
        "account": account, "cash": c, "positions_value": round(pos_value, 2),
        "equity": equity, "starting_capital": STARTING_CAPITAL,
        "total_pnl": round(equity - STARTING_CAPITAL, 2),
        "total_return_pct": round((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 4),
        "positions": sorted(pos_list, key=lambda r: -r["market_value"]),
    }


def save_backtest(label: str, result: dict, *, run_id: str | None = None,
                  account: str = ACCOUNT) -> int:
    """Persist a backtest the agent ran (evidence panel reads desk_backtests)."""
    rows = _store().insert("desk_backtests", {
        "account": account, "run_id": run_id, "label": label,
        "spec": {k: result.get(k) for k in ("rule", "symbols", "schedule", "start", "end")},
        "result": {k: result.get(k) for k in (
            "return_pct", "sharpe", "max_drawdown_pct", "benchmark_return_pct",
            "excess_return_pct", "num_trades", "days", "final_equity")},
        "ts": _utcnow()})
    return int(rows[0]["id"]) if rows else 0


def main(argv: list[str] | None = None) -> None:
    import argparse

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

    store = _store()
    if args.cmd == "state":
        print(json.dumps(state(store), indent=2))
    elif args.cmd == "mark":
        print(json.dumps(mark(store), indent=2))
    elif args.cmd == "record":
        print(json.dumps(record_trade(
            store, symbol=args.symbol, side=args.side, shares=args.shares,
            price=args.price, rationale=args.rationale, run_id=args.run_id,
            latest_close=args.latest_close), indent=2))


if __name__ == "__main__":
    main()

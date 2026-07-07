"""The paper book — live-quote fills, rebuild positions, mark to market.

The fill ledger (``desk_trades``) is the source of truth. Cash is always
recomputed from it; ``desk_positions`` is a projection rebuilt from the same
ledger, so the account can never silently drift. Long-only, paper only,
fractional shares, no leverage.

THE HONESTY CONTRACT (REBUILD-V3.md): a live fill prices off the real-time
Alpaca SIP quote at the moment it books — BUY at the live ask, SELL at the
live bid, ± slippage — and the quote snapshot {bid, ask, mid, t} is stamped
on the fill row (``fill_quote``). ``fill`` refuses to book when the market
is closed or the quote fails sanity. The legacy ``record`` path (explicit
price) survives for tests/backfills and keeps its close-band guard.

Persistence goes through ``agent.store`` (pg or rest transport) — the same
integrity logic runs whether the book lives behind raw Postgres or the
Supabase Data API.

CLI (the agent calls these via Bash; all output is JSON):
  python -m agent.ledger state
  python -m agent.ledger fill --symbol NVDA --side buy --shares 12.5 \
      --rationale "momentum breakout" --run-id 2026-07-07T14:30
  python -m agent.ledger fill --symbol NVDA --side buy --notional 5000 ...
  python -m agent.ledger mark
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from agent.models import ACCOUNT, STARTING_CAPITAL

# Legacy guard (record path): reject a price more than this fraction away
# from the latest known close — a stale/garbled quote, not a real move.
FILL_BAND = 0.25
# Live guard (fill path): the booked price must sit essentially ON the quote —
# within this fraction outside the [bid, ask] it priced from.
LIVE_BAND = 0.005
# Slippage applied to live fills, in basis points (buy pays up, sell gives up).
SLIPPAGE_BP = 1.0
# Ignore sub-satoshi share residue from fractional math.
EPS_SHARES = 1e-9


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
    """Average-cost open lots from the trade ledger (fractional shares)."""
    book: dict[str, dict] = {}
    for t in trades:
        b = book.setdefault(t["symbol"], {"shares": 0.0, "cost": 0.0, "opened_at": t.get("ts")})
        qty = float(t["shares"])
        if t["side"] == "BUY":
            if b["shares"] <= EPS_SHARES:
                b["opened_at"] = t.get("ts")
            b["shares"] += qty
            b["cost"] += qty * float(t["price"])
        else:  # SELL — reduce average-cost basis proportionally
            if b["shares"] > EPS_SHARES:
                avg = b["cost"] / b["shares"]
                sold = min(qty, b["shares"])
                b["shares"] -= sold
                b["cost"] -= sold * avg
                if b["shares"] <= EPS_SHARES:
                    b["shares"] = 0.0
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
        if b["shares"] <= EPS_SHARES:
            continue
        shares = round(b["shares"], 6)
        avg = round(b["cost"] / b["shares"], 4)
        rows.append({"account": account, "symbol": sym, "shares": shares,
                     "avg_price": avg, "last_price": None,
                     "opened_at": b["opened_at"], "marked_at": None})
        out[sym] = {"shares": shares, "avg_price": avg,
                    "opened_at": b["opened_at"]}
    if rows:
        store.insert("desk_positions", rows, returning=False)
    return out


def _held_shares(store, symbol: str, account: str = ACCOUNT) -> float:
    rows = store.select("desk_positions", filters={"account": account, "symbol": symbol})
    return float(rows[0]["shares"]) if rows else 0.0


# ── writes ──────────────────────────────────────────────────


def record_trade(store=None, *, symbol: str, side: str, shares: float, price: float,
                 rationale: str | None = None, run_id: str | None = None,
                 latest_close: float | None = None, fill_quote: dict | None = None,
                 account: str = ACCOUNT) -> dict:
    """Append one fill, then rebuild positions. Returns a result dict.

    Guards: positive fractional shares; long-only (a SELL is capped at the
    held quantity); a BUY is refused if it would overdraw cash. Price sanity:
    with a ``fill_quote`` the price must sit essentially ON the quote
    (LIVE_BAND outside [bid, ask]); without one, the legacy close-band applies.
    """
    store = store or _store()
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    if side not in ("BUY", "SELL"):
        return {"ok": False, "error": f"bad side {side!r}"}
    shares = round(float(shares), 6)
    price = float(price)
    if shares <= EPS_SHARES or price <= 0:
        return {"ok": False, "error": "shares and price must be positive"}

    if fill_quote and fill_quote.get("bid") and fill_quote.get("ask"):
        bid, ask = float(fill_quote["bid"]), float(fill_quote["ask"])
        if not (bid * (1 - LIVE_BAND) <= price <= ask * (1 + LIVE_BAND)):
            return {"ok": False, "error": "fill rejected: price off the live quote",
                    "price": price, "bid": bid, "ask": ask, "band": LIVE_BAND}
    else:
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
        if held <= EPS_SHARES:
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
        "rationale": rationale, "fill_quote": fill_quote,
        "ts": _utcnow()}, returning=False)
    rebuild_positions(store, account)
    return {"ok": True, "symbol": symbol, "side": side, "shares": shares,
            "price": round(price, 4), "dollars": gross, "fill_quote": fill_quote,
            "cash_after": cash(store, account)}


def _live_quote(symbol: str) -> dict:
    """One live SIP quote via REST (the fill path's price source). Raises on
    missing keys/quote — the caller turns that into a clean rejection."""
    from agent import broker

    q = broker.Broker().quotes([symbol]).get(symbol)
    if not q or not q.get("bid") or not q.get("ask"):
        raise ValueError(f"no live quote for {symbol}")
    return q


def live_fill(store=None, *, symbol: str, side: str, shares: float | None = None,
              notional: float | None = None, rationale: str | None = None,
              run_id: str | None = None, slippage_bp: float = SLIPPAGE_BP,
              account: str = ACCOUNT) -> dict:
    """Book a fill AT THE LIVE QUOTE — the only execution path the agent uses.

    Reads the real-time SIP quote, prices the correct side (BUY at ask,
    SELL at bid, ± slippage), stamps the quote snapshot on the fill, and
    refuses to book when the market is closed or the quote is degenerate.
    Exactly one of ``shares`` / ``notional`` sizes the order (fractional ok).
    """
    from agent import broker

    store = store or _store()
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    if (shares is None) == (notional is None):
        return {"ok": False, "error": "pass exactly one of shares or notional"}

    try:
        b = broker.Broker()
        if not b.is_market_open():
            return {"ok": False, "error": "market closed — live fills only trade "
                    "during regular hours (no stale/after-hours pricing)"}
        q = _live_quote(symbol)
    except Exception as exc:  # noqa: BLE001 — clean rejection, not a stacktrace
        return {"ok": False, "error": f"live quote unavailable: {exc}"}

    bid, ask = float(q["bid"]), float(q["ask"])
    if bid <= 0 or ask <= 0 or ask < bid or (ask - bid) / ask > 0.05:
        return {"ok": False, "error": "degenerate quote (crossed or >5% spread)",
                "bid": bid, "ask": ask}

    slip = slippage_bp / 10_000.0
    price = round(ask * (1 + slip), 4) if side == "BUY" else round(bid * (1 - slip), 4)
    if shares is None:
        shares = round(float(notional) / price, 6)

    snapshot = {"bid": bid, "ask": ask, "mid": q.get("mid"), "t": q.get("t"),
                "src": "alpaca_sip_rest", "slippage_bp": slippage_bp}
    return record_trade(store, symbol=symbol, side=side, shares=shares,
                        price=price, rationale=rationale, run_id=run_id,
                        fill_quote=snapshot, account=account)


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


def _live_mids(symbols: list[str]) -> dict[str, float]:
    """Live SIP mids for marking. Empty dict on any failure → close fallback."""
    if not symbols:
        return {}
    try:
        from agent import broker

        qs = broker.Broker().quotes(symbols)
        return {s: q["mid"] for s, q in qs.items() if q.get("mid")}
    except Exception:
        return {}


def mark(store=None, *, prices: dict[str, float] | None = None,
         account: str = ACCOUNT) -> dict:
    """Mark open positions and append an equity snapshot.

    Prefers LIVE SIP mids (even after hours — the last quote is still the best
    mark); falls back to the latest close per symbol only when live quotes are
    unavailable."""
    store = store or _store()
    positions = rebuild_positions(store, account)
    if prices is None:
        prices = _live_mids(list(positions))
        for sym, px in _latest_closes([s for s in positions if s not in prices]).items():
            prices.setdefault(sym, px)
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
    fil = sub.add_parser("fill", help="book a fill AT THE LIVE QUOTE (the agent's path)")
    fil.add_argument("--symbol", required=True)
    fil.add_argument("--side", required=True, choices=["buy", "sell", "BUY", "SELL"])
    fil.add_argument("--shares", default=None, type=float)
    fil.add_argument("--notional", default=None, type=float)
    fil.add_argument("--rationale", default=None)
    fil.add_argument("--run-id", default=None)
    fil.add_argument("--slippage-bp", default=SLIPPAGE_BP, type=float)
    rec = sub.add_parser("record", help="legacy explicit-price fill (tests/backfills)")
    rec.add_argument("--symbol", required=True)
    rec.add_argument("--side", required=True, choices=["BUY", "SELL"])
    rec.add_argument("--shares", required=True, type=float)
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
    elif args.cmd == "fill":
        print(json.dumps(live_fill(
            store, symbol=args.symbol, side=args.side, shares=args.shares,
            notional=args.notional, rationale=args.rationale, run_id=args.run_id,
            slippage_bp=args.slippage_bp), indent=2))
    elif args.cmd == "record":
        print(json.dumps(record_trade(
            store, symbol=args.symbol, side=args.side, shares=args.shares,
            price=args.price, rationale=args.rationale, run_id=args.run_id,
            latest_close=args.latest_close), indent=2))


if __name__ == "__main__":
    main()

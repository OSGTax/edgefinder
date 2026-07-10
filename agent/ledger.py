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
# within this fraction outside the [bid, ask] it priced from. OPRA books quote
# in cents so the equity 0.5% band rejects legitimate fills on cheap contracts;
# options get a wider fraction OR an absolute-cents floor, whichever is looser.
LIVE_BAND = 0.005
LIVE_BAND_OPT = 0.02
LIVE_BAND_OPT_CENTS = 0.05
# Slippage applied to live fills, in basis points (buy pays up, sell gives up).
SLIPPAGE_BP = 1.0
# Ignore sub-satoshi share residue from fractional math.
EPS_SHARES = 1e-9
# Options: one contract controls 100 shares of the underlying.
MULTIPLIER = 100
# Extended-hours fills are allowed for equities, off entirely for options.
# The equity spread cap tightens outside RTH — thinner tape, wider quotes.
MAX_SPREAD_EQ = 0.05
MAX_SPREAD_EQ_EXT = 0.02
MAX_SPREAD_OPT = 0.50
# Crypto: 24/7, no options, fractional (lot size ≪ 1 unit). Alpaca crypto
# spreads on the top pairs (BTC/USD, ETH/USD) hover well under 1%; low-volume
# pairs can reach 2-3%. 3% cap catches genuine dislocation without over-
# rejecting.
MAX_SPREAD_CRYPTO = 0.03
# Quote-freshness at fill time — reject if the quote timestamp is older than
# this many seconds. OPRA quotes update less often than SIP; the option cap is
# looser to match reality (thin contracts genuinely quote once per minute).
# Crypto quotes tick continuously, so the equity cap fits.
MAX_QUOTE_AGE_SEC_EQ = 30.0
MAX_QUOTE_AGE_SEC_OPT = 120.0
MAX_QUOTE_AGE_SEC_CRYPTO = 30.0


def _mult(symbol: str) -> int:
    from agent import occ

    return MULTIPLIER if occ.is_option(symbol) else 1


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
    """Average-cost open lots from the trade ledger.

    Equities are long-only fractional lots (record_trade guarantees sells never
    exceed holdings). Option contracts are SIGNED: a SELL beyond the long lot
    opens a short leg (negative shares) — record_trade only permits that when
    the short is covered, so here we just do honest signed arithmetic. ``cost``
    tracks |basis|; crossing through zero re-opens the lot at the fill price.
    """
    from agent import occ

    book: dict[str, dict] = {}
    for t in trades:
        sym = t["symbol"]
        b = book.setdefault(sym, {"shares": 0.0, "cost": 0.0, "opened_at": t.get("ts")})
        qty = float(t["shares"])
        signed = qty if t["side"] == "BUY" else -qty
        cur = b["shares"]
        if abs(cur) <= EPS_SHARES:
            b.update(shares=signed, cost=abs(signed) * float(t["price"]),
                     opened_at=t.get("ts"))
            continue
        if (cur > 0) == (signed > 0):  # extending the same direction
            b["shares"] = cur + signed
            b["cost"] += abs(signed) * float(t["price"])
            continue
        # reducing / crossing
        closing = min(abs(signed), abs(cur))
        avg = b["cost"] / abs(cur)
        b["cost"] -= closing * avg
        b["shares"] = cur + signed
        if abs(b["shares"]) <= EPS_SHARES:
            b["shares"], b["cost"] = 0.0, 0.0
        elif (b["shares"] > 0) != (cur > 0):  # crossed zero → new lot at fill px
            b["cost"] = abs(b["shares"]) * float(t["price"])
            b["opened_at"] = t.get("ts")
        # equities can never actually go short (guarded upstream); clamp residue
        if not occ.is_option(sym) and b["shares"] < 0:
            b["shares"], b["cost"] = 0.0, 0.0
    return book


def _realized_pnl(trades: list[dict]):
    """Replay the avg-cost stream and ACCUMULATE realized P&L on reductions
    (``_compute_book`` discards it — this is the read-only twin that keeps it).

    Returns ``(by_run_symbol, by_symbol)`` where a reduction of ``closing``
    units books ``closing * (px - avg) * sign * mult`` — sign +1 closing a
    long, -1 closing a short (buying back cheaper than you sold = profit).
    Realized P&L is attributed to the run that booked the CLOSING fill,
    priced against the global average cost at that moment — approximate when
    several runs built the lot; the per-symbol totals are exact.
    """
    from agent import occ  # noqa: F401 — via _mult

    by_run_symbol: dict[tuple[str | None, str], float] = {}
    by_symbol: dict[str, float] = {}
    book: dict[str, dict] = {}
    for t in trades:
        sym = t["symbol"]
        b = book.setdefault(sym, {"shares": 0.0, "cost": 0.0})
        qty = float(t["shares"])
        signed = qty if t["side"] == "BUY" else -qty
        cur = b["shares"]
        if abs(cur) <= EPS_SHARES or (cur > 0) == (signed > 0):
            b["shares"] = cur + signed
            b["cost"] += abs(signed) * float(t["price"])
            continue
        closing = min(abs(signed), abs(cur))
        avg = b["cost"] / abs(cur)
        sign = 1.0 if cur > 0 else -1.0
        pnl = closing * (float(t["price"]) - avg) * sign * _mult(sym)
        key = (t.get("run_id"), sym)
        by_run_symbol[key] = by_run_symbol.get(key, 0.0) + pnl
        by_symbol[sym] = by_symbol.get(sym, 0.0) + pnl
        b["cost"] -= closing * avg
        b["shares"] = cur + signed
        if abs(b["shares"]) <= EPS_SHARES:
            b["shares"], b["cost"] = 0.0, 0.0
        elif (b["shares"] > 0) != (cur > 0):
            b["cost"] = abs(b["shares"]) * float(t["price"])
    return by_run_symbol, by_symbol


def _spy_closes(store, *, since: str) -> list[tuple[str, float]]:
    """SPY daily closes (ascending) from ``daily_bars``, with a lookback buffer
    so a window starting on a weekend/holiday still finds a baseline close.

    Price return only — dividends are excluded, which matches the book (the
    ledger books no dividend cash), so the comparison is like-for-like.
    ``index_daily`` is NOT used: it froze at the 2026-06-22 cutover."""
    from datetime import date as _date, timedelta as _td

    start = _date.fromisoformat(since[:10]) - _td(days=10)
    rows = store.select("daily_bars", columns="date,close",
                        filters={"symbol": "SPY", "date": ("gte", start)},
                        order=[("date", "asc")])
    return [(str(r["date"])[:10], float(r["close"]))
            for r in rows if r.get("close")]


def _spy_window_pct(spy: list[tuple[str, float]], start_date: str) -> float | None:
    """SPY price change from the last close ON/BEFORE ``start_date`` to the
    latest stored close. None when no baseline exists in the series."""
    base = None
    for d, c in spy:
        if d <= start_date[:10]:
            base = c
        else:
            break
    if not base:
        return None
    return round((spy[-1][1] - base) / base * 100, 2)


def outcomes(store=None, *, days: int = 30, run_id: str | None = None,
             account: str = ACCOUNT) -> dict:
    """How past decisions actually aged — the grounding for wiki lessons.

    Joins each decision's picks to its fills (run_id + symbol) and reports
    realized P&L (closing-run attribution, see ``_realized_pnl``), current
    open exposure, and ``since_this_run_pct`` — the current mark vs THAT
    run's own average fill price (exact per pick, the primary reflection
    signal). Every window also carries the SPY move over the same period
    (``spy_same_window_pct``) and the difference (``alpha_pct``) — raw P&L
    on a long book is mostly beta; alpha is the skill signal reflection
    should grade. Settlement rows (run_id='settlement') are bucketed
    separately; trades with no run_id are counted, never silently dropped.
    No network: marks come from desk_positions.last_price, same as
    ``state()``."""
    from agent import occ

    store = store or _store()
    trades = _trades(store, account)
    by_run_symbol, by_symbol = _realized_pnl(trades)
    positions = {r["symbol"]: r for r in
                 store.select("desk_positions", filters={"account": account})}

    cutoff = _utcnow() - __import__("datetime").timedelta(days=days)
    decisions = store.select("desk_decisions", filters={"account": account},
                             order=[("ts", "desc")], limit=200)
    decisions = [d for d in decisions
                 if (run_id and d["run_id"] == run_id)
                 or (not run_id and str(d.get("ts") or "") >= str(cutoff))]

    inception = str(trades[0]["ts"])[:10] if trades else None
    window_starts = [str(d.get("ts") or "")[:10]
                     for d in decisions if d.get("ts")]
    if inception:
        window_starts.append(inception)
    spy = _spy_closes(store, since=min(window_starts)) if window_starts else []

    fills_by_run: dict[tuple[str | None, str], list[dict]] = {}
    for t in trades:
        fills_by_run.setdefault((t.get("run_id"), t["symbol"]), []).append(t)

    runs = []
    for d in decisions:
        rid = d["run_id"]
        run_spy_pct = (_spy_window_pct(spy, str(d.get("ts") or ""))
                       if d.get("ts") else None)
        picks_out = []
        for p in (d.get("picks") or []):
            sym = str(p.get("symbol") or "").upper()
            fills = [{"side": f["side"], "shares": f["shares"], "price": f["price"]}
                     for f in fills_by_run.get((rid, sym), [])]
            buys = [f for f in fills if f["side"] == "BUY"]
            entry_avg = (sum(f["shares"] * f["price"] for f in buys)
                         / sum(f["shares"] for f in buys)) if buys else None
            pos = positions.get(sym)
            mark = (pos.get("last_price") or pos.get("avg_price")) if pos else None
            open_now = None
            if pos and abs(float(pos["shares"])) > EPS_SHARES:
                m = _mult(sym)
                open_now = {"shares": float(pos["shares"]),
                            "avg_price": pos["avg_price"], "last_price": mark,
                            "unrealized_pnl": round(float(pos["shares"])
                                                    * ((mark or pos["avg_price"])
                                                       - pos["avg_price"]) * m, 2)}
            since_pct = (round((mark - entry_avg) / entry_avg * 100, 2)
                         if (entry_avg and mark) else None)
            alpha = (round(since_pct - run_spy_pct, 2)
                     if (since_pct is not None and run_spy_pct is not None)
                     else None)
            picks_out.append({
                "symbol": sym, "action": p.get("action"),
                "why_now": p.get("why_now"), "rationale": p.get("rationale"),
                "prediction": p.get("prediction"),
                "horizon_days": p.get("horizon_days"), "kill": p.get("kill"),
                "fills": fills, "entry_avg_px": round(entry_avg, 4) if entry_avg else None,
                "realized_pnl": round(by_run_symbol.get((rid, sym), 0.0), 2),
                "open_now": open_now, "since_this_run_pct": since_pct,
                "spy_same_window_pct": run_spy_pct, "alpha_pct": alpha})
        runs.append({"run_id": rid, "ts": str(d.get("ts") or ""),
                     "regime": d.get("regime"), "summary": d.get("summary"),
                     "picks": picks_out, "rejected": d.get("rejected") or [],
                     "spy_same_window_pct": run_spy_pct,
                     "run_realized_pnl": round(sum(
                         v for (r, _), v in by_run_symbol.items() if r == rid), 2)})

    settlement_pnl = round(sum(v for (r, _), v in by_run_symbol.items()
                               if r == "settlement"), 2)
    unattributed = sum(1 for t in trades if not t.get("run_id"))
    symbols_out = []
    for sym in sorted(set(by_symbol) | set(positions)):
        pos = positions.get(sym)
        m = _mult(sym)
        mark = (pos.get("last_price") or pos.get("avg_price")) if pos else None
        symbols_out.append({
            "symbol": sym, "realized_pnl": round(by_symbol.get(sym, 0.0), 2),
            "unrealized_pnl": (round(float(pos["shares"])
                                     * ((mark or pos["avg_price"]) - pos["avg_price"])
                                     * m, 2) if pos else 0.0),
            "open_shares": float(pos["shares"]) if pos else 0.0,
            "is_option": occ.is_option(sym)})
    book = None
    if inception:
        c = STARTING_CAPITAL
        for t in trades:
            c += float(t["dollars"]) * (1 if t["side"] == "SELL" else -1)
        pos_value = sum(
            float(p["shares"]) * float(p.get("last_price") or p["avg_price"])
            * _mult(s) for s, p in positions.items())
        book_pct = round((c + pos_value - STARTING_CAPITAL)
                         / STARTING_CAPITAL * 100, 2)
        spy_pct = _spy_window_pct(spy, inception)
        book = {"inception": inception, "since_inception_pct": book_pct,
                "spy_since_inception_pct": spy_pct,
                "alpha_pct": (round(book_pct - spy_pct, 2)
                              if spy_pct is not None else None)}

    return {"as_of": str(_utcnow()), "days": days,
            "convention": "realized P&L is attributed to the run that booked "
                          "the CLOSING fill, priced against the global average "
                          "cost at that moment — approximate when several runs "
                          "built the lot; per-symbol totals are exact. "
                          "since_this_run_pct compares the current mark to that "
                          "run's own average fill and is exact per pick. "
                          "spy_same_window_pct / alpha_pct benchmark each window "
                          "against SPY closes from daily_bars (price return, "
                          "dividends excluded on both sides — the book books no "
                          "dividend cash either); a long book's raw P&L is "
                          "mostly market beta, so grade alpha, not dollars.",
            "book": book, "runs": runs, "symbols": symbols_out,
            "settlement": {"realized_pnl": settlement_pnl},
            "unattributed_trades": unattributed}


def rebuild_positions(store, account: str = ACCOUNT) -> dict[str, dict]:
    """Recompute open lots from the ledger and rewrite desk_positions.

    Idempotent: wipes and rewrites the projection so it always equals the
    ledger. Returns ``{symbol: {shares, avg_price, opened_at}}``.
    """
    book = _compute_book(_trades(store, account))
    store.delete("desk_positions", {"account": account})
    rows, out = [], {}
    for sym, b in book.items():
        if abs(b["shares"]) <= EPS_SHARES:
            continue
        shares = round(b["shares"], 6)  # negative = covered short option leg
        avg = round(b["cost"] / abs(b["shares"]), 4)
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


# ── options risk math (defined-risk only — REBUILD-V3 charter) ──


def _positions_map(store, account: str = ACCOUNT) -> dict[str, float]:
    rows = store.select("desk_positions", filters={"account": account})
    return {r["symbol"]: float(r["shares"]) for r in rows}


def _option_legs(positions: dict[str, float], underlying: str, type_: str):
    """(short_qty, long_qty, latest_short_expiry, max_short_strike) for one
    underlying+type across all contracts. Long legs only count as coverage
    when they expire ON/AFTER the latest short expiry (they must outlive it)."""
    from agent import occ

    shorts, longs = [], []
    for sym, sh in positions.items():
        if not occ.is_option(sym) or abs(sh) <= EPS_SHARES:
            continue
        p = occ.parse(sym)
        if p["underlying"] != underlying or p["type"] != type_:
            continue
        (shorts if sh < 0 else longs).append((p, abs(sh)))
    short_qty = sum(q for _, q in shorts)
    latest_short_exp = max((p["expiry"] for p, _ in shorts), default=None)
    max_short_strike = max((p["strike"] for p, _ in shorts), default=0.0)
    long_qty = sum(q for p, q in longs
                   if latest_short_exp is None or p["expiry"] >= latest_short_exp)
    return short_qty, long_qty, latest_short_exp, max_short_strike


def _csp_reserved(positions: dict[str, float]) -> float:
    """Cash reserved to secure uncovered short puts, per underlying:
    max(0, short_puts − covering long puts) × max short strike × 100.
    Conservative on purpose — the reservation may overstate, never understate."""
    from agent import occ

    reserved = 0.0
    for und in {occ.parse(s)["underlying"] for s, sh in positions.items()
                if occ.is_option(s) and sh < -EPS_SHARES}:
        short_q, long_q, _, max_k = _option_legs(positions, und, "P")
        uncovered = max(0.0, short_q - long_q)
        reserved += uncovered * max_k * MULTIPLIER
    return reserved


def free_cash(store, account: str = ACCOUNT) -> float:
    """Spendable cash = ledger cash − CSP reservations."""
    return round(cash(store, account) - _csp_reserved(_positions_map(store, account)), 2)


def _check_option_sell(positions: dict[str, float], symbol: str,
                       sell_qty: float, cash_now: float) -> str | None:
    """Validate ANY option SELL against the full POST-FILL book (None = ok).

    One simulation covers every path — opening/extending a short, legging out
    of a spread's LONG leg (which strands the short: the P2-options verifier's
    bypass), and a cross-through sell that closes a long and opens a short in
    one order. Calls: every short call on the underlying must stay covered by
    shares/100 + surviving long calls. Puts: the post-fill CSP reservation
    must still fit inside cash."""
    from agent import occ

    p = occ.parse(symbol)
    und = p["underlying"]
    sim = dict(positions)
    sim[symbol] = sim.get(symbol, 0.0) - sell_qty  # the whole fill, signed
    if p["type"] == "C":
        short_q, long_q, _, _ = _option_legs(sim, und, "C")
        if short_q <= 0:
            return None
        shares_cover = max(0.0, sim.get(und, 0.0)) / MULTIPLIER
        if shares_cover + long_q + 1e-9 < short_q:
            return (f"uncovered short call forbidden: {short_q:g} short calls on {und} "
                    f"vs coverage {shares_cover:g} (shares/100) + {long_q:g} long calls"
                    " — close the short leg first")
        return None
    reserved_after = _csp_reserved(sim)
    if reserved_after > cash_now + 1e-6:
        return (f"cash-secured put requires ${reserved_after:,.2f} reserved "
                f"but cash is ${cash_now:,.2f} — close the short leg first")
    return None


def _check_equity_sell_keeps_calls_covered(positions: dict[str, float],
                                           symbol: str, sell_qty: float) -> str | None:
    """Selling shares must not strand short calls uncovered."""
    sim = dict(positions)
    sim[symbol] = sim.get(symbol, 0.0) - sell_qty
    short_q, long_q, _, _ = _option_legs(sim, symbol, "C")
    if short_q <= 0:
        return None
    shares_cover = max(0.0, sim.get(symbol, 0.0)) / MULTIPLIER
    if shares_cover + long_q + 1e-9 < short_q:
        return (f"sell would strand {short_q:g} short {symbol} calls uncovered "
                f"— close/roll the calls first")
    return None


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
    from agent import occ

    store = store or _store()
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    is_opt = occ.is_option(symbol)
    if side not in ("BUY", "SELL"):
        return {"ok": False, "error": f"bad side {side!r}"}
    shares = round(float(shares), 6)
    price = float(price)
    if shares <= EPS_SHARES or price <= 0:
        return {"ok": False, "error": "shares and price must be positive"}
    if is_opt and abs(shares - round(shares)) > EPS_SHARES:
        return {"ok": False, "error": "options trade in whole contracts"}
    if is_opt and occ.parse(symbol)["expiry"] < _utcnow().date():
        return {"ok": False, "error": f"{symbol} is expired — run `settle`"}

    if fill_quote is not None:
        # a live fill's snapshot must be complete — no falling back to the
        # loose close band with a half-formed quote
        if not (fill_quote.get("bid") and fill_quote.get("ask")):
            return {"ok": False, "error": "fill rejected: fill_quote missing bid/ask"}
        bid, ask = float(fill_quote["bid"]), float(fill_quote["ask"])
        # Options: a wider fractional band OR an absolute-cents floor —
        # whichever puts the price farther from the mid. OPRA's minimum tick
        # is $0.01/$0.05, so 0.5% of a $0.35 contract is <1 cent and rejects
        # legitimate one-tick-through fills.
        if is_opt:
            slack = max(bid * LIVE_BAND_OPT, LIVE_BAND_OPT_CENTS)
            lo, hi = bid - slack, ask + slack
            band_repr = f"max({LIVE_BAND_OPT:.0%}, ${LIVE_BAND_OPT_CENTS:.2f})"
        else:
            lo, hi = bid * (1 - LIVE_BAND), ask * (1 + LIVE_BAND)
            band_repr = f"{LIVE_BAND:.1%}"
        if not (lo <= price <= hi):
            return {"ok": False, "error": "fill rejected: price off the live quote",
                    "price": price, "bid": bid, "ask": ask, "band": band_repr}
    elif is_opt:
        return {"ok": False, "error": "option fills require a live fill_quote"}
    else:
        if latest_close is None:
            latest_close = _latest_close(symbol)
        if latest_close and latest_close > 0:
            dev = abs(price - latest_close) / latest_close
            if dev > FILL_BAND:
                return {"ok": False, "error": "fill rejected by sanity guard",
                        "price": price, "latest_close": round(latest_close, 4),
                        "deviation": round(dev, 4), "band": FILL_BAND}

    positions = _positions_map(store, account)
    cash_now = cash(store, account)
    mult = MULTIPLIER if is_opt else 1

    if side == "SELL":
        held = positions.get(symbol, 0.0)
        if is_opt:
            # every option sell is validated against the full post-fill book —
            # opening shorts AND selling a covering long leg both go through here
            err = _check_option_sell(positions, symbol, shares, cash_now)
            if err:
                return {"ok": False, "error": err}
        else:
            if held <= EPS_SHARES:
                return {"ok": False, "error": f"no open position in {symbol} to sell"}
            if shares > held:
                shares = held  # long-only equities: cap at holdings
            err = _check_equity_sell_keeps_calls_covered(positions, symbol, shares)
            if err:
                return {"ok": False, "error": err}

    gross = round(shares * price * mult, 2)
    if side == "BUY":
        spendable = round(cash_now - _csp_reserved(positions), 2)
        if gross > spendable + 1e-6:
            return {"ok": False, "error": "insufficient free cash for buy",
                    "needed": gross, "cash": cash_now, "free_cash": spendable}

    store.insert("desk_trades", {
        "account": account, "run_id": run_id, "symbol": symbol, "side": side,
        "shares": shares, "price": round(price, 4), "dollars": gross,
        "rationale": rationale, "fill_quote": fill_quote,
        "ts": _utcnow()}, returning=False)
    rebuild_positions(store, account)
    return {"ok": True, "symbol": symbol, "side": side, "shares": shares,
            "price": round(price, 4), "dollars": gross, "multiplier": mult,
            "fill_quote": fill_quote, "cash_after": cash(store, account)}


def _live_quote(symbol: str) -> dict:
    """One live quote via REST — SIP for equities, OPRA for option contracts,
    Alpaca crypto endpoint for pairs with a slash (BTC/USD, ETH/USD, …).
    Raises on missing keys/quote — the caller turns that into a rejection."""
    from agent import broker, occ

    b = broker.Broker()
    if occ.is_option(symbol):
        q = b.option_quotes([symbol]).get(symbol)
    else:
        # broker.quotes() auto-routes crypto pairs to the crypto endpoint;
        # equities take the SIP/IEX path. Single-lookup either way.
        q = b.quotes([symbol]).get(symbol)
    if not q or not q.get("bid") or not q.get("ask"):
        raise ValueError(f"no live quote for {symbol}")
    return q


def _quote_age_sec(t) -> float | None:
    """Seconds since a quote's timestamp (any of: ISO string, datetime,
    ns/µs/s epoch). None when unparseable — caller handles missing gracefully.
    """
    if t is None:
        return None
    from datetime import datetime, timezone
    dt = None
    if hasattr(t, "isoformat"):
        dt = t
    elif isinstance(t, str):
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        except ValueError:
            return None
    elif isinstance(t, (int, float)):
        # heuristic: sec vs ms vs µs vs ns by magnitude
        n = float(t)
        if n > 1e17:
            n /= 1e9
        elif n > 1e14:
            n /= 1e6
        elif n > 1e11:
            n /= 1e3
        dt = datetime.fromtimestamp(n, tz=timezone.utc)
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds()


def live_fill(store=None, *, symbol: str, side: str, shares: float | None = None,
              notional: float | None = None, rationale: str | None = None,
              run_id: str | None = None, slippage_bp: float = SLIPPAGE_BP,
              account: str = ACCOUNT) -> dict:
    """Book a fill AT THE LIVE QUOTE — the only execution path the agent uses.

    Reads the real-time SIP quote, prices the correct side (BUY at ask,
    SELL at bid, ± slippage), stamps the quote snapshot on the fill, and
    refuses to book when the market is closed, the quote is degenerate, or
    the quote is stale. Extended hours are allowed for equities with tighter
    spread guards; options are refused outside RTH (OPRA book is genuinely
    bad pre-open/post-close). **Crypto** pairs (BTC/USD, ETH/USD, …) trade
    24/7 and skip the RTH/close-soon gates; their spread cap is 3% and no
    options structure is possible.  Exactly one of ``shares`` / ``notional``
    sizes the order (fractional ok — required for high-priced crypto).
    """
    from agent import broker, occ

    store = store or _store()
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    is_opt = occ.is_option(symbol)
    is_cx = broker.is_crypto(symbol)
    if (shares is None) == (notional is None):
        return {"ok": False, "error": "pass exactly one of shares or notional"}

    try:
        b = broker.Broker()
        sess = b.session(symbol) if hasattr(b, "session") else (
            "regular" if b.is_market_open() else "closed")
        if sess == "closed":
            return {"ok": False, "error":
                    "market closed — live fills only trade in RTH or extended hours"}
        if sess == "extended" and is_opt:
            return {"ok": False, "error":
                    "options fills are RTH-only — OPRA book is too thin outside "
                    "regular hours"}
        # Late-day discipline: don't open a new position we can't sell today.
        # Crypto is 24/7 so the "can't sell today" premise doesn't apply.
        if sess == "regular" and side == "BUY" and hasattr(b, "is_close_soon") \
                and b.is_close_soon(minutes=15):
            return {"ok": False, "error":
                    "close in <15m — refusing to open a position we can't exit today"}
        q = _live_quote(symbol)
    except Exception as exc:  # noqa: BLE001 — clean rejection, not a stacktrace
        return {"ok": False, "error": f"live quote unavailable: {exc}"}

    bid, ask = float(q["bid"]), float(q["ask"])
    if is_cx:
        max_spread = MAX_SPREAD_CRYPTO
    elif is_opt:
        max_spread = MAX_SPREAD_OPT
    elif sess == "extended":
        max_spread = MAX_SPREAD_EQ_EXT
    else:
        max_spread = MAX_SPREAD_EQ
    if bid <= 0 or ask <= 0 or ask < bid or (ask - bid) / ask > max_spread:
        return {"ok": False, "error":
                f"degenerate quote (crossed or >{max_spread:.0%} spread)",
                "bid": bid, "ask": ask, "session": sess}

    # Quote-freshness — the missing defense against a feed stuck open on an
    # old print. Only rejects when we can actually measure it; a quote without
    # a timestamp is unusual (SDK always sets one) but not an outright reject.
    age = _quote_age_sec(q.get("t"))
    if is_cx:
        max_age = MAX_QUOTE_AGE_SEC_CRYPTO
    elif is_opt:
        max_age = MAX_QUOTE_AGE_SEC_OPT
    else:
        max_age = MAX_QUOTE_AGE_SEC_EQ
    if age is not None and age > max_age:
        return {"ok": False, "error":
                f"stale quote ({age:.0f}s old, cap {max_age:.0f}s) — feed hiccup?",
                "bid": bid, "ask": ask, "session": sess}

    slip = slippage_bp / 10_000.0
    price = round(ask * (1 + slip), 4) if side == "BUY" else round(bid * (1 - slip), 4)
    if shares is None:
        per_unit = price * (MULTIPLIER if is_opt else 1)
        shares = (float(notional) // per_unit) if is_opt else round(float(notional) / price, 6)
        if shares <= 0:
            return {"ok": False, "error": "notional too small for one contract",
                    "contract_cost": per_unit}

    if is_opt:
        src = "alpaca_opra_rest"
    elif is_cx:
        src = "alpaca_crypto_rest"
    else:
        src = "alpaca_sip_rest"
    snapshot = {"bid": bid, "ask": ask, "mid": q.get("mid"), "t": q.get("t"),
                "src": src, "slippage_bp": slippage_bp, "session": sess}
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
    """Live mids for marking (SIP for equities, OPRA for contracts).
    Empty dict on any failure → close/basis fallback."""
    if not symbols:
        return {}
    from agent import occ

    opts = [s for s in symbols if occ.is_option(s)]
    eqs = [s for s in symbols if not occ.is_option(s)]
    out: dict[str, float] = {}
    try:
        from agent import broker

        b = broker.Broker()
        if eqs:
            out.update({s: q["mid"] for s, q in b.quotes(eqs).items() if q.get("mid")})
        if opts:
            out.update({s: q["mid"] for s, q in b.option_quotes(opts).items()
                        if q.get("mid")})
    except Exception:
        pass
    return out


def _book_settlement(store, *, symbol: str, side: str, qty: float, price: float,
                     rationale: str, underlying_px: float,
                     account: str = ACCOUNT) -> None:
    """Directly book a settlement row (bypasses the live-quote guards — expiry
    settles at intrinsic value, which may legitimately be 0)."""
    store.insert("desk_trades", {
        "account": account, "run_id": "settlement", "symbol": symbol,
        "side": side, "shares": qty, "price": round(price, 4),
        "dollars": round(qty * price * _mult(symbol), 2),
        "rationale": rationale,
        "fill_quote": {"src": "expiry_settlement",
                       "underlying_px": round(underlying_px, 4)},
        "ts": _utcnow()}, returning=False)


def settle(store=None, *, account: str = ACCOUNT) -> dict:
    """Settle expired option positions honestly — run at the top of every cycle.

    For each open contract past expiry, using the underlying's price:
    - long ITM  → cash-settled exercise: SELL at intrinsic value.
    - long OTM  → expires worthless: SELL at 0.
    - short call ITM (covered by shares) → assignment: shares called away —
      book an equity SELL of 100/contract at the strike, close the call at 0.
    - short put ITM → assignment: book an equity BUY of 100/contract at the
      strike (the CSP reservation funds it), close the put at 0.
    - short OTM → expires worthless: BUY back at 0 (premium kept).
    Every action is an append-only ledger row with src=expiry_settlement.
    """
    from agent import occ

    store = store or _store()
    today = _utcnow().date()
    positions = _positions_map(store, account)
    actions: list[dict] = []
    for sym, sh in sorted(positions.items()):
        if not occ.is_option(sym) or abs(sh) <= EPS_SHARES:
            continue
        p = occ.parse(sym)
        if p["expiry"] >= today:
            continue
        und = p["underlying"]
        und_px = (_live_mids([und]).get(und) or _latest_close(und) or 0.0)
        if und_px <= 0:
            actions.append({"symbol": sym, "action": "SKIPPED",
                            "error": f"no price for underlying {und}"})
            continue
        iv = occ.intrinsic(sym, und_px)
        qty = abs(sh)
        desc = occ.describe(sym)
        if sh > 0:  # long leg
            _book_settlement(store, symbol=sym, side="SELL", qty=qty, price=iv,
                             rationale=f"expiry settlement: {desc} "
                                       f"{'exercised (cash-settled intrinsic)' if iv > 0 else 'expired worthless'}",
                             underlying_px=und_px, account=account)
            actions.append({"symbol": sym, "action": "long settled",
                            "intrinsic": round(iv, 4), "qty": qty})
        else:  # short leg
            # Share assignment only when shares/cash actually back the short;
            # a SPREAD-covered short cash-settles at intrinsic (its long leg
            # settles on its own row), never creating a phantom share position.
            shares_backed = (p["type"] == "C"
                             and max(0.0, positions.get(und, 0.0)) >= qty * MULTIPLIER)
            cash_backed = False
            if p["type"] == "P":
                short_q, long_q, _, _ = _option_legs(positions, und, "P")
                cash_backed = long_q + 1e-9 < short_q  # not fully spread-covered
            if iv > 0 and p["type"] == "C" and shares_backed:
                _book_settlement(store, symbol=und, side="SELL",
                                 qty=qty * MULTIPLIER, price=p["strike"],
                                 rationale=f"assignment: shares called away at "
                                           f"${p['strike']:g} ({desc})",
                                 underlying_px=und_px, account=account)
                close_px = 0.0
            elif iv > 0 and p["type"] == "P" and cash_backed:
                _book_settlement(store, symbol=und, side="BUY",
                                 qty=qty * MULTIPLIER, price=p["strike"],
                                 rationale=f"assignment: shares put to us at "
                                           f"${p['strike']:g} ({desc})",
                                 underlying_px=und_px, account=account)
                close_px = 0.0
            else:
                close_px = iv  # cash-settle (spread-covered, or OTM → 0)
            _book_settlement(store, symbol=sym, side="BUY", qty=qty, price=close_px,
                             rationale=f"expiry settlement: {desc} short leg "
                                       f"{'assigned' if (iv > 0 and close_px == 0.0) else ('cash-settled' if iv > 0 else 'expired worthless')}",
                             underlying_px=und_px, account=account)
            actions.append({"symbol": sym, "action": "short settled",
                            "itm": iv > 0, "qty": qty})
    if actions:
        rebuild_positions(store, account)
    return {"settled": actions, "as_of": str(today)}


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
        pos_value += p["shares"] * px * _mult(sym)  # signed; options ×100
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
        m = _mult(p["symbol"])
        mv = round(p["shares"] * mark_px * m, 2)  # signed; short legs negative
        pos_value += mv
        pos_list.append({
            "symbol": p["symbol"], "shares": p["shares"],
            "avg_price": round(p["avg_price"], 4), "last_price": round(mark_px, 4),
            "market_value": mv,
            "cost_basis": round(p["shares"] * p["avg_price"] * m, 2),
            "unrealized_pnl": round(p["shares"] * (mark_px - p["avg_price"]) * m, 2),
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
    sub.add_parser("settle", help="settle expired option positions (run each cycle)")
    oc = sub.add_parser("outcomes",
                        help="how past picks aged vs what was said (reflection input)")
    oc.add_argument("--days", type=int, default=30)
    oc.add_argument("--run-id", default=None)
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
    elif args.cmd == "settle":
        print(json.dumps(settle(store), indent=2))
    elif args.cmd == "outcomes":
        print(json.dumps(outcomes(store, days=args.days, run_id=args.run_id),
                         indent=2))
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

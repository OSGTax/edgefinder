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


def _fq(t: dict) -> dict:
    """A trade row's fill_quote as a dict (transports may hand back JSON text)."""
    fq = t.get("fill_quote")
    if isinstance(fq, str):
        try:
            fq = json.loads(fq)
        except ValueError:
            fq = None
    return fq if isinstance(fq, dict) else {}


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

    Equities are long-only fractional lots. record_trade caps sells at
    holdings, but a double-writer race (streamer hard stop + trading cycle,
    or two streamers during a deploy overlap) can still land an over-sell
    row — the replay therefore CLAMPS equity lots at zero and never
    fabricates a short. Option contracts are SIGNED: a SELL beyond the long
    lot opens a short leg (negative shares) — record_trade only permits that
    when the short is covered, so options keep honest signed arithmetic.
    ``cost`` tracks |basis|; crossing through zero re-opens the lot at the
    fill price.

    SPLIT ADJUSTMENTS (fill_quote.src == "split_adjustment", booked by
    ``settle``) change the unit count, never the basis: shares move by the
    split delta while ``cost`` and ``opened_at`` stay put, so avg_price
    rebases by exactly the split ratio and no P&L is fabricated.
    """
    from agent import occ

    book: dict[str, dict] = {}
    for t in trades:
        sym = t["symbol"]
        b = book.setdefault(sym, {"shares": 0.0, "cost": 0.0, "opened_at": t.get("ts")})
        qty = float(t["shares"])
        signed = qty if t["side"] == "BUY" else -qty
        cur = b["shares"]
        if _fq(t).get("src") == "split_adjustment":
            b["shares"] = cur + signed
            if abs(b["shares"]) <= EPS_SHARES:
                b["shares"], b["cost"] = 0.0, 0.0
            continue
        if abs(cur) <= EPS_SHARES:
            if signed < 0 and not occ.is_option(sym):
                # An equity SELL on a flat book (a duplicate exit from a
                # writer race) must degrade to FLAT — opening a short lot
                # here would double-credit cash AND display a position the
                # desk would then try to "cover".
                b["shares"], b["cost"] = 0.0, 0.0
                continue
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

    Split-adjustment rows are unit-count changes, not trades: they move
    shares with cost untouched and book ZERO realized P&L (dividend rows
    carry 0 shares, so they naturally book zero here too — dividend cash
    lands in the book's equity, not in per-symbol realized P&L).
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
        if _fq(t).get("src") == "split_adjustment":
            b["shares"] = cur + signed  # cost unchanged — no P&L from a split
            continue
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


def _adjust_closes_for_dividends(closes: list[tuple[str, float]],
                                 divs: list[tuple[str, float]]
                                 ) -> list[tuple[str, float]]:
    """CRSP-style total-return back-adjustment, pandas-free (the ledger has
    no pandas dependency — this mirrors the semantics of
    ``edgefinder.engine.data.adjust_for_dividends`` on plain (iso-date,
    close) tuples).

    For each ex-date, every close strictly BEFORE it is scaled by
    ``1 − dividend / close_the_day_before_ex``; the LAST close is unchanged,
    so a buy-and-hold of the adjusted series equals the dividend-reinvested
    return. Ex-dates outside the series and degenerate rows (dividend >=
    prev close, prev close <= 0) are skipped. With no dividend rows at all
    the series passes through untouched — total return degrades to price
    return, never a crash."""
    if not closes or not divs:
        return closes
    import bisect

    dates = [d for d, _ in closes]
    factors = [1.0] * len(closes)
    for ex, amount in divs:
        idx = bisect.bisect_left(dates, str(ex)[:10])
        if idx <= 0 or idx >= len(dates):
            continue
        prev_close = closes[idx - 1][1]
        if prev_close <= 0 or amount >= prev_close:
            continue
        f = 1.0 - amount / prev_close
        for i in range(idx):
            factors[i] *= f
    return [(d, c * f) for (d, c), f in zip(closes, factors)]


def _spy_closes(store, *, since: str) -> list[tuple[str, float]]:
    """SPY daily closes (ascending) from ``daily_bars``, back-adjusted to
    TOTAL RETURN with the SPY rows from ``dividends``, with a lookback
    buffer so a window starting on a weekend/holiday still finds a baseline
    close.

    Total return, because the book is total return too: ``settle`` credits
    dividend cash on held names at the ex-date, so a price-only SPY would
    hand the book the benchmark's own yield as fake alpha. Missing SPY
    dividend rows degrade gracefully to price return (see
    ``_adjust_closes_for_dividends``). ``index_daily`` is NOT used: it
    froze at the 2026-06-22 cutover."""
    from datetime import date as _date, timedelta as _td

    start = _date.fromisoformat(since[:10]) - _td(days=10)
    rows = store.select("daily_bars", columns="date,close",
                        filters={"symbol": "SPY", "date": ("gte", start)},
                        order=[("date", "asc")])
    closes = [(str(r["date"])[:10], float(r["close"]))
              for r in rows if r.get("close")]
    try:
        div_rows = store.select("dividends", columns="ex_date,cash_amount",
                                filters={"symbol": "SPY",
                                         "ex_date": ("gte", start)})
    except Exception:  # noqa: BLE001 — no dividend table/rows → price return
        div_rows = []
    divs = sorted((str(r["ex_date"])[:10], float(r["cash_amount"]))
                  for r in div_rows if r.get("cash_amount"))
    return _adjust_closes_for_dividends(closes, divs)


def _et_date(ts) -> str | None:
    """The ET calendar date of a naive-UTC desk timestamp.

    Windows are trading-day windows: a 19:30 ET decision is already next-day
    in UTC, and dating it by the UTC calendar would baseline SPY off the NEXT
    session's close — a full day of index movement misattributed to alpha."""
    from zoneinfo import ZoneInfo

    if ts is None:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            return str(ts)[:10]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return str(ts.astimezone(ZoneInfo("America/New_York")).date())


def _spy_window_pct(spy: list[tuple[str, float]], start_date: str,
                    end_date: str | None = None) -> float | None:
    """SPY price change over a trading window, honestly bounded.

    Baseline: the last close STRICTLY BEFORE ``start_date`` — the final print
    before the window's day opened. (A close ON the start date is 16:00 ET,
    AFTER an intraday entry — and for a same-day window it would be the
    endpoint itself, yielding a confident fake 0.00.) Endpoint: the last
    close on/before ``end_date`` when given (a closed pick's exit day), else
    the latest stored close. Returns None when the window has no baseline or
    no completed span (baseline row == endpoint row) — None means "too young
    to benchmark", never zero."""
    base = base_d = None
    for d, c in spy:
        if d < start_date[:10]:
            base, base_d = c, d
        else:
            break
    if not base:
        return None
    end, end_d = spy[-1][1], spy[-1][0]
    if end_date is not None:
        bounded = [(d, c) for d, c in spy if d <= end_date[:10]]
        if not bounded:
            return None
        end_d, end = bounded[-1]
    if end_d == base_d:
        return None
    return round((end - base) / base * 100, 2)


def _booked_split_events(trades: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """{symbol: [(execution_date, to/from factor), ...]} from the ledger's own
    ``split_adjustment`` rows (``fill_quote`` carries ratio + execution_date).
    Only what the book actually folded in can be netted out of its prices —
    the raw ticker_splits table may hold rows settle never booked."""
    out: dict[str, list[tuple[str, float]]] = {}
    for t in trades:
        fq = _fq(t)
        if fq.get("src") != "split_adjustment":
            continue
        try:
            to_s, frm_s = str(fq.get("ratio") or "").split(":")
            factor = float(to_s) / float(frm_s)
        except (ValueError, ZeroDivisionError):
            continue
        d = str(fq.get("execution_date") or "")[:10]
        if d and factor > 0:
            out.setdefault(t["symbol"], []).append((d, factor))
    for evs in out.values():
        evs.sort()
    return out


def _split_factor_since(events: list[tuple[str, float]] | None, d: str) -> float:
    """Cumulative to/from factor of booked splits executing STRICTLY AFTER
    ``d`` — how many current-basis units one unit dated ``d`` became. A fill
    ON the execution date is already post-split (the tape rebases at the
    open), so it takes no factor — mirrors ``_shares_held_asof``'s
    strictly-before entitlement rule."""
    f = 1.0
    for ed, x in (events or []):
        if ed > d[:10]:
            f *= x
    return f


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
    should grade. Settlement rows (run_id='settlement') and hard-stop exits
    (run_id='hardstop:<watch id>') are bucketed separately; trades with no
    run_id are counted, never silently dropped.
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

    inception = _et_date(trades[0]["ts"]) if trades else None
    window_starts = [w for w in (_et_date(d.get("ts")) for d in decisions) if w]
    if inception:
        window_starts.append(inception)
    spy = _spy_closes(store, since=min(window_starts)) if window_starts else []

    fills_by_run: dict[tuple[str | None, str], list[dict]] = {}
    for t in trades:
        fills_by_run.setdefault((t.get("run_id"), t["symbol"]), []).append(t)
    split_events = _booked_split_events(trades)

    runs = []
    for d in decisions:
        rid = d["run_id"]
        run_date = _et_date(d.get("ts"))
        run_spy_pct = _spy_window_pct(spy, run_date) if run_date else None
        picks_out = []
        for p in (d.get("picks") or []):
            sym = str(p.get("symbol") or "").upper()
            if sym == "BOOK":
                continue  # whole-book stance note — no fills, nothing to grade per-name
            raw_fills = fills_by_run.get((rid, sym), [])
            fills = [{"side": f["side"], "shares": f["shares"], "price": f["price"]}
                     for f in raw_fills]
            # Grading math runs on the CURRENT share basis: a split booked
            # between a fill and the mark rebases the tape, and grading a
            # pre-split entry against a post-split mark reads as a fake −50%
            # a flat position never earned. Each fill is rescaled by the
            # cumulative booked-split factor since its date (shares × f,
            # price / f) — ``fills`` above stays raw, as booked (receipts).
            evs = split_events.get(sym)
            adj = [{"side": f["side"],
                    "shares": f["shares"] * _split_factor_since(
                        evs, _et_date(f.get("ts")) or ""),
                    "price": f["price"] / _split_factor_since(
                        evs, _et_date(f.get("ts")) or "")}
                   for f in raw_fills] if evs else fills
            buys = [f for f in adj if f["side"] == "BUY"]
            sells = [f for f in adj if f["side"] == "SELL"]
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
            # A round trip closed within this run gets an EXACT realized
            # return and an exit-bounded SPY window — the grading target the
            # weekly reflection prioritizes, previously None.
            closed_pct = exit_spy_pct = None
            bought = sum(f["shares"] for f in buys)
            sold = sum(f["shares"] for f in sells)
            if buys and sells and abs(bought - sold) <= EPS_SHARES:
                sell_avg = sum(f["shares"] * f["price"] for f in sells) / sold
                closed_pct = round((sell_avg - entry_avg) / entry_avg * 100, 2)
                exit_date = _et_date(raw_fills[-1].get("ts"))
                if run_date and exit_date:
                    exit_spy_pct = _spy_window_pct(spy, run_date, exit_date)
            is_opt = occ.is_option(sym)
            live_pct = closed_pct if closed_pct is not None else since_pct
            spy_pct = exit_spy_pct if closed_pct is not None else run_spy_pct
            # Options: premium %-moves carry leverage and theta — subtracting
            # an index move from them is not alpha. Grade options on realized
            # dollars + thesis instead.
            alpha = (round(live_pct - spy_pct, 2)
                     if (live_pct is not None and spy_pct is not None
                         and not is_opt) else None)
            picks_out.append({
                "symbol": sym, "action": p.get("action"),
                "is_option": is_opt,
                "why_now": p.get("why_now"), "rationale": p.get("rationale"),
                "prediction": p.get("prediction"),
                "horizon_days": p.get("horizon_days"), "kill": p.get("kill"),
                "fills": fills, "entry_avg_px": round(entry_avg, 4) if entry_avg else None,
                "realized_pnl": round(by_run_symbol.get((rid, sym), 0.0), 2),
                "open_now": open_now, "since_this_run_pct": since_pct,
                "closed_return_pct": closed_pct,
                "spy_same_window_pct": spy_pct, "alpha_pct": alpha})
        # How many completed SPY sessions the window spans — alpha on fewer
        # than 2 is inside the benchmark's own noise; the skills must not
        # grade it as skill.
        sessions = 0
        if run_date and spy:
            base_d = None
            for sd, _ in spy:
                if sd < run_date:
                    base_d = sd
                else:
                    break
            if base_d is not None:
                sessions = sum(1 for sd, _ in spy if sd > base_d)
        runs.append({"run_id": rid, "ts": str(d.get("ts") or ""),
                     "regime": d.get("regime"), "summary": d.get("summary"),
                     "picks": picks_out, "rejected": d.get("rejected") or [],
                     "spy_same_window_pct": run_spy_pct,
                     "spy_window_sessions": sessions,
                     "run_realized_pnl": round(sum(
                         v for (r, _), v in by_run_symbol.items() if r == rid), 2)})

    settlement_pnl = round(sum(v for (r, _), v in by_run_symbol.items()
                               if r == "settlement"), 2)
    # Hard-stop exits book under run_id "hardstop:<watch id>" — no decision
    # row carries that id, so without its own bucket every stop loss would
    # vanish from per-run grading exactly when it matters most.
    hardstop_pnl = round(sum(v for (r, _), v in by_run_symbol.items()
                             if str(r or "").startswith("hardstop:")), 2)
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
                          "run's own average fill and is exact per pick; "
                          "entry_avg_px and round-trip matching are expressed "
                          "on the CURRENT share basis (booked split adjustments "
                          "between a fill and the mark rescale that run's own "
                          "fills, so a flat position grades ~0 after a split, "
                          "not a fake -50). spy_same_window_pct / alpha_pct "
                          "benchmark each window against SPY closes from "
                          "daily_bars back-adjusted to TOTAL RETURN with SPY's "
                          "dividend rows — like-for-like with the book, which "
                          "credits dividend cash at the ex-date; missing "
                          "dividend rows degrade to price return. SPY baseline "
                          "= the last close STRICTLY BEFORE the window's ET "
                          "start date; endpoint = exit-day close for round "
                          "trips closed in-run (closed_return_pct), else the "
                          "latest stored close. None means "
                          "too-young-to-benchmark, never zero. Alpha under "
                          "spy_window_sessions < 2 is inside benchmark noise — "
                          "do not grade it as skill. Options carry alpha_pct = "
                          "null by design: premium %-moves embed leverage/"
                          "theta; grade them on realized dollars and thesis. "
                          "Hard-stop exits (run_id 'hardstop:<id>') are "
                          "bucketed under 'hardstop', like settlement. A long "
                          "book's raw P&L is mostly market beta, so grade "
                          "alpha, not dollars.",
            "book": book, "runs": runs, "symbols": symbols_out,
            "settlement": {"realized_pnl": settlement_pnl},
            "hardstop": {"realized_pnl": hardstop_pnl},
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
    bad pre-open/post-close). NOTE: a post-close (after-hours) extended BUY
    cannot be exited until the NEXT session's tape — an overnight hold by
    construction, matching pre-market semantics — so size it accordingly.
    **Crypto** pairs (BTC/USD, ETH/USD, …) trade
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


def _bar_close_on(store, symbol: str, day) -> float | None:
    """The stored daily close for ``symbol`` ON ``day`` exactly (None if that
    bar is missing). Transport-safe: a plain equality select on daily_bars."""
    try:
        rows = store.select("daily_bars", columns="close",
                            filters={"symbol": symbol, "date": day}, limit=1)
    except Exception:
        return None
    if rows and rows[0].get("close"):
        px = float(rows[0]["close"])
        return px if px > 0 else None
    return None


def _book_settlement(store, *, symbol: str, side: str, qty: float, price: float,
                     rationale: str, underlying_px: float, basis: str,
                     account: str = ACCOUNT) -> None:
    """Directly book a settlement row (bypasses the live-quote guards — expiry
    settles at intrinsic value, which may legitimately be 0). ``basis`` says
    which price settled it: 'expiry_close' (the honest default) or
    'runtime_price' (fallback when the expiry-day bar is missing)."""
    store.insert("desk_trades", {
        "account": account, "run_id": "settlement", "symbol": symbol,
        "side": side, "shares": qty, "price": round(price, 4),
        "dollars": round(qty * price * _mult(symbol), 2),
        "rationale": rationale,
        "fill_quote": {"src": "expiry_settlement",
                       "underlying_px": round(underlying_px, 4),
                       "settle_basis": basis},
        "ts": _utcnow()}, returning=False)


def settle(store=None, *, account: str = ACCOUNT) -> dict:
    """Settle expired options AND fold in equity corporate actions — run at
    the top of every cycle.

    OPTIONS: for each open contract past expiry, priced off the underlying's
    EXPIRY-DAY daily close from ``daily_bars`` (settle_basis=expiry_close —
    a Friday expiry must not settle at Monday's price; the old run-time
    pricing gave a free weekend look-back). KNOWN LIMIT: the expiry_close
    basis requires the underlying in the daily_bars hot set — only when
    that bar is missing does it fall back to the live mid / latest close
    (settle_basis=runtime_price), and the basis used is stamped on every
    settlement row:
    - long ITM  → cash-settled exercise: SELL at intrinsic value.
    - long OTM  → expires worthless: SELL at 0.
    - short call ITM (covered by shares) → assignment: shares called away —
      book an equity SELL of 100/contract at the strike, close the call at 0.
    - short put ITM → assignment: book an equity BUY of 100/contract at the
      strike (the CSP reservation funds it), close the put at 0.
    - short OTM → expires worthless: BUY back at 0 (premium kept).
    Every action is an append-only ledger row with src=expiry_settlement.

    EQUITIES: splits and dividends on open positions are booked as 0-price
    adjustment rows — see ``_settle_equity_corp_actions``.
    """
    from agent import occ

    store = store or _store()
    today = _utcnow().date()
    positions = _positions_map(store, account)
    actions: list[dict] = []
    expiry_px: dict[tuple[str, object], tuple[float, str]] = {}
    for sym, sh in sorted(positions.items()):
        if not occ.is_option(sym) or abs(sh) <= EPS_SHARES:
            continue
        p = occ.parse(sym)
        if p["expiry"] >= today:
            continue
        und = p["underlying"]
        key = (und, p["expiry"])
        if key not in expiry_px:
            close = _bar_close_on(store, und, p["expiry"])
            if close:
                expiry_px[key] = (close, "expiry_close")
            else:
                expiry_px[key] = (
                    (_live_mids([und]).get(und) or _latest_close(und) or 0.0),
                    "runtime_price")
        und_px, basis = expiry_px[key]
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
                             underlying_px=und_px, basis=basis, account=account)
            actions.append({"symbol": sym, "action": "long settled",
                            "intrinsic": round(iv, 4), "qty": qty,
                            "basis": basis})
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
                                 underlying_px=und_px, basis=basis, account=account)
                close_px = 0.0
            elif iv > 0 and p["type"] == "P" and cash_backed:
                _book_settlement(store, symbol=und, side="BUY",
                                 qty=qty * MULTIPLIER, price=p["strike"],
                                 rationale=f"assignment: shares put to us at "
                                           f"${p['strike']:g} ({desc})",
                                 underlying_px=und_px, basis=basis, account=account)
                close_px = 0.0
            else:
                close_px = iv  # cash-settle (spread-covered, or OTM → 0)
            _book_settlement(store, symbol=sym, side="BUY", qty=qty, price=close_px,
                             rationale=f"expiry settlement: {desc} short leg "
                                       f"{'assigned' if (iv > 0 and close_px == 0.0) else ('cash-settled' if iv > 0 else 'expired worthless')}",
                             underlying_px=und_px, basis=basis, account=account)
            actions.append({"symbol": sym, "action": "short settled",
                            "itm": iv > 0, "qty": qty, "basis": basis})
    if actions:
        rebuild_positions(store, account)
    corp = _settle_equity_corp_actions(store, account=account)
    if corp.get("splits") or corp.get("dividends"):
        rebuild_positions(store, account)
    return {"settled": actions, "corp_actions": corp, "as_of": str(today)}


# ── equity corporate actions (splits + dividends on the live book) ──


def _adj_effective_date(t: dict) -> str | None:
    """The market-effective date of a corp-action adjustment row (None for a
    normal fill). Adjustment rows are BOOKED whenever settle next runs, so
    replays must date them by the action's own date, not the booking ts."""
    fq = _fq(t)
    if fq.get("src") == "split_adjustment":
        return str(fq.get("execution_date") or "")[:10] or None
    if fq.get("src") == "dividend":
        return str(fq.get("ex_date") or "")[:10] or None
    return None


def _shares_held_asof(trades: list[dict], symbol: str, on_date: str) -> float:
    """Net shares of ``symbol`` held going INTO ``on_date``: the signed sum of
    every fill dated STRICTLY BEFORE it (ET date of the fill ts; adjustment
    rows count at their own effective date). Strictly-before is the honest
    entitlement rule — shares bought ON an ex-date earn no dividend, and a
    split converts the units held at the prior close."""
    held = 0.0
    for t in trades:
        if t["symbol"] != symbol:
            continue
        d = _adj_effective_date(t) or _et_date(t.get("ts"))
        if d is None or d >= on_date[:10]:
            continue
        held += float(t["shares"]) if t["side"] == "BUY" else -float(t["shares"])
    return held


def _book_adjustment(store, *, symbol: str, side: str, qty: float,
                     dollars: float, rationale: str, fill_quote: dict,
                     account: str = ACCOUNT) -> dict:
    """Directly book one corp-action adjustment row (price 0 by convention:
    splits move shares with no cash; dividends move cash with no shares).
    Returns the row dict so the caller can extend its in-memory replay."""
    row = {"account": account, "run_id": "settlement", "symbol": symbol,
           "side": side, "shares": round(qty, 6), "price": 0.0,
           "dollars": round(dollars, 2), "rationale": rationale,
           "fill_quote": fill_quote, "ts": _utcnow()}
    store.insert("desk_trades", row, returning=False)
    return row


def _ledger_shares_now(store, symbol: str, account: str = ACCOUNT) -> float:
    """Net open shares of ``symbol`` recomputed from a FRESH ledger read —
    not the desk_positions projection (which may be mid-rebuild) and not an
    in-memory replay that predates a concurrent writer's rows."""
    rows = store.select("desk_trades",
                        filters={"account": account, "symbol": symbol})
    rows.sort(key=lambda r: (str(r.get("ts")), r.get("id") or 0))
    b = _compute_book(rows).get(symbol)
    return b["shares"] if b else 0.0


def _corp_actions_for_symbol(store, sym: str, sym_trades: list[dict],
                             opened: str, today: str, *,
                             account: str = ACCOUNT
                             ) -> tuple[int, int, list[dict]]:
    """The per-symbol splits+dividends pass shared by the full sweep inside
    ``settle`` and the streamer's targeted ``settle_corp_actions_for``.

    ``sym_trades`` must be the symbol's ledger rows in replay order; it is
    extended in place as adjustment rows book, so later entitlements in the
    same pass see earlier adjustments (a dividend after a split pays on the
    post-split count). Returns (splits_booked, dividends_booked, details).
    """
    from datetime import date as _date

    splits_booked = dividends_booked = 0
    details: list[dict] = []
    done = {(_fq(t).get("src"), _adj_effective_date(t))
            for t in sym_trades if _adj_effective_date(t)}

    # splits — few rows per symbol ever; window client-side. Reverse splits
    # keep the exact fractional share count (KNOWN LIMIT: no cash-in-lieu —
    # a real broker would pay out the fraction; the paper book holds it).
    try:
        split_rows = store.select("ticker_splits", filters={"symbol": sym})
    except Exception:
        split_rows = []
    split_rows = [r for r in split_rows
                  if opened <= str(r.get("execution_date"))[:10] <= today]
    split_rows.sort(key=lambda r: str(r.get("execution_date"))[:10])
    for r in split_rows:
        exec_date = str(r["execution_date"])[:10]
        frm, to = float(r.get("split_from") or 0), float(r.get("split_to") or 0)
        if frm <= 0 or to <= 0 or ("split_adjustment", exec_date) in done:
            continue
        held = _shares_held_asof(sym_trades, sym, exec_date)
        delta = held * (to / frm - 1.0)
        if abs(delta) <= EPS_SHARES:
            continue
        # Book the delta ONLY onto a position that is STILL open RIGHT NOW:
        # a concurrent hard-stop sell (the streamer) may have zeroed the lot
        # after this pass read the book, and a split delta landing on a
        # closed lot fabricates a phantom position at avg_price 0.
        # Prevention is layered — execute_hard_stop runs this pass FIRST
        # and the skill's cycle ordering runs settle before any manual
        # sells — and this fresh ledger re-read closes the remaining window.
        if _ledger_shares_now(store, sym, account) <= EPS_SHARES:
            continue
        ratio = f"{to:g}:{frm:g}"
        row = _book_adjustment(
            store, symbol=sym, side="BUY" if delta > 0 else "SELL",
            qty=abs(delta), dollars=0.0,
            rationale=f"split adjustment: {sym} {ratio} split executed "
                      f"{exec_date} — {held:g} shares become "
                      f"{held + delta:g}, cost basis unchanged",
            fill_quote={"src": "split_adjustment",
                        "execution_date": exec_date, "ratio": ratio},
            account=account)
        sym_trades.append(row)
        done.add(("split_adjustment", exec_date))
        splits_booked += 1
        details.append({"symbol": sym, "action": "split", "ratio": ratio,
                        "execution_date": exec_date,
                        "share_delta": round(delta, 6)})

    # dividends — ex-date window bounded by the lot's open (range filter
    # is transport-safe on the Date column; Phase A added list-of-specs)
    try:
        div_rows = store.select(
            "dividends",
            filters={"symbol": sym,
                     "ex_date": [("gte", _date.fromisoformat(opened)),
                                 ("lte", _date.fromisoformat(today))]})
    except Exception:
        div_rows = []
    div_rows.sort(key=lambda r: str(r.get("ex_date"))[:10])
    for r in div_rows:
        ex = str(r["ex_date"])[:10]
        cash_amount = float(r.get("cash_amount") or 0.0)
        if cash_amount <= 0 or ("dividend", ex) in done:
            continue
        held = _shares_held_asof(sym_trades, sym, ex)
        credit = round(held * cash_amount, 2)
        if held <= EPS_SHARES or credit <= 0:
            continue
        row = _book_adjustment(
            store, symbol=sym, side="SELL", qty=0.0, dollars=credit,
            rationale=f"dividend: {sym} ${cash_amount:g}/share, ex-date "
                      f"{ex} on {held:g} shares held — ${credit:,.2f} "
                      "credited (ex-date basis; real cash pays later)",
            fill_quote={"src": "dividend", "ex_date": ex,
                        "cash_per_share": cash_amount,
                        "shares_asof": round(held, 6)},
            account=account)
        sym_trades.append(row)
        done.add(("dividend", ex))
        dividends_booked += 1
        details.append({"symbol": sym, "action": "dividend", "ex_date": ex,
                        "cash_per_share": cash_amount, "credit": credit})
    return splits_booked, dividends_booked, details


def settle_corp_actions_for(store, symbol: str, *,
                            account: str = ACCOUNT) -> dict:
    """Idempotent SINGLE-SYMBOL equity corp-actions pass (splits+dividends).

    The streamer's hard-stop executor calls this BEFORE sizing a
    full-position exit: a split that executed at today's open rebases the
    tape, and a stop armed below the pre-split price trips instantly — the
    sale must be sized off the POST-split share count, so the split books
    first, then the caller re-reads the position. Same math and same
    idempotence keys as the full pass inside ``settle``; running both is
    safe. Rebuilds the positions projection when anything booked. Options
    and crypto pairs have no equity corp actions; a closed or unknown
    position books nothing."""
    from agent import broker, occ

    symbol = symbol.upper().strip()
    out = {"splits": 0, "dividends": 0, "details": []}
    if occ.is_option(symbol) or broker.is_crypto(symbol):
        return out
    trades = _trades(store, account)
    b = _compute_book(trades).get(symbol)
    if not b or b["shares"] <= EPS_SHARES:
        return out
    opened = _et_date(b.get("opened_at"))
    if not opened:
        return out
    today = _et_date(_utcnow()) or str(_utcnow().date())
    sym_trades = [t for t in trades if t["symbol"] == symbol]
    s, d, details = _corp_actions_for_symbol(store, symbol, sym_trades,
                                             opened, today, account=account)
    if s or d:
        rebuild_positions(store, account)
    return {"splits": s, "dividends": d, "details": details}


def _settle_equity_corp_actions(store, *, account: str = ACCOUNT) -> dict:
    """Fold real-world corporate actions into open EQUITY positions, using
    the same 0-dollar ledger-row convention option expiry already uses.

    SPLITS (ticker_splits): a held N:1 split otherwise fabricates a fake
    ≈−(1−1/N) loss the moment the quote rebases. For each split executed
    while the current lot was open, book a share delta of
    ``held-as-of-the-execution-date × (to/from − 1)`` at price 0 / dollars 0
    (BUY for a forward split, SELL for a reverse) — cost basis is untouched,
    so avg_price rebases by the ratio and market value / unrealized P&L are
    unchanged. Each delta books only if the position is still open at that
    moment (fresh re-read — see ``_corp_actions_for_symbol``). KNOWN LIMIT:
    reverse-split fractional shares are kept exactly; no cash-in-lieu.

    DIVIDENDS (dividends): credit ``shares held going into the ex-date ×
    cash_amount`` as a SELL of 0 shares with ``dollars=credit`` — pure cash,
    offsetting the ex-date price drop. Booked on the EX-DATE basis: real
    cash arrives on the pay date, but a mark-to-market paper book that
    waited would show a phantom dip between ex and pay, so crediting at ex
    is the honest simplification. Shares are replayed from the ledger AS OF
    the ex-date, so a position resized since then still pays on what was
    actually held. KNOWN LIMIT: only currently-open positions are scanned,
    so a position fully exited between an ex-date and the next settle drops
    that entitlement (small, conservative — never overstates the book).

    IDEMPOTENT: every adjustment row carries its key in fill_quote
    ({src, execution_date|ex_date}); booked keys are skipped by reading the
    symbol's own rows back (client-side — a JSON-field filter isn't
    transport-safe, and rows per symbol are few). Lookback is bounded to
    the current lot's earliest open fill — positions are short-lived.
    Options are skipped (occ), crypto pairs are skipped (no corp actions).
    """
    from agent import broker, occ

    trades = _trades(store, account)
    book = _compute_book(trades)
    today = _et_date(_utcnow()) or str(_utcnow().date())
    splits_booked = dividends_booked = 0
    details: list[dict] = []
    for sym in sorted(book):
        b = book[sym]
        if (b["shares"] <= EPS_SHARES or occ.is_option(sym)
                or broker.is_crypto(sym)):
            continue
        opened = _et_date(b.get("opened_at"))
        if not opened:
            continue
        sym_trades = [t for t in trades if t["symbol"] == sym]
        s, d, det = _corp_actions_for_symbol(store, sym, sym_trades,
                                             opened, today, account=account)
        splits_booked += s
        dividends_booked += d
        details.extend(det)
    return {"splits": splits_booked, "dividends": dividends_booked,
            "details": details}


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

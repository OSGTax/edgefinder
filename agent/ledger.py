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
  python -m agent.ledger grade      # machine facts per pick → desk_outcomes
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
# Crypto quotes tick continuously, so the equity cap fits. A quote whose
# timestamp is missing or unparseable FAILS CLOSED — an age we cannot measure
# is an age we must assume is bad.
MAX_QUOTE_AGE_SEC_EQ = 30.0
MAX_QUOTE_AGE_SEC_OPT = 120.0
MAX_QUOTE_AGE_SEC_CRYPTO = 30.0
# Live-fill friction gates (EQUITIES only — options liquidity is the chain's
# problem, already judged by the OPRA spread/staleness caps above, and crypto
# pairs have no daily_bars history to gate against):
# - LAST-CLOSE SANITY: the live price must sit within LIVE_CLOSE_BAND of the
#   latest STORED daily close or the fill refuses — a symbol mixup, a garbled
#   feed, or a tape event that deserves a deliberate decision, not a reflex
#   fill. Override with --allow-price-deviation when the move is real and
#   intended; a name with no stored close (new listing) warns and allows.
# - ADV SIZE GATE: one order may take at most ADV_MAX_NOTIONAL_PCT of the
#   ADV_SESSIONS-session average dollar volume — a $50k fill in a name that
#   trades $1M/day would never fill at the touch in the real world, and paper
#   alpha earned that way is fiction. Override with --allow-illiquid, on the
#   record; short/missing history (fewer than ADV_MIN_SESSIONS stored bars)
#   warns and allows.
LIVE_CLOSE_BAND = 0.20
# The close-band's reference must itself be fresh: when the latest stored
# daily close is older than this many sessions (weekday count off the bar
# date), the band degrades to WARN-AND-ALLOW — a data-ingest outage must not
# brick trading behind a reference price nobody is updating.
CLOSE_BAND_STALE_SESSIONS = 5
ADV_SESSIONS = 20
ADV_MIN_SESSIONS = 10
ADV_MAX_NOTIONAL_PCT = 0.01
# Options pay a flat per-contract fee (OPRA/regulatory + commission) on every
# fill booked through record_trade: BUY dollars = gross + fee, SELL dollars =
# gross − fee (floored at 0), with the fee stamped inside fill_quote. The
# per-share ``price`` stays the quote price — the fee lives in ``dollars``,
# so cash replay from the ledger stays exact. Settlement and corp-action rows
# bypass record_trade on purpose: they are accounting events, not orders, and
# stay fee-free.
OPTION_FEE_PER_CONTRACT = 0.65
# Marks: when more than this % of position value is marked at COST BASIS
# (no live quote, no stored close), the equity snapshot is flagged degraded —
# still written (an outage must not stop the curve), but loudly visible.
MARK_DEGRADED_COST_PCT = 20.0


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
    lands in the book's equity, not in per-symbol realized P&L). Option
    per-contract fees live in ``dollars``/cash only, never in ``price``, so
    realized P&L here is gross of fees; the equity curve nets them via cash.

    EQUITY LONG-ONLY RULES mirror ``_compute_book`` exactly: an equity SELL
    on a flat book (a duplicate exit from a writer race — a reachable row,
    see _compute_book's docstring) books ZERO and opens nothing, and any
    negative equity residue clamps flat. Without the mirror, one duplicate
    row opens a phantom short in THIS replay only and every later fill on
    the symbol attributes P&L one phase out of sync, forever.
    """
    from agent import occ

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
        if abs(cur) <= EPS_SHARES:
            if signed < 0 and not occ.is_option(sym):
                continue  # duplicate equity exit on a flat book — no lot, no P&L
            b["shares"] = cur + signed
            b["cost"] += abs(signed) * float(t["price"])
            continue
        if (cur > 0) == (signed > 0):  # extending the same direction
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
        if not occ.is_option(sym) and b["shares"] < 0:
            b["shares"], b["cost"] = 0.0, 0.0  # equities never go short
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
    # The window/run filter lives in the QUERY. A blind newest-200 cap here
    # silently truncated every window to ~7-13 market days at the v9.12
    # wake-chain cadence (~15-30 decisions/day) — open picks aged out of
    # grading and --run-id lookups for older runs returned nothing.
    if run_id:
        decisions = store.select("desk_decisions",
                                 filters={"account": account,
                                          "run_id": run_id})
    else:
        decisions = store.select("desk_decisions",
                                 filters={"account": account,
                                          "ts": ("gte", cutoff)},
                                 order=[("ts", "desc")])

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
            is_opt = occ.is_option(sym)
            entry_avg = (sum(f["shares"] * f["price"] for f in buys)
                         / sum(f["shares"] for f in buys)) if buys else None
            # OPTION picks (M3) grade off the fee-inclusive ``dollars`` on
            # the fills, not price×shares: every option fill pays a
            # per-contract fee inside dollars, and a return computed gross
            # of it flatters cheap contracts. entry_avg becomes the
            # fee-inclusive per-share cost, so since/closed percentages are
            # net-of-fee by construction (equities unchanged — no fees in
            # their dollars). Options have no split adjustments, so raw
            # fills are already on the current basis.
            buy_units = sold_units = buy_dollars = sell_dollars = 0.0
            short_opened = False
            if is_opt and raw_fills:
                buy_units = sum(float(f["shares"]) for f in raw_fills
                                if f["side"] == "BUY")
                buy_dollars = sum(float(f["dollars"]) for f in raw_fills
                                  if f["side"] == "BUY")
                sold_units = sum(float(f["shares"]) for f in raw_fills
                                 if f["side"] == "SELL")
                sell_dollars = sum(float(f["dollars"]) for f in raw_fills
                                   if f["side"] == "SELL")
                # A pick that OPENED short (sold-to-open CSP / covered call)
                # enters at the credit received. Deriving entry from the BUY
                # fills would grade the round trip with entry and exit
                # swapped — the buyback labelled as the entry.
                short_opened = raw_fills[0]["side"] == "SELL"
                if short_opened and sold_units > 0 and sell_dollars > 0:
                    entry_avg = sell_dollars / (sold_units * MULTIPLIER)
                elif buy_units > 0 and buy_dollars > 0:
                    entry_avg = buy_dollars / (buy_units * MULTIPLIER)
            pos = positions.get(sym)
            mark = (pos.get("last_price") or pos.get("avg_price")) if pos else None
            open_now = None
            if pos and abs(float(pos["shares"])) > EPS_SHARES:
                m = _mult(sym)
                open_now = {"shares": float(pos["shares"]),
                            "avg_price": pos["avg_price"], "last_price": mark,
                            # a projection row with no mark yet prices at cost
                            # basis — fake-flat; graders treat it like a
                            # cost-tier mark (see grade's M2 guard)
                            "mark_is_cost": pos.get("last_price") is None,
                            "unrealized_pnl": round(float(pos["shares"])
                                                    * ((mark or pos["avg_price"])
                                                       - pos["avg_price"]) * m, 2)}
            since_pct = None
            if entry_avg and mark:
                # short-opened picks profit as the premium DECAYS below the
                # credit received — the sign flips
                chg = (entry_avg - mark) if short_opened else (mark - entry_avg)
                since_pct = round(chg / entry_avg * 100, 2)
            # A round trip closed within this run gets an EXACT realized
            # return and an exit-bounded SPY window — the grading target the
            # weekly reflection prioritizes, previously None.
            closed_pct = exit_spy_pct = exit_date = None
            bought = sum(f["shares"] for f in buys)
            sold = sum(f["shares"] for f in sells)
            if buys and sells and abs(bought - sold) <= EPS_SHARES:
                if is_opt and short_opened and sell_dollars > 0:
                    # credit round trip: premium kept over premium received
                    closed_pct = round((sell_dollars - buy_dollars)
                                       / sell_dollars * 100, 2)
                elif is_opt and buy_dollars > 0:
                    # fee-inclusive round trip: dollars in vs dollars out
                    closed_pct = round((sell_dollars - buy_dollars)
                                       / buy_dollars * 100, 2)
                else:
                    sell_avg = sum(f["shares"] * f["price"] for f in sells) / sold
                    closed_pct = round((sell_avg - entry_avg) / entry_avg * 100, 2)
                exit_date = _et_date(raw_fills[-1].get("ts"))
                if run_date and exit_date:
                    exit_spy_pct = _spy_window_pct(spy, run_date, exit_date)
            # per-pick realized: options net of fees via dollars (partial
            # sells prorate the fee-inclusive cost basis); equities keep the
            # closing-run avg-cost attribution.
            realized = round(by_run_symbol.get((rid, sym), 0.0), 2)
            if (is_opt and buy_units > 0 and buy_dollars > 0
                    and 0 < sold_units <= buy_units + EPS_SHARES):
                realized = round(sell_dollars
                                 - (sold_units / buy_units) * buy_dollars, 2)
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
                "realized_pnl": realized,
                "open_now": open_now, "since_this_run_pct": since_pct,
                "closed_return_pct": closed_pct, "exit_date": exit_date,
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
                          "OPTION pick numbers are NET OF FEES: entry_avg_px "
                          "is the fee-inclusive per-share cost from the "
                          "fills' dollars, so since/closed percentages and "
                          "the pick's realized_pnl include the per-contract "
                          "fee (per-symbol totals and run_realized_pnl stay "
                          "gross — fees net out in cash). "
                          "Hard-stop exits (run_id 'hardstop:<id>') are "
                          "bucketed under 'hardstop', like settlement. A long "
                          "book's raw P&L is mostly market beta, so grade "
                          "alpha, not dollars.",
            "book": book, "runs": runs, "symbols": symbols_out,
            "settlement": {"realized_pnl": settlement_pnl},
            "hardstop": {"realized_pnl": hardstop_pnl},
            "unattributed_trades": unattributed}


def rebuild_positions(store, account: str = ACCOUNT) -> dict[str, dict]:
    """Recompute open lots from the ledger and reconcile desk_positions.

    Idempotent: the projection equals the ledger afterward. The reconcile is
    NON-DESTRUCTIVE — surviving symbols update in place (keeping
    ``last_price``/``marked_at``: a fill must not wipe the marks every
    reader between fills prices the book with), new symbols insert, and
    only symbols the ledger says are flat are deleted. The old
    wipe-then-rewrite exposed an empty book to every reader between its two
    statements — /portfolio's 10s cache could latch a cash-only equity —
    and left ``last_price`` NULL until the next mark, which graders read as
    a fake-flat cost mark. Returns ``{symbol: {shares, avg_price,
    opened_at}}``.
    """
    from agent.store import is_duplicate_key_error

    book = _compute_book(_trades(store, account))
    try:
        existing = {r["symbol"]: r for r in
                    store.select("desk_positions", filters={"account": account})}
    except Exception:  # noqa: BLE001 — fresh DB/first run: nothing to keep
        existing = {}
    out: dict[str, dict] = {}
    for sym, b in book.items():
        if abs(b["shares"]) <= EPS_SHARES:
            continue
        shares = round(b["shares"], 6)  # negative = covered short option leg
        avg = round(b["cost"] / abs(b["shares"]), 4)
        prev = existing.get(sym)
        if prev is None:
            try:
                store.insert("desk_positions",
                             {"account": account, "symbol": sym,
                              "shares": shares, "avg_price": avg,
                              "last_price": None,
                              "opened_at": b["opened_at"],
                              "marked_at": None}, returning=False)
            except Exception as exc:  # noqa: BLE001 — concurrent writer race
                if not is_duplicate_key_error(exc):
                    raise
                store.update("desk_positions",
                             {"account": account, "symbol": sym},
                             {"shares": shares, "avg_price": avg,
                              "opened_at": b["opened_at"]}, returning=False)
        elif (abs(float(prev.get("shares") or 0.0) - shares) > EPS_SHARES
                or abs(float(prev.get("avg_price") or 0.0) - avg) > 1e-9):
            store.update("desk_positions", {"account": account, "symbol": sym},
                         {"shares": shares, "avg_price": avg,
                          "opened_at": b["opened_at"]}, returning=False)
        out[sym] = {"shares": shares, "avg_price": avg,
                    "opened_at": b["opened_at"]}
    for sym in existing:
        if sym not in out:
            store.delete("desk_positions", {"account": account, "symbol": sym})
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
    """Cash reserved against short puts, per underlying — the worst-case
    expiry obligation, not a token:

    - an UNCOVERED short put reserves its full strike × 100 (cash-secured);
    - a SPREAD-COVERED short put (paired with a surviving long put that
      outlives it) reserves the spread's max loss:
      (short strike − long strike)⁺ × 100 per paired contract. The old
      zero-reservation for covered shorts let a wide put credit spread
      settle a fully-deployed book into deeply negative cash — de facto
      leverage on a no-leverage ledger.

    Pairing is greedy highest-strike-short against highest-strike eligible
    long, which never under-reserves relative to the true portfolio max
    loss (and replaces the old max-strike-times-all-uncovered formula that
    over-reserved multi-strike CSPs)."""
    from agent import occ

    reserved = 0.0
    unds = {occ.parse(s)["underlying"] for s, sh in positions.items()
            if occ.is_option(s) and sh < -EPS_SHARES}
    for und in unds:
        shorts: list[tuple[float, object, float]] = []
        longs: list[list] = []  # [strike, expiry, qty] — qty drawn by pairing
        for sym, sh in positions.items():
            if not occ.is_option(sym) or abs(sh) <= EPS_SHARES:
                continue
            p = occ.parse(sym)
            if p["underlying"] != und or p["type"] != "P":
                continue
            if sh < 0:
                shorts.append((p["strike"], p["expiry"], -sh))
            else:
                longs.append([p["strike"], p["expiry"], sh])
        shorts.sort(key=lambda x: -x[0])
        longs.sort(key=lambda x: -x[0])
        for k_short, exp, q_short in shorts:
            remaining = q_short
            for leg in longs:
                if remaining <= EPS_SHARES:
                    break
                k_long, l_exp, q_long = leg
                if q_long <= EPS_SHARES or l_exp < exp:
                    continue  # a long that dies first covers nothing at expiry
                take = min(q_long, remaining)
                reserved += take * max(0.0, k_short - k_long) * MULTIPLIER
                leg[2] -= take
                remaining -= take
            reserved += remaining * k_short * MULTIPLIER
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

    OPTION FEES: every option fill pays OPTION_FEE_PER_CONTRACT × contracts —
    added to a BUY's dollars, subtracted from a SELL's (floored at 0) — and
    the fee breakdown is stamped inside ``fill_quote``. ``price`` stays the
    per-share quote price; only ``dollars`` (and therefore cash) carries the
    fee, so the ledger replay stays exact. Settlement/corp-action rows book
    directly (not through here) and are fee-free by design.
    """
    from agent import occ

    store = store or _store()
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    is_opt = occ.is_option(symbol)
    if not is_opt and occ.is_adjusted_occ(symbol):
        # A digit-rooted adjusted OCC symbol (AAPL1…) is an option is_option
        # does NOT recognize — booked here it would price a 100-multiplier
        # contract as a 1x equity with no expiry settlement, no defined-risk
        # checks, and no fee. Fail closed.
        return {"ok": False, "error":
                f"{symbol} looks like an ADJUSTED OCC option symbol — "
                "adjusted contracts (non-standard deliverables) are "
                "unsupported; trade the standard contract instead"}
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
    fee = 0.0
    if is_opt:
        fee = round(OPTION_FEE_PER_CONTRACT * shares, 2)
        fill_quote = {**(fill_quote or {}),
                      "fee": {"per_contract": OPTION_FEE_PER_CONTRACT,
                              "contracts": shares, "total": fee}}
    dollars = (round(gross + fee, 2) if side == "BUY"
               else max(0.0, round(gross - fee, 2)))
    if side == "BUY":
        # Reservation on the SIMULATED post-fill book (mirrors
        # _check_option_sell): buying back a short put, or buying a long put
        # that covers one, RELEASES its reservation — gating the buy on the
        # PRE-fill reservation bricked exactly the close/roll the 5-DTE
        # discipline demands, when cash was deployed and the put ITM.
        res_positions = positions
        if is_opt:
            res_positions = dict(positions)
            res_positions[symbol] = res_positions.get(symbol, 0.0) + shares
        spendable = round(cash_now - _csp_reserved(res_positions), 2)
        if dollars > spendable + 1e-6:
            return {"ok": False, "error": "insufficient free cash for buy",
                    "needed": dollars, "cash": cash_now, "free_cash": spendable}

    store.insert("desk_trades", {
        "account": account, "run_id": run_id, "symbol": symbol, "side": side,
        "shares": shares, "price": round(price, 4), "dollars": dollars,
        "rationale": rationale, "fill_quote": fill_quote,
        "ts": _utcnow()}, returning=False)
    rebuild_positions(store, account)
    out = {"ok": True, "symbol": symbol, "side": side, "shares": shares,
           "price": round(price, 4), "dollars": dollars, "multiplier": mult,
           "fill_quote": fill_quote, "cash_after": cash(store, account)}
    if is_opt:
        out["fee"] = fee
    return out


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


def _latest_stored_close(store, symbol: str) -> tuple[float | None, str | None]:
    """``(close, bar_date)`` of the most recent daily close STORED for
    ``symbol`` — a plain order-by-date select on daily_bars, transport-safe
    on both lanes (unlike ``_latest_close``, which routes through the
    indicator engine). ``(None, None)`` when the symbol has no bars (new
    listing) or the read fails. The bar date rides along so the caller can
    judge whether the reference itself is stale (see live_fill's
    warn-and-allow degrade)."""
    try:
        rows = store.select("daily_bars", columns="date,close",
                            filters={"symbol": symbol},
                            order=[("date", "desc")], limit=1)
    except Exception:  # noqa: BLE001 — no close row → the caller warns+allows
        return None, None
    if rows and rows[0].get("close"):
        px = float(rows[0]["close"])
        if px > 0:
            return px, str(rows[0].get("date"))[:10]
    return None, None


def _weekday_sessions_since(bar_date: str | None, today) -> int:
    """Approximate TRADING SESSIONS elapsed strictly after ``bar_date`` up to
    ``today``: weekdays (Mon–Fri) counted, exchange holidays NOT subtracted —
    around a holiday this over-counts by one and degrades the close band a
    session early, which errs toward allowing (the honest direction for an
    outage guard). An unparseable bar date counts as maximally stale."""
    from datetime import date as _date, timedelta as _td

    try:
        d = _date.fromisoformat(str(bar_date)[:10])
    except (TypeError, ValueError):
        return 10_000
    n = 0
    while d < today:
        d += _td(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _avg_dollar_volume(store, symbol: str) -> tuple[float | None, int]:
    """(20-session average dollar volume, sessions found) from daily_bars —
    close × volume over the latest ADV_SESSIONS stored bars. Returns
    ``(None, n)`` when fewer than ADV_MIN_SESSIONS bars exist (new listing /
    symbol outside the hot set) so the gate warns and allows instead of
    blocking a name we simply haven't ingested. Same math as the market
    tools' dollar-volume ranking, computed locally on the ledger's own
    transport seam."""
    try:
        rows = store.select("daily_bars", columns="date,close,volume",
                            filters={"symbol": symbol},
                            order=[("date", "desc")], limit=ADV_SESSIONS)
    except Exception:  # noqa: BLE001 — unreadable history → warn-and-allow
        return None, 0
    vals = [float(r["close"]) * float(r.get("volume") or 0.0)
            for r in rows if r.get("close")]
    if len(vals) < ADV_MIN_SESSIONS:
        return None, len(vals)
    return sum(vals) / len(vals), len(vals)


def live_fill(store=None, *, symbol: str, side: str, shares: float | None = None,
              notional: float | None = None, rationale: str | None = None,
              run_id: str | None = None, slippage_bp: float = SLIPPAGE_BP,
              allow_price_deviation: bool = False, allow_illiquid: bool = False,
              account: str = ACCOUNT) -> dict:
    """Book a fill AT THE LIVE QUOTE — the only execution path the agent uses.

    Reads the real-time SIP quote, prices the correct side (BUY at ask,
    SELL at bid, ± slippage), stamps the quote snapshot on the fill, and
    refuses to book when the market is closed, the quote is degenerate, or
    the quote is stale (a quote whose timestamp cannot be parsed FAILS
    CLOSED). Extended hours are allowed for equities with tighter
    spread guards; options are refused outside RTH (OPRA book is genuinely
    bad pre-open/post-close). NOTE: a post-close (after-hours) extended BUY
    cannot be exited until the NEXT session's tape — an overnight hold by
    construction, matching pre-market semantics — so size it accordingly.
    **Crypto** pairs (BTC/USD, ETH/USD, …) trade
    24/7 and skip the RTH/close-soon gates; their spread cap is 3% and no
    options structure is possible.  Exactly one of ``shares`` / ``notional``
    sizes the order (fractional ok — required for high-priced crypto).

    EQUITY FRICTION GATES (see the constants block): the priced fill must
    sit within LIVE_CLOSE_BAND of the latest stored daily close
    (``allow_price_deviation`` overrides; no stored close warns and allows,
    and a reference close older than CLOSE_BAND_STALE_SESSIONS degrades the
    band to warn-and-allow — a data outage must not brick trading) and the
    order notional may take at most ADV_MAX_NOTIONAL_PCT of the 20-session
    average dollar volume (``allow_illiquid`` overrides; short history warns
    and allows). Both gates apply to BUYs and SELLs alike — dumping into no
    bid is as fictional as buying at the touch. Options and crypto skip both
    gates: options liquidity is judged on the chain itself (OPRA spread/
    staleness caps), and crypto has no daily_bars history to gate against.

    DESIGN NOTE — gates vs protective exits: these bands exist to veto
    ENTRIES (a reflex buy into a garbled quote, a size that could never
    fill). A PROTECTIVE EXIT — the streamer's hard stop — traverses the
    same gates but passes BOTH overrides explicitly: the canonical stop
    scenario (a >20% earnings gap) and a full-position exit over the ADV
    cap must never be vetoed by the very gates meant to protect entries.
    Honesty is preserved by the receipt, not by refusing the exit — every
    override and skipped/degraded gate lands as ``warnings`` inside the
    persisted ``fill_quote``.
    """
    from agent import broker, occ

    store = store or _store()
    symbol = symbol.upper().strip()
    side = side.upper().strip()
    is_opt = occ.is_option(symbol)
    is_cx = broker.is_crypto(symbol)
    # Negative slippage would book INSIDE the quote — price improvement is
    # fiction on a paper book (the honesty contract prices BUYs at the ask
    # and SELLs at the bid, never better).
    slippage_bp = max(0.0, float(slippage_bp))
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
        # Crypto is 24/7 so the "can't sell today" premise doesn't apply —
        # and buying back a SHORT option leg is an EXIT, not a new position:
        # the gate must not strand a short put through the close into expiry.
        if sess == "regular" and side == "BUY" and hasattr(b, "is_close_soon") \
                and b.is_close_soon(minutes=15) \
                and not (is_opt and _held_shares(store, symbol) < -EPS_SHARES):
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

    # Quote-freshness — the defense against a feed stuck open on an old
    # print. FAIL-CLOSED: a quote whose timestamp is missing or unparseable
    # has an age we cannot measure, and an unmeasurable age must be assumed
    # bad — never a free pass around the staleness cap.
    age = _quote_age_sec(q.get("t"))
    if age is None:
        return {"ok": False, "error":
                "quote timestamp missing/unparseable — cannot verify "
                "freshness, refusing to fill",
                "bid": bid, "ask": ask, "session": sess}
    if is_cx:
        max_age = MAX_QUOTE_AGE_SEC_CRYPTO
    elif is_opt:
        max_age = MAX_QUOTE_AGE_SEC_OPT
    else:
        max_age = MAX_QUOTE_AGE_SEC_EQ
    if age > max_age:
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

    # Equity friction gates — an INDEPENDENT price reference (yesterday's
    # stored close) and a size-vs-liquidity check. The in-quote band inside
    # record_trade checks the price against the very quote it was computed
    # from, which a wrong quote passes trivially; these gates are the checks
    # a wrong quote or an unfillable size actually fails.
    warnings: list[str] = []
    if not (is_opt or is_cx):
        last_close, close_date = _latest_stored_close(store, symbol)
        if last_close is None:
            warnings.append(f"no stored daily close for {symbol} (new listing "
                            "or outside the hot set) — last-close sanity gate "
                            "skipped")
        else:
            from datetime import date as _date

            today = _date.fromisoformat(
                _et_date(_utcnow()) or str(_utcnow().date()))
            # A split executing AFTER the stored close rebases the tape: the
            # raw reference would veto every same-day fill on the split name
            # (or wave a genuinely wrong price through on a reverse split).
            # Rescale the reference onto the current share basis.
            try:
                srows = store.select("ticker_splits",
                                     filters={"symbol": symbol})
            except Exception:  # noqa: BLE001 — guard falls back to raw close
                srows = []
            for sr in srows:
                ed = str(sr.get("execution_date") or "")[:10]
                frm = float(sr.get("split_from") or 0)
                to = float(sr.get("split_to") or 0)
                if frm > 0 and to > 0 and close_date < ed <= str(today):
                    last_close = last_close * frm / to
                    warnings.append(
                        f"last-close reference rebased for the {to:g}:{frm:g} "
                        f"{symbol} split executed {ed}")
            stale_sessions = _weekday_sessions_since(close_date, today)
            dev = abs(price / last_close - 1.0)
            if stale_sessions > CLOSE_BAND_STALE_SESSIONS:
                # the reference itself is stale — a data-ingest outage must
                # not brick trading behind a close nobody is updating
                warnings.append(
                    f"latest stored close for {symbol} is ~{stale_sessions} "
                    f"sessions old (bar date {close_date}) — last-close "
                    "band degraded to warn-and-allow")
                if dev > LIVE_CLOSE_BAND:
                    warnings.append(f"price deviation {dev:.1%} vs stale "
                                    f"close {last_close:g} allowed "
                                    "(degraded band)")
            elif dev > LIVE_CLOSE_BAND and not allow_price_deviation:
                return {"ok": False, "error":
                        f"price {price:g} is {dev:.1%} from the latest stored "
                        f"close {last_close:g} (band {LIVE_CLOSE_BAND:.0%}) — "
                        "a real move this big deserves a deliberate override "
                        "(--allow-price-deviation), not a reflex fill",
                        "price": price, "latest_close": round(last_close, 4),
                        "deviation": round(dev, 4), "band": LIVE_CLOSE_BAND}
            elif dev > LIVE_CLOSE_BAND:
                warnings.append(f"price deviation {dev:.1%} vs close "
                                f"{last_close:g} allowed by override")
        order_notional = shares * price
        adv, adv_sessions = _avg_dollar_volume(store, symbol)
        if adv is None:
            warnings.append(f"only {adv_sessions} stored session(s) for "
                            f"{symbol} — ADV size gate skipped")
        elif order_notional > ADV_MAX_NOTIONAL_PCT * adv:
            if not allow_illiquid:
                return {"ok": False, "error":
                        f"order notional ${order_notional:,.0f} exceeds "
                        f"{ADV_MAX_NOTIONAL_PCT:.0%} of {symbol}'s "
                        f"{adv_sessions}-session average dollar volume "
                        f"(${adv:,.0f}) — this size would not fill at the "
                        "touch; downsize or override with --allow-illiquid",
                        "notional": round(order_notional, 2),
                        "adv": round(adv, 2),
                        "max_pct": ADV_MAX_NOTIONAL_PCT}
            warnings.append(f"notional ${order_notional:,.0f} > "
                            f"{ADV_MAX_NOTIONAL_PCT:.0%} of ADV "
                            f"(${adv:,.0f}) allowed by override")

    if is_opt:
        src = "alpaca_opra_rest"
    elif is_cx:
        src = "alpaca_crypto_rest"
    else:
        src = "alpaca_sip_rest"
    snapshot = {"bid": bid, "ask": ask, "mid": q.get("mid"), "t": q.get("t"),
                "src": src, "slippage_bp": slippage_bp, "session": sess}
    if warnings:
        # the receipt, not just the return value: overridden/degraded gates
        # must be visible on the persisted row (H1 — a protective exit that
        # crossed a band shows it crossed the band)
        snapshot["warnings"] = list(warnings)
    res = record_trade(store, symbol=symbol, side=side, shares=shares,
                       price=price, rationale=rationale, run_id=run_id,
                       fill_quote=snapshot, account=account)
    if warnings:
        res.setdefault("warnings", []).extend(warnings)
    return res


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
    'runtime_price' (fallback when the expiry-day bar is missing).
    Settlement rows are FEE-FREE by design: expiry settlement is an
    accounting event, not an order (the per-contract fee lives only on fills
    booked through record_trade)."""
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

    COVERAGE IS AN ALLOCATION, NOT A PER-LEG CHECK: every expired short
    draws its backing DOWN from a mutable per-underlying working state —
    surviving long legs (same type, expiring on/after the short) classify
    it spread-covered first; only the remainder may share-assign (calls,
    up to the shares still unallocated) or cash-assign (puts, only when
    cash actually funds the purchase — else it cash-settles at intrinsic,
    the honest loss without overdrawing into shares nobody chose). The old
    per-leg checks re-read the same STATIC aggregates, so a covered call
    plus a call vertical on one underlying share-assigned TWICE — sourcing
    the second sale from shares already called away and fabricating cash
    on the append-only ledger.

    KNOWN LIMIT: options settle ONLY at expiry — there is no
    early-assignment model. A deep-ITM American short (classically a
    covered call through an ex-dividend date) would realistically be
    assigned early; the paper book holds it to expiry, which slightly
    flatters short-premium structures.

    EQUITIES: splits and dividends on open positions are booked as 0-price
    adjustment rows — see ``_settle_equity_corp_actions``.
    """
    from agent import occ

    store = store or _store()
    today = _utcnow().date()
    positions = _positions_map(store, account)
    actions: list[dict] = []
    expiry_px: dict[tuple[str, object], tuple[float, str]] = {}

    def _px_for(p: dict) -> tuple[float, str]:
        key = (p["underlying"], p["expiry"])
        if key not in expiry_px:
            close = _bar_close_on(store, p["underlying"], p["expiry"])
            if close:
                expiry_px[key] = (close, "expiry_close")
            else:
                expiry_px[key] = ((_live_mids([p["underlying"]]).get(p["underlying"])
                                   or _latest_close(p["underlying"]) or 0.0),
                                  "runtime_price")
        return expiry_px[key]

    expired: list[tuple[str, float, dict]] = []
    for sym, sh in sorted(positions.items()):
        if not occ.is_option(sym) or abs(sh) <= EPS_SHARES:
            continue
        p = occ.parse(sym)
        if p["expiry"] < today:
            expired.append((sym, sh, p))

    # The mutable coverage allocation (see the docstring): shares and
    # surviving long legs are DRAWN DOWN as shorts classify against them,
    # so two shorts can never both claim the same backing.
    shares_avail: dict[str, float] = {}
    long_pool: dict[tuple[str, str], list[list]] = {}
    for sym2, sh2 in positions.items():
        if occ.is_option(sym2) and sh2 > EPS_SHARES:
            pp = occ.parse(sym2)
            long_pool.setdefault((pp["underlying"], pp["type"]), []).append(
                [pp["expiry"], sh2])
    for legs in long_pool.values():
        legs.sort()  # consume the shortest-lived eligible cover first

    shorts: list[tuple[str, float, dict]] = []
    for sym, sh, p in expired:
        und_px, basis = _px_for(p)
        if und_px <= 0:
            actions.append({"symbol": sym, "action": "SKIPPED",
                            "error": f"no price for underlying {p['underlying']}"})
            continue
        if sh < 0:
            shorts.append((sym, abs(sh), p))
            continue
        iv = occ.intrinsic(sym, und_px)
        qty = abs(sh)
        desc = occ.describe(sym)
        _book_settlement(store, symbol=sym, side="SELL", qty=qty, price=iv,
                         rationale=f"expiry settlement: {desc} "
                                   f"{'exercised (cash-settled intrinsic)' if iv > 0 else 'expired worthless'}",
                         underlying_px=und_px, basis=basis, account=account)
        actions.append({"symbol": sym, "action": "long settled",
                        "intrinsic": round(iv, 4), "qty": qty,
                        "basis": basis})

    for sym, qty, p in sorted(shorts, key=lambda x: (x[2]["expiry"], x[0])):
        und = p["underlying"]
        und_px, basis = _px_for(p)
        iv = occ.intrinsic(sym, und_px)
        desc = occ.describe(sym)
        # 1) spread cover first: net against surviving long legs of the same
        #    type that expire ON/AFTER this short (eligibility is per-SHORT —
        #    the old global latest-short-expiry filter discounted an expired
        #    long exactly when a later-dated short existed on the underlying)
        covered = 0.0
        for leg in long_pool.get((und, p["type"]), []):
            if covered >= qty - EPS_SHARES:
                break
            l_exp, l_qty = leg
            if l_qty <= EPS_SHARES or l_exp < p["expiry"]:
                continue
            take = min(l_qty, qty - covered)
            leg[1] -= take
            covered += take
        remainder = qty - covered
        # 2) the uncovered remainder assigns — calls against the shares still
        #    unallocated, puts against cash that actually funds the purchase
        assigned = 0.0
        cash_short = False
        if iv > 0 and remainder > EPS_SHARES:
            if p["type"] == "C":
                avail = shares_avail.setdefault(
                    und, max(0.0, positions.get(und, 0.0)))
                assigned = min(remainder, avail // MULTIPLIER)
                if assigned > 0:
                    shares_avail[und] = avail - assigned * MULTIPLIER
                    _book_settlement(store, symbol=und, side="SELL",
                                     qty=assigned * MULTIPLIER, price=p["strike"],
                                     rationale=f"assignment: shares called away "
                                               f"at ${p['strike']:g} ({desc})",
                                     underlying_px=und_px, basis=basis,
                                     account=account)
            else:
                cost = remainder * p["strike"] * MULTIPLIER
                if cost <= cash(store, account) + 1e-6:
                    assigned = remainder
                    _book_settlement(store, symbol=und, side="BUY",
                                     qty=assigned * MULTIPLIER, price=p["strike"],
                                     rationale=f"assignment: shares put to us "
                                               f"at ${p['strike']:g} ({desc})",
                                     underlying_px=und_px, basis=basis,
                                     account=account)
                else:
                    cash_short = True  # cash-settle below at intrinsic instead
        # 3) close the option leg itself: assigned contracts at 0 (the
        #    strike-priced share row carries the economics), the rest at
        #    intrinsic (spread-covered / cash-settled; OTM → 0)
        if assigned > 0:
            _book_settlement(store, symbol=sym, side="BUY", qty=assigned,
                             price=0.0,
                             rationale=f"expiry settlement: {desc} short leg "
                                       "assigned",
                             underlying_px=und_px, basis=basis, account=account)
        cash_qty = qty - assigned
        if cash_qty > EPS_SHARES:
            _book_settlement(store, symbol=sym, side="BUY", qty=cash_qty,
                             price=iv,
                             rationale=f"expiry settlement: {desc} short leg "
                                       f"{'cash-settled' if iv > 0 else 'expired worthless'}",
                             underlying_px=und_px, basis=basis, account=account)
        act = {"symbol": sym, "action": "short settled", "itm": iv > 0,
               "qty": qty, "assigned": round(assigned, 6),
               "spread_covered": round(covered, 6), "basis": basis}
        if cash_short:
            act["note"] = ("insufficient cash for put assignment — "
                           "cash-settled at intrinsic")
        actions.append(act)
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
    unavailable, then to COST BASIS as the last resort.

    PROVENANCE: every snapshot records which tier marked each position in
    ``desk_equity.mark_meta`` — ``{"sources": {live, close, cost},
    "cost_marked": [...], "cost_marked_value_pct": x}``. A cost-basis mark is
    fake-flat P&L, and a snapshot written during a data outage used to embed
    that silently forever. When cost-marked value exceeds
    MARK_DEGRADED_COST_PCT of position value the meta carries
    ``"degraded": true`` — the snapshot still writes (an outage must not
    stop the equity curve; visibility is the fix) and the CLI warns loudly.
    Caller-supplied ``prices`` count as live (they are explicit marks)."""
    store = store or _store()
    positions = rebuild_positions(store, account)
    src: dict[str, str] = {}
    if prices is None:
        prices = _live_mids(list(positions))
        src = {s: "live" for s in prices}
        for sym, px in _latest_closes([s for s in positions if s not in prices]).items():
            prices.setdefault(sym, px)
            src.setdefault(sym, "close")
    else:
        src = {s: "live" for s in prices}
    now = _utcnow()
    pos_value = 0.0
    counts = {"live": 0, "close": 0, "cost": 0}
    cost_marked: list[str] = []
    cost_value = total_value = 0.0
    for sym, p in positions.items():
        tier = src.get(sym, "live") if prices.get(sym) else "cost"
        px = prices.get(sym) or p["avg_price"]
        px = round(float(px), 4)
        mv = p["shares"] * px * _mult(sym)  # signed; options ×100
        pos_value += mv
        total_value += abs(mv)
        counts[tier] += 1
        if tier == "cost":
            cost_marked.append(sym)
            cost_value += abs(mv)
        store.update("desk_positions", {"account": account, "symbol": sym},
                     {"last_price": px, "marked_at": now}, returning=False)
    cost_pct = round(cost_value / total_value * 100, 2) if total_value else 0.0
    mark_meta: dict = {"sources": counts, "cost_marked": sorted(cost_marked),
                       "cost_marked_value_pct": cost_pct}
    if cost_pct > MARK_DEGRADED_COST_PCT:
        mark_meta["degraded"] = True
    c = cash(store, account)
    equity = round(c + pos_value, 2)
    snap = {
        "account": account, "ts": now, "cash": c,
        "positions_value": round(pos_value, 2), "equity": equity,
        "return_pct": round((equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 4),
        "mark_meta": mark_meta,
    }
    try:
        store.insert("desk_equity", snap, returning=False)
    except Exception as exc:  # noqa: BLE001 — classify, then re-raise others
        # Pre-deploy grace: a DB that predates the mark_meta column must not
        # crash the mark mid-write — the equity point matters more than its
        # provenance for the one deploy where they disagree.
        if "mark_meta" not in str(exc):
            raise
        import sys

        print("WARNING: desk_equity has no mark_meta column (schema not "
              "migrated) — snapshot written WITHOUT provenance; deploy "
              "(render_start runs the idempotent ALTER) or run "
              "scripts/setup_db.py", file=sys.stderr)
        store.insert("desk_equity",
                     {k: v for k, v in snap.items() if k != "mark_meta"},
                     returning=False)
    return state(store, account)


def _latest_mark_meta(store, account: str = ACCOUNT) -> dict | None:
    """The latest equity snapshot's mark provenance (None before any mark,
    or for snapshots that predate the mark_meta column). Transports may hand
    JSON back as text — parse defensively, same as ``_fq``."""
    try:
        rows = store.select("desk_equity", filters={"account": account},
                            order=[("ts", "desc"), ("id", "desc")], limit=1)
    except Exception:  # noqa: BLE001 — provenance is a bonus, never a crash
        return None
    if not rows:
        return None
    meta = rows[0].get("mark_meta")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = None
    return meta if isinstance(meta, dict) else None


def state(store=None, account: str = ACCOUNT) -> dict:
    """Full account snapshot: cash, positions (marked), equity, P&L — plus
    the latest mark's provenance summary (``mark_meta``, see ``mark``)."""
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
        "mark_meta": _latest_mark_meta(store, account),
    }


# ── the learning loop: machine-graded pick facts (desk_outcomes) ──


def _plausible_kill(level: float | None,
                    entry_px: float | None) -> float | None:
    """Long-only stop plausibility: a kill is an exit level on a long
    position, so with the pick's entry price known, a parsed level outside
    [0.2×, 2×] of entry is a parse ARTIFACT (a year, a share count, a price
    target for a different unit), not a stop — return None and let the
    reflection judge the free text itself. Without an entry price there is
    nothing to check against, so the level passes through."""
    if level is None or level <= 0:
        return None
    if entry_px and entry_px > 0 \
            and not (0.2 * entry_px <= level <= 2.0 * entry_px):
        return None
    return level


def _parse_kill(kill, entry_px: float | None = None) -> float | None:
    """A pick's ``kill`` as a price level, when it parses unambiguously.

    Kills are free text by design ("closes below $385", "thesis breaks on a
    guidance cut"). A bare numeric kill is the level; in text, a single
    $-prefixed number wins (so "$385 within 10 sessions" still parses), else
    a single bare number — AFTER excluding numbers that are provably not
    price levels:
    - percentages ("drops 8% in a day" — 8 is a move size, not a level);
    - numbers glued to indicator/time-unit tokens ("100DMA", "50-day",
      "10 sessions" — lookback lengths and spans).
    Count words in prose ("two closes below 190") never register — only
    digits are candidates, so 190 parses. Surviving candidates then face the
    long-only plausibility check (``_plausible_kill``: within [0.2×, 2×] of
    the entry price when known). Zero or several candidates, or an
    implausible one, is honest ambiguity → None, and kill_breached stays
    null (the reflection judges the free text from the closes itself)
    rather than a confidently wrong fact."""
    if isinstance(kill, (int, float)) and not isinstance(kill, bool):
        return _plausible_kill(float(kill), entry_px)
    if not isinstance(kill, str):
        return None
    import re

    unit_after = re.compile(
        r"^\s*(?:%|[\s\-]*(?:x\b|dma\b|s?ma\b|ema\b|days?\b|sessions?\b|"
        r"weeks?\b|months?\b|hours?\b))", re.I)
    dollar: list[str] = []
    bare: list[str] = []
    for m in re.finditer(r"(\$)?\s*(\d[\d,]*(?:\.\d+)?)", kill):
        if unit_after.match(kill[m.end():]):
            continue  # a percentage / indicator length / time span
        (dollar if m.group(1) else bare).append(m.group(2))
    nums = dollar if len(dollar) == 1 else (dollar + bare)
    if len(nums) != 1:
        return None
    try:
        v = float(nums[0].replace(",", ""))
    except ValueError:
        return None
    return _plausible_kill(v, entry_px)


def _kill_breached(store, symbol: str, level: float, start_date: str,
                   end_date: str,
                   split_events: list[tuple[str, float]] | None = None
                   ) -> bool | None:
    """Did any stored daily close between entry and grade date touch the
    kill? Long-book semantics: a kill is a BELOW level and touch counts
    (close <= level), mirroring the tripwire sweep's at-or-below trip rule.
    The kill was stated on the ENTRY-day price basis, so a booked split
    between entry and a close rebases the level with the tape
    (``split_events`` are the ledger's own booked split rows). None when no
    closes are stored in the window — nothing to judge, never a fake no."""
    from datetime import date as _date

    try:
        rows = store.select(
            "daily_bars", columns="date,close",
            filters={"symbol": symbol,
                     "date": [("gte", _date.fromisoformat(start_date[:10])),
                              ("lte", _date.fromisoformat(end_date[:10]))]},
            order=[("date", "asc")])
    except Exception:  # noqa: BLE001 — unreadable history → null, not a guess
        return None
    closes = [(str(r["date"])[:10], float(r["close"]))
              for r in rows if r.get("close")]
    if not closes:
        return None
    for d, c in closes:
        f = 1.0
        for ed, x in (split_events or []):
            if start_date[:10] < ed <= d:
                f *= x
        if c <= level / f:
            return True
    return False


def _trade_key(t: dict) -> tuple:
    """The ledger's replay-order key — the same (ts, id) ordering _trades
    sorts by, usable for strictly-before/after comparisons."""
    return (str(t.get("ts")), t.get("id") or 0)


def _exit_kind_for(dominant_run, pick_run_id: str) -> str:
    """Classify how a pick closed, by the run that booked the (dominant)
    closing fills: same_run | cross_run | hardstop | settlement."""
    r = str(dominant_run or "")
    if r.startswith("hardstop:"):
        return "hardstop"
    if r == "settlement":
        return "settlement"
    return "same_run" if r == pick_run_id else "cross_run"


def _reconstruct_exit(trades: list[dict], sym: str, entry_fills: list[dict],
                      split_events: list[tuple[str, float]] | None,
                      is_opt: bool) -> dict | None:
    """How a pick's position actually CLOSED when the closing fills lived
    outside its own run (a hard stop, a later run's exit, expiry
    settlement).

    Walks the symbol's ledger rows in replay order, finds the FIRST moment
    the net position returns to flat strictly after the pick's entry fills
    (long-pick semantics: net <= 0 counts as flat — grade only grades picks
    with BUY entries), and averages every closing SELL between the last
    entry fill and that flat point — ANY run_id — split-adjusted onto the
    current share basis. Corp-action rows are bookkeeping, not exits: split
    share-deltas move the running position but never enter the average, and
    0-share dividend credits are ignored. For OPTION picks the exit prices
    from the fee-inclusive ``dollars`` (fee-net proceeds per share),
    matching the fee-inclusive entry basis.

    Returns None when the position never went flat after the entry (still
    open, or the ledger is odd); else ``{exit_avg_px, exit_units, exit_date,
    flat_key, dominant_run}``.
    """
    if not entry_fills:
        return None
    last_entry = max(_trade_key(t) for t in entry_fills)
    running = 0.0
    closing: list[dict] = []
    flat_at = None
    for t in trades:  # already in replay order (_trades sorts)
        if t["symbol"] != sym:
            continue
        qty = float(t["shares"])
        running += qty if t["side"] == "BUY" else -qty
        if _trade_key(t) <= last_entry:
            continue
        if (t["side"] == "SELL" and qty > EPS_SHARES
                and _fq(t).get("src") not in ("split_adjustment", "dividend")):
            closing.append(t)
        if running <= EPS_SHARES:
            flat_at = t
            break
    if flat_at is None or not closing:
        return None
    units = value = 0.0
    run_units: dict = {}
    for t in closing:
        f = _split_factor_since(split_events, _et_date(t.get("ts")) or "")
        sh = float(t["shares"]) * f
        if is_opt:  # fee-net per-share proceeds (options never split-adjust)
            px = float(t["dollars"]) / (float(t["shares"]) * MULTIPLIER)
        else:
            px = float(t["price"]) / f
        units += sh
        value += sh * px
        rid = t.get("run_id")
        run_units[rid] = run_units.get(rid, 0.0) + sh
    if units <= EPS_SHARES:
        return None
    return {"exit_avg_px": value / units, "exit_units": units,
            "exit_date": _et_date(flat_at.get("ts")),
            "flat_key": _trade_key(flat_at),
            "dominant_run": max(run_units, key=lambda r: run_units[r])}


def _commitment_breach(store, symbol: str, direction: str, level: float,
                       start_date: str, end_date: str,
                       split_events: list[tuple[str, float]] | None = None
                       ) -> tuple[str, float] | None:
    """The FIRST stored daily close between the commitment's creation and its
    deadline that touches its level in the stated direction — (date, close),
    else None. Sibling of ``_kill_breached`` but two-sided: a ``below``
    commitment (a re-entry floor, a stop) triggers on close <= level; an
    ``above`` commitment (a re-add ceiling, "reclaims $325") triggers on
    close >= level. Split-aware on the creation-day basis, same as the kill
    sweep. None when no closes are stored — nothing to judge, never a fake
    no."""
    from datetime import date as _date

    try:
        rows = store.select(
            "daily_bars", columns="date,close",
            filters={"symbol": symbol,
                     "date": [("gte", _date.fromisoformat(start_date[:10])),
                              ("lte", _date.fromisoformat(end_date[:10]))]},
            order=[("date", "asc")])
    except Exception:  # noqa: BLE001 — unreadable history → null, not a guess
        return None
    for r in rows:
        if not r.get("close"):
            continue
        d, c = str(r["date"])[:10], float(r["close"])
        f = 1.0
        for ed, x in (split_events or []):
            if start_date[:10] < ed <= d:
                f *= x
        lvl = level / f
        if (direction == "below" and c <= lvl) or \
           (direction == "above" and c >= lvl):
            return d, round(c, 4)
    return None


def sweep_commitments(store=None, *, account: str = ACCOUNT,
                      split_events: dict | None = None,
                      trades: list | None = None,
                      today: str | None = None) -> dict:
    """Machine-check open commitments (the structured falsification clauses on
    trim/exit picks) against stored closes — the fix for free-text re-add
    promises that used to go silently unchecked. A priced commitment whose
    level is touched in its direction flips to ``fired`` with the breaching
    date + close; an unpriced or price-untouched commitment past its ``until``
    deadline flips to ``expired``. Fired-and-unhonored commitments are what
    ``brain context`` then surfaces as obligations. Idempotent: only ``open``
    rows are swept, and grade calls this each pass."""
    from datetime import date as _date

    store = store or _store()
    try:
        rows = store.select("desk_commitments",
                            filters={"account": account, "status": "open"},
                            order=[("id", "asc")], limit=500)
    except Exception as exc:  # noqa: BLE001 — pre-deploy grace, like grade()
        from agent.store import is_missing_table_error

        if is_missing_table_error(exc):
            return {"ok": True, "swept": 0, "fired": 0, "expired": 0,
                    "note": "desk_commitments not migrated yet — skipped"}
        raise
    if not rows:
        return {"ok": True, "swept": 0, "fired": 0, "expired": 0}

    if trades is None:
        trades = _trades(store, account)
    if split_events is None:
        split_events = _booked_split_events(trades)
    today = today or _et_date(_utcnow()) or str(_utcnow().date())

    # commitment start = its creating decision's date (the level was stated
    # then); fall back to created_at, then to nothing swept.
    dec_ts: dict[str, str] = {}
    for d in store.select("desk_decisions", filters={"account": account},
                          columns="run_id,ts", order=[("ts", "desc")],
                          limit=200):
        dec_ts.setdefault(str(d.get("run_id")), _et_date(d.get("ts")) or "")

    fired = expired = 0
    for c in rows:
        sym = c.get("symbol")
        start = dec_ts.get(str(c.get("run_id"))) \
            or _et_date(c.get("created_at")) or ""
        until = c.get("until")
        until = str(until)[:10] if until else None
        hit = None
        if (c.get("level") is not None and c.get("direction") in
                ("above", "below") and start):
            end = min(until, today) if until else today
            if end >= start:
                hit = _commitment_breach(store, sym, c["direction"],
                                         float(c["level"]), start, end,
                                         split_events.get(sym))
        if hit:
            store.update("desk_commitments", {"id": c["id"]},
                         {"status": "fired", "fired_date":
                          _date.fromisoformat(hit[0]), "fired_close": hit[1]},
                         returning=False)
            fired += 1
        elif until and today > until:
            # deadline passed with no breach (or nothing to price-check) —
            # the window closed; the clause did not fire.
            store.update("desk_commitments", {"id": c["id"]},
                         {"status": "expired"}, returning=False)
            expired += 1
    return {"ok": True, "swept": len(rows), "fired": fired, "expired": expired}


# Open picks are always re-graded; --days only bounds how far back CLOSED
# rows are refreshed (their facts are final once written).
GRADE_OPEN_LOOKBACK_DAYS = 3650


def grade(store=None, *, days: int = 30, run_id: str | None = None,
          account: str = ACCOUNT) -> dict:
    """Materialize per-pick MACHINE FACTS into ``desk_outcomes`` — the
    learning loop's durable scoreboard, run before every reflection so
    grading starts from numbers, not memory.

    One row per (account, run_id, symbol), UPDATED IN PLACE on re-grade
    (``grade_date`` tracks the latest pass) — boring and transport-safe, and
    the reflection agent's ``verdict``/``verdict_note`` survive re-grading
    because grade never touches those two columns (``agent.brain verdict``
    is their only writer). Facts come from ``outcomes()`` — already
    split-aware and TR-SPY-benchmarked, with BOOK stances excluded and
    settlement/hard-stop rows bucketed out of runs — so only picks with
    entry (BUY) fills in their own run are graded: a hold/trim/exit pick
    books no entry and has no per-pick entry fact to grade.

    WINDOW (L4): ``days`` bounds CLOSED-row re-grades only — a closed row
    older than the window whose facts are already stored is skipped. Every
    still-OPEN pick is re-graded on every pass regardless of ``days``: an
    open position must never age out of grading.

    EXITS (H3): a pick whose position is flat but whose own run booked no
    round trip — a hard-stop exit, a later run's sell, expiry settlement —
    is NOT left with null facts: the exit is reconstructed from the actual
    closing sell fills (any run_id, split-adjusted, fee-net for options; see
    ``_reconstruct_exit``) and the row carries ``exit_avg_px``,
    ``exit_kind`` (same_run | cross_run | hardstop | settlement, by the
    dominant closing run), a real ``since_pct``/``alpha_pct`` over the
    entry→flat window, and ``realized_pnl``. realized_pnl is the
    per-symbol avg-cost replay between entry and flat — exact when the pick
    owned the lot alone, approximate when other runs built or held the same
    name concurrently (documented approximation, same replay machinery as
    ``_realized_pnl``).

    DEGRADED MARKS (M2): when the latest equity snapshot's ``mark_meta``
    says a pick's symbol was marked at COST BASIS, the mark-derived facts
    (since_pct / alpha_pct / mark_px) are written as NULL and the row's
    ``degraded`` flag is set — a fake-flat mark must not grade a pick. A
    later clean re-grade overwrites both.

    Per row: entry_avg_px and mark_px (basis 'exit' for closed picks with a
    reconstructed or same-run exit, else 'mark'), since_pct / spy_pct /
    alpha_pct, horizon_days and horizon_elapsed (completed SPY sessions
    since the decision vs the horizon), kill_level (parsed from the pick's
    free-text kill when unambiguous and plausible vs entry — see
    ``_parse_kill``) and kill_breached (see ``_kill_breached``),
    status open|closed."""
    from datetime import date as _date, timedelta as _td

    store = store or _store()
    # Pre-deploy grace (L1): a missing desk_outcomes table must exit with an
    # actionable message, not a stack trace mid-reflection.
    try:
        store.select("desk_outcomes", filters={"account": account}, limit=1)
    except Exception as exc:  # noqa: BLE001 — classify, then re-raise others
        from agent.store import is_missing_table_error

        # Classified by type/code — a transient connection error whose str()
        # merely mentions the table (SQLAlchemy embeds the SQL) re-raises.
        if is_missing_table_error(exc):
            return {"ok": False, "error":
                    "desk_outcomes is unreachable — schema not migrated; "
                    "deploy (render_start runs the idempotent DDL) or run "
                    "scripts/setup_db.py", "detail": str(exc)[:200]}
        raise
    out = outcomes(store, days=(days if run_id else GRADE_OPEN_LOOKBACK_DAYS),
                   run_id=run_id, account=account)
    trades = _trades(store, account)
    split_events = _booked_split_events(trades)
    today = _et_date(_utcnow()) or str(_utcnow().date())
    now = _utcnow()
    cutoff = str(now - _td(days=days))
    meta = _latest_mark_meta(store, account) or {}
    cost_marked = set(meta.get("cost_marked") or [])
    spy_cache: list | None = None

    def _spy():  # lazy: only exit reconstructions need a second SPY read
        nonlocal spy_cache
        if spy_cache is None:
            rd = [d for d in (_et_date(r.get("ts")) for r in out["runs"]) if d]
            spy_cache = _spy_closes(store, since=min(rd)) if rd else []
        return spy_cache

    graded: list[dict] = []
    skipped_closed = 0
    for run in out["runs"]:
        rid = run["run_id"]
        run_ts = str(run.get("ts") or "")
        run_date = _et_date(run.get("ts"))
        sessions = run.get("spy_window_sessions") or 0
        for p in run["picks"]:
            entry = p.get("entry_avg_px")
            if entry is None:
                continue
            sym = p["symbol"]
            is_opt = bool(p.get("is_option"))
            closed = p.get("closed_return_pct")
            since = closed if closed is not None else p.get("since_this_run_pct")
            spy_pct = p.get("spy_same_window_pct")
            alpha = p.get("alpha_pct")
            status = ("closed" if (closed is not None
                                   or p.get("open_now") is None) else "open")
            existing = store.select(
                "desk_outcomes",
                filters={"account": account, "run_id": rid, "symbol": sym},
                limit=1)
            # L4: a closed row outside the window with facts already stored
            # is final — skip the re-grade; open picks always refresh.
            if (not run_id and status == "closed" and existing
                    and existing[0].get("status") == "closed"
                    and run_ts and run_ts < cutoff):
                skipped_closed += 1
                continue
            exit_kind = exit_avg = realized = rec = None
            mark_basis = "exit" if closed is not None else "mark"
            if closed is not None:
                exit_kind = "same_run"
                exit_avg = round(entry * (1 + closed / 100), 4)
                realized = p.get("realized_pnl")
            elif status == "closed":
                # H3: flat book, no same-run round trip — the exit lives in
                # other rows (hard stop, later run, settlement). Reconstruct
                # it so a stop-out grades with numbers, not nulls.
                entry_fills = [t for t in trades
                               if t.get("run_id") == rid
                               and t["symbol"] == sym and t["side"] == "BUY"
                               and _fq(t).get("src") not in
                               ("split_adjustment", "dividend")]
                rec = _reconstruct_exit(trades, sym, entry_fills,
                                        split_events.get(sym), is_opt)
                if rec:
                    exit_avg = round(rec["exit_avg_px"], 4)
                    exit_kind = _exit_kind_for(rec["dominant_run"], rid)
                    since = round((rec["exit_avg_px"] - entry) / entry * 100, 2)
                    mark_basis = "exit"
                    espy = (_spy_window_pct(_spy(), run_date, rec["exit_date"])
                            if run_date and rec.get("exit_date") else None)
                    spy_pct = espy
                    alpha = (round(since - espy, 2)
                             if (espy is not None and not is_opt) else None)
                    if is_opt:
                        # fee-net proceeds vs fee-inclusive cost, both per
                        # share on the same basis
                        realized = round(rec["exit_units"]
                                         * (rec["exit_avg_px"] - entry)
                                         * MULTIPLIER, 2)
                    else:
                        sym_tr = [t for t in trades if t["symbol"] == sym]
                        last_entry = max(_trade_key(t) for t in entry_fills)
                        realized = round(
                            _realized_pnl([t for t in sym_tr if _trade_key(t)
                                           <= rec["flat_key"]])[1].get(sym, 0.0)
                            - _realized_pnl([t for t in sym_tr if _trade_key(t)
                                             <= last_entry])[1].get(sym, 0.0), 2)
            if mark_basis == "exit":
                mark_px = exit_avg
            else:
                mark_px = (round(entry * (1 + since / 100), 4)
                           if since is not None else None)
            degraded = False
            if mark_basis == "mark" and (
                    sym in cost_marked
                    or (p.get("open_now") or {}).get("mark_is_cost")):
                # M2: the latest snapshot priced this symbol at cost basis —
                # OR the projection row has no mark at all (rebuilt before
                # any mark() ran): either way the "mark" is the entry price,
                # fake-flat; write nulls, flag the row.
                since = alpha = mark_px = None
                degraded = True
            h = p.get("horizon_days")
            h = (int(h) if isinstance(h, (int, float))
                 and not isinstance(h, bool) else None)
            # kill plausibility runs against the RAW (as-booked) entry — the
            # kill was stated on the entry-day price basis
            raw_buys = [f for f in (p.get("fills") or [])
                        if f.get("side") == "BUY"]
            raw_units = sum(float(f["shares"]) for f in raw_buys)
            raw_entry = (sum(float(f["shares"]) * float(f["price"])
                             for f in raw_buys) / raw_units
                         if raw_units > 0 else None)
            kill_level = _parse_kill(p.get("kill"), raw_entry)
            breached = None
            if kill_level is not None and run_date:
                # For CLOSED picks the breach window ends at the EXIT — a
                # dip (or an unbooked post-exit split) AFTER the trade
                # finished must not stamp a discipline failure on it.
                kill_end = today
                if status == "closed":
                    kill_end = (p.get("exit_date")
                                or (rec or {}).get("exit_date") or today)
                breached = _kill_breached(store, sym, kill_level, run_date,
                                          kill_end, split_events.get(sym))
            values = {
                "grade_date": _date.fromisoformat(today),
                "entry_avg_px": round(entry, 4),
                "mark_px": mark_px,
                "mark_basis": mark_basis,
                "since_pct": since,
                "spy_pct": spy_pct,
                "alpha_pct": alpha,
                "exit_kind": exit_kind, "exit_avg_px": exit_avg,
                "realized_pnl": realized, "degraded": degraded,
                "horizon_days": h,
                "horizon_elapsed": (sessions >= h) if h else None,
                "kill_level": kill_level, "kill_breached": breached,
                "status": status, "graded_at": now}
            if existing:
                store.update("desk_outcomes", {"id": existing[0]["id"]},
                             values, returning=False)
            else:
                try:
                    store.insert("desk_outcomes",
                                 {"account": account, "run_id": rid,
                                  "symbol": sym, **values}, returning=False)
                except Exception as exc:  # noqa: BLE001 — race classifier
                    from agent.store import is_duplicate_key_error

                    if not is_duplicate_key_error(exc):
                        raise
                    # L6: a concurrent grade won the insert race on the
                    # (account, run_id, symbol) unique key → update instead.
                    rows = store.select(
                        "desk_outcomes",
                        filters={"account": account, "run_id": rid,
                                 "symbol": sym}, limit=1)
                    if rows:
                        store.update("desk_outcomes", {"id": rows[0]["id"]},
                                     values, returning=False)
            graded.append({"run_id": rid, "symbol": sym, "status": status,
                           "since_pct": since,
                           "alpha_pct": alpha,
                           "exit_kind": exit_kind,
                           "degraded": degraded or None,
                           "horizon_elapsed": (sessions >= h) if h else None,
                           "kill_level": kill_level,
                           "kill_breached": breached})
    # Same pass, same split-aware close data: sweep the structured
    # commitments (trim/exit falsification clauses) so a fired re-add promise
    # becomes a tracked obligation instead of silent free text.
    commitments = sweep_commitments(store, account=account,
                                    split_events=split_events, trades=trades,
                                    today=today)
    return {"ok": True, "as_of": today, "graded": len(graded),
            "closed_rows_outside_window": skipped_closed, "rows": graded,
            "commitments": commitments}


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


def _warn_if_degraded(st: dict) -> None:
    """A loud stderr warning when the latest mark is running on cost basis —
    stdout stays clean JSON for the caller."""
    import sys

    meta = st.get("mark_meta") or {}
    if meta.get("degraded"):
        print("WARNING: MARKS DEGRADED — "
              f"{meta.get('cost_marked_value_pct')}% of position value is "
              f"marked at COST BASIS ({', '.join(meta.get('cost_marked') or [])}"
              "); the equity point is fake-flat for those names. Fix the "
              "quote/close feed before trusting P&L.", file=sys.stderr)


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
    gd = sub.add_parser("grade",
                        help="materialize machine-graded pick facts into "
                             "desk_outcomes (run before reflection)")
    gd.add_argument("--days", type=int, default=30,
                    help="bounds CLOSED-row re-grades only; still-open picks "
                         "always refresh regardless of the window")
    gd.add_argument("--run-id", default=None)
    fil = sub.add_parser("fill", help="book a fill AT THE LIVE QUOTE (the agent's path)")
    fil.add_argument("--symbol", required=True)
    fil.add_argument("--side", required=True, choices=["buy", "sell", "BUY", "SELL"])
    fil.add_argument("--shares", default=None, type=float)
    fil.add_argument("--notional", default=None, type=float)
    fil.add_argument("--rationale", default=None)
    fil.add_argument("--run-id", default=None)
    fil.add_argument("--slippage-bp", default=SLIPPAGE_BP, type=float)
    fil.add_argument("--allow-price-deviation", action="store_true",
                     help="override the last-close sanity band (>20% from the "
                          "stored close) — use only for a real, named move")
    fil.add_argument("--allow-illiquid", action="store_true",
                     help="override the ADV size gate (notional >1% of the "
                          "20-session average dollar volume)")
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
        out = state(store)
        _warn_if_degraded(out)
        print(json.dumps(out, indent=2))
    elif args.cmd == "mark":
        out = mark(store)
        _warn_if_degraded(out)
        print(json.dumps(out, indent=2))
    elif args.cmd == "settle":
        print(json.dumps(settle(store), indent=2))
    elif args.cmd == "outcomes":
        print(json.dumps(outcomes(store, days=args.days, run_id=args.run_id),
                         indent=2))
    elif args.cmd == "grade":
        print(json.dumps(grade(store, days=args.days, run_id=args.run_id),
                         indent=2))
    elif args.cmd == "fill":
        print(json.dumps(live_fill(
            store, symbol=args.symbol, side=args.side, shares=args.shares,
            notional=args.notional, rationale=args.rationale, run_id=args.run_id,
            slippage_bp=args.slippage_bp,
            allow_price_deviation=args.allow_price_deviation,
            allow_illiquid=args.allow_illiquid), indent=2))
    elif args.cmd == "record":
        print(json.dumps(record_trade(
            store, symbol=args.symbol, side=args.side, shares=args.shares,
            price=args.price, rationale=args.rationale, run_id=args.run_id,
            latest_close=args.latest_close), indent=2))


if __name__ == "__main__":
    main()

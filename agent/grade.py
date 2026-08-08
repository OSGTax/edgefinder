"""Pick grading against the Alpaca paper book (REBUILD-V4).

The successor to ``agent.ledger``'s outcomes/grade/sweep machinery, with the
same OUTPUT semantics (``desk_outcomes`` columns keep their names and
meanings) and new SOURCES:

- fills come from the ``desk_orders`` mirror (Alpaca is the book of record;
  ``agent.trade reconcile`` keeps the mirror converged) joined to picks by
  the ``(run_id, symbol)`` key that ``client_order_id`` attribution
  preserves;
- open positions and marks come from Alpaca (``agent.trade`` positions —
  ``current_price``), passed in or fetched live;
- split rebasing reads ``ticker_splits`` (the ledger's own booked-split
  rows are gone — Alpaca rebases its positions silently, so OUR stored fill
  prices are what need adjusting);
- the SPY benchmark is **symmetric price return** (charter V4): the paper
  broker credits no dividends on the book, so the benchmark must not carry
  its dividends either — like-for-like, in the less flattering direction
  for SPY but the honest one for the comparison.

Grading conventions preserved: baseline strictly before the window's ET
start date; None means too-young-to-benchmark, never zero; alpha under
``spy_window_sessions < 2`` is benchmark noise; options carry
``alpha_pct = null`` (premium %-moves embed leverage/theta — grade them on
realized dollars and thesis); short-opened option picks (CSP / covered
call) enter at the credit received and profit as premium decays.

T+1 RULE (paper): option non-trade activities (OPEXP/OPASN/OPEXC/OPXRC)
land in the activities feed only the NEXT day, while positions update
instantly. Settlement is therefore detected from position-disappearance at
or after expiry, and the activity rows refine the row on a later pass —
grading never waits on same-day NTAs.

CLI (JSON out):
  python -m agent.grade run [--days N] [--run-id R]
  python -m agent.grade outcomes [--days N]
  python -m agent.grade sweep
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from agent.models import ACCOUNT

logger = logging.getLogger(__name__)

MULTIPLIER = 100  # option contract multiplier
EPS_UNITS = 1e-6
# Open picks are always re-graded; --days only bounds how far back CLOSED
# rows are refreshed (their facts are final once written).
GRADE_OPEN_LOOKBACK_DAYS = 3650

__all__ = ["outcomes", "grade", "sweep_commitments", "spy_price_closes",
           "fills_from_orders"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _store():
    from agent.store import get_store

    return get_store()


def _mult(symbol: str) -> int:
    from agent import occ

    return MULTIPLIER if occ.is_option(symbol) else 1


def _et_date(ts) -> str | None:
    """The ET calendar date of a timestamp (naive-UTC datetime or ISO text).
    Windows are trading-day windows: a 19:30 ET decision is already
    next-day in UTC, and dating it by the UTC calendar would baseline SPY
    off the NEXT session's close."""
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


# ── fills: the desk_orders mirror in replay shape ────────────────────────


def fills_from_orders(store, account: str = ACCOUNT) -> list[dict]:
    """Mirror rows with actual executions → replay-ordered fill dicts:
    ``{id, ts, run_id, symbol, side BUY|SELL, shares, price, dollars, kind}``.

    mleg PARENT rows are skipped (their legs carry the per-contract fills —
    counting both would double every spread). Partially-filled-then-canceled
    orders count for exactly what filled. ``dollars`` is qty × price × mult
    — the paper engine simulates no commissions, so there is no fee term.
    """
    rows = store.select("desk_orders", filters={"account": account})
    fills: list[dict] = []
    for r in rows:
        fq = r.get("filled_qty")
        px = r.get("filled_avg_price")
        if not fq or not px or float(fq) <= EPS_UNITS:
            continue
        if (r.get("order_class") == "mleg") and not r.get("parent_order_id"):
            continue  # parent shell — the legs carry the fills
        sym = r["symbol"]
        side = str(r.get("side") or "").upper()
        if side not in ("BUY", "SELL"):
            continue
        fills.append({
            "id": r.get("id"), "ts": r.get("filled_at") or r.get("submitted_at"),
            "run_id": r.get("run_id"), "symbol": sym, "side": side,
            "shares": float(fq), "price": float(px),
            "dollars": float(fq) * float(px) * _mult(sym),
            "kind": r.get("kind")})
    fills.sort(key=lambda f: (str(f.get("ts")), f.get("id") or 0))
    return fills


def _trade_key(t: dict) -> tuple:
    return (str(t.get("ts")), t.get("id") or 0)


# ── split events from ticker_splits ──────────────────────────────────────


def split_events_from_table(store, symbols: set[str] | None = None,
                            *, since: str | None = None
                            ) -> dict[str, list[tuple[str, float]]]:
    """{symbol: [(execution_date, to/from factor), ...]} from the
    ``ticker_splits`` market table. Alpaca rebases its own positions and
    order history silently; OUR stored fill prices are on the as-filled
    basis and need this to compare across a split."""
    filters: dict = {}
    if symbols is not None:
        if not symbols:
            return {}
        filters["symbol"] = ("in", sorted(symbols))
    if since:
        filters["execution_date"] = ("gte", since[:10])
    try:
        rows = store.select("ticker_splits",
                            columns="symbol,execution_date,split_from,split_to",
                            filters=filters or None)
    except Exception:  # noqa: BLE001 — no split table → no rebasing, not a crash
        return {}
    out: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        frm, to = r.get("split_from"), r.get("split_to")
        try:
            factor = float(to) / float(frm)
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        d = str(r.get("execution_date") or "")[:10]
        if d and factor > 0:
            out.setdefault(str(r["symbol"]).upper(), []).append((d, factor))
    for evs in out.values():
        evs.sort()
    return out


def _split_factor_since(events: list[tuple[str, float]] | None, d: str) -> float:
    """Cumulative to/from factor of splits executing STRICTLY AFTER ``d`` —
    a fill ON the execution date is already post-split (the tape rebases at
    the open)."""
    f = 1.0
    for ed, x in (events or []):
        if ed > d[:10]:
            f *= x
    return f


# ── SPY benchmark: symmetric price return (charter V4) ───────────────────


def spy_price_closes(store, *, since: str) -> list[tuple[str, float]]:
    """SPY daily closes (ascending) from ``daily_bars`` — PRICE RETURN, no
    dividend back-adjustment, deliberately: the Alpaca paper book receives
    no dividends, so a total-return SPY would hand the benchmark a yield
    the book cannot earn. Symmetry is the honesty principle; the missed
    dividends are disclosed on the dashboard instead. A lookback buffer
    keeps a weekend/holiday window start from losing its baseline."""
    from datetime import date as _date, timedelta as _td

    start = _date.fromisoformat(since[:10]) - _td(days=10)
    rows = store.select("daily_bars", columns="date,close",
                        filters={"symbol": "SPY", "date": ("gte", start)},
                        order=[("date", "asc")])
    return [(str(r["date"])[:10], float(r["close"]))
            for r in rows if r.get("close")]


def _spy_window_pct(spy: list[tuple[str, float]], start_date: str,
                    end_date: str | None = None) -> float | None:
    """SPY price change over a trading window, honestly bounded. Baseline:
    the last close STRICTLY BEFORE ``start_date``; endpoint: the last close
    on/before ``end_date`` when given, else the latest stored close. None
    when there is no baseline or no completed span — too young to
    benchmark, never zero."""
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


# ── realized P&L replay (avg-cost, closing-run attribution) ──────────────


def _realized_pnl(fills: list[dict]):
    """Replay the avg-cost stream and accumulate realized P&L on reductions.
    Returns ``(by_run_symbol, by_symbol)``; attribution goes to the run that
    booked the CLOSING fill, priced against the global average cost at that
    moment — approximate when several runs built the lot; per-symbol totals
    are exact. Equities clamp at flat (the account is long-only by server
    config — a phantom short here would desync every later attribution);
    option lots are signed (short legs are real, covered by construction)."""
    from agent import occ

    by_run_symbol: dict[tuple[str | None, str], float] = {}
    by_symbol: dict[str, float] = {}
    book: dict[str, dict] = {}
    for t in fills:
        sym = t["symbol"]
        b = book.setdefault(sym, {"units": 0.0, "cost": 0.0})
        qty = float(t["shares"])
        signed = qty if t["side"] == "BUY" else -qty
        cur = b["units"]
        if abs(cur) <= EPS_UNITS:
            if signed < 0 and not occ.is_option(sym):
                continue  # sell on a flat equity book — no lot, no P&L
            b["units"] = cur + signed
            b["cost"] += abs(signed) * float(t["price"])
            continue
        if (cur > 0) == (signed > 0):
            b["units"] = cur + signed
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
        b["units"] = cur + signed
        if abs(b["units"]) <= EPS_UNITS:
            b["units"], b["cost"] = 0.0, 0.0
        elif (b["units"] > 0) != (cur > 0):
            b["cost"] = abs(b["units"]) * float(t["price"])
        if not occ.is_option(sym) and b["units"] < 0:
            b["units"], b["cost"] = 0.0, 0.0
    return by_run_symbol, by_symbol


# ── positions: the Alpaca side of the join ───────────────────────────────


def _position_map(positions: list[dict] | None) -> dict[str, dict]:
    return {str(p.get("symbol") or "").upper(): p for p in (positions or [])
            if abs(float(p.get("qty") or 0.0)) > EPS_UNITS}


def _fetch_positions() -> tuple[list[dict] | None, str | None]:
    """Live Alpaca positions when trade creds exist; (None, why) otherwise —
    callers must treat None as 'marks unavailable', never as 'book flat'."""
    from agent.trade import Trade, trade_enabled

    if not trade_enabled():
        return None, "trade creds not set — marks unavailable"
    try:
        return Trade().positions(), None
    except Exception as exc:  # noqa: BLE001 — a dead broker degrades, loudly
        return None, f"positions fetch failed: {type(exc).__name__}: {exc}"


# ── outcomes: how past decisions actually aged ───────────────────────────


def outcomes(store=None, *, days: int = 30, run_id: str | None = None,
             account: str = ACCOUNT, positions: list[dict] | None = None,
             equity: float | None = None) -> dict:
    """Joins each decision's picks to its mirror fills (run_id + symbol) and
    reports entry basis, realized P&L, current open exposure and mark
    (Alpaca ``current_price``), ``since_this_run_pct``, closed round trips,
    and the SPY-price-return window + alpha. The grounding for grading and
    the wiki — same shape as the V3 ledger's outcomes, minus the fee
    machinery (paper simulates no commissions) and the settlement/hardstop
    run buckets (stops now book under the run that ARMED them; expiry books
    no fill at all — see grade()'s settlement handling)."""
    from agent import occ

    store = store or _store()
    pos_note = None
    if positions is None:
        positions, pos_note = _fetch_positions()
    pos_map = _position_map(positions)
    marks_available = positions is not None

    fills = fills_from_orders(store, account)
    by_run_symbol, by_symbol = _realized_pnl(fills)

    from datetime import timedelta as _td
    cutoff = _utcnow() - _td(days=days)
    if run_id:
        decisions = store.select("desk_decisions",
                                 filters={"account": account, "run_id": run_id})
    else:
        decisions = store.select("desk_decisions",
                                 filters={"account": account,
                                          "ts": ("gte", cutoff)},
                                 order=[("ts", "desc")])

    inception = _et_date(fills[0]["ts"]) if fills else None
    window_starts = [w for w in (_et_date(d.get("ts")) for d in decisions) if w]
    if inception:
        window_starts.append(inception)
    spy = spy_price_closes(store, since=min(window_starts)) if window_starts else []

    fills_by_run: dict[tuple[str | None, str], list[dict]] = {}
    for t in fills:
        fills_by_run.setdefault((t.get("run_id"), t["symbol"]), []).append(t)
    split_events = split_events_from_table(
        store, {t["symbol"] for t in fills} | set(pos_map),
        since=min(window_starts) if window_starts else None)

    runs = []
    for d in decisions:
        rid = d["run_id"]
        run_date = _et_date(d.get("ts"))
        run_spy_pct = _spy_window_pct(spy, run_date) if run_date else None
        picks_out = []
        for p in (d.get("picks") or []):
            sym = str(p.get("symbol") or "").upper()
            if sym == "BOOK":
                continue  # whole-book stance note — nothing to grade per-name
            raw_fills = fills_by_run.get((rid, sym), [])
            fills_disp = [{"side": f["side"], "shares": f["shares"],
                           "price": f["price"]} for f in raw_fills]
            evs = split_events.get(sym)
            adj = [{"side": f["side"],
                    "shares": f["shares"] * _split_factor_since(
                        evs, _et_date(f.get("ts")) or ""),
                    "price": f["price"] / _split_factor_since(
                        evs, _et_date(f.get("ts")) or "")}
                   for f in raw_fills] if evs else fills_disp
            buys = [f for f in adj if f["side"] == "BUY"]
            sells = [f for f in adj if f["side"] == "SELL"]
            is_opt = occ.is_option(sym)
            entry_avg = (sum(f["shares"] * f["price"] for f in buys)
                         / sum(f["shares"] for f in buys)) if buys else None
            # A pick that OPENED short (sold-to-open CSP / covered call)
            # enters at the credit received; deriving entry from BUY fills
            # would grade the round trip with entry and exit swapped.
            short_opened = bool(raw_fills) and raw_fills[0]["side"] == "SELL" \
                and is_opt
            buy_units = sum(f["shares"] for f in raw_fills if f["side"] == "BUY")
            buy_dollars = sum(f["dollars"] for f in raw_fills
                              if f["side"] == "BUY")
            sold_units = sum(f["shares"] for f in raw_fills
                             if f["side"] == "SELL")
            sell_dollars = sum(f["dollars"] for f in raw_fills
                               if f["side"] == "SELL")
            if is_opt:
                if short_opened and sold_units > 0 and sell_dollars > 0:
                    entry_avg = sell_dollars / (sold_units * MULTIPLIER)
                elif buy_units > 0 and buy_dollars > 0:
                    entry_avg = buy_dollars / (buy_units * MULTIPLIER)
            pos = pos_map.get(sym)
            mark = pos.get("current_price") if pos else None
            open_now = None
            if pos:
                qty = float(pos.get("qty") or 0.0)
                avg_entry = pos.get("avg_entry_price")
                open_now = {"shares": qty, "avg_price": avg_entry,
                            "last_price": mark,
                            # Alpaca returned no price — fake-flat, the M2
                            # degraded-mark guard nulls mark-derived facts
                            "mark_is_cost": mark is None,
                            "unrealized_pnl": pos.get("unrealized_pl")}
            since_pct = None
            if entry_avg and mark:
                chg = (entry_avg - mark) if short_opened else (mark - entry_avg)
                since_pct = round(chg / entry_avg * 100, 2)
            closed_pct = exit_spy_pct = exit_date = None
            bought = sum(f["shares"] for f in buys)
            sold = sum(f["shares"] for f in sells)
            if buys and sells and abs(bought - sold) <= EPS_UNITS:
                if is_opt and short_opened and sell_dollars > 0:
                    closed_pct = round((sell_dollars - buy_dollars)
                                       / sell_dollars * 100, 2)
                elif is_opt and buy_dollars > 0:
                    closed_pct = round((sell_dollars - buy_dollars)
                                       / buy_dollars * 100, 2)
                else:
                    sell_avg = sum(f["shares"] * f["price"] for f in sells) / sold
                    closed_pct = round((sell_avg - entry_avg) / entry_avg * 100, 2)
                exit_date = _et_date(raw_fills[-1].get("ts"))
                if run_date and exit_date:
                    exit_spy_pct = _spy_window_pct(spy, run_date, exit_date)
            realized = round(by_run_symbol.get((rid, sym), 0.0), 2)
            live_pct = closed_pct if closed_pct is not None else since_pct
            spy_pct = exit_spy_pct if closed_pct is not None else run_spy_pct
            # Options: premium %-moves carry leverage and theta — subtracting
            # an index move from them is not alpha.
            alpha = (round(live_pct - spy_pct, 2)
                     if (live_pct is not None and spy_pct is not None
                         and not is_opt) else None)
            picks_out.append({
                "symbol": sym, "action": p.get("action"), "is_option": is_opt,
                "why_now": p.get("why_now"), "rationale": p.get("rationale"),
                "prediction": p.get("prediction"),
                "horizon_days": p.get("horizon_days"), "kill": p.get("kill"),
                "claims": p.get("claims"),
                "fills": fills_disp,
                "entry_avg_px": round(entry_avg, 4) if entry_avg else None,
                "short_opened": short_opened or None,
                "realized_pnl": realized,
                "open_now": open_now, "since_this_run_pct": since_pct,
                "closed_return_pct": closed_pct, "exit_date": exit_date,
                "spy_same_window_pct": spy_pct, "alpha_pct": alpha})
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
                         v for (r, _), v in by_run_symbol.items()
                         if r == rid), 2)})

    unattributed = sum(1 for t in fills if not t.get("run_id"))
    symbols_out = []
    for sym in sorted(set(by_symbol) | set(pos_map)):
        pos = pos_map.get(sym)
        symbols_out.append({
            "symbol": sym, "realized_pnl": round(by_symbol.get(sym, 0.0), 2),
            "unrealized_pnl": (pos.get("unrealized_pl") if pos else 0.0),
            "open_shares": float(pos.get("qty")) if pos else 0.0,
            "is_option": occ.is_option(sym)})
    book = None
    if inception and equity is not None:
        from config.settings import settings

        start_cap = float(settings.starting_capital)
        book_pct = (round((equity - start_cap) / start_cap * 100, 2)
                    if start_cap else None)
        spy_pct = _spy_window_pct(spy, inception)
        book = {"inception": inception, "since_inception_pct": book_pct,
                "spy_since_inception_pct": spy_pct,
                "alpha_pct": (round(book_pct - spy_pct, 2)
                              if (book_pct is not None and spy_pct is not None)
                              else None)}

    return {"as_of": str(_utcnow()), "days": days,
            "convention": "Fills come from the Alpaca-paper mirror "
                          "(desk_orders), joined to picks by (run_id, "
                          "symbol) via client_order_id attribution. Realized "
                          "P&L is attributed to the run that booked the "
                          "CLOSING fill, priced against the global average "
                          "cost at that moment — approximate when several "
                          "runs built the lot; per-symbol totals are exact. "
                          "since_this_run_pct compares the Alpaca mark "
                          "(current_price) to that run's own average fill "
                          "and is exact per pick; entry_avg_px and round-"
                          "trip matching are expressed on the CURRENT share "
                          "basis (ticker_splits events between a fill and "
                          "the mark rescale that run's own fills). "
                          "spy_same_window_pct / alpha_pct benchmark each "
                          "window against SPY PRICE-RETURN closes — "
                          "symmetric with the book, which the paper broker "
                          "pays no dividends into (charter V4; the missed "
                          "dividends are disclosed, not embedded). Baseline "
                          "= last close STRICTLY BEFORE the window's ET "
                          "start; endpoint = exit-day close for closed "
                          "round trips, else the latest stored close. None "
                          "means too-young-to-benchmark, never zero. Alpha "
                          "under spy_window_sessions < 2 is inside "
                          "benchmark noise — do not grade it as skill. "
                          "Options carry alpha_pct = null by design. Stop "
                          "exits book under the run that ARMED the stop; "
                          "option expiry books no fill (see grade's "
                          "settlement facts). A long book's raw P&L is "
                          "mostly market beta — grade alpha, not dollars.",
            "book": book, "runs": runs, "symbols": symbols_out,
            "marks_available": marks_available,
            "positions_note": pos_note,
            "unattributed_fills": unattributed}


# ── exit reconstruction (cross-run / stop exits) ─────────────────────────


def _exit_kind_for(dominant_run, pick_run_id: str,
                   closing_fills: list[dict]) -> str:
    """same_run | cross_run | hardstop, by the closing fills. A protective
    stop books under the run that ARMED it (client_order_id attribution),
    so the stop is recognized by the mirrored order's kind, not by a magic
    run id."""
    if any((f.get("kind") == "stop") for f in closing_fills):
        return "hardstop"
    return "same_run" if str(dominant_run or "") == pick_run_id else "cross_run"


def _reconstruct_exit(fills: list[dict], sym: str, entry_fills: list[dict],
                      split_events: list[tuple[str, float]] | None,
                      is_opt: bool) -> dict | None:
    """How a pick's position actually CLOSED when the closing fills lived
    outside its own run (a stop, a later run's exit). Walks the symbol's
    fills in replay order, finds the first flat point strictly after the
    entry fills, and averages every closing SELL up to it — any run_id,
    split-adjusted onto the current basis. None when the position never
    went flat (still open) or no closing sells exist (expiry settlement —
    grade handles that separately)."""
    if not entry_fills:
        return None
    last_entry = max(_trade_key(t) for t in entry_fills)
    running = 0.0
    closing: list[dict] = []
    flat_at = None
    for t in fills:
        if t["symbol"] != sym:
            continue
        qty = float(t["shares"])
        running += qty if t["side"] == "BUY" else -qty
        if _trade_key(t) <= last_entry:
            continue
        if t["side"] == "SELL" and qty > EPS_UNITS:
            closing.append(t)
        if running <= EPS_UNITS:
            flat_at = t
            break
    if flat_at is None or not closing:
        return None
    units = value = 0.0
    run_units: dict = {}
    for t in closing:
        f = _split_factor_since(split_events, _et_date(t.get("ts")) or "")
        sh = float(t["shares"]) * f
        px = float(t["price"]) if is_opt else float(t["price"]) / f
        units += sh
        value += sh * px
        rid = t.get("run_id")
        run_units[rid] = run_units.get(rid, 0.0) + sh
    if units <= EPS_UNITS:
        return None
    return {"exit_avg_px": value / units, "exit_units": units,
            "exit_date": _et_date(flat_at.get("ts")),
            "flat_key": _trade_key(flat_at),
            "dominant_run": max(run_units, key=lambda r: run_units[r]),
            "closing_fills": closing}


# ── kill / commitment level checks (daily_bars, split-aware) ─────────────


def _plausible_kill(level: float | None, entry_px: float | None) -> float | None:
    """Long-only stop plausibility: a parsed level outside [0.2×, 2×] of the
    entry is a parse artifact (a year, a share count), not a stop."""
    if level is None or level <= 0:
        return None
    if entry_px and entry_px > 0 \
            and not (0.2 * entry_px <= level <= 2.0 * entry_px):
        return None
    return level


def _parse_kill(kill, entry_px: float | None = None) -> float | None:
    """A pick's free-text ``kill`` as a price level, when unambiguous.
    Percentages and indicator/time-unit numbers ("8%", "100DMA",
    "10 sessions") are excluded; a single $-prefixed number wins, else a
    single bare number; ambiguity → None (the reflection judges the free
    text itself) rather than a confidently wrong fact."""
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
            continue
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
    kill (close <= level, long-book semantics)? The kill was stated on the
    entry-day price basis, so splits between entry and a close rebase the
    level. None when no closes are stored — nothing to judge, never a fake
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


def _commitment_breach(store, symbol: str, direction: str, level: float,
                       start_date: str, end_date: str,
                       split_events: list[tuple[str, float]] | None = None
                       ) -> tuple[str, float] | None:
    """First stored daily close touching the commitment's level in its
    direction — (date, close), else None. Two-sided sibling of
    ``_kill_breached``."""
    from datetime import date as _date

    try:
        rows = store.select(
            "daily_bars", columns="date,close",
            filters={"symbol": symbol,
                     "date": [("gte", _date.fromisoformat(start_date[:10])),
                              ("lte", _date.fromisoformat(end_date[:10]))]},
            order=[("date", "asc")])
    except Exception:  # noqa: BLE001
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
                      today: str | None = None) -> dict:
    """Machine-check open commitments (structured trim/exit falsification
    clauses) against stored closes. A touched level flips to ``fired`` with
    the breaching date + close; a passed deadline flips to ``expired``.
    Fired-and-unhonored commitments surface in brain.context as
    obligations. Idempotent — only ``open`` rows are swept."""
    from datetime import date as _date

    store = store or _store()
    try:
        rows = store.select("desk_commitments",
                            filters={"account": account, "status": "open"},
                            order=[("id", "asc")], limit=500)
    except Exception as exc:  # noqa: BLE001 — pre-deploy grace
        from agent.store import is_missing_table_error

        if is_missing_table_error(exc):
            return {"ok": True, "swept": 0, "fired": 0, "expired": 0,
                    "note": "desk_commitments not migrated yet — skipped"}
        raise
    if not rows:
        return {"ok": True, "swept": 0, "fired": 0, "expired": 0}

    if split_events is None:
        split_events = split_events_from_table(
            store, {str(c.get("symbol")).upper() for c in rows})
    today = today or _et_date(_utcnow()) or str(_utcnow().date())

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
                         {"status": "fired",
                          "fired_date": _date.fromisoformat(hit[0]),
                          "fired_close": hit[1]}, returning=False)
            fired += 1
        elif until and today > until:
            store.update("desk_commitments", {"id": c["id"]},
                         {"status": "expired"}, returning=False)
            expired += 1
    return {"ok": True, "swept": len(rows), "fired": fired, "expired": expired}


# ── grade: materialize machine facts into desk_outcomes ──────────────────


def _settlement_facts(store, sym: str, entry_avg: float, raw_fills: list[dict],
                      account: str) -> dict | None:
    """Facts for an option pick whose position vanished WITHOUT closing
    fills — expiry settlement on the Alpaca side. OPEXP (expired worthless)
    grades at 0; OPASN/OPEXC/OPXRC (assignment/exercise) leave exit price
    facts null until the T+1 activity rows carry enough to refine — partial
    facts now beat wrong facts forever. Detection is position-absence at/
    after expiry (positions update instantly on paper); the activity rows
    land next day and refine the SAME row on the next grade pass."""
    from datetime import date as _date

    from agent import occ

    try:
        expiry = occ.parse(sym)["expiry"]
    except ValueError:
        return None
    from agent.broker import _today_et

    if expiry >= _today_et():
        return None  # not expired — position absence means something else
    acts = []
    try:
        acts = store.select("desk_activities",
                            filters={"account": account, "symbol": sym},
                            order=[("date", "asc")])
    except Exception:  # noqa: BLE001 — mirror not synced yet → detection only
        acts = []
    kinds = {a.get("activity_type") for a in acts}
    if "OPEXP" in kinds or not kinds & {"OPASN", "OPEXC", "OPXRC"}:
        # Expired worthless (or no NTA row yet — T+1): a LONG option's value
        # went to 0; a short-opened pick kept the whole credit.
        exit_px = 0.0
        return {"exit_avg_px": exit_px, "exit_kind": "settlement",
                "exit_date": expiry.isoformat(),
                "refined": "OPEXP" in kinds}
    # Assignment/exercise: the per-contract close-out value isn't in the
    # position stream; leave price facts null, keep the classification.
    return {"exit_avg_px": None, "exit_kind": "settlement",
            "exit_date": expiry.isoformat(), "refined": True}


def grade(store=None, *, days: int = 30, run_id: str | None = None,
          account: str = ACCOUNT, positions: list[dict] | None = None) -> dict:
    """Materialize per-pick MACHINE FACTS into ``desk_outcomes`` — one row
    per (account, run_id, symbol), updated in place; the reflection's
    ``verdict``/``verdict_note`` columns are never touched here
    (``agent.brain verdict`` is their only writer).

    Preserved from V3: only picks with entry (BUY, or sell-to-open) fills
    in their own run are graded; ``days`` bounds CLOSED-row re-grades only
    (open picks never age out); cross-run/stop exits are reconstructed from
    actual closing fills; a mark Alpaca returned no price for writes NULL
    mark-facts with ``degraded`` set (fake-flat must not grade a pick);
    kill windows end at the exit for closed picks. New here: exit_kind
    ``hardstop`` is recognized from the closing order's kind (stops book
    under the run that ARMED them), ``settlement`` from position-absence at
    expiry with T+1 activity refinement, and ``cutover`` rows (written once
    by the Era-1 freeze) are never re-graded."""
    from datetime import date as _date, timedelta as _td

    store = store or _store()
    try:
        store.select("desk_outcomes", filters={"account": account}, limit=1)
    except Exception as exc:  # noqa: BLE001 — classify, then re-raise others
        from agent.store import is_missing_table_error

        if is_missing_table_error(exc):
            return {"ok": False, "error":
                    "desk_outcomes is unreachable — schema not migrated; "
                    "deploy or run scripts/setup_db.py",
                    "detail": str(exc)[:200]}
        raise
    pos_note = None
    if positions is None:
        positions, pos_note = _fetch_positions()
    out = outcomes(store, days=(days if run_id else GRADE_OPEN_LOOKBACK_DAYS),
                   run_id=run_id, account=account, positions=positions or [])
    marks_available = positions is not None
    fills = fills_from_orders(store, account)
    split_events = split_events_from_table(
        store, {t["symbol"] for t in fills})
    today = _et_date(_utcnow()) or str(_utcnow().date())
    now = _utcnow()
    cutoff = str(now - _td(days=days))
    spy_cache: list | None = None

    def _spy():
        nonlocal spy_cache
        if spy_cache is None:
            rd = [d for d in (_et_date(r.get("ts")) for r in out["runs"]) if d]
            spy_cache = spy_price_closes(store, since=min(rd)) if rd else []
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
            # Open/closed: an open position keeps the row open; with marks
            # UNAVAILABLE (no positions read) absence proves nothing — the
            # row is left as-is rather than falsely closed.
            if closed is None and p.get("open_now") is None \
                    and not marks_available:
                continue
            status = ("closed" if (closed is not None
                                   or p.get("open_now") is None) else "open")
            existing = store.select(
                "desk_outcomes",
                filters={"account": account, "run_id": rid, "symbol": sym},
                limit=1)
            if existing and existing[0].get("exit_kind") == "cutover":
                continue  # Era-1 freeze rows are final by construction
            if (not run_id and status == "closed" and existing
                    and existing[0].get("status") == "closed"
                    and run_ts and run_ts < cutoff):
                skipped_closed += 1
                continue
            exit_kind = exit_avg = realized = rec = None
            mark_basis = "exit" if closed is not None else "mark"
            if closed is not None:
                # A same-run round trip — but if the closing fill was the
                # protective stop this run ARMED, the honest label is
                # hardstop, not same_run: the exit was mechanical.
                own_sells = [t for t in fills if t.get("run_id") == rid
                             and t["symbol"] == sym and t["side"] == "SELL"]
                exit_kind = ("hardstop" if any(t.get("kind") == "stop"
                                               for t in own_sells)
                             else "same_run")
                exit_avg = round(entry * (1 + closed / 100), 4)
                realized = p.get("realized_pnl")
            elif status == "closed":
                entry_fills = [t for t in fills
                               if t.get("run_id") == rid
                               and t["symbol"] == sym and t["side"] == "BUY"]
                if not entry_fills and p.get("short_opened"):
                    entry_fills = [t for t in fills if t.get("run_id") == rid
                                   and t["symbol"] == sym]
                rec = _reconstruct_exit(fills, sym, entry_fills,
                                        split_events.get(sym), is_opt)
                if rec:
                    exit_avg = round(rec["exit_avg_px"], 4)
                    exit_kind = _exit_kind_for(rec["dominant_run"], rid,
                                               rec["closing_fills"])
                    since = round((rec["exit_avg_px"] - entry) / entry * 100, 2)
                    mark_basis = "exit"
                    espy = (_spy_window_pct(_spy(), run_date, rec["exit_date"])
                            if run_date and rec.get("exit_date") else None)
                    spy_pct = espy
                    alpha = (round(since - espy, 2)
                             if (espy is not None and not is_opt) else None)
                    realized = round(rec["exit_units"]
                                     * (rec["exit_avg_px"] - entry)
                                     * _mult(sym), 2)
                elif is_opt:
                    st = _settlement_facts(store, sym, entry, [], account)
                    if st:
                        exit_kind = "settlement"
                        exit_avg = st["exit_avg_px"]
                        mark_basis = "exit"
                        if exit_avg is not None:
                            chg = ((entry - exit_avg) if p.get("short_opened")
                                   else (exit_avg - entry))
                            since = round(chg / entry * 100, 2)
                            units = sum(f["shares"] for f in
                                        (p.get("fills") or [])
                                        if f["side"] == ("SELL" if
                                                         p.get("short_opened")
                                                         else "BUY"))
                            realized = round(units * chg * MULTIPLIER, 2)
                        else:
                            since = None
                        spy_pct = alpha = None
            if mark_basis == "exit":
                mark_px = exit_avg
            else:
                mark_px = (round(entry * (1 + since / 100), 4)
                           if since is not None else None)
            degraded = False
            if mark_basis == "mark" and (p.get("open_now") or {}).get(
                    "mark_is_cost"):
                since = alpha = mark_px = None
                degraded = True
            h = p.get("horizon_days")
            h = (int(h) if isinstance(h, (int, float))
                 and not isinstance(h, bool) else None)
            raw_buys = [f for f in (p.get("fills") or [])
                        if f.get("side") == "BUY"]
            raw_units = sum(float(f["shares"]) for f in raw_buys)
            raw_entry = (sum(float(f["shares"]) * float(f["price"])
                             for f in raw_buys) / raw_units
                         if raw_units > 0 else None)
            kill_level = _parse_kill(p.get("kill"), raw_entry)
            breached = None
            if kill_level is not None and run_date:
                kill_end = today
                if status == "closed":
                    kill_end = (p.get("exit_date")
                                or (rec or {}).get("exit_date") or today)
                breached = _kill_breached(store, sym, kill_level, run_date,
                                          kill_end, split_events.get(sym))
            values = {
                "grade_date": _date.fromisoformat(today),
                "entry_avg_px": round(entry, 4),
                "mark_px": mark_px, "mark_basis": mark_basis,
                "since_pct": since, "spy_pct": spy_pct, "alpha_pct": alpha,
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
                    rows = store.select(
                        "desk_outcomes",
                        filters={"account": account, "run_id": rid,
                                 "symbol": sym}, limit=1)
                    if rows:
                        store.update("desk_outcomes", {"id": rows[0]["id"]},
                                     values, returning=False)
            graded.append({"run_id": rid, "symbol": sym, "status": status,
                           "since_pct": since, "alpha_pct": alpha,
                           "exit_kind": exit_kind,
                           "degraded": degraded or None,
                           "horizon_elapsed": (sessions >= h) if h else None,
                           "kill_level": kill_level,
                           "kill_breached": breached})
    commitments = sweep_commitments(store, account=account,
                                    split_events=split_events, today=today)
    return {"ok": True, "as_of": today, "graded": len(graded),
            "closed_rows_outside_window": skipped_closed, "rows": graded,
            "marks_available": marks_available, "positions_note": pos_note,
            "commitments": commitments}


def save_backtest(label: str, result: dict, *, run_id: str | None = None,
                  account: str = ACCOUNT) -> int:
    """Persist a backtest the agent ran (evidence panel reads desk_backtests).
    Lived in the V3 ledger; homed here with the rest of the evidence layer."""
    rows = _store().insert("desk_backtests", {
        "account": account, "run_id": run_id, "label": label,
        "spec": {k: result.get(k) for k in ("rule", "symbols", "schedule",
                                            "start", "end")},
        "result": {k: result.get(k) for k in (
            "return_pct", "sharpe", "max_drawdown_pct", "benchmark_return_pct",
            "excess_return_pct", "num_trades", "days", "final_equity")},
        "ts": _utcnow()})
    return int(rows[0]["id"]) if rows else 0


# ── CLI ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("run", help="grade picks → desk_outcomes")
    g.add_argument("--days", type=int, default=30)
    g.add_argument("--run-id", default=None)
    o = sub.add_parser("outcomes", help="how past decisions aged (read-only)")
    o.add_argument("--days", type=int, default=30)
    o.add_argument("--run-id", default=None)
    sub.add_parser("sweep", help="sweep open commitments only")
    args = p.parse_args(argv)

    if args.cmd == "run":
        out = grade(days=args.days, run_id=args.run_id)
    elif args.cmd == "outcomes":
        equity = None
        try:
            from agent.trade import Trade, trade_enabled

            if trade_enabled():
                equity = Trade().account().get("equity")
        except Exception:  # noqa: BLE001 — book section degrades to absent
            equity = None
        out = outcomes(days=args.days, run_id=args.run_id, equity=equity)
    else:
        out = sweep_commitments()
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

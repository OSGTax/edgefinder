"""Options intelligence — the chain summary + the growing IV data bank.

Two jobs:

1. **Live chain summary** (``get_summary``): pull the OPRA chain for an
   underlying via ``agent.broker`` and reduce it to what a trader actually
   reads — spot, the focus expiry, ATM IV, the straddle-implied expected
   move, 25-delta skew, and a strikes table around the money. Served to the
   desk/symbol pages by ``/api/desk/options/{symbol}`` (60s in-process cache
   so page loads don't hammer OPRA).

2. **The data bank** (``persist_snapshot``): one row per (underlying, day)
   into ``desk_options_snap`` — ATM IV, expected move, skew, spot. Written
   automatically by ``agent.refresh --source alpaca`` (i.e. every agent
   cycle, first one of the day wins), so IV history accumulates from today
   forward and the charts gain an IV series the agent can also use for
   IV-rank reasoning as the bank grows.

The reduction math is pure (``summarize_chain``) and unit-tested; only
``get_summary`` touches the network.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

# focus expiries at least this many days out (skip 0-2 DTE noise)
MIN_FOCUS_DTE = 3
# strikes table width around spot
TABLE_MONEYNESS = 0.10
_CACHE_TTL = 60.0
_cache: dict[str, tuple[float, dict]] = {}


def _mid(row: dict) -> float | None:
    if row.get("mid") is not None:
        return row["mid"]
    b, a = row.get("bid"), row.get("ask")
    return round((b + a) / 2, 4) if (b and a) else None


def summarize_chain(rows: list[dict], spot: float, *,
                    today: date | None = None) -> dict:
    """Reduce broker.option_chain rows to the trader's read (pure).

    Returns {spot, expiry, dte, atm_strike, atm_iv, expected_move_pct,
    expected_move_dollars, skew_25d, calls_table, puts_table, expiries}.
    """
    today = today or date.today()
    out: dict = {"spot": spot, "available": bool(rows), "expiries": []}
    if not rows or not spot:
        out["available"] = False
        return out

    by_expiry: dict[str, list[dict]] = {}
    for r in rows:
        by_expiry.setdefault(r["expiry"], []).append(r)
    out["expiries"] = sorted(by_expiry)

    focus = next((e for e in sorted(by_expiry)
                  if (date.fromisoformat(e) - today).days >= MIN_FOCUS_DTE),
                 sorted(by_expiry)[-1])
    frows = by_expiry[focus]
    dte = (date.fromisoformat(focus) - today).days
    out.update(expiry=focus, dte=dte)

    calls = [r for r in frows if r["type"] == "C"]
    puts = [r for r in frows if r["type"] == "P"]
    strikes = sorted({r["strike"] for r in frows})
    if not strikes:
        out["available"] = False
        return out
    atm = min(strikes, key=lambda k: abs(k - spot))
    out["atm_strike"] = atm

    def at(rows_, strike):
        return next((r for r in rows_ if r["strike"] == strike), None)

    ac, ap = at(calls, atm), at(puts, atm)
    ivs = [r["iv"] for r in (ac, ap) if r and r.get("iv")]
    out["atm_iv"] = round(sum(ivs) / len(ivs), 4) if ivs else None

    # expected move ≈ ATM straddle cost / spot
    cm, pm = (_mid(ac) if ac else None), (_mid(ap) if ap else None)
    if cm and pm:
        out["expected_move_dollars"] = round(cm + pm, 2)
        out["expected_move_pct"] = round((cm + pm) / spot * 100, 2)
    else:
        out["expected_move_dollars"] = out["expected_move_pct"] = None

    # 25-delta skew: put IV − call IV at |delta| nearest 0.25 (put fear gauge)
    def d25(rows_):
        cands = [r for r in rows_ if r.get("delta") is not None and r.get("iv")]
        return min(cands, key=lambda r: abs(abs(r["delta"]) - 0.25), default=None)

    c25, p25 = d25(calls), d25(puts)
    out["skew_25d"] = (round(p25["iv"] - c25["iv"], 4)
                       if (c25 and p25) else None)

    lo, hi = spot * (1 - TABLE_MONEYNESS), spot * (1 + TABLE_MONEYNESS)

    def table(rows_):
        return [{k: r.get(k) for k in
                 ("symbol", "strike", "bid", "ask", "iv", "delta", "theta")}
                for r in sorted(rows_, key=lambda r: r["strike"])
                if lo <= r["strike"] <= hi]

    out["calls_table"] = table(calls)
    out["puts_table"] = table(puts)
    return out


def get_summary(symbol: str, *, dte_max: int = 45) -> dict:
    """Live chain summary for an underlying (60s cached). Graceful when keys
    are absent or the chain fails: {"available": False, "error": ...}."""
    symbol = symbol.upper().strip()
    now = time.time()
    hit = _cache.get(symbol)
    if hit and now - hit[0] < _CACHE_TTL:
        return hit[1]
    try:
        from agent import broker

        if not broker.enabled():
            return {"available": False, "symbol": symbol,
                    "error": "no market-data keys on this host"}
        b = broker.Broker()
        uq = b.quotes([symbol]).get(symbol) or {}
        spot = uq.get("mid") or uq.get("ask") or uq.get("bid")
        rows = b.option_chain(symbol, dte_max=dte_max)
        out = summarize_chain(rows, spot)
        out["symbol"] = symbol
        out["as_of"] = datetime.now(timezone.utc).isoformat()
        _cache[symbol] = (now, out)
        return out
    except Exception as exc:  # noqa: BLE001 — the page must degrade, not 500
        logger.warning("options summary failed for %s: %s", symbol, exc)
        return {"available": False, "symbol": symbol,
                "error": f"{type(exc).__name__}: {exc}"}


def persist_snapshot(store, summary: dict, *, snap_date: date | None = None) -> bool:
    """Write one desk_options_snap row per (underlying, day) — the data bank.
    First write of the day wins; later calls are no-ops (one canonical row per
    day, on purpose — and the refresh's session gate makes that row an RTH
    read, with ``captured_at`` as the receipt). Returns True if written."""
    if not summary.get("available"):
        return False
    snap_date = snap_date or date.today()
    sym = summary["symbol"]
    existing = store.select("desk_options_snap",
                            filters={"symbol": sym, "snap_date": snap_date}, limit=1)
    if existing:
        return False
    store.insert("desk_options_snap", {
        "symbol": sym, "snap_date": snap_date, "spot": summary.get("spot"),
        "atm_iv": summary.get("atm_iv"),
        "expected_move_pct": summary.get("expected_move_pct"),
        "skew_25d": summary.get("skew_25d"),
        "dte": summary.get("dte"), "expiry": summary.get("expiry"),
        "captured_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }, returning=False)
    return True


def history(store, symbol: str, limit: int = 250) -> list[dict]:
    """The IV data bank series for one underlying, oldest→newest."""
    rows = store.select("desk_options_snap", filters={"symbol": symbol.upper()},
                        order=[("snap_date", "desc")], limit=limit)
    rows.reverse()
    return [{"date": str(r["snap_date"])[:10], "spot": r.get("spot"),
             "atm_iv": r.get("atm_iv"),
             "expected_move_pct": r.get("expected_move_pct"),
             "skew_25d": r.get("skew_25d"), "dte": r.get("dte")} for r in rows]

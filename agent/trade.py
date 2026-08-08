"""Alpaca PAPER-account order lifecycle — the desk's execution arm (REBUILD-V4).

The paper account IS the book of record: fills, positions, cash, equity,
corporate actions, and option expiry are Alpaca's. This module submits and
manages orders, mirrors everything it sees into local tables
(``desk_orders`` / ``desk_activities`` / ``desk_portfolio_history`` — Alpaca's
retention of closed history is undocumented, so we keep our own copy), and
serves the account reads the brain and dashboard consume.

PAPER-ONLY BY CONSTRUCTION, three layers deep:
1. Trade credentials are their own settings (``EDGEFINDER_ALPACA_TRADE_KEY``/
   ``_SECRET``) with NO fallback to the data keys — a paper key pair cannot
   authenticate against the live API at all.
2. ``Trade`` refuses to construct when ``settings.alpaca_paper`` is falsy.
3. The SDK client is always built ``paper=True`` — never from a variable.

ATTRIBUTION (the knowledge loop hangs on this): every order we submit stamps
``client_order_id = "<run_id>:<seq>"``. The symbol is deliberately NOT encoded
— it is recovered from the order/leg object — so OCC contracts and crypto
pairs (BTC/USD) need no escaping, and grade()'s ``(run_id, symbol)`` joins
survive the migration byte-for-byte. mleg legs mirror with
``parent_order_id`` set and the parent's run_id/seq propagated.

The mirror is a cache/archive, never the arbiter: ``reconcile`` re-syncs at
every cycle start and Alpaca wins on conflict.

CLI (JSON out — the trading skill drives these via Bash):
  python -m agent.trade account | positions | state
  python -m agent.trade submit --symbol NVDA --side buy --notional 5000 \
      --type market --run-id 2026-08-17T14:30-r7kq
  python -m agent.trade submit --legs '[{"symbol":"...","ratio_qty":1,...}]' \
      --qty 1 --type limit --limit-price 1.25 --run-id <RID>
  python -m agent.trade arm-stop --symbol NVDA --stop-price 150 --run-id <RID>
  python -m agent.trade reconcile [--since ISO]
  python -m agent.trade activities --sync
  python -m agent.trade probe --suite cutover
"""

from __future__ import annotations

import json
import logging
import time as _time

from config.settings import settings

logger = logging.getLogger(__name__)

# Order statuses we treat as final — polling stops here.
TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "replaced"}
# A GTC order silently auto-cancels at 90 days on Alpaca; warn well before.
GTC_WARN_AGE_DAYS = 80
# Default seconds `submit` polls for a terminal status before returning
# whatever state the order is in (the order stays working either way).
SUBMIT_POLL_SECS = 10

__all__ = [
    "Trade", "trade_enabled", "resolve_trade_creds", "make_client_order_id",
    "parse_client_order_id", "asset_class_of", "validate_order",
    "normalize_order_full", "normalize_activity", "state",
]


def resolve_trade_creds() -> dict:
    """Paper TRADE credentials — settings only, deliberately no env fallback
    to the data keys. Empty strings when unset."""
    return {"key": (settings.alpaca_trade_key or "").strip(),
            "secret": (settings.alpaca_trade_secret or "").strip()}


def trade_enabled() -> bool:
    c = resolve_trade_creds()
    return bool(c["key"] and c["secret"])


# ── pure helpers (SDK-free, network-free → unit-tested) ──────


def make_client_order_id(run_id: str, seq: int) -> str:
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required for a client_order_id")
    if not (1 <= int(seq) <= 99):
        raise ValueError(f"seq out of range 1-99: {seq}")
    cid = f"{rid}:{int(seq):02d}"
    if len(cid) > 128:
        raise ValueError(f"client_order_id too long ({len(cid)} > 128): {cid}")
    return cid


def parse_client_order_id(cid: str) -> dict | None:
    """``"<run_id>:<seq>"`` → {run_id, seq}, or None for a foreign id
    (Alpaca auto-generates ids for mleg legs and dashboard orders)."""
    if not cid or ":" not in cid:
        return None
    rid, _, tail = cid.rpartition(":")
    if not rid or not tail.isdigit() or len(tail) != 2:
        return None
    return {"run_id": rid, "seq": int(tail)}


def asset_class_of(symbol: str) -> str:
    """'crypto' | 'us_option' | 'us_equity' from the symbol shape — the same
    tells the rest of the codebase uses (slash = crypto, OCC = option)."""
    from agent import occ
    from agent.broker import is_crypto

    s = (symbol or "").strip().upper()
    if is_crypto(s):
        return "crypto"
    if occ.is_option(s):
        return "us_option"
    return "us_equity"


def _is_whole(x) -> bool:
    try:
        return float(x) == int(float(x))
    except (TypeError, ValueError):
        return False


def validate_order(*, symbol: str | None, side: str, qty=None, notional=None,
                   order_type: str = "market", tif: str = "day",
                   extended_hours: bool = False, legs: list[dict] | None = None,
                   today=None) -> list[str]:
    """Order-shape legality per asset class — the thin pre-submit gate that
    survived REBUILD-V4. Everything risk-shaped (long-only, leverage,
    defined-risk options) is SERVER-enforced by the account configuration
    (no_shorting, max_margin_multiplier=1, Level 3 max); this checks only
    what Alpaca would accept-then-reject slowly or, worse, accept: expired /
    adjusted-OCC symbols and per-asset-class type/TIF legality. Returns a
    list of human-readable errors; empty = submit it."""
    from datetime import date as _date

    from agent import occ

    errors: list[str] = []
    side = (side or "").strip().lower()
    order_type = (order_type or "").strip().lower()
    tif = (tif or "").strip().lower()
    today = today or _date.today()

    if side not in ("buy", "sell"):
        errors.append(f"side must be buy or sell, got {side!r}")
    if (qty is None) == (notional is None):
        errors.append("exactly one of qty / notional is required")
    for label, v in (("qty", qty), ("notional", notional)):
        if v is not None:
            try:
                if float(v) <= 0:
                    errors.append(f"{label} must be positive")
            except (TypeError, ValueError):
                errors.append(f"{label} is not a number: {v!r}")

    def _check_contract(sym: str, where: str):
        s = (sym or "").strip().upper()
        if occ.is_adjusted_occ(s):
            errors.append(f"{where}: adjusted OCC contract {s} — non-standard "
                          "deliverable, fail closed")
            return
        if not occ.is_option(s):
            errors.append(f"{where}: not a valid OCC option symbol: {s!r}")
            return
        if occ.parse(s)["expiry"] < today:
            errors.append(f"{where}: contract {s} is expired")

    if legs:
        # Multi-leg options order. All legs option contracts; Alpaca accepts
        # 2-4 legs, market/limit only (stop is single-leg only), day/gtc,
        # whole-contract quantities, no notional, never extended hours.
        if not 2 <= len(legs) <= 4:
            errors.append(f"mleg orders take 2-4 legs, got {len(legs)}")
        for i, leg in enumerate(legs):
            _check_contract(leg.get("symbol"), f"leg {i + 1}")
            rq = leg.get("ratio_qty")
            if not _is_whole(rq) or (rq is not None and float(rq) <= 0):
                errors.append(f"leg {i + 1}: ratio_qty must be a positive whole "
                              f"number, got {rq!r}")
            if (leg.get("side") or "").lower() not in ("buy", "sell"):
                errors.append(f"leg {i + 1}: side must be buy or sell")
            if (leg.get("position_intent") or "").lower() not in (
                    "buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"):
                errors.append(f"leg {i + 1}: position_intent must be one of "
                              "buy_to_open/buy_to_close/sell_to_open/sell_to_close")
        if order_type not in ("market", "limit"):
            errors.append(f"mleg order type must be market or limit, got {order_type!r}")
        if tif not in ("day", "gtc"):
            errors.append(f"options TIF must be day or gtc, got {tif!r}")
        if extended_hours:
            errors.append("options never trade extended hours")
        if notional is not None:
            errors.append("mleg orders take qty (spread count), not notional")
        if qty is not None and not _is_whole(qty):
            errors.append("mleg qty must be a whole number of spreads")
        return errors

    if not (symbol or "").strip():
        errors.append("symbol is required (or pass legs for an mleg order)")
        return errors

    cls = asset_class_of(symbol)
    if cls == "crypto":
        if order_type not in ("market", "limit", "stop_limit"):
            errors.append(f"crypto order type must be market/limit/stop_limit, "
                          f"got {order_type!r}")
        if tif not in ("gtc", "ioc"):
            errors.append(f"crypto TIF must be gtc or ioc (no 'day'), got {tif!r}")
        if extended_hours:
            errors.append("extended_hours does not apply to crypto (24/7 already)")
    elif cls == "us_option":
        _check_contract(symbol, "contract")
        if order_type not in ("market", "limit", "stop", "stop_limit"):
            errors.append(f"option order type must be market/limit/stop/stop_limit, "
                          f"got {order_type!r}")
        if tif not in ("day", "gtc"):
            errors.append(f"options TIF must be day or gtc, got {tif!r}")
        if extended_hours:
            errors.append("options never trade extended hours")
        if notional is not None:
            errors.append("options take qty (whole contracts), not notional")
        if qty is not None and not _is_whole(qty):
            errors.append("option qty must be whole contracts")
    else:  # us_equity
        # An OCC-shaped symbol with adjustment digits is NOT an equity — it
        # must never fall through to the equity path.
        from agent import occ as _occ
        if _occ.is_adjusted_occ(symbol):
            errors.append(f"adjusted OCC contract {symbol.strip().upper()} — "
                          "non-standard deliverable, fail closed")
            return errors
        if order_type not in ("market", "limit", "stop", "stop_limit",
                              "trailing_stop"):
            errors.append(f"unknown equity order type {order_type!r}")
        if tif not in ("day", "gtc", "opg", "cls", "ioc", "fok"):
            errors.append(f"unknown equity TIF {tif!r}")
        if notional is not None and order_type != "market":
            errors.append("notional (dollar) orders must be market orders")
        if qty is not None and not _is_whole(qty) and tif != "day":
            errors.append("fractional-share orders must be TIF day")
        if extended_hours and (order_type != "limit" or tif not in ("day", "gtc")):
            errors.append("extended-hours orders must be limit day/gtc")
    return errors


def _val(v):
    """Enum → value, everything else → itself (alpaca-py enums are str+Enum;
    str() yields 'OrderSide.BUY', .value yields 'buy')."""
    return getattr(v, "value", v)


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _iso(v):
    if v is None:
        return None
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


def normalize_order_full(o) -> dict:
    """Alpaca order object/dict → the full mirror shape, legs included.
    Pure — the single seam between SDK objects and everything downstream."""
    g = o.get if isinstance(o, dict) else lambda k, d=None: getattr(o, k, d)
    sym = (g("symbol") or "").upper()
    out = {
        "alpaca_order_id": str(g("id")) if g("id") is not None else None,
        "client_order_id": g("client_order_id"),
        "symbol": sym,
        "asset_class": _val(g("asset_class")) or (asset_class_of(sym) if sym else None),
        "side": (str(_val(g("side"))).lower() or None) if g("side") else None,
        "order_type": (str(_val(g("order_type") or g("type"))).lower() or None)
                      if (g("order_type") or g("type")) else None,
        "tif": (str(_val(g("time_in_force"))).lower() or None)
               if g("time_in_force") else None,
        "order_class": (str(_val(g("order_class"))).lower() or None)
                       if g("order_class") else None,
        "limit_price": _f(g("limit_price")),
        "stop_price": _f(g("stop_price")),
        "qty": _f(g("qty")),
        "notional": _f(g("notional")),
        "status": (str(_val(g("status"))).lower() or None) if g("status") else None,
        "filled_qty": _f(g("filled_qty")),
        "filled_avg_price": _f(g("filled_avg_price")),
        "submitted_at": _iso(g("submitted_at")),
        "filled_at": _iso(g("filled_at")),
        "canceled_at": _iso(g("canceled_at")),
    }
    legs = g("legs")
    out["legs"] = [normalize_order_full(leg) for leg in legs] if legs else []
    return out


def normalize_activity(a) -> dict:
    """Alpaca account-activity dict/object → the desk_activities shape.
    FILLs carry transaction_time; non-trade activities carry date."""
    g = a.get if isinstance(a, dict) else lambda k, d=None: getattr(a, k, d)
    tt = _iso(g("transaction_time"))
    date = _iso(g("date")) or (tt[:10] if tt else None)
    return {
        "alpaca_activity_id": str(g("id")) if g("id") is not None else None,
        "activity_type": str(_val(g("activity_type")) or ""),
        "date": date[:10] if date else None,
        "symbol": ((g("symbol") or "").upper() or None),
        "side": (str(_val(g("side"))).lower() or None) if g("side") else None,
        "qty": _f(g("qty")) or _f(g("cum_qty")),
        "price": _f(g("price")) or _f(g("per_share_amount")),
        "net_amount": _f(g("net_amount")),
        "alpaca_order_id": (str(g("order_id")) if g("order_id") else None),
        "raw": {k: _iso(v) if hasattr(v, "isoformat") else _val(v)
                for k, v in (a.items() if isinstance(a, dict) else [])},
    }


def infer_kind(norm: dict) -> str:
    """entry | exit | stop from the normalized order — the mirror's coarse
    classifier (grade() reasons from sides and positions, not from this)."""
    if (norm.get("order_type") or "") in ("stop", "stop_limit", "trailing_stop"):
        return "stop"
    return "entry" if norm.get("side") == "buy" else "exit"


# ── mirror (desk_orders / desk_activities) ───────────────────


def next_seq(store, run_id: str, *, account: str | None = None) -> int:
    from agent.models import ACCOUNT

    rows = store.select("desk_orders", filters={"account": account or ACCOUNT,
                                                "run_id": run_id},
                        order=[("seq", "desc")], limit=1, columns="seq")
    prev = rows[0]["seq"] if rows and rows[0].get("seq") is not None else 0
    return int(prev) + 1


def mirror_order(store, norm: dict, *, run_id: str | None = None,
                 seq: int | None = None, kind: str | None = None,
                 account: str | None = None) -> None:
    """Upsert one normalized order (and its legs) into desk_orders by
    alpaca_order_id. run_id/seq: passed on our own submits; recovered from
    client_order_id on re-syncs; legs inherit the parent's."""
    from agent.models import ACCOUNT
    from agent.store import is_duplicate_key_error

    acct = account or ACCOUNT
    if run_id is None:
        parsed = parse_client_order_id(norm.get("client_order_id") or "")
        if parsed:
            run_id, seq = parsed["run_id"], parsed["seq"]

    def _write(n: dict, parent_id: str | None):
        row = {k: n.get(k) for k in (
            "alpaca_order_id", "client_order_id", "symbol", "asset_class",
            "side", "order_type", "tif", "order_class", "limit_price",
            "stop_price", "qty", "notional", "status", "filled_qty",
            "filled_avg_price", "submitted_at", "filled_at", "canceled_at")}
        row.update({"account": acct, "run_id": run_id, "seq": seq,
                    "parent_order_id": parent_id,
                    "kind": kind or infer_kind(n),
                    "raw": {k: v for k, v in n.items() if k != "legs"}})
        if not row["alpaca_order_id"]:
            return
        existing = store.select("desk_orders",
                                filters={"alpaca_order_id": row["alpaca_order_id"]},
                                limit=1, columns="id")
        if existing:
            update = dict(row)
            update.pop("alpaca_order_id", None)
            store.update("desk_orders",
                         {"alpaca_order_id": row["alpaca_order_id"]}, update,
                         returning=False)
            return
        try:
            store.insert("desk_orders", row, returning=False)
        except Exception as exc:  # lost the check-then-insert race — update
            if not is_duplicate_key_error(exc):
                raise
            update = dict(row)
            update.pop("alpaca_order_id", None)
            store.update("desk_orders",
                         {"alpaca_order_id": row["alpaca_order_id"]}, update,
                         returning=False)

    _write(norm, None)
    for leg in norm.get("legs") or []:
        _write(leg, norm.get("alpaca_order_id"))


def sync_activities(store, activities: list[dict], *,
                    account: str | None = None) -> int:
    """Insert new activities; duplicates (unique alpaca_activity_id) are
    skipped silently. Returns how many landed."""
    from agent.models import ACCOUNT
    from agent.store import is_duplicate_key_error

    acct = account or ACCOUNT
    added = 0
    for act in activities:
        n = act if "alpaca_activity_id" in act else normalize_activity(act)
        if not n.get("alpaca_activity_id"):
            continue
        row = dict(n)
        row["account"] = acct
        try:
            store.insert("desk_activities", row, returning=False)
            added += 1
        except Exception as exc:
            if not is_duplicate_key_error(exc):
                raise
    return added


# ── the trade client (lazy SDK; paper-only) ──────────────────


class Trade:
    """Order lifecycle against the Alpaca PAPER account. Inject ``client``
    (any object with the TradingClient surface) for tests."""

    def __init__(self, client=None, *, store=None):
        if not settings.alpaca_paper:
            raise RuntimeError(
                "REFUSED: EDGEFINDER_ALPACA_PAPER is false. agent.trade is "
                "paper-only by charter (REBUILD-V4) — there is no live mode.")
        self._client = client
        self._store = store

    @property
    def client(self):
        if self._client is None:
            c = resolve_trade_creds()
            if not (c["key"] and c["secret"]):
                raise RuntimeError(
                    "Alpaca TRADE creds missing — set EDGEFINDER_ALPACA_TRADE_KEY"
                    "/_SECRET (the PAPER account's key pair; the data keys are "
                    "not a fallback, by design).")
            from alpaca.trading.client import TradingClient

            # paper=True is hard-coded, not read from config: even if every
            # other guard failed, a paper key cannot authenticate against
            # the live endpoint.
            self._client = TradingClient(c["key"], c["secret"], paper=True)
        return self._client

    @property
    def store(self):
        if self._store is None:
            from agent.store import get_store

            self._store = get_store()
        return self._store

    # -- reads --

    def account(self) -> dict:
        a = self.client.get_account()
        g = lambda k: getattr(a, k, None)  # noqa: E731
        return {
            "account_number": g("account_number"),
            "status": str(_val(g("status"))),
            "currency": g("currency"),
            "cash": _f(g("cash")),
            "equity": _f(g("equity")),
            "last_equity": _f(g("last_equity")),
            "buying_power": _f(g("buying_power")),
            "options_buying_power": _f(g("options_buying_power")),
            "options_approved_level": g("options_approved_level"),
            "options_trading_level": g("options_trading_level"),
            "shorting_enabled": bool(g("shorting_enabled")),
            "multiplier": g("multiplier"),
            "long_market_value": _f(g("long_market_value")),
            "short_market_value": _f(g("short_market_value")),
            "paper": True,
        }

    def positions(self) -> list[dict]:
        out = []
        for p in self.client.get_all_positions():
            g = p.get if isinstance(p, dict) else lambda k, d=None: getattr(p, k, d)
            sym = (g("symbol") or "").upper()
            out.append({
                "symbol": sym,
                "asset_class": _val(g("asset_class")) or asset_class_of(sym),
                "qty": _f(g("qty")),
                "qty_available": _f(g("qty_available")),
                "avg_entry_price": _f(g("avg_entry_price")),
                "current_price": _f(g("current_price")),
                "market_value": _f(g("market_value")),
                "cost_basis": _f(g("cost_basis")),
                "unrealized_pl": _f(g("unrealized_pl")),
                "unrealized_plpc": _f(g("unrealized_plpc")),
                "change_today": _f(g("change_today")),
                "side": str(_val(g("side")) or "long").lower(),
            })
        return out

    def get_order(self, *, order_id: str | None = None,
                  client_order_id: str | None = None, mirror: bool = True) -> dict:
        if order_id:
            o = self.client.get_order_by_id(order_id)
        elif client_order_id:
            o = self.client.get_order_by_client_id(client_order_id)
        else:
            raise ValueError("pass order_id or client_order_id")
        norm = normalize_order_full(o)
        if mirror:
            mirror_order(self.store, norm)
        return norm

    def orders(self, *, status: str = "all", limit: int = 100,
               symbols: list[str] | None = None, mirror: bool = False) -> list[dict]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        qs = {"open": QueryOrderStatus.OPEN, "closed": QueryOrderStatus.CLOSED,
              "all": QueryOrderStatus.ALL}[status]
        req = GetOrdersRequest(status=qs, limit=limit, nested=True,
                               symbols=symbols)
        out = [normalize_order_full(o) for o in self.client.get_orders(req)]
        if mirror:
            for n in out:
                mirror_order(self.store, n)
        return out

    def activities(self, *, after: str | None = None,
                   activity_types: str | None = None,
                   page_size: int = 100) -> list[dict]:
        """Raw account activities via the REST passthrough (the installed SDK
        has no typed method for /account/activities). Cursor-paginates until
        exhausted."""
        params: dict = {"page_size": page_size, "direction": "asc"}
        if after:
            params["after"] = after
        if activity_types:
            params["activity_types"] = activity_types
        out: list[dict] = []
        while True:
            batch = self.client.get("/account/activities", params) or []
            out.extend(batch)
            if len(batch) < page_size:
                return out
            last_id = (batch[-1].get("id") if isinstance(batch[-1], dict)
                       else getattr(batch[-1], "id", None))
            if not last_id:
                return out
            params["page_token"] = str(last_id)

    def portfolio_history(self, *, period: str = "1M", timeframe: str = "1D",
                          intraday_reporting: str = "market_hours") -> dict:
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe,
                                         intraday_reporting=intraday_reporting)
        h = self.client.get_portfolio_history(req)
        g = h.get if isinstance(h, dict) else lambda k, d=None: getattr(h, k, d)
        return {"timestamp": list(g("timestamp") or []),
                "equity": [_f(x) for x in (g("equity") or [])],
                "profit_loss": [_f(x) for x in (g("profit_loss") or [])],
                "profit_loss_pct": [_f(x) for x in (g("profit_loss_pct") or [])],
                "base_value": _f(g("base_value")),
                "timeframe": str(g("timeframe") or timeframe)}

    # -- writes --

    def submit(self, *, symbol: str | None = None, side: str = "buy", qty=None,
               notional=None, order_type: str = "market", tif: str = "day",
               limit_price=None, stop_price=None, extended_hours: bool = False,
               legs: list[dict] | None = None, run_id: str, kind: str | None = None,
               wait_secs: float = SUBMIT_POLL_SECS) -> dict:
        """Validate → stamp client_order_id → submit → poll briefly → mirror.
        Returns {order, errors}; errors non-empty means NOTHING was sent."""
        from agent.broker import _today_et

        errors = validate_order(symbol=symbol, side=side, qty=qty,
                                notional=notional, order_type=order_type,
                                tif=tif, extended_hours=extended_hours,
                                legs=legs, today=_today_et())
        if errors:
            return {"order": None, "errors": errors}

        seq = next_seq(self.store, run_id)
        cid = make_client_order_id(run_id, seq)
        req = self._build_request(symbol=symbol, side=side, qty=qty,
                                  notional=notional, order_type=order_type,
                                  tif=tif, limit_price=limit_price,
                                  stop_price=stop_price,
                                  extended_hours=extended_hours, legs=legs,
                                  client_order_id=cid)
        try:
            o = self.client.submit_order(req)
        except Exception as exc:
            # Ambiguous failures (timeout mid-flight): the order may have
            # landed. Look it up by our id before reporting failure.
            try:
                o = self.client.get_order_by_client_id(cid)
            except Exception:
                return {"order": None,
                        "errors": [f"submit failed: {type(exc).__name__}: {exc}"]}

        norm = self._poll_until_settled(normalize_order_full(o), wait_secs)
        mirror_order(self.store, norm, run_id=run_id, seq=seq, kind=kind)
        return {"order": norm, "errors": []}

    def _poll_until_settled(self, norm: dict, wait_secs: float) -> dict:
        deadline = _time.monotonic() + max(0.0, wait_secs)
        while (norm.get("status") not in TERMINAL_STATUSES
               and _time.monotonic() < deadline):
            _time.sleep(1.0)
            try:
                norm = normalize_order_full(
                    self.client.get_order_by_id(norm["alpaca_order_id"]))
            except Exception:  # transient read failure — keep last state
                break
        return norm

    def _build_request(self, *, symbol, side, qty, notional, order_type, tif,
                       limit_price, stop_price, extended_hours, legs,
                       client_order_id):
        from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
        from alpaca.trading.requests import (LimitOrderRequest,
                                             MarketOrderRequest,
                                             OptionLegRequest,
                                             StopLimitOrderRequest,
                                             StopOrderRequest,
                                             TrailingStopOrderRequest)

        tif_e = TimeInForce(tif)
        side_e = OrderSide(side)
        common: dict = {"time_in_force": tif_e,
                        "client_order_id": client_order_id}
        if legs:
            leg_objs = [OptionLegRequest(
                symbol=leg["symbol"].strip().upper(),
                ratio_qty=int(leg["ratio_qty"]),
                side=OrderSide(str(leg["side"]).lower()),
                position_intent=str(leg["position_intent"]).lower(),
            ) for leg in legs]
            common.update({"order_class": OrderClass.MLEG, "legs": leg_objs,
                           "qty": int(float(qty))})
            if order_type == "limit":
                return LimitOrderRequest(limit_price=float(limit_price),
                                         **common)
            return MarketOrderRequest(**common)

        common.update({"symbol": symbol.strip().upper(), "side": side_e,
                       "extended_hours": bool(extended_hours)})
        if qty is not None:
            common["qty"] = float(qty)
        else:
            common["notional"] = float(notional)
        if order_type == "market":
            return MarketOrderRequest(**common)
        if order_type == "limit":
            return LimitOrderRequest(limit_price=float(limit_price), **common)
        if order_type == "stop":
            return StopOrderRequest(stop_price=float(stop_price), **common)
        if order_type == "stop_limit":
            return StopLimitOrderRequest(stop_price=float(stop_price),
                                         limit_price=float(limit_price),
                                         **common)
        if order_type == "trailing_stop":
            return TrailingStopOrderRequest(**common)
        raise ValueError(f"unknown order type {order_type!r}")

    def cancel(self, order_id: str) -> dict:
        self.client.cancel_order_by_id(order_id)
        try:
            return self.get_order(order_id=order_id)
        except Exception:
            return {"alpaca_order_id": order_id, "status": "pending_cancel"}

    def arm_stop(self, *, symbol: str, stop_price: float, run_id: str,
                 qty: float | None = None, limit_price: float | None = None) -> dict:
        """One resting GTC protective stop per equity position. Cancels any
        prior open stop on the symbol first (replace semantics), sizes to
        qty_available by default (shares locked under covered calls are not
        sellable — Alpaca's qty_available already excludes them). Refuses
        options and crypto: no stop on mleg, no plain stop on crypto —
        those exits are managed at cycle cadence, eyes open."""
        cls = asset_class_of(symbol)
        if cls != "us_equity":
            return {"order": None, "errors": [
                f"arm-stop is equities-only ({cls} exits are managed at cycle "
                "cadence — Alpaca has no resting stop for this asset class "
                "in our shape)"]}
        sym = symbol.strip().upper()
        pos = {p["symbol"]: p for p in self.positions()}.get(sym)
        if not pos or not (pos.get("qty") or 0) > 0:
            return {"order": None, "errors": [f"no long position in {sym}"]}
        avail = pos.get("qty_available")
        use_qty = float(qty) if qty is not None else float(avail or 0)
        if use_qty <= 0:
            return {"order": None, "errors": [
                f"{sym}: qty_available is 0 — every share is locked "
                "(open order or covered call)"]}
        if avail is not None and use_qty > float(avail):
            return {"order": None, "errors": [
                f"{sym}: requested {use_qty:g} exceeds qty_available "
                f"{float(avail):g} (covered-call/open-order lock)"]}
        # Fractional stops must be TIF day (Alpaca rule) — a resting
        # protective stop must be GTC, so round DOWN to whole shares.
        if use_qty != int(use_qty):
            use_qty = float(int(use_qty))
            if use_qty <= 0:
                return {"order": None, "errors": [
                    f"{sym}: position is under one whole share — a GTC stop "
                    "cannot rest on a fractional quantity"]}
        canceled = []
        for o in self.orders(status="open", symbols=[sym]):
            if (o.get("order_type") or "") in ("stop", "stop_limit") \
                    and o.get("side") == "sell":
                self.client.cancel_order_by_id(o["alpaca_order_id"])
                canceled.append(o["alpaca_order_id"])
        res = self.submit(symbol=sym, side="sell", qty=use_qty,
                          order_type=("stop_limit" if limit_price else "stop"),
                          tif="gtc", stop_price=stop_price,
                          limit_price=limit_price, run_id=run_id, kind="stop",
                          wait_secs=2)
        res["replaced"] = canceled
        return res

    # -- composed reads --

    def reconcile(self, *, order_limit: int = 100) -> dict:
        """Cycle-start sync: mirror recent orders + new activities, then
        report what changed and what needs attention. Alpaca wins on every
        conflict — this is the moment the mirror re-converges."""
        from datetime import datetime, timedelta, timezone

        orders = self.orders(status="all", limit=order_limit, mirror=True)
        # Activities cursor: resync a 3-day overlap; the unique id dedupes.
        rows = self.store.select("desk_activities", order=[("date", "desc")],
                                 limit=1, columns="date")
        after = None
        if rows and rows[0].get("date"):
            from datetime import date as _d
            y, m, d = str(rows[0]["date"])[:10].split("-")
            after = (_d(int(y), int(m), int(d)) - timedelta(days=3)).isoformat()
        acts = [normalize_activity(a) for a in self.activities(after=after)]
        added = sync_activities(self.store, acts)

        now = datetime.now(timezone.utc)
        day_ago = (now - timedelta(hours=24)).isoformat()
        filled_recent = [o for o in orders
                         if o.get("status") == "filled"
                         and (o.get("filled_at") or "") >= day_ago]
        open_orders = [o for o in orders if o.get("status") in
                       ("new", "accepted", "partially_filled", "pending_new",
                        "accepted_for_bidding", "held")]
        stop_warnings = []
        for o in open_orders:
            if (o.get("order_type") or "") not in ("stop", "stop_limit"):
                continue
            sub = o.get("submitted_at")
            if not sub:
                continue
            try:
                sub_dt = datetime.fromisoformat(str(sub).replace("Z", "+00:00"))
                age = (now - sub_dt).days
            except ValueError:
                continue
            if age >= GTC_WARN_AGE_DAYS:
                stop_warnings.append(
                    {"symbol": o["symbol"], "alpaca_order_id": o["alpaca_order_id"],
                     "age_days": age,
                     "note": f"GTC auto-cancels at 90 days — re-arm before day 90"})
        return {"account": self.account(),
                "orders_synced": len(orders),
                "activities_added": added,
                "fills_last_24h": filled_recent,
                "open_orders": open_orders,
                "gtc_stop_warnings": stop_warnings}

    def state(self) -> dict:
        """The account header everything reads — the `agent.ledger state`
        successor. total_pnl/return measure Era 2 against
        EDGEFINDER_STARTING_CAPITAL (set at cutover to Era-1 final equity)."""
        acct = self.account()
        positions = self.positions()
        equity = acct.get("equity") or 0.0
        start = float(settings.starting_capital)
        for p in positions:
            mv = p.get("market_value") or 0.0
            p["weight"] = round(mv / equity, 6) if equity else None
        return {"account": "agent", "paper": True,
                "cash": acct.get("cash"), "equity": equity,
                "buying_power": acct.get("buying_power"),
                "options_buying_power": acct.get("options_buying_power"),
                "starting_capital": start,
                "total_pnl": round(equity - start, 2),
                "total_return_pct": (round((equity - start) / start * 100, 4)
                                     if start else None),
                "positions": positions,
                "positions_value": round(sum(p.get("market_value") or 0.0
                                             for p in positions), 2)}

    def snapshot_portfolio(self) -> dict:
        """Write today's ET row into desk_portfolio_history (idempotent —
        one row per date, updated in place). The nightly refresh calls this;
        it is also the split-guard's baseline."""
        from agent.broker import _today_et
        from agent.models import ACCOUNT
        from agent.store import is_duplicate_key_error

        acct = self.account()
        pos_map = {p["symbol"]: {"qty": p.get("qty"),
                                 "avg_entry_price": p.get("avg_entry_price")}
                   for p in self.positions()}
        row = {"account": ACCOUNT, "snap_date": _today_et().isoformat(),
               "equity": acct.get("equity"), "cash": acct.get("cash"),
               "profit_loss": (round((acct.get("equity") or 0.0)
                                     - float(settings.starting_capital), 2)),
               "base_value": float(settings.starting_capital),
               "positions": pos_map}
        try:
            self.store.insert("desk_portfolio_history", row, returning=False)
        except Exception as exc:
            if not is_duplicate_key_error(exc):
                raise
            self.store.update("desk_portfolio_history",
                              {"account": ACCOUNT, "snap_date": row["snap_date"]},
                              {k: v for k, v in row.items()
                               if k not in ("account", "snap_date")},
                              returning=False)
        return row

    def apply_paper_config(self) -> dict:
        """Enforce the charter at the BROKER: long-only + no leverage. Run
        once at cutover (runbook step 6) and safe to re-run any time."""
        self.client.patch("/account/configurations",
                          {"no_shorting": True, "max_margin_multiplier": "1"})
        cfg = self.client.get("/account/configurations", None)
        acct = self.account()
        return {"configurations": cfg,
                "options_approved_level": acct.get("options_approved_level"),
                "options_trading_level": acct.get("options_trading_level"),
                "shorting_enabled": acct.get("shorting_enabled"),
                "multiplier": acct.get("multiplier")}

    # -- the cutover probe suite --

    def probe_cutover(self, *, journal: bool = True) -> dict:
        """Empirical pre-cutover checks (runbook step 8): the facts the docs
        left unverified, answered against the real paper account. Safe by
        construction — the only order placed is a $1 limit far below any
        market, canceled immediately."""
        results: dict[str, dict] = {}

        def _run(name, fn):
            try:
                results[name] = {"ok": True, "result": fn()}
            except Exception as exc:  # noqa: BLE001 — a probe never aborts the suite
                results[name] = {"ok": False,
                                 "error": f"{type(exc).__name__}: {exc}"}

        _run("account", self.account)
        _run("config", self.apply_paper_config)

        def _roundtrip():
            res = self.submit(symbol="SPY", side="buy", qty=1,
                              order_type="limit", tif="day", limit_price=1.00,
                              run_id="probe-cutover", wait_secs=2)
            if res["errors"]:
                return {"submitted": False, "errors": res["errors"]}
            oid = res["order"]["alpaca_order_id"]
            cid = res["order"]["client_order_id"]
            back = self.get_order(client_order_id=cid)
            self.cancel(oid)
            return {"submitted": True, "client_order_id": cid,
                    "lookup_matches": back.get("alpaca_order_id") == oid,
                    "canceled": True}

        _run("far_limit_roundtrip", _roundtrip)

        def _sip_on_trade_keys():
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.enums import DataFeed

            c = resolve_trade_creds()
            data = StockHistoricalDataClient(c["key"], c["secret"])
            req = StockLatestQuoteRequest(symbol_or_symbols=["SPY"],
                                          feed=DataFeed.SIP)
            res = data.get_stock_latest_quote(req)
            q = res.get("SPY")
            return {"sip_allowed": True,
                    "quote_t": _iso(getattr(q, "timestamp", None))}

        _run("sip_entitlement_trade_keys", _sip_on_trade_keys)

        if journal:
            try:
                from agent.models import ACCOUNT
                self.store.insert("desk_journal", {
                    "account": ACCOUNT, "kind": "note",
                    "title": "V4 cutover probe results",
                    "body": json.dumps(results, default=str)[:8000],
                }, returning=False)
            except Exception:  # noqa: BLE001 — the probe output is the deliverable
                logger.warning("probe: journal write failed", exc_info=True)
        return results


def state() -> dict:
    """Module-level convenience for brain.context / preflight."""
    return Trade().state()


# ── CLI ──────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("account")
    sub.add_parser("positions")
    sub.add_parser("state")
    sub.add_parser("config", help="apply + show the paper account configuration "
                                  "(no_shorting, multiplier=1)")

    s = sub.add_parser("submit", help="submit one order (validated, mirrored)")
    s.add_argument("--symbol", default=None)
    s.add_argument("--side", default="buy", choices=["buy", "sell"])
    s.add_argument("--qty", type=float, default=None)
    s.add_argument("--notional", type=float, default=None)
    s.add_argument("--type", dest="order_type", default="market",
                   choices=["market", "limit", "stop", "stop_limit",
                            "trailing_stop"])
    s.add_argument("--tif", default="day",
                   choices=["day", "gtc", "opg", "cls", "ioc", "fok"])
    s.add_argument("--limit-price", type=float, default=None)
    s.add_argument("--stop-price", type=float, default=None)
    s.add_argument("--extended", action="store_true",
                   help="extended-hours eligible (limit day/gtc only)")
    s.add_argument("--legs", default=None,
                   help='mleg legs as JSON: [{"symbol","ratio_qty","side",'
                        '"position_intent"},...]')
    s.add_argument("--kind", default=None, choices=["entry", "exit", "stop"])
    s.add_argument("--run-id", required=True)
    s.add_argument("--wait", type=float, default=SUBMIT_POLL_SECS)

    o = sub.add_parser("order", help="fetch one order (mirrors it)")
    o.add_argument("--id", default=None)
    o.add_argument("--client-order-id", default=None)

    c = sub.add_parser("cancel")
    c.add_argument("--id", required=True)

    ol = sub.add_parser("orders")
    ol.add_argument("--status", default="all", choices=["all", "open", "closed"])
    ol.add_argument("--limit", type=int, default=50)
    ol.add_argument("--symbols", default=None, help="comma-separated filter")

    a = sub.add_parser("activities")
    a.add_argument("--types", default=None,
                   help="comma-separated activity types (FILL,SSP,OPASN,...)")
    a.add_argument("--after", default=None, help="ISO date lower bound")
    a.add_argument("--sync", action="store_true",
                   help="also insert new rows into desk_activities")

    ph = sub.add_parser("portfolio-history")
    ph.add_argument("--period", default="1M")
    ph.add_argument("--timeframe", default="1D")

    st = sub.add_parser("arm-stop", help="one resting GTC protective stop per "
                                         "equity position (replace semantics)")
    st.add_argument("--symbol", required=True)
    st.add_argument("--stop-price", type=float, required=True)
    st.add_argument("--limit-price", type=float, default=None,
                    help="make it a stop-limit")
    st.add_argument("--qty", type=float, default=None,
                    help="default: full qty_available")
    st.add_argument("--run-id", required=True)

    r = sub.add_parser("reconcile", help="cycle-start mirror sync + report")
    r.add_argument("--order-limit", type=int, default=100)

    sub.add_parser("snapshot", help="write today's desk_portfolio_history row")

    pr = sub.add_parser("probe", help="empirical checks against the real "
                                      "paper account")
    pr.add_argument("--suite", default="cutover", choices=["cutover"])

    args = p.parse_args(argv)

    if not trade_enabled():
        print(json.dumps({"error": "alpaca TRADE creds not set",
                          "need": ["EDGEFINDER_ALPACA_TRADE_KEY",
                                   "EDGEFINDER_ALPACA_TRADE_SECRET"],
                          "note": "paper account keys — the data keys are not "
                                  "a fallback, by design"}))
        return 2
    t = Trade()
    if args.cmd == "account":
        out = t.account()
    elif args.cmd == "positions":
        out = t.positions()
    elif args.cmd == "state":
        out = t.state()
    elif args.cmd == "config":
        out = t.apply_paper_config()
    elif args.cmd == "submit":
        legs = json.loads(args.legs) if args.legs else None
        out = t.submit(symbol=args.symbol, side=args.side, qty=args.qty,
                       notional=args.notional, order_type=args.order_type,
                       tif=args.tif, limit_price=args.limit_price,
                       stop_price=args.stop_price, extended_hours=args.extended,
                       legs=legs, run_id=args.run_id, kind=args.kind,
                       wait_secs=args.wait)
    elif args.cmd == "order":
        out = t.get_order(order_id=args.id, client_order_id=args.client_order_id)
    elif args.cmd == "cancel":
        out = t.cancel(args.id)
    elif args.cmd == "orders":
        syms = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
                if args.symbols else None)
        out = t.orders(status=args.status, limit=args.limit, symbols=syms)
    elif args.cmd == "activities":
        raw = t.activities(after=args.after, activity_types=args.types)
        norm = [normalize_activity(x) for x in raw]
        if args.sync:
            added = sync_activities(t.store, norm)
            out = {"fetched": len(norm), "added": added}
        else:
            out = norm
    elif args.cmd == "portfolio-history":
        out = t.portfolio_history(period=args.period, timeframe=args.timeframe)
    elif args.cmd == "arm-stop":
        out = t.arm_stop(symbol=args.symbol, stop_price=args.stop_price,
                         limit_price=args.limit_price, qty=args.qty,
                         run_id=args.run_id)
    elif args.cmd == "reconcile":
        out = t.reconcile(order_limit=args.order_limit)
    elif args.cmd == "snapshot":
        out = t.snapshot_portfolio()
    elif args.cmd == "probe":
        out = t.probe_cutover()
    else:  # pragma: no cover
        out = {"error": "unknown command"}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

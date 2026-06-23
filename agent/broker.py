"""Alpaca broker wrapper — the account of record for live paper trading.

This is the foundation of the live rebuild (see LIVE-REBUILD-PLAN.md). The
agent no longer keeps its own authoritative ledger; instead it trades a real
Alpaca **paper** account that fills against the live market, and we read the
account/positions/fills back for display. Quotes come from Alpaca's real-time
market-data API (SIP feed). Everything here is plain HTTPS/WebSocket, so it
works both on Render (always-on streamer) and inside the hourly Routine
(submits orders + reads state over REST).

Design:
- The SDK (`alpaca-py`) is imported **lazily** so this module imports cleanly
  without it installed (CI / unit tests use mocks); install with the `live`
  extra. The pure helpers (`build_order`, `normalize_*`) have no SDK or network
  dependency and carry the testable logic.
- Credentials resolve from settings (`EDGEFINDER_ALPACA_*`) first, then the
  SDK's native `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, then `ALPACA_*`.

CLI (once paper keys are set):
  python -m agent.broker account
  python -m agent.broker quote --symbols NVDA,AAPL
  python -m agent.broker positions
  python -m agent.broker buy --symbol NVDA --notional 1000   # paper order
"""

from __future__ import annotations

import json
import logging
import os

from config.settings import settings

logger = logging.getLogger(__name__)

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"


# ── credentials ──────────────────────────────────────────────

def resolve_creds() -> dict:
    """Resolve Alpaca creds from settings, then native SDK env vars. Returns
    {key, secret, paper, feed}; key/secret are "" when unset."""
    key = (settings.alpaca_api_key
           or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY") or "")
    secret = (settings.alpaca_api_secret
              or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET") or "")
    return {"key": key.strip(), "secret": secret.strip(),
            "paper": bool(settings.alpaca_paper),
            "feed": (settings.alpaca_data_feed or "sip").strip().lower()}


def enabled() -> bool:
    """True when paper keys are present — the live engine can run."""
    c = resolve_creds()
    return bool(c["key"] and c["secret"])


# ── pure helpers (SDK-free, network-free → unit-tested) ──────

def build_order(symbol: str, side: str, *, qty: float | None = None,
                notional: float | None = None, type: str = "market",
                limit_price: float | None = None,
                time_in_force: str = "day") -> dict:
    """Validate + normalize an order into a plain spec dict (the SDK request is
    built from this at submit time). Exactly one of qty / notional is required;
    notional (fractional dollars) only makes sense for market orders."""
    symbol = (symbol or "").strip().upper()
    side = (side or "").strip().lower()
    type = (type or "market").strip().lower()
    if not symbol:
        raise ValueError("order: symbol required")
    if side not in ("buy", "sell"):
        raise ValueError(f"order: side must be buy/sell, got {side!r}")
    if (qty is None) == (notional is None):
        raise ValueError("order: pass exactly one of qty or notional")
    if type not in ("market", "limit"):
        raise ValueError(f"order: type must be market/limit, got {type!r}")
    if type == "limit" and not limit_price:
        raise ValueError("order: limit order needs limit_price")
    if notional is not None and type != "market":
        raise ValueError("order: notional is only valid for market orders")
    if qty is not None and qty <= 0:
        raise ValueError("order: qty must be positive")
    if notional is not None and notional <= 0:
        raise ValueError("order: notional must be positive")
    spec: dict = {"symbol": symbol, "side": side, "type": type,
                  "time_in_force": time_in_force}
    if qty is not None:
        spec["qty"] = float(qty)
    if notional is not None:
        spec["notional"] = round(float(notional), 2)
    if type == "limit":
        spec["limit_price"] = float(limit_price)
    return spec


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def normalize_position(p) -> dict:
    """Alpaca position object/dict → plain dict for the desk."""
    g = p.get if isinstance(p, dict) else lambda k, d=None: getattr(p, k, d)
    qty = _f(g("qty"))
    return {
        "symbol": g("symbol"),
        "qty": qty,
        "avg_entry_price": _f(g("avg_entry_price")),
        "current_price": _f(g("current_price")),
        "market_value": _f(g("market_value")),
        "cost_basis": _f(g("cost_basis")),
        "unrealized_pl": _f(g("unrealized_pl")),
        "unrealized_plpc": _f(g("unrealized_plpc")),
        "side": g("side") or ("long" if (qty or 0) >= 0 else "short"),
    }


def normalize_order(o) -> dict:
    """Alpaca order object/dict → plain dict (fill price + status for the feed)."""
    g = o.get if isinstance(o, dict) else lambda k, d=None: getattr(o, k, d)
    return {
        "id": str(g("id")) if g("id") is not None else None,
        "symbol": g("symbol"),
        "side": str(g("side")).lower().replace("orderside.", "") if g("side") else None,
        "qty": _f(g("qty")),
        "filled_qty": _f(g("filled_qty")),
        "filled_avg_price": _f(g("filled_avg_price")),
        "type": str(g("order_type") or g("type") or "").lower().replace("ordertype.", "") or None,
        "status": str(g("status")).lower().replace("orderstatus.", "") if g("status") else None,
        "submitted_at": str(g("submitted_at")) if g("submitted_at") else None,
        "filled_at": str(g("filled_at")) if g("filled_at") else None,
    }


def normalize_quote(symbol: str, q) -> dict:
    """Alpaca latest-quote object → {symbol, bid, ask, mid, ...}."""
    g = q.get if isinstance(q, dict) else lambda k, d=None: getattr(q, k, d)
    bid = _f(g("bid_price")) or _f(g("bp"))
    ask = _f(g("ask_price")) or _f(g("ap"))
    mid = round((bid + ask) / 2, 4) if (bid and ask) else (ask or bid)
    return {"symbol": symbol, "bid": bid, "ask": ask, "mid": mid,
            "bid_size": _f(g("bid_size")) or _f(g("bs")),
            "ask_size": _f(g("ask_size")) or _f(g("as")),
            "t": str(g("timestamp")) if g("timestamp") else None}


# ── the broker (lazy SDK; live calls) ────────────────────────

class Broker:
    """Thin Alpaca paper-trading + market-data client. Construct once and reuse;
    the SDK clients are built lazily on first use."""

    def __init__(self, creds: dict | None = None):
        self._c = creds or resolve_creds()
        if not (self._c["key"] and self._c["secret"]):
            raise RuntimeError(
                "Alpaca creds missing — set EDGEFINDER_ALPACA_API_KEY/_SECRET "
                "(or APCA_API_KEY_ID/APCA_API_SECRET_KEY).")
        self._trading = None
        self._data = None

    # -- lazy SDK clients --
    @property
    def trading(self):
        if self._trading is None:
            from alpaca.trading.client import TradingClient
            self._trading = TradingClient(self._c["key"], self._c["secret"],
                                          paper=self._c["paper"])
        return self._trading

    @property
    def data(self):
        if self._data is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._data = StockHistoricalDataClient(self._c["key"], self._c["secret"])
        return self._data

    # -- reads --
    def account(self) -> dict:
        a = self.trading.get_account()
        g = lambda k: getattr(a, k, None)  # noqa: E731
        return {"account_number": g("account_number"), "status": str(g("status")),
                "cash": _f(g("cash")), "equity": _f(g("equity")),
                "buying_power": _f(g("buying_power")),
                "portfolio_value": _f(g("portfolio_value")),
                "long_market_value": _f(g("long_market_value")),
                "paper": self._c["paper"]}

    def positions(self) -> list[dict]:
        return [normalize_position(p) for p in self.trading.get_all_positions()]

    def quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Latest real-time quotes for symbols (SIP/IEX per settings)."""
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed

        feed = DataFeed.SIP if self._c["feed"] == "sip" else DataFeed.IEX
        req = StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=feed)
        res = self.data.get_stock_latest_quote(req)
        return {sym: normalize_quote(sym, q) for sym, q in res.items()}

    def is_market_open(self) -> bool:
        return bool(getattr(self.trading.get_clock(), "is_open", False))

    def recent_orders(self, limit: int = 50) -> list[dict]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        return [normalize_order(o) for o in self.trading.get_orders(req)]

    # -- writes --
    def submit(self, spec: dict) -> dict:
        """Submit an order from a build_order() spec. Returns the normalized
        order (with fill price once filled)."""
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side = OrderSide.BUY if spec["side"] == "buy" else OrderSide.SELL
        tif = TimeInForce(spec.get("time_in_force", "day"))
        common = {"symbol": spec["symbol"], "side": side, "time_in_force": tif}
        if "qty" in spec:
            common["qty"] = spec["qty"]
        else:
            common["notional"] = spec["notional"]
        if spec["type"] == "limit":
            req = LimitOrderRequest(limit_price=spec["limit_price"], **common)
        else:
            req = MarketOrderRequest(**common)
        return normalize_order(self.trading.submit_order(req))


# ── CLI ──────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("account")
    sub.add_parser("positions")
    sub.add_parser("orders")
    sub.add_parser("clock")
    q = sub.add_parser("quote"); q.add_argument("--symbols", required=True)
    for name in ("buy", "sell"):
        o = sub.add_parser(name)
        o.add_argument("--symbol", required=True)
        o.add_argument("--qty", type=float)
        o.add_argument("--notional", type=float)
        o.add_argument("--limit", type=float, dest="limit_price")
    args = p.parse_args(argv)

    if not enabled():
        print(json.dumps({"error": "alpaca creds not set",
                          "need": ["EDGEFINDER_ALPACA_API_KEY", "EDGEFINDER_ALPACA_API_SECRET"]}))
        return 2
    b = Broker()
    if args.cmd == "account":
        out = b.account()
    elif args.cmd == "positions":
        out = b.positions()
    elif args.cmd == "orders":
        out = b.recent_orders()
    elif args.cmd == "clock":
        out = {"is_open": b.is_market_open()}
    elif args.cmd == "quote":
        out = b.quotes([s.strip().upper() for s in args.symbols.split(",") if s.strip()])
    elif args.cmd in ("buy", "sell"):
        spec = build_order(args.symbol, args.cmd, qty=args.qty, notional=args.notional,
                           type="limit" if args.limit_price else "market",
                           limit_price=args.limit_price)
        out = b.submit(spec)
    else:  # pragma: no cover
        out = {"error": "unknown command"}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

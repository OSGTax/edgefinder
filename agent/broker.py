"""Alpaca DATA-READER — live SIP quotes + market clock for the agent's own book.

READ-ONLY by design (see REBUILD-V3.md). The agent trades its OWN paper ledger
(``agent.ledger``); Alpaca supplies the live market data that prices those
fills: real-time SIP quotes, the market clock, and account diagnostics for the
key-health check. This wrapper NEVER submits, modifies, or cancels orders at
Alpaca — there are no write methods, deliberately.

Design:
- The SDK (`alpaca-py`) is imported **lazily** so this module imports cleanly
  without it installed (CI / unit tests use mocks). The pure normalizers have
  no SDK or network dependency and carry the testable logic.
- Credentials resolve from settings (`EDGEFINDER_ALPACA_*`) first, then the
  SDK's native `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`, then `ALPACA_*`.

CLI (once paper keys are set):
  python -m agent.broker account
  python -m agent.broker quote --symbols NVDA,AAPL
  python -m agent.broker clock
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


def normalize_asset(a) -> dict:
    """Alpaca asset object → {symbol, name, exchange, tradable, ...} (pure).

    ``has_options`` reads the asset's ``attributes`` list (Alpaca flags an
    optionable name with the string ``"has_options"``), so the same normalizer
    powers the optionable-underlying enumerator with no extra call."""
    g = a.get if isinstance(a, dict) else lambda k, d=None: getattr(a, k, d)
    attrs = list(g("attributes") or [])
    return {
        "symbol": (g("symbol") or "").upper(),
        "name": g("name"),
        "exchange": str(g("exchange") or ""),
        "tradable": bool(g("tradable")),
        "fractionable": bool(g("fractionable")),
        "shortable": bool(g("shortable")),
        "has_options": "has_options" in attrs,
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

    @property
    def option_data(self):
        if getattr(self, "_option_data", None) is None:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self._option_data = OptionHistoricalDataClient(self._c["key"],
                                                           self._c["secret"])
        return self._option_data

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

    def list_assets(self, *, optionable: bool = False,
                    fractionable_only: bool = False) -> list[dict]:
        """Every ACTIVE, TRADABLE US equity/ETF — the whole investable catalog
        Alpaca offers (~13k names), the direct replacement for Polygon's
        grouped-daily universe. ``optionable`` keeps only names with listed
        options (~6k); ``fractionable_only`` keeps names that fill in dollar
        notional cleanly. Read-only; one paginated Trading-API call."""
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetStatus, AssetClass

        req = GetAssetsRequest(status=AssetStatus.ACTIVE,
                               asset_class=AssetClass.US_EQUITY)
        out = []
        for raw in self.trading.get_all_assets(req):
            a = normalize_asset(raw)
            if not a["tradable"]:
                continue
            if optionable and not a["has_options"]:
                continue
            if fractionable_only and not a["fractionable"]:
                continue
            out.append(a)
        return out

    def is_market_open(self) -> bool:
        return bool(getattr(self.trading.get_clock(), "is_open", False))

    def recent_orders(self, limit: int = 50) -> list[dict]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
        return [normalize_order(o) for o in self.trading.get_orders(req)]

    # -- options (OPRA, included in Algo Trader Plus) --
    def option_chain(self, underlying: str, *, dte_max: int = 60,
                     moneyness: float = 0.20) -> list[dict]:
        """The option chain around the money: contracts within ±moneyness of
        the live underlying price, expiring within dte_max days. Each row:
        occ symbol, type, strike, expiry, dte, bid/ask/mid, IV, delta/theta.
        Sorted by (expiry, strike)."""
        from datetime import date as _date, timedelta as _td

        from alpaca.data.requests import OptionChainRequest

        underlying = underlying.upper().strip()
        uq = self.quotes([underlying]).get(underlying) or {}
        px = uq.get("mid") or uq.get("ask") or uq.get("bid")
        req_kwargs = {"underlying_symbol": underlying,
                      "expiration_date_lte": _date.today() + _td(days=dte_max)}
        if px:
            req_kwargs["strike_price_gte"] = round(px * (1 - moneyness), 2)
            req_kwargs["strike_price_lte"] = round(px * (1 + moneyness), 2)
        snaps = self.option_data.get_option_chain(OptionChainRequest(**req_kwargs))
        from agent import occ
        out = []
        for sym, snap in snaps.items():
            try:
                p = occ.parse(sym)
            except ValueError:
                continue
            q = getattr(snap, "latest_quote", None)
            g = getattr(snap, "greeks", None)
            bid = _f(getattr(q, "bid_price", None)) if q else None
            ask = _f(getattr(q, "ask_price", None)) if q else None
            out.append({
                "symbol": sym, "type": p["type"], "strike": p["strike"],
                "expiry": p["expiry"].isoformat(),
                "dte": (p["expiry"] - _date.today()).days,
                "bid": bid, "ask": ask,
                "mid": round((bid + ask) / 2, 4) if (bid and ask) else None,
                "iv": _f(getattr(snap, "implied_volatility", None)),
                "delta": _f(getattr(g, "delta", None)) if g else None,
                "theta": _f(getattr(g, "theta", None)) if g else None,
            })
        out.sort(key=lambda r: (r["expiry"], r["strike"], r["type"]))
        return out

    def option_quotes(self, occ_symbols: list[str]) -> dict[str, dict]:
        """Live bid/ask for specific option contracts (OPRA)."""
        from alpaca.data.requests import OptionLatestQuoteRequest

        req = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbols)
        res = self.option_data.get_option_latest_quote(req)
        return {sym: normalize_quote(sym, q) for sym, q in res.items()}

    # NO write methods, by design. This wrapper is DATA-READER ONLY: the agent
    # trades its OWN paper ledger (agent.ledger) priced off these live quotes.
    # It never submits, modifies, or cancels orders at Alpaca.


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
    q = sub.add_parser("quote")
    q.add_argument("--symbols", default=None, help="equity symbols, comma-separated")
    q.add_argument("--contracts", default=None, help="OCC option symbols, comma-separated")
    ch = sub.add_parser("chain", help="option chain around the money (OPRA)")
    ch.add_argument("--symbol", required=True)
    ch.add_argument("--dte-max", type=int, default=60)
    ch.add_argument("--moneyness", type=float, default=0.20)
    asrt = sub.add_parser("assets", help="enumerate the tradable universe")
    asrt.add_argument("--optionable", action="store_true",
                      help="only names with listed options")
    asrt.add_argument("--fractionable", action="store_true",
                      help="only names that fill in dollar notional")
    asrt.add_argument("--limit", type=int, default=None,
                      help="cap the returned list (symbols only when set)")
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
        out = {}
        if args.symbols:
            out.update(b.quotes([s.strip().upper() for s in args.symbols.split(",") if s.strip()]))
        if args.contracts:
            out.update(b.option_quotes([s.strip().upper() for s in args.contracts.split(",") if s.strip()]))
        if not out:
            out = {"error": "pass --symbols and/or --contracts"}
    elif args.cmd == "chain":
        out = b.option_chain(args.symbol, dte_max=args.dte_max,
                             moneyness=args.moneyness)
    elif args.cmd == "assets":
        assets = b.list_assets(optionable=args.optionable,
                               fractionable_only=args.fractionable)
        if args.limit is not None:
            out = {"count": len(assets),
                   "symbols": [a["symbol"] for a in assets[:args.limit]]}
        else:
            out = {"count": len(assets), "assets": assets}
    else:  # pragma: no cover
        out = {"error": "unknown command"}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

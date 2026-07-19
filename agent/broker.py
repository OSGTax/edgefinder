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


def is_crypto(symbol: str) -> bool:
    """Alpaca crypto pairs carry a slash (BTC/USD, ETH/USD, DOGE/USD).
    Equities and OCC option symbols never do — the slash is the tell,
    same test the SDK uses to route quote requests."""
    return "/" in (symbol or "")


def _et_utc_offset(utc_dt):
    """UTC offset for US Eastern time at ``utc_dt`` (DST-aware, dependency-free).
    DST runs from the 2nd Sunday of March through the 1st Sunday of November;
    ET is UTC−4 during DST, UTC−5 otherwise.
    """
    from datetime import date, timedelta
    y = utc_dt.year

    def _nth_sunday(month, n):
        d = date(y, month, 1)
        d += timedelta(days=(6 - d.weekday()) % 7)  # first Sunday
        return d + timedelta(days=7 * (n - 1))

    dst_start = _nth_sunday(3, 2)  # 2nd Sunday of March, 02:00 local
    dst_end = _nth_sunday(11, 1)   # 1st Sunday of November, 02:00 local
    # crude: switch at 07:00 UTC on those days (02:00 ET) — good enough for the
    # ~2h ambiguity window nobody trades in anyway
    from datetime import datetime as _dt
    dst_on = (_dt(y, dst_start.month, dst_start.day, 7)
              <= utc_dt.replace(tzinfo=None)
              < _dt(y, dst_end.month, dst_end.day, 7))
    return timedelta(hours=-4 if dst_on else -5)


def _today_et():
    """Today's ET calendar date. Process-local ``date.today()`` is already
    'tomorrow' after ~20:00 ET on the UTC hosts this runs on — which shifted
    option DTE math and the corporate-announcement window by a day every
    evening."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).date()


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


def normalize_news(n) -> dict:
    """Alpaca news object → the desk's ticker_news shape (pure). ``symbols`` is
    the article's tagged tickers, fanned out one row per symbol on write."""
    g = n.get if isinstance(n, dict) else lambda k, d=None: getattr(n, k, d)
    ca = g("created_at")
    published = (ca.isoformat() if hasattr(ca, "isoformat")
                 else (str(ca) if ca else None))
    return {
        "id": g("id"),
        "title": g("headline"),
        "author": g("author"),
        "publisher": g("source"),
        "url": g("url"),
        "description": (g("summary") or "")[:1000],
        "published_utc": published,
        "symbols": [str(s).upper() for s in (g("symbols") or [])],
    }


def normalize_corp_action(a) -> dict:
    """Alpaca corporate-announcement object → a flat dict (pure). Covers the
    types the desk cares about: cash dividends and stock splits."""
    g = a.get if isinstance(a, dict) else lambda k, d=None: getattr(a, k, d)
    ca_type = g("ca_type")
    ca_type = (ca_type.value if hasattr(ca_type, "value")
               else (str(ca_type) if ca_type else None))

    def _d(x):
        return x.isoformat() if hasattr(x, "isoformat") else (str(x) if x else None)

    return {
        "ca_type": ca_type,
        "symbol": ((g("initiating_symbol") or g("target_symbol") or "")).upper(),
        "ex_date": _d(g("ex_date")),
        "payable_date": _d(g("payable_date")),
        "record_date": _d(g("record_date")),
        "cash": _f(g("cash")),
        "old_rate": _f(g("old_rate")),
        "new_rate": _f(g("new_rate")),
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

    @property
    def crypto_data(self):
        if getattr(self, "_crypto_data", None) is None:
            from alpaca.data.historical.crypto import CryptoHistoricalDataClient
            # Crypto data endpoint doesn't require creds for reads but pass
            # them so the same rate-limit bucket is used across product lines.
            self._crypto_data = CryptoHistoricalDataClient(self._c["key"],
                                                           self._c["secret"])
        return self._crypto_data

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
        """Latest real-time quotes. Automatically routes crypto pairs
        (BTC/USD, ETH/USD, …) to the crypto data endpoint and equities to
        SIP/IEX per settings. Each caller can pass a mixed list — the
        result dict keys are the input symbols."""
        eq = [s for s in symbols if not is_crypto(s)]
        cx = [s for s in symbols if is_crypto(s)]
        out: dict[str, dict] = {}
        if eq:
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.enums import DataFeed
            feed = DataFeed.SIP if self._c["feed"] == "sip" else DataFeed.IEX
            req = StockLatestQuoteRequest(symbol_or_symbols=eq, feed=feed)
            res = self.data.get_stock_latest_quote(req)
            out.update({sym: normalize_quote(sym, q) for sym, q in res.items()})
        if cx:
            from alpaca.data.requests import CryptoLatestQuoteRequest
            req = CryptoLatestQuoteRequest(symbol_or_symbols=cx)
            res = self.crypto_data.get_crypto_latest_quote(req)
            out.update({sym: normalize_quote(sym, q) for sym, q in res.items()})
        return out

    def intraday_bars(self, symbols: list[str], *, timeframe: str = "15Min",
                      limit: int = 32) -> dict[str, list[dict]]:
        """Recent intraday bars — the structure between yesterday's daily bar
        and this second's quote, for short-term work. Read-only, never stored
        (daily_bars stays daily; this is a live glance, not an asset). Routes
        crypto pairs to the crypto endpoint. ``timeframe``: 1Min/5Min/15Min/
        30Min/1Hour."""
        from datetime import datetime, timedelta, timezone

        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        amounts = {"1Min": (1, TimeFrameUnit.Minute),
                   "5Min": (5, TimeFrameUnit.Minute),
                   "15Min": (15, TimeFrameUnit.Minute),
                   "30Min": (30, TimeFrameUnit.Minute),
                   "1Hour": (1, TimeFrameUnit.Hour)}
        if timeframe not in amounts:
            raise ValueError(f"timeframe must be one of {sorted(amounts)}")
        amount, unit = amounts[timeframe]
        tf = TimeFrame(amount, unit)
        # Enough lookback to fill `limit` bars across a weekend/overnight gap.
        minutes = amount * (60 if unit == TimeFrameUnit.Hour else 1)
        start = (datetime.now(timezone.utc)
                 - timedelta(minutes=minutes * limit * 4 + 4320))

        def _rows(res, syms):
            out = {}
            for sym in syms:
                bars = res.data.get(sym) or []
                out[sym] = [{"t": str(b.timestamp), "o": _f(b.open),
                             "h": _f(b.high), "l": _f(b.low),
                             "c": _f(b.close), "v": _f(b.volume)}
                            for b in bars][-limit:]
            return out

        eq = [s for s in symbols if not is_crypto(s)]
        cx = [s for s in symbols if is_crypto(s)]
        out: dict[str, list[dict]] = {}
        if eq:
            from alpaca.data.enums import DataFeed
            feed = DataFeed.SIP if self._c["feed"] == "sip" else DataFeed.IEX
            res = self.data.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=eq, timeframe=tf, start=start, feed=feed))
            out.update(_rows(res, eq))
        if cx:
            res = self.crypto_data.get_crypto_bars(CryptoBarsRequest(
                symbol_or_symbols=cx, timeframe=tf, start=start))
            out.update(_rows(res, cx))
        return out

    def list_assets(self, *, optionable: bool = False,
                    fractionable_only: bool = False,
                    asset_class: str = "us_equity") -> list[dict]:
        """Every ACTIVE, TRADABLE asset in ``asset_class``. Defaults to US
        equities/ETFs (~13k names) — the direct replacement for Polygon's
        grouped-daily universe. Pass ``asset_class="crypto"`` for the ~30
        crypto pairs Alpaca supports on the paper account (BTC/USD,
        ETH/USD, …). ``optionable`` keeps only names with listed options
        (equities only); ``fractionable_only`` keeps names that fill in
        dollar notional cleanly (all crypto is fractionable). Read-only;
        one paginated Trading-API call."""
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetStatus, AssetClass

        cls = (AssetClass.CRYPTO if str(asset_class).lower() == "crypto"
               else AssetClass.US_EQUITY)
        req = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=cls)
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

    def news(self, symbols, *, limit: int = 50) -> list[dict]:
        """Recent headlines for ``symbols`` (Alpaca's news API — Benzinga feed),
        each normalized to the desk's ticker_news shape. One batched call; the
        Alpaca replacement for the retired Polygon news source."""
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        if getattr(self, "_news_client", None) is None:
            self._news_client = NewsClient(self._c["key"], self._c["secret"])
        syms = (",".join(symbols) if isinstance(symbols, (list, tuple))
                else str(symbols))
        req = NewsRequest(symbols=syms, limit=limit, exclude_contentless=True,
                          include_content=False)
        res = self._news_client.get_news(req)
        items = res.data.get("news", []) if hasattr(res, "data") else (res or [])
        return [normalize_news(n) for n in items]

    def corporate_announcements(self, *, since=None, until=None,
                                symbol: str | None = None) -> list[dict]:
        """Cash dividends + stock splits from Alpaca (the retired Polygon
        corporate-actions replacement). Alpaca caps the window at 90 days, so
        callers pass a ≤90-day (since, until). Read-only."""
        from datetime import timedelta

        from alpaca.trading.enums import CorporateActionType
        from alpaca.trading.requests import GetCorporateAnnouncementsRequest

        since = since or (_today_et() - timedelta(days=45))
        until = until or (_today_et() + timedelta(days=45))
        req = GetCorporateAnnouncementsRequest(
            ca_types=[CorporateActionType.DIVIDEND, CorporateActionType.SPLIT],
            since=since, until=until, symbol=symbol)
        return [normalize_corp_action(a)
                for a in self.trading.get_corporate_announcements(req)]

    def is_market_open(self) -> bool:
        return bool(getattr(self.trading.get_clock(), "is_open", False))

    def calendar_day(self, day) -> dict | None:
        """The equity trading calendar's row for ``day`` (an ET date):
        ``{"date", "open", "close"}`` with open/close as naive ET datetimes
        (16:00 close normally, 13:00 on half-days), or None when the market
        has no session that day (weekend/holiday). Cached per date — the
        calendar never changes intraday."""
        key = str(day)
        cache = getattr(self, "_calendar_cache", None)
        if cache is None:
            cache = self._calendar_cache = {}
        if key not in cache:
            from alpaca.trading.requests import GetCalendarRequest

            rows = self.trading.get_calendar(GetCalendarRequest(start=day, end=day))
            row = next((r for r in (rows or [])
                        if str(getattr(r, "date", None)) == key), None)
            cache[key] = ({"date": key, "open": row.open, "close": row.close}
                          if row is not None else None)
        return cache[key]

    def session(self, symbol: str | None = None, *, now_utc=None) -> str:
        """Which session are we in RIGHT NOW: 'regular' | 'extended' |
        'closed' | 'crypto'.

        Crypto pairs (BTC/USD, ETH/USD, …) trade 24/7 with no equity
        clock — pass the symbol to get 'crypto' instead of the equity
        session. Without a symbol the answer is for equity/options.

        Regular = Alpaca's clock says is_open (09:30–16:00 ET on a trading
        day). Extended = 04:00–20:00 ET on a trading day but outside RTH —
        Alpaca supports these hours for equities. Closed = weekends,
        holidays, or overnight. Options fills are blocked outside 'regular'
        (OPRA book is genuinely bad outside RTH — enforced in
        ledger.live_fill).
        """
        if symbol is not None and is_crypto(symbol):
            return "crypto"
        clk = self.trading.get_clock()
        if bool(getattr(clk, "is_open", False)):
            return "regular"
        # `next_open` is a tz-aware UTC datetime; when it's today's ET-date's
        # session, we are pre-market on a trading day. Post-market comes from
        # the trading CALENDAR. Anything else (weekend, holiday,
        # overnight-past-8pm) is closed.
        from datetime import datetime, timezone
        now = now_utc or datetime.now(timezone.utc)
        et_offset = _et_utc_offset(now)  # respects DST
        now_et = (now + et_offset).replace(tzinfo=None)
        nxt_open = getattr(clk, "next_open", None)

        def _to_et(dt):
            if dt is None:
                return None
            if getattr(dt, "tzinfo", None) is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return (dt.astimezone(timezone.utc) + et_offset).replace(tzinfo=None)

        no_ = _to_et(nxt_open)
        # Pre-market: today (ET) matches next_open's ET date AND we're past 04:00.
        if no_ and now_et.date() == no_.date() and now_et.hour >= 4 and now_et < no_:
            return "extended"
        # Post-market: once RTH ends, Alpaca's clock flips next_close to the
        # NEXT session's close, so the old "next_close is today" test was
        # unsatisfiable — dead code that silently refused every 16:00-20:00
        # exit as 'closed'. Derive it from the trading calendar instead: the
        # market HAD a session today (ET) and we're between its actual close
        # (16:00, or 13:00 on half-days — the calendar row knows) and 20:00.
        if now_et.hour < 20:
            try:
                cal = self.calendar_day(now_et.date())
            except Exception:
                cal = None  # calendar unavailable → fail closed, never open
            if cal is not None and now_et >= cal["close"]:
                return "extended"
        return "closed"

    def is_close_soon(self, minutes: int = 15) -> bool:
        """True when the RTH close is within ``minutes`` — refuse to open new
        positions this close to the bell, since we can't sell them until
        tomorrow (or next Monday on a Friday). Uses Alpaca's clock as truth.
        """
        clk = self.trading.get_clock()
        if not bool(getattr(clk, "is_open", False)):
            return False
        from datetime import datetime, timezone, timedelta
        nc = getattr(clk, "next_close", None)
        if nc is None:
            return False
        if getattr(nc, "tzinfo", None) is None:
            nc = nc.replace(tzinfo=timezone.utc)
        return (nc - datetime.now(timezone.utc)) <= timedelta(minutes=minutes)

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
        from datetime import timedelta as _td

        from alpaca.data.requests import OptionChainRequest

        underlying = underlying.upper().strip()
        et_today = _today_et()
        uq = self.quotes([underlying]).get(underlying) or {}
        px = uq.get("mid") or uq.get("ask") or uq.get("bid")
        req_kwargs = {"underlying_symbol": underlying,
                      "expiration_date_lte": et_today + _td(days=dte_max)}
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
                "dte": (p["expiry"] - et_today).days,
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
    ss = sub.add_parser("session",
                         help="regular | extended | closed | crypto (per Alpaca clock)")
    ss.add_argument("--symbol", default=None,
                    help="pass a crypto pair (e.g. BTC/USD) to get 'crypto' instead of equity session")
    q = sub.add_parser("quote")
    q.add_argument("--symbols", default=None,
                   help="equity or crypto symbols, comma-separated (crypto uses a slash: BTC/USD)")
    q.add_argument("--contracts", default=None, help="OCC option symbols, comma-separated")
    ch = sub.add_parser("chain", help="option chain around the money (OPRA)")
    ch.add_argument("--symbol", required=True)
    ch.add_argument("--dte-max", type=int, default=60)
    ch.add_argument("--moneyness", type=float, default=0.20)
    ib = sub.add_parser("bars", help="recent INTRADAY bars (live glance, not stored)")
    ib.add_argument("--symbols", required=True,
                    help="comma-separated; crypto pairs use a slash (BTC/USD)")
    ib.add_argument("--timeframe", default="15Min",
                    choices=["1Min", "5Min", "15Min", "30Min", "1Hour"])
    ib.add_argument("--limit", type=int, default=32)
    asrt = sub.add_parser("assets", help="enumerate the tradable universe")
    asrt.add_argument("--optionable", action="store_true",
                      help="only names with listed options (equities only)")
    asrt.add_argument("--fractionable", action="store_true",
                      help="only names that fill in dollar notional")
    asrt.add_argument("--crypto", action="store_true",
                      help="list crypto pairs instead of US equities")
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
    elif args.cmd == "session":
        sym = (args.symbol or "").strip().upper() or None
        out = {"symbol": sym,
               "session": b.session(sym),
               "close_soon": (b.is_close_soon()
                              if (sym is None or not is_crypto(sym))
                              and b.is_market_open() else False)}
    elif args.cmd == "quote":
        out = {}
        if args.symbols:
            out.update(b.quotes([s.strip().upper() for s in args.symbols.split(",") if s.strip()]))
        if args.contracts:
            out.update(b.option_quotes([s.strip().upper() for s in args.contracts.split(",") if s.strip()]))
        if not out:
            out = {"error": "pass --symbols and/or --contracts"}
    elif args.cmd == "bars":
        out = b.intraday_bars(
            [s.strip().upper() for s in args.symbols.split(",") if s.strip()],
            timeframe=args.timeframe, limit=args.limit)
    elif args.cmd == "chain":
        out = b.option_chain(args.symbol, dte_max=args.dte_max,
                             moneyness=args.moneyness)
    elif args.cmd == "assets":
        assets = b.list_assets(
            optionable=args.optionable,
            fractionable_only=args.fractionable,
            asset_class=("crypto" if args.crypto else "us_equity"),
        )
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

"""Test doubles for the Alpaca trading surface (REBUILD-V4).

``FakeTradingClient`` implements the slice of ``alpaca-py``'s TradingClient
that ``agent.trade`` consumes — order submit/lookup/cancel, positions,
account, the ``/account/activities`` REST passthrough, portfolio history,
and account configurations. Orders are plain dicts (``agent.trade``'s
normalizers accept dicts and SDK objects alike); the account is a namespace
so ``getattr`` reads work.

Fill behavior: ``fill_mode="instant"`` fills marketable orders at
``prices[symbol]`` immediately; ``"working"`` leaves them status ``new``
(the caller polls, as against the real paper engine).
"""

from __future__ import annotations

from types import SimpleNamespace


class FakeTradingClient:
    def __init__(self, *, prices: dict[str, float] | None = None,
                 equity: float = 100_000.0, cash: float = 100_000.0,
                 positions: list[dict] | None = None,
                 activities: list[dict] | None = None,
                 fill_mode: str = "instant",
                 reject_duplicate_ids: bool = True):
        self.prices = dict(prices or {})
        self.equity = equity
        self.cash = cash
        self._positions = list(positions or [])
        self._activities = list(activities or [])
        self.fill_mode = fill_mode
        self.reject_duplicate_ids = reject_duplicate_ids
        self.orders: dict[str, dict] = {}
        self.submitted_requests: list = []
        self.canceled: list[str] = []
        self.config: dict = {}
        self.fail_next_submit: Exception | None = None
        self._seq = 0

    # -- helpers --

    def _next_id(self) -> str:
        self._seq += 1
        return f"o-{self._seq}"

    @staticmethod
    def _val(v):
        return getattr(v, "value", v)

    def add_order(self, **kw) -> dict:
        """Seed an order directly (e.g. a pre-existing open stop)."""
        o = {"id": kw.get("id") or self._next_id(), "client_order_id": None,
             "symbol": None, "side": "buy", "qty": None, "notional": None,
             "order_type": "market", "time_in_force": "day",
             "order_class": "simple", "limit_price": None, "stop_price": None,
             "status": "new", "filled_qty": None, "filled_avg_price": None,
             "submitted_at": "2026-08-08T14:00:00+00:00", "filled_at": None,
             "canceled_at": None, "legs": []}
        o.update(kw)
        self.orders[o["id"]] = o
        return o

    # -- TradingClient surface --

    def get_account(self):
        return SimpleNamespace(
            account_number="PA-FAKE", status="ACTIVE", currency="USD",
            cash=self.cash, equity=self.equity, last_equity=self.equity,
            buying_power=self.cash, options_buying_power=self.cash,
            options_approved_level=3, options_trading_level=3,
            shorting_enabled=False, multiplier="1",
            long_market_value=self.equity - self.cash, short_market_value=0.0)

    def get_all_positions(self):
        return list(self._positions)

    def submit_order(self, req):
        if self.fail_next_submit is not None:
            exc, self.fail_next_submit = self.fail_next_submit, None
            raise exc
        self.submitted_requests.append(req)
        g = lambda k, d=None: getattr(req, k, d)  # noqa: E731
        cid = g("client_order_id")
        if self.reject_duplicate_ids and cid and any(
                o.get("client_order_id") == cid for o in self.orders.values()):
            raise ValueError(f"client_order_id must be unique: {cid}")
        legs = []
        for leg in (g("legs") or []):
            legs.append({"id": self._next_id(),
                         "client_order_id": f"auto-{self._seq}",
                         "symbol": self._val(getattr(leg, "symbol", None)),
                         "side": str(self._val(getattr(leg, "side", ""))).lower(),
                         "qty": getattr(leg, "ratio_qty", None),
                         "order_type": str(self._val(g("type") or "limit")).lower(),
                         "time_in_force": str(self._val(g("time_in_force"))).lower(),
                         "order_class": "mleg", "status": "new",
                         "filled_qty": None, "filled_avg_price": None,
                         "limit_price": None, "stop_price": None,
                         "submitted_at": "2026-08-08T14:30:00+00:00",
                         "filled_at": None, "canceled_at": None, "legs": []})
        sym = (g("symbol") or (legs[0]["symbol"] if legs else None))
        o = {"id": self._next_id(), "client_order_id": cid,
             "symbol": sym,
             "side": str(self._val(g("side") or "buy")).lower(),
             "qty": float(g("qty")) if g("qty") is not None else None,
             "notional": float(g("notional")) if g("notional") is not None else None,
             "order_type": str(self._val(g("type") or "market")).lower(),
             "time_in_force": str(self._val(g("time_in_force") or "day")).lower(),
             "order_class": str(self._val(g("order_class") or "simple")).lower(),
             "limit_price": g("limit_price"), "stop_price": g("stop_price"),
             "status": "new", "filled_qty": None, "filled_avg_price": None,
             "submitted_at": "2026-08-08T14:30:00+00:00", "filled_at": None,
             "canceled_at": None, "legs": legs}
        marketable = (o["order_type"] == "market"
                      or (o["order_type"] == "limit"
                          and o["symbol"] in self.prices
                          and o["limit_price"] is not None
                          and ((o["side"] == "buy"
                                and o["limit_price"] >= self.prices[o["symbol"]])
                               or (o["side"] == "sell"
                                   and o["limit_price"] <= self.prices[o["symbol"]]))))
        if self.fill_mode == "instant" and marketable \
                and o["order_type"] not in ("stop", "stop_limit"):
            px = self.prices.get(o["symbol"], 100.0)
            qty = o["qty"] if o["qty"] is not None else \
                (round(o["notional"] / px, 9) if o["notional"] else None)
            o.update({"status": "filled", "filled_qty": qty,
                      "filled_avg_price": px,
                      "filled_at": "2026-08-08T14:30:01+00:00"})
            for leg in o["legs"]:
                leg.update({"status": "filled", "filled_qty": leg["qty"],
                            "filled_avg_price": self.prices.get(leg["symbol"], 1.0),
                            "filled_at": o["filled_at"]})
        self.orders[o["id"]] = o
        return o

    def get_order_by_id(self, order_id):
        return self.orders[str(order_id)]

    def get_order_by_client_id(self, cid):
        for o in self.orders.values():
            if o.get("client_order_id") == cid:
                return o
        raise KeyError(f"no order with client_order_id {cid}")

    def get_orders(self, req=None):
        status = str(self._val(getattr(req, "status", "all") or "all")).lower()
        symbols = getattr(req, "symbols", None)
        out = []
        for o in self.orders.values():
            if status == "open" and o["status"] not in (
                    "new", "accepted", "partially_filled", "pending_new", "held"):
                continue
            if status == "closed" and o["status"] in (
                    "new", "accepted", "partially_filled", "pending_new", "held"):
                continue
            if symbols and o.get("symbol") not in symbols:
                continue
            out.append(o)
        return out

    def cancel_order_by_id(self, order_id):
        o = self.orders[str(order_id)]
        if o["status"] not in ("filled",):
            o["status"] = "canceled"
            o["canceled_at"] = "2026-08-08T14:31:00+00:00"
        self.canceled.append(str(order_id))

    # -- REST passthrough (activities + configurations) --

    def get(self, path, params=None):
        if path == "/account/activities":
            params = params or {}
            after = params.get("after")
            page_size = int(params.get("page_size", 100))
            token = params.get("page_token")
            rows = [a for a in self._activities
                    if not after or (a.get("date") or a.get(
                        "transaction_time", ""))[:10] >= str(after)[:10]]
            rows.sort(key=lambda a: str(a.get("id")))
            if token:
                rows = [a for a in rows if str(a.get("id")) > str(token)]
            return rows[:page_size]
        if path == "/account/configurations":
            return dict(self.config)
        raise KeyError(f"FakeTradingClient.get: unhandled path {path}")

    def patch(self, path, body=None):
        if path == "/account/configurations":
            self.config.update(body or {})
            return dict(self.config)
        raise KeyError(f"FakeTradingClient.patch: unhandled path {path}")

    def get_portfolio_history(self, req=None):
        return SimpleNamespace(
            timestamp=[1754600400, 1754686800],
            equity=[self.equity, self.equity],
            profit_loss=[0.0, 0.0], profit_loss_pct=[0.0, 0.0],
            base_value=self.equity, timeframe="1D")

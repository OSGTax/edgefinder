"""Unit-test the SDK-free, network-free logic of the Alpaca broker wrapper.

The live calls need paper keys (Phase 1 proof); here we lock down the pure
parts: credential resolution, order validation/spec building, and the
normalizers that turn Alpaca objects into the desk's plain dicts.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import broker


def test_enabled_and_creds(monkeypatch):
    # nothing set anywhere → disabled
    monkeypatch.setattr(broker.settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(broker.settings, "alpaca_api_secret", "", raising=False)
    for v in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"):
        monkeypatch.delenv(v, raising=False)
    assert broker.enabled() is False
    with pytest.raises(RuntimeError):
        broker.Broker()

    # native SDK env vars are accepted as a fallback
    monkeypatch.setenv("APCA_API_KEY_ID", "k123")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s456")
    c = broker.resolve_creds()
    assert c["key"] == "k123" and c["secret"] == "s456"
    assert broker.enabled() is True

    # settings take precedence over env
    monkeypatch.setattr(broker.settings, "alpaca_api_key", "fromsettings", raising=False)
    assert broker.resolve_creds()["key"] == "fromsettings"


def test_broker_is_read_only():
    """The data-reader contract: no order-write surface exists, at all."""
    assert not hasattr(broker, "build_order")
    assert not hasattr(broker.Broker, "submit")
    for name in dir(broker.Broker):
        assert not any(w in name.lower() for w in ("submit", "cancel", "replace")), name


def test_normalize_position():
    p = SimpleNamespace(symbol="NVDA", qty="100", avg_entry_price="120.0",
                        current_price="130.0", market_value="13000.0",
                        cost_basis="12000.0", unrealized_pl="1000.0",
                        unrealized_plpc="0.0833", side="long")
    n = broker.normalize_position(p)
    assert n["symbol"] == "NVDA" and n["qty"] == 100.0
    assert n["unrealized_pl"] == 1000.0 and n["current_price"] == 130.0


def test_normalize_order_fill():
    o = SimpleNamespace(id="abc-1", symbol="NVDA", side="OrderSide.BUY", qty="8",
                        filled_qty="8", filled_avg_price="131.27",
                        order_type="OrderType.MARKET", status="OrderStatus.FILLED",
                        submitted_at="2026-06-23T14:30:00Z", filled_at="2026-06-23T14:30:01Z")
    n = broker.normalize_order(o)
    assert n["id"] == "abc-1" and n["side"] == "buy" and n["status"] == "filled"
    assert n["filled_avg_price"] == 131.27 and n["type"] == "market"


def test_normalize_quote_mid():
    q = SimpleNamespace(bid_price=130.0, ask_price=130.4, bid_size=3, ask_size=5,
                        timestamp="2026-06-23T14:30:00Z")
    n = broker.normalize_quote("NVDA", q)
    assert n["bid"] == 130.0 and n["ask"] == 130.4 and n["mid"] == 130.2


def test_normalize_asset_optionable_and_flags():
    a = SimpleNamespace(symbol="aapl", name="Apple Inc.",
                        exchange="NASDAQ", tradable=True, fractionable=True,
                        shortable=True, attributes=["has_options", "overnight_tradable"])
    out = broker.normalize_asset(a)
    assert out["symbol"] == "AAPL"           # upper-cased
    assert out["exchange"] == "NASDAQ"
    assert out["tradable"] is True and out["fractionable"] is True
    assert out["has_options"] is True         # read from attributes list


def test_normalize_asset_non_optionable_defaults():
    a = SimpleNamespace(symbol="XYZ", name=None, exchange="NYSE",
                        tradable=False, fractionable=False, shortable=False,
                        attributes=None)
    out = broker.normalize_asset(a)
    assert out["has_options"] is False        # empty attributes → not optionable
    assert out["tradable"] is False
    assert out["symbol"] == "XYZ"


def test_normalize_news_maps_alpaca_to_ticker_news_shape():
    n = SimpleNamespace(id=42, headline="Lilly pops on trial data", author="J. Doe",
                        source="benzinga", summary="Phase 3 win.",
                        url="https://x/y", symbols=["lly", "nvo"],
                        created_at=SimpleNamespace(isoformat=lambda: "2026-07-08T16:00:00+00:00"))
    out = broker.normalize_news(n)
    assert out["title"] == "Lilly pops on trial data"
    assert out["publisher"] == "benzinga"        # source → publisher
    assert out["url"] == "https://x/y"
    assert out["published_utc"] == "2026-07-08T16:00:00+00:00"
    assert out["symbols"] == ["LLY", "NVO"]       # upper-cased


def test_broker_session_post_market_from_calendar(monkeypatch):
    """F9: after 16:00 Alpaca's clock points next_close at the NEXT session,
    so the old 'next_close is today' post-market test was unsatisfiable dead
    code — 16:00-20:00 ET exits were refused as 'closed'. Post-market now
    derives from the trading CALENDAR: a session happened today (ET) and now
    is between its actual close (16:00, or 13:00 on half-days) and 20:00."""
    from datetime import date, datetime, timezone

    from agent import broker

    class _Trading:
        def __init__(self, is_open, nxt_open, cal_rows):
            self._c = SimpleNamespace(is_open=is_open, next_open=nxt_open,
                                      next_close=None)
            self._cal = cal_rows

        def get_clock(self):
            return self._c

        def get_calendar(self, req=None):
            return self._cal

    monkeypatch.setenv("EDGEFINDER_ALPACA_API_KEY", "k")
    monkeypatch.setenv("EDGEFINDER_ALPACA_API_SECRET", "s")
    from config.settings import settings as _s
    _s.alpaca_api_key = "k"; _s.alpaca_api_secret = "s"

    def cal_row(d, close_h, close_m=0):
        return SimpleNamespace(
            date=d,
            open=datetime(d.year, d.month, d.day, 9, 30),
            close=datetime(d.year, d.month, d.day, close_h, close_m))

    def make(is_open, nxt_open, cal_rows):
        b = broker.Broker()
        b._trading = _Trading(is_open, nxt_open, cal_rows)
        return b

    day = date(2026, 7, 8)                            # a Wednesday; EDT = UTC-4
    next_open = datetime(2026, 7, 9, 13, 30, tzinfo=timezone.utc)  # Thu 09:30 ET

    # regular day, 17:30 ET (21:30 UTC) → extended (after-hours)
    b = make(False, next_open, [cal_row(day, 16)])
    at = datetime(2026, 7, 8, 21, 30, tzinfo=timezone.utc)
    assert b.session(now_utc=at) == "extended"

    # half-day (13:00 close), 13:30 ET → extended
    b = make(False, next_open, [cal_row(day, 13)])
    at = datetime(2026, 7, 8, 17, 30, tzinfo=timezone.utc)
    assert b.session(now_utc=at) == "extended"

    # regular day, 21:00 ET → closed (extended hours end at 20:00)
    b = make(False, next_open, [cal_row(day, 16)])
    at = datetime(2026, 7, 9, 1, 0, tzinfo=timezone.utc)   # 21:00 ET on the 8th
    assert b.session(now_utc=at) == "closed"

    # weekend, 17:30 ET → closed (no calendar session that day)
    sat = date(2026, 7, 11)
    b = make(False, datetime(2026, 7, 13, 13, 30, tzinfo=timezone.utc), [])
    at = datetime(2026, 7, 11, 21, 30, tzinfo=timezone.utc)
    assert b.session(now_utc=at) == "closed"
    assert sat.weekday() == 5  # the scenario really is a Saturday

    # regular day 15:00 ET with the clock open → regular (clock wins)
    b = make(True, None, [cal_row(day, 16)])
    at = datetime(2026, 7, 8, 19, 0, tzinfo=timezone.utc)
    assert b.session(now_utc=at) == "regular"

    # pre-market path intact: 08:00 ET before the same day's 09:30 open
    b = make(False, datetime(2026, 7, 8, 13, 30, tzinfo=timezone.utc),
             [cal_row(day, 16)])
    at = datetime(2026, 7, 8, 12, 0, tzinfo=timezone.utc)
    assert b.session(now_utc=at) == "extended"

    # 01:00 ET on a trading day (overnight) → closed, not post-market
    b = make(False, datetime(2026, 7, 8, 13, 30, tzinfo=timezone.utc),
             [cal_row(day, 16)])
    at = datetime(2026, 7, 8, 5, 0, tzinfo=timezone.utc)
    assert b.session(now_utc=at) == "closed"


def test_normalize_corp_action_dividend_and_split():
    div = SimpleNamespace(ca_type=SimpleNamespace(value="dividend"),
                          initiating_symbol="lly", target_symbol=None,
                          ex_date=SimpleNamespace(isoformat=lambda: "2026-08-15"),
                          payable_date=None, record_date=None,
                          cash=1.73, old_rate=None, new_rate=None)
    out = broker.normalize_corp_action(div)
    assert out["ca_type"] == "dividend" and out["symbol"] == "LLY"
    assert out["ex_date"] == "2026-08-15" and out["cash"] == 1.73

    split = SimpleNamespace(ca_type="split", initiating_symbol="NVDA", target_symbol=None,
                            ex_date=SimpleNamespace(isoformat=lambda: "2026-06-10"),
                            payable_date=None, record_date=None,
                            cash=None, old_rate=1.0, new_rate=10.0)
    o2 = broker.normalize_corp_action(split)
    assert o2["ca_type"] == "split" and o2["old_rate"] == 1.0 and o2["new_rate"] == 10.0

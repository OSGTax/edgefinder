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


def test_build_order_market_and_limit():
    m = broker.build_order("nvda", "BUY", notional=1000)
    assert m == {"symbol": "NVDA", "side": "buy", "type": "market",
                 "time_in_force": "day", "notional": 1000.0}
    q = broker.build_order("AAPL", "sell", qty=10, type="limit", limit_price=210.5)
    assert q["qty"] == 10.0 and q["limit_price"] == 210.5 and q["type"] == "limit"


def test_build_order_validation():
    with pytest.raises(ValueError):  # need exactly one of qty/notional
        broker.build_order("NVDA", "buy")
    with pytest.raises(ValueError):  # ...not both
        broker.build_order("NVDA", "buy", qty=1, notional=100)
    with pytest.raises(ValueError):  # bad side
        broker.build_order("NVDA", "hodl", qty=1)
    with pytest.raises(ValueError):  # limit needs a price
        broker.build_order("NVDA", "buy", qty=1, type="limit")
    with pytest.raises(ValueError):  # notional only for market
        broker.build_order("NVDA", "buy", notional=100, type="limit", limit_price=5)
    with pytest.raises(ValueError):  # positive sizes
        broker.build_order("NVDA", "buy", qty=-1)


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

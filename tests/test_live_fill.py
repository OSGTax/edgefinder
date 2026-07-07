"""Phase-2 tests: fractional shares, the live-quote fill path, bar top-up rows.

live_fill's broker calls are monkeypatched — the REAL quote path is proven on
Render (broker-health + the P2 verifier's scratch fill); here we lock down the
pricing/guard logic.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'fill.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None  # reset the singleton for this DB
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


class FakeBroker:
    """Market open, NVDA quoted 130.00 / 130.20."""
    def __init__(self, open_=True, bid=130.0, ask=130.2):
        self._open, self._bid, self._ask = open_, bid, ask

    def is_market_open(self):
        return self._open

    def quotes(self, symbols):
        return {s: {"symbol": s, "bid": self._bid, "ask": self._ask,
                    "mid": round((self._bid + self._ask) / 2, 4),
                    "t": "2026-07-07T16:00:00Z"} for s in symbols}


def _patch_broker(monkeypatch, fake):
    import agent.broker as broker
    monkeypatch.setattr(broker, "Broker", lambda *a, **k: fake)
    monkeypatch.setattr(broker, "enabled", lambda: True)


def test_live_fill_buy_prices_at_ask_plus_slip(store, monkeypatch):
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker())
    r = ledger.live_fill(store, symbol="nvda", side="buy", notional=5000,
                         rationale="test", run_id="T1")
    assert r["ok"], r
    assert r["price"] == round(130.2 * 1.0001, 4)          # ask + 1bp
    assert abs(r["shares"] * r["price"] - 5000) < 1.0      # notional sizing
    assert r["shares"] != int(r["shares"])                  # fractional booked
    assert r["fill_quote"]["bid"] == 130.0 and r["fill_quote"]["ask"] == 130.2
    assert r["fill_quote"]["src"] == "alpaca_sip_rest"

    # the snapshot survives the round-trip to the ledger row
    rows = store.select("desk_trades", filters={"symbol": "NVDA"})
    assert rows and rows[0]["fill_quote"]["ask"] == 130.2


def test_live_fill_sell_prices_at_bid_minus_slip(store, monkeypatch):
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker())
    ledger.live_fill(store, symbol="NVDA", side="buy", shares=10, run_id="T1")
    r = ledger.live_fill(store, symbol="NVDA", side="sell", shares=4, run_id="T1")
    assert r["ok"], r
    assert r["price"] == round(130.0 * 0.9999, 4)           # bid - 1bp
    held = store.select("desk_positions", filters={"symbol": "NVDA"})[0]
    assert abs(float(held["shares"]) - 6) < 1e-9


def test_live_fill_rejects_when_market_closed(store, monkeypatch):
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker(open_=False))
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert not r["ok"] and "market closed" in r["error"]


def test_live_fill_rejects_degenerate_quote(store, monkeypatch):
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker(bid=130.0, ask=140.0))  # >5% spread
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert not r["ok"] and "degenerate" in r["error"]


def test_record_trade_live_band_guard(store):
    from agent import ledger
    q = {"bid": 130.0, "ask": 130.2, "mid": 130.1, "t": "x", "src": "test"}
    off = ledger.record_trade(store, symbol="NVDA", side="BUY", shares=1,
                              price=140.0, fill_quote=q)
    assert not off["ok"] and "off the live quote" in off["error"]
    on = ledger.record_trade(store, symbol="NVDA", side="BUY", shares=1,
                             price=130.21, fill_quote=q)
    assert on["ok"]


def test_fractional_cash_integrity(store):
    from agent import ledger
    q = {"bid": 100.0, "ask": 100.0, "mid": 100.0, "t": "x", "src": "test"}
    ledger.record_trade(store, symbol="SPY", side="BUY", shares=2.5,
                        price=100.0, fill_quote=q)
    assert ledger.cash(store) == 100_000.0 - 250.0
    ledger.record_trade(store, symbol="SPY", side="SELL", shares=1.25,
                        price=100.0, fill_quote=q)
    assert ledger.cash(store) == 100_000.0 - 125.0
    pos = store.select("desk_positions", filters={"symbol": "SPY"})[0]
    assert abs(float(pos["shares"]) - 1.25) < 1e-9


def test_alpaca_bars_to_rows():
    from agent.refresh import alpaca_bars_to_rows
    bars = [SimpleNamespace(timestamp=datetime(2026, 7, 3, 4, 0),
                            open=100.0, high=101.0, low=99.5, close=100.5,
                            volume=1_000_000, trade_count=5000.0),  # FLOAT, as Alpaca sends
            SimpleNamespace(timestamp=None, close=1.0)]  # bad row skipped
    rows = alpaca_bars_to_rows(bars, "SPY")
    assert len(rows) == 1
    r = rows[0]
    assert r["symbol"] == "SPY" and str(r["date"]) == "2026-07-03"
    assert r["close"] == 100.5 and r["source"] == "alpaca_daily"
    # regression (agent's first routine run): trade_count floats must be
    # coerced — the INTEGER column rejects them on the REST lane
    assert r["transactions"] == 5000 and isinstance(r["transactions"], int)


def test_no_alpaca_order_writes_anywhere():
    """Contract (P0 verifier caveat): no repo code reaches Alpaca order writes,
    even via the raw .trading client."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for d in ("agent", "dashboard", "edgefinder", "scripts", "config"):
        for f in (root / d).rglob("*.py"):
            src = f.read_text()
            for bad in ("submit_order", "cancel_order", "replace_order",
                        "close_position", "close_all_positions"):
                assert bad not in src, f"{f}: forbidden Alpaca write '{bad}'"


def test_record_trade_rejects_incomplete_fill_quote(store):
    """Regression (P2 verifier): a half-formed snapshot must reject outright,
    never fall back to the loose close band."""
    from agent import ledger
    r = ledger.record_trade(store, symbol="NVDA", side="BUY", shares=1,
                            price=130.0, fill_quote={"ask": 130.2})
    assert not r["ok"] and "missing bid/ask" in r["error"]

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
    """Market open, NVDA quoted 130.00 / 130.20 with a fresh-now timestamp."""
    def __init__(self, open_=True, bid=130.0, ask=130.2, session_=None,
                 close_soon=False, quote_t=None):
        self._open, self._bid, self._ask = open_, bid, ask
        # Default session tracks is_market_open unless explicitly set — so a
        # test that only says `open_=False` still sees the "closed" session.
        self._session = session_ if session_ is not None else (
            "regular" if open_ else "closed")
        self._close_soon = close_soon
        self._quote_t = quote_t

    def is_market_open(self):
        return self._open

    def session(self):
        return self._session

    def is_close_soon(self, minutes=15):
        return self._close_soon

    def quotes(self, symbols):
        from datetime import datetime, timezone
        t = self._quote_t or datetime.now(timezone.utc).isoformat()
        return {s: {"symbol": s, "bid": self._bid, "ask": self._ask,
                    "mid": round((self._bid + self._ask) / 2, 4),
                    "t": t} for s in symbols}

    def option_quotes(self, symbols):
        return self.quotes(symbols)


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


# ── Phase 1: extended hours, freshness, options-band, early-close ────


def test_live_fill_allows_extended_hours_for_equities(store, monkeypatch):
    """Ext-hours (pre-market or post-close) is allowed on equities with a
    tighter spread cap. Session gets stamped on the fill_quote."""
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker(open_=False, session_="extended",
                                          bid=130.0, ask=131.0))  # 0.76% < 2%
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert r["ok"], r
    assert r["fill_quote"]["session"] == "extended"


def test_live_fill_ext_hours_tightens_equity_spread_cap(store, monkeypatch):
    """5% is legal in RTH but rejected in ext-hours (cap 2%)."""
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker(open_=False, session_="extended",
                                          bid=130.0, ask=134.0))  # 3.0% > 2%
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert not r["ok"] and "degenerate" in r["error"]


def test_live_fill_refuses_options_in_extended_hours(store, monkeypatch):
    """Options fills are RTH-only regardless of spread — OPRA book is bad."""
    from agent import ledger, occ
    from datetime import date, timedelta
    sym = occ.build("NVDA", date.today() + timedelta(days=45), "C", 200)
    _patch_broker(monkeypatch, FakeBroker(open_=False, session_="extended",
                                          bid=5.4, ask=5.5))
    r = ledger.live_fill(store, symbol=sym, side="buy", shares=1)
    assert not r["ok"] and "RTH-only" in r["error"]


def test_live_fill_rejects_stale_quote(store, monkeypatch):
    """A quote timestamp older than MAX_QUOTE_AGE_SEC_EQ is rejected — the
    only defense against a feed stuck open on yesterday's book."""
    from agent import ledger
    from datetime import datetime, timezone, timedelta
    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    _patch_broker(monkeypatch, FakeBroker(quote_t=old))
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert not r["ok"] and "stale quote" in r["error"]


def test_live_fill_early_close_refuses_new_position(store, monkeypatch):
    """Within 15 min of close, refuse to open a new position — nothing to
    sell into by 16:00, and we don't want to hold overnight by accident."""
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker(close_soon=True))
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert not r["ok"] and "close in <15m" in r["error"]
    # sells are still allowed close-to-bell (we want to reduce, not hold)
    ledger.record_trade(store, symbol="NVDA", side="BUY", shares=10,
                        price=130.1, fill_quote={"bid": 130.0, "ask": 130.2,
                                                 "mid": 130.1, "t": "x",
                                                 "src": "test"})
    sell = ledger.live_fill(store, symbol="NVDA", side="sell", shares=4)
    assert sell["ok"], sell


def test_record_trade_options_band_wider(store):
    """Option LIVE_BAND is max(2%, $0.05) — cheap contracts get the cents floor.
    Old 0.5% band would have rejected a 1-tick-through fill at $0.36 (band=$0.002)."""
    from agent import ledger, occ
    from datetime import date, timedelta
    sym = occ.build("NVDA", date.today() + timedelta(days=45), "C", 200)
    fq = {"bid": 0.35, "ask": 0.40, "mid": 0.375, "t": "x", "src": "test"}
    # a $0.42 fill sits inside max(2%, $0.05) → 0.40 + 0.05 = $0.45 ceiling
    ok = ledger.record_trade(store, symbol=sym, side="BUY", shares=1,
                             price=0.42, fill_quote=fq)
    assert ok["ok"], ok
    off = ledger.record_trade(store, symbol=sym, side="BUY", shares=1,
                              price=0.55, fill_quote=fq)
    assert not off["ok"] and "off the live quote" in off["error"]


def test_broker_session_helper_states(monkeypatch):
    """Broker.session() returns regular/extended/closed from clock + ET wall."""
    from datetime import datetime, timezone, timedelta

    from agent import broker

    class _Trading:
        def __init__(self, is_open, nxt_open, nxt_close):
            self._c = SimpleNamespace(is_open=is_open, next_open=nxt_open,
                                      next_close=nxt_close)
        def get_clock(self):
            return self._c

    monkeypatch.setenv("EDGEFINDER_ALPACA_API_KEY", "k")
    monkeypatch.setenv("EDGEFINDER_ALPACA_API_SECRET", "s")
    # Force settings to re-read
    from config.settings import settings as _s
    _s.alpaca_api_key = "k"; _s.alpaca_api_secret = "s"

    b = broker.Broker()
    # regular hours: is_open=True
    b._trading = _Trading(True, None, None)
    assert b.session() == "regular"
    # closed: is_open False and both next_open/next_close far future/past
    b._trading = _Trading(False,
                          datetime(2099, 1, 1, 14, 30, tzinfo=timezone.utc),
                          datetime(2099, 1, 1, 21, 0, tzinfo=timezone.utc))
    assert b.session() == "closed"


def test_broker_is_close_soon(monkeypatch):
    from datetime import datetime, timezone, timedelta

    from agent import broker

    class _Trading:
        def __init__(self, is_open, nxt_close):
            self._c = SimpleNamespace(is_open=is_open, next_close=nxt_close)
        def get_clock(self):
            return self._c

    monkeypatch.setenv("EDGEFINDER_ALPACA_API_KEY", "k")
    monkeypatch.setenv("EDGEFINDER_ALPACA_API_SECRET", "s")
    from config.settings import settings as _s
    _s.alpaca_api_key = "k"; _s.alpaca_api_secret = "s"

    b = broker.Broker()
    soon = datetime.now(timezone.utc) + timedelta(minutes=10)
    later = datetime.now(timezone.utc) + timedelta(hours=3)
    b._trading = _Trading(True, soon)
    assert b.is_close_soon(minutes=15) is True
    b._trading = _Trading(True, later)
    assert b.is_close_soon(minutes=15) is False
    # not open → never "close soon"
    b._trading = _Trading(False, soon)
    assert b.is_close_soon(minutes=15) is False

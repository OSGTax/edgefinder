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

    def session(self, symbol=None):
        # Match the real broker: crypto symbols report 'crypto'; anything else
        # returns the fixtured session.
        if symbol and "/" in symbol:
            return "crypto"
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


# ── C1: friction + sanity gates on the live-fill path ────────


def _seed_bars(store, symbol, closes_volumes):
    """Seed (close, volume) daily bars ending yesterday, oldest first."""
    from datetime import date, timedelta
    d = date.today() - timedelta(days=len(closes_volumes))
    for close, vol in closes_volumes:
        d += timedelta(days=1)
        store.insert("daily_bars", {"symbol": symbol, "date": d, "open": close,
                                    "high": close, "low": close, "close": close,
                                    "volume": vol, "source": "test"},
                     returning=False)


@pytest.fixture()
def store_bars(store):
    """The fill-test store with the market-data tables also created."""
    from edgefinder.db.engine import Base, get_engine
    import edgefinder.db.models  # noqa: F401 — daily_bars
    Base.metadata.create_all(get_engine())
    return store


def test_live_fill_rejects_price_far_from_stored_close(store_bars, monkeypatch):
    """C1a: the quote says 130 but the stored close says 90 — a 44% jump is
    either a bad feed or a move that deserves a deliberate override, not a
    reflex fill. The independent close reference is the check the in-quote
    band (which prices off the same quote) can never fail."""
    from agent import ledger
    _seed_bars(store_bars, "NVDA", [(90.0, 5_000_000)] * 20)
    _patch_broker(monkeypatch, FakeBroker())         # 130.0/130.2 quote
    r = ledger.live_fill(store_bars, symbol="NVDA", side="buy", shares=1)
    assert not r["ok"] and "latest stored close" in r["error"]
    assert r["latest_close"] == 90.0 and r["band"] == ledger.LIVE_CLOSE_BAND
    # the opt-in override books it, with the override on the record
    ok = ledger.live_fill(store_bars, symbol="NVDA", side="buy", shares=1,
                          allow_price_deviation=True)
    assert ok["ok"], ok
    assert any("override" in w for w in ok["warnings"])


def test_live_fill_no_stored_close_warns_and_allows(store_bars, monkeypatch):
    """C1a: a new listing has no close row — warn, never block."""
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker())
    r = ledger.live_fill(store_bars, symbol="NVDA", side="buy", shares=1)
    assert r["ok"], r
    assert any("no stored daily close" in w for w in r["warnings"])


def test_live_fill_rejects_unparseable_quote_timestamp(store, monkeypatch):
    """C1b: an age we cannot measure FAILS CLOSED — previously a quote with
    a garbled timestamp walked straight past the staleness cap."""
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker(quote_t="not-a-timestamp"))
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert not r["ok"] and "timestamp missing/unparseable" in r["error"]


def test_live_fill_adv_size_gate(store_bars, monkeypatch):
    """C1c: an order may take at most 1% of the 20-session average dollar
    volume — ADV here is 130 × 10k = $1.3M/day, so the cap is $13k."""
    from agent import ledger
    _seed_bars(store_bars, "NVDA", [(130.0, 10_000)] * 20)
    _patch_broker(monkeypatch, FakeBroker())         # quote ≈ stored close
    big = ledger.live_fill(store_bars, symbol="NVDA", side="buy",
                           notional=20_000)
    assert not big["ok"] and "average dollar volume" in big["error"]
    assert big["adv"] == pytest.approx(1_300_000.0)
    # inside the cap books clean, with no warnings
    ok = ledger.live_fill(store_bars, symbol="NVDA", side="buy", notional=5_000)
    assert ok["ok"] and "warnings" not in ok
    # the override books the oversize, on the record
    forced = ledger.live_fill(store_bars, symbol="NVDA", side="buy",
                              notional=20_000, allow_illiquid=True)
    assert forced["ok"]
    assert any("ADV" in w or "override" in w for w in forced["warnings"])


def test_live_fill_adv_gate_short_history_warns_and_allows(store_bars, monkeypatch):
    from agent import ledger
    _seed_bars(store_bars, "NVDA", [(130.0, 100)] * 5)  # < ADV_MIN_SESSIONS
    _patch_broker(monkeypatch, FakeBroker())
    r = ledger.live_fill(store_bars, symbol="NVDA", side="buy", notional=20_000)
    assert r["ok"], r
    assert any("ADV size gate skipped" in w for w in r["warnings"])


def test_live_fill_gates_skip_options_and_crypto(store, monkeypatch):
    """C1a/c: options liquidity is the chain's problem (OPRA spread and
    staleness caps); crypto has no daily_bars history — neither asset class
    hits the equity friction gates."""
    from agent import ledger, occ
    from datetime import date, timedelta
    _patch_broker(monkeypatch, FakeBroker(open_=False, session_="closed",
                                          bid=60_000.0, ask=60_050.0))
    r = ledger.live_fill(store, symbol="BTC/USD", side="buy", notional=1000)
    assert r["ok"] and "warnings" not in r
    sym = occ.build("NVDA", date.today() + timedelta(days=45), "C", 200)
    _patch_broker(monkeypatch, FakeBroker(bid=5.4, ask=5.5))
    ro = ledger.live_fill(store, symbol=sym, side="buy", shares=1)
    assert ro["ok"] and "warnings" not in ro


def test_option_live_fill_fee_and_cash_replay(store, monkeypatch):
    """C1d: option fills pay a flat per-contract fee inside ``dollars`` (BUY
    adds, SELL subtracts, price untouched) and the replayed cash from the
    ledger matches the fill's own cash_after exactly — fee invariance."""
    from agent import ledger, occ
    from datetime import date, timedelta
    FEE = ledger.OPTION_FEE_PER_CONTRACT
    sym = occ.build("NVDA", date.today() + timedelta(days=45), "C", 200)
    _patch_broker(monkeypatch, FakeBroker(bid=5.0, ask=5.1))
    buy = ledger.live_fill(store, symbol=sym, side="buy", shares=2)
    assert buy["ok"], buy
    assert buy["price"] == round(5.1 * 1.0001, 4)      # fee never in price
    assert buy["dollars"] == round(2 * buy["price"] * 100 + 2 * FEE, 2)
    assert buy["fill_quote"]["fee"]["total"] == 2 * FEE
    assert buy["cash_after"] == ledger.cash(store)     # replay invariance
    sell = ledger.live_fill(store, symbol=sym, side="sell", shares=2)
    assert sell["ok"], sell
    assert sell["dollars"] == round(2 * sell["price"] * 100 - 2 * FEE, 2)
    assert sell["cash_after"] == ledger.cash(store)
    assert ledger.cash(store) == pytest.approx(
        100_000.0 - buy["dollars"] + sell["dollars"])


def test_option_sell_fee_floors_at_zero(store):
    """C1d: a near-worthless sell can't go cash-negative on the fee."""
    from agent import ledger, occ
    from datetime import date, timedelta
    sym = occ.build("NVDA", date.today() + timedelta(days=45), "C", 200)
    fq = {"bid": 0.01, "ask": 0.02, "mid": 0.015, "t": "x", "src": "test"}
    buy = ledger.record_trade(store, symbol=sym, side="BUY", shares=1,
                              price=0.02, fill_quote=fq)
    assert buy["ok"]
    sell = ledger.record_trade(store, symbol=sym, side="SELL", shares=1,
                               price=0.01, fill_quote=fq)  # gross $1 < 2×fee? no: 1 − 0.65
    assert sell["ok"] and sell["dollars"] == max(
        0.0, round(1.0 - ledger.OPTION_FEE_PER_CONTRACT, 2))
    # a truly worthless exit (settlement handles 0-price, but the floor is
    # the arithmetic guarantee): gross $0.5 − $0.65 fee → 0, never negative
    buy2 = ledger.record_trade(store, symbol=sym, side="BUY", shares=1,
                               price=0.02, fill_quote=fq)
    assert buy2["ok"]
    fq2 = {"bid": 0.005, "ask": 0.02, "mid": 0.012, "t": "x", "src": "test"}
    tiny = ledger.record_trade(store, symbol=sym, side="SELL", shares=1,
                               price=0.005, fill_quote=fq2)
    assert tiny["ok"] and tiny["dollars"] == 0.0


def test_settlement_rows_stay_fee_free(store, monkeypatch):
    """C1d: expiry settlement is an accounting event, not an order — its
    rows carry exact intrinsic dollars with no fee."""
    from agent import ledger, occ
    from datetime import date, timedelta
    past = date.today() - timedelta(days=3)
    sym = occ.build("NVDA", past, "C", 180)
    store.insert("desk_trades", {"account": "agent", "symbol": sym,
                                 "side": "BUY", "shares": 1, "price": 5.0,
                                 "dollars": 500.0, "run_id": "T",
                                 "ts": ledger._utcnow()}, returning=False)
    ledger.rebuild_positions(store)
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {"NVDA": 195.0})
    monkeypatch.setattr(ledger, "_latest_close", lambda s: 195.0)
    monkeypatch.setattr(ledger, "_bar_close_on", lambda *a, **k: None)
    ledger.settle(store)
    row = store.select("desk_trades",
                       filters={"symbol": sym, "run_id": "settlement"})[0]
    assert row["dollars"] == 1500.0                    # intrinsic × 100, no fee
    assert "fee" not in (row["fill_quote"] or {})


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
    # monkeypatch so the settings singleton is restored (no test bleed)
    from config.settings import settings as _s
    monkeypatch.setattr(_s, "alpaca_api_key", "k", raising=False)
    monkeypatch.setattr(_s, "alpaca_api_secret", "s", raising=False)

    b = broker.Broker()
    # regular hours: is_open=True
    b._trading = _Trading(True, None, None)
    assert b.session() == "regular"
    # closed: is_open False and both next_open/next_close far future/past
    b._trading = _Trading(False,
                          datetime(2099, 1, 1, 14, 30, tzinfo=timezone.utc),
                          datetime(2099, 1, 1, 21, 0, tzinfo=timezone.utc))
    assert b.session() == "closed"


def test_live_fill_crypto_ignores_market_closed(store, monkeypatch):
    """Crypto is 24/7 — the "market closed" gate on the equity clock must
    not apply. The fill goes through and stamps session='crypto' on the row."""
    from agent import ledger
    # is_market_open()=False is what a weekend/overnight equity clock returns;
    # session()="closed" would be returned for an EQUITY symbol under this
    # FakeBroker, but the crypto short-circuit inside session() flips it.
    _patch_broker(monkeypatch, FakeBroker(open_=False, session_="closed",
                                          bid=60_000.0, ask=60_050.0))
    r = ledger.live_fill(store, symbol="btc/usd", side="buy", notional=1000)
    assert r["ok"], r
    assert r["fill_quote"]["session"] == "crypto"
    assert r["fill_quote"]["src"] == "alpaca_crypto_rest"
    # fractional shares: $1000 / (60050 * 1.0001) ≈ 0.0166 BTC
    assert 0 < r["shares"] < 0.02


def test_live_fill_crypto_spread_cap(store, monkeypatch):
    """Crypto gets its own 3% spread cap — tighter than options, looser than
    equities. A 5% spread that would be fine for equity RTH is refused here."""
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker(open_=False, session_="closed",
                                          bid=60_000.0, ask=63_500.0))  # ~5.5%
    r = ledger.live_fill(store, symbol="BTC/USD", side="buy", shares=0.01)
    assert not r["ok"] and "degenerate" in r["error"]


def test_is_crypto_helper():
    from agent import broker
    assert broker.is_crypto("BTC/USD")
    assert broker.is_crypto("ETH/USD")
    assert not broker.is_crypto("NVDA")
    assert not broker.is_crypto("NVDA270116C00200000")  # OCC option, no slash
    assert not broker.is_crypto("")
    assert not broker.is_crypto(None)


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
    # monkeypatch so the settings singleton is restored (no test bleed)
    from config.settings import settings as _s
    monkeypatch.setattr(_s, "alpaca_api_key", "k", raising=False)
    monkeypatch.setattr(_s, "alpaca_api_secret", "s", raising=False)

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

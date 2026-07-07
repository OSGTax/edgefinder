"""O1 tests: OCC helpers, option fills (×100), covered-only shorts, CSP cash
reservation, spread coverage, and expiry settlement in the agent's own ledger.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from agent import occ

TODAY = date.today()
FUT = TODAY + timedelta(days=45)          # a live expiry
PAST = TODAY - timedelta(days=3)          # an expired one


def C(und, strike, expiry=FUT):
    return occ.build(und, expiry, "C", strike)


def P(und, strike, expiry=FUT):
    return occ.build(und, expiry, "P", strike)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'opt.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def q(bid, ask):
    return {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4),
            "t": "x", "src": "test"}


# ── OCC helpers ──────────────────────────────────────────────

def test_occ_roundtrip_and_describe():
    sym = occ.build("NVDA", date(2027, 1, 16), "C", 200)
    assert sym == "NVDA270116C00200000"
    p = occ.parse(sym)
    assert (p["underlying"], p["type"], p["strike"]) == ("NVDA", "C", 200.0)
    assert p["expiry"] == date(2027, 1, 16)
    assert occ.describe(sym) == "NVDA $200C 2027-01-16"
    assert occ.is_option(sym) and not occ.is_option("NVDA")
    assert occ.intrinsic(sym, 215.0) == 15.0 and occ.intrinsic(sym, 190.0) == 0.0
    put = occ.build("SPY", date(2026, 8, 21), "P", 700.5)
    assert occ.parse(put)["strike"] == 700.5
    assert occ.intrinsic(put, 690.0) == 10.5


# ── option fills ─────────────────────────────────────────────

def test_long_call_multiplier_and_cash(store):
    from agent import ledger
    sym = C("NVDA", 200)
    r = ledger.record_trade(store, symbol=sym, side="BUY", shares=2,
                            price=5.50, fill_quote=q(5.40, 5.50))
    assert r["ok"] and r["multiplier"] == 100
    assert r["dollars"] == 2 * 5.50 * 100 == 1100.0
    assert ledger.cash(store) == 100_000.0 - 1100.0
    # whole contracts only
    bad = ledger.record_trade(store, symbol=sym, side="BUY", shares=1.5,
                              price=5.50, fill_quote=q(5.40, 5.50))
    assert not bad["ok"] and "whole contracts" in bad["error"]
    # options demand a live snapshot
    noq = ledger.record_trade(store, symbol=sym, side="BUY", shares=1, price=5.5)
    assert not noq["ok"] and "require a live fill_quote" in noq["error"]


def test_naked_short_call_rejected_covered_allowed(store):
    from agent import ledger
    sym = C("NVDA", 200)
    naked = ledger.record_trade(store, symbol=sym, side="SELL", shares=1,
                                price=5.0, fill_quote=q(4.9, 5.0))
    assert not naked["ok"] and "uncovered short call" in naked["error"]
    # buy 100 shares → 1 covered call allowed, 2 not
    ledger.record_trade(store, symbol="NVDA", side="BUY", shares=100,
                        price=195.0, fill_quote=q(194.9, 195.0))
    two = ledger.record_trade(store, symbol=sym, side="SELL", shares=2,
                              price=5.0, fill_quote=q(4.9, 5.0))
    assert not two["ok"]
    one = ledger.record_trade(store, symbol=sym, side="SELL", shares=1,
                              price=5.0, fill_quote=q(4.9, 5.0))
    assert one["ok"] and one["dollars"] == 500.0
    pos = {r["symbol"]: r["shares"] for r in store.select("desk_positions")}
    assert pos[sym] == -1  # short leg carried as negative contracts
    # selling the shares out from under the call is refused
    strand = ledger.record_trade(store, symbol="NVDA", side="SELL", shares=100,
                                 price=195.0, fill_quote=q(194.9, 195.0))
    assert not strand["ok"] and "strand" in strand["error"]


def test_cash_secured_put_reservation(store):
    from agent import ledger
    sym = P("SPY", 700)  # secures 70,000
    r = ledger.record_trade(store, symbol=sym, side="SELL", shares=1,
                            price=10.0, fill_quote=q(9.9, 10.0))
    assert r["ok"]
    # premium landed, but 70k is reserved
    assert ledger.cash(store) == 100_000.0 + 1000.0
    assert ledger.free_cash(store) == 101_000.0 - 70_000.0
    # a second one (140k total) exceeds cash → rejected
    r2 = ledger.record_trade(store, symbol=sym, side="SELL", shares=1,
                             price=10.0, fill_quote=q(9.9, 10.0))
    assert not r2["ok"] and "cash-secured put requires" in r2["error"]
    # buys are limited to FREE cash (can't spend the reservation)
    big = ledger.record_trade(store, symbol="NVDA", side="BUY", shares=300,
                              price=195.0, fill_quote=q(194.9, 195.0))
    assert not big["ok"] and "free cash" in big["error"]


def test_vertical_spread_coverage(store):
    from agent import ledger
    long_leg, short_leg = C("NVDA", 210), C("NVDA", 200)
    # short first without the long → rejected
    r = ledger.record_trade(store, symbol=short_leg, side="SELL", shares=1,
                            price=6.0, fill_quote=q(5.9, 6.0))
    assert not r["ok"]
    # buy the long leg, then the short is spread-covered
    ledger.record_trade(store, symbol=long_leg, side="BUY", shares=1,
                        price=3.0, fill_quote=q(2.9, 3.0))
    r2 = ledger.record_trade(store, symbol=short_leg, side="SELL", shares=1,
                             price=6.0, fill_quote=q(5.9, 6.0))
    assert r2["ok"]  # defined-risk credit spread
    # put debit/credit spread coverage works the same way
    lp, sp = P("SPY", 690), P("SPY", 700)
    ledger.record_trade(store, symbol=lp, side="BUY", shares=1,
                        price=8.0, fill_quote=q(7.9, 8.0))
    r3 = ledger.record_trade(store, symbol=sp, side="SELL", shares=1,
                             price=12.0, fill_quote=q(11.9, 12.0))
    assert r3["ok"]
    assert ledger.free_cash(store) == ledger.cash(store)  # spread ⇒ no CSP reserve


def test_expired_contract_rejected_at_fill(store):
    from agent import ledger
    r = ledger.record_trade(store, symbol=C("NVDA", 200, PAST), side="BUY",
                            shares=1, price=5.0, fill_quote=q(4.9, 5.0))
    assert not r["ok"] and "expired" in r["error"]


# ── settlement ───────────────────────────────────────────────

def _patch_underlying(monkeypatch, prices):
    from agent import ledger
    monkeypatch.setattr(ledger, "_live_mids",
                        lambda syms: {s: prices[s] for s in syms if s in prices})
    monkeypatch.setattr(ledger, "_latest_close", lambda s: prices.get(s))


def test_settle_long_itm_and_otm(store, monkeypatch):
    from agent import ledger
    itm, otm = C("NVDA", 180, PAST), C("NVDA", 250, PAST)
    for sym in (itm, otm):
        store.insert("desk_trades", {"account": "agent", "symbol": sym,
                                     "side": "BUY", "shares": 1, "price": 5.0,
                                     "dollars": 500.0, "run_id": "T",
                                     "ts": ledger._utcnow()}, returning=False)
    ledger.rebuild_positions(store)
    _patch_underlying(monkeypatch, {"NVDA": 195.0})
    out = ledger.settle(store)
    assert len(out["settled"]) == 2
    pos = {r["symbol"] for r in store.select("desk_positions")}
    assert itm not in pos and otm not in pos  # both gone
    # cash: -1000 premium, ITM paid out (195-180)*100 = 1500, OTM zero
    assert ledger.cash(store) == 100_000.0 - 1000.0 + 1500.0


def test_settle_covered_call_assignment(store, monkeypatch):
    from agent import ledger
    ledger.record_trade(store, symbol="NVDA", side="BUY", shares=100,
                        price=150.0, fill_quote=q(149.9, 150.0))
    sym = C("NVDA", 180, PAST)
    store.insert("desk_trades", {"account": "agent", "symbol": sym, "side": "SELL",
                                 "shares": 1, "price": 5.0, "dollars": 500.0,
                                 "run_id": "T", "ts": ledger._utcnow()}, returning=False)
    ledger.rebuild_positions(store)
    _patch_underlying(monkeypatch, {"NVDA": 195.0})
    ledger.settle(store)
    pos = {r["symbol"]: r["shares"] for r in store.select("desk_positions")}
    assert sym not in pos and "NVDA" not in pos  # shares called away
    # cash: -15000 buy +500 premium +18000 assignment sale
    assert ledger.cash(store) == 100_000.0 - 15_000.0 + 500.0 + 18_000.0


def test_settle_csp_assignment_books_shares(store, monkeypatch):
    from agent import ledger
    sym = P("SPY", 700, PAST)
    store.insert("desk_trades", {"account": "agent", "symbol": sym, "side": "SELL",
                                 "shares": 1, "price": 10.0, "dollars": 1000.0,
                                 "run_id": "T", "ts": ledger._utcnow()}, returning=False)
    ledger.rebuild_positions(store)
    _patch_underlying(monkeypatch, {"SPY": 680.0})
    ledger.settle(store)
    pos = {r["symbol"]: r["shares"] for r in store.select("desk_positions")}
    assert pos.get("SPY") == 100.0  # shares put to us at the strike
    assert sym not in pos
    assert ledger.cash(store) == 100_000.0 + 1000.0 - 70_000.0


def test_settle_spread_covered_short_cash_settles(store, monkeypatch):
    from agent import ledger
    long_leg, short_leg = C("NVDA", 210, PAST), C("NVDA", 200, PAST)
    for sym, side, px in ((long_leg, "BUY", 3.0), (short_leg, "SELL", 6.0)):
        store.insert("desk_trades", {"account": "agent", "symbol": sym, "side": side,
                                     "shares": 1, "price": px, "dollars": px * 100,
                                     "run_id": "T", "ts": ledger._utcnow()}, returning=False)
    ledger.rebuild_positions(store)
    _patch_underlying(monkeypatch, {"NVDA": 220.0})   # both ITM
    ledger.settle(store)
    pos = {r["symbol"] for r in store.select("desk_positions")}
    assert not any(occ.is_option(s) for s in pos)
    assert "NVDA" not in pos                          # NO phantom share position
    # cash: +300 net credit at entry... (-300 +600) then long pays +1000, short costs -2000
    assert ledger.cash(store) == 100_000.0 + 300.0 + 1000.0 - 2000.0

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
    import edgefinder.db.models  # noqa: F401 — daily_bars for expiry-close settlement
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


# The fill/settlement/coverage tests that used to live here tested
# agent.ledger, which REBUILD-V4 deleted: defined-risk coverage is enforced
# by the Alpaca account itself (Level 3 max, all-legs-covered mleg, CSP
# buying power), expiry settles on the broker's side (graded by
# agent.grade's settlement facts), and order-shape legality lives in
# agent.trade.validate_order (tested in test_trade.py).

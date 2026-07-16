"""C2: mark provenance — every equity snapshot records WHICH pricing tier
marked each position (live mid → daily close → cost basis), so a snapshot
written during a quote/data outage (cost marks = fake-flat P&L) is visibly
flagged instead of silently embedded in the curve forever.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'mark.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def q(px):
    return {"bid": px, "ask": px, "mid": px, "t": "x", "src": "test"}


def _buy(store, symbol, shares, price):
    from agent import ledger
    r = ledger.record_trade(store, symbol=symbol, side="BUY", shares=shares,
                            price=price, fill_quote=q(price))
    assert r["ok"], r


def test_mark_meta_counts_each_fallback_tier(store, monkeypatch):
    """One position per tier: AAA gets a live mid, BBB a stored close, CCC
    nothing (cost basis) — the snapshot's mark_meta counts all three and
    names the cost-marked symbol."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)   # $1,000
    _buy(store, "BBB", 10, 100.0)   # $1,000
    _buy(store, "CCC", 10, 100.0)   # $1,000
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {"AAA": 110.0})
    monkeypatch.setattr(ledger, "_latest_closes",
                        lambda syms: {s: 105.0 for s in syms if s == "BBB"})
    st = ledger.mark(store)
    meta = st["mark_meta"]
    assert meta["sources"] == {"live": 1, "close": 1, "cost": 1}
    assert meta["cost_marked"] == ["CCC"]
    # $1,000 of $3,150 marked value ≈ 31.7% at cost → degraded
    assert meta["cost_marked_value_pct"] == pytest.approx(31.75, abs=0.01)
    assert meta["degraded"] is True
    # the snapshot row itself carries the same meta (durable provenance)
    row = store.select("desk_equity", filters={"account": "agent"})[0]
    assert row["mark_meta"]["sources"] == {"live": 1, "close": 1, "cost": 1}


def test_mark_meta_not_degraded_under_threshold(store, monkeypatch):
    """Cost-marked value under MARK_DEGRADED_COST_PCT: recorded, not flagged."""
    from agent import ledger
    _buy(store, "AAA", 100, 100.0)  # $10,000
    _buy(store, "CCC", 10, 100.0)   # $1,000 → ~9% at cost
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {"AAA": 100.0})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    st = ledger.mark(store)
    meta = st["mark_meta"]
    assert meta["sources"] == {"live": 1, "close": 0, "cost": 1}
    assert meta["cost_marked"] == ["CCC"]
    assert meta["cost_marked_value_pct"] < ledger.MARK_DEGRADED_COST_PCT
    assert "degraded" not in meta


def test_mark_explicit_prices_count_as_live(store):
    """Caller-supplied prices are explicit marks — the test/backfill path
    keeps a clean provenance record."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)
    st = ledger.mark(store, prices={"AAA": 101.0})
    assert st["mark_meta"]["sources"] == {"live": 1, "close": 0, "cost": 0}
    assert st["mark_meta"]["cost_marked"] == []
    assert st["mark_meta"]["cost_marked_value_pct"] == 0.0


def test_state_surfaces_latest_mark_meta(store, monkeypatch):
    """`ledger state` carries the LATEST snapshot's provenance so a degraded
    mark is visible on every subsequent read, not only at mark time."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    ledger.mark(store)                      # everything at cost → degraded
    st = ledger.state(store)
    assert st["mark_meta"]["degraded"] is True
    assert st["mark_meta"]["cost_marked_value_pct"] == 100.0
    # before any mark at all, the key is present and honest
    store.delete("desk_equity", {"account": "agent"})
    assert ledger.state(store)["mark_meta"] is None


def test_degraded_mark_still_writes_the_snapshot(store, monkeypatch):
    """An outage must not stop the equity curve — degraded marks write,
    flagged, and the CLI warning helper fires on them."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    st = ledger.mark(store)
    assert len(store.select("desk_equity", filters={"account": "agent"})) == 1
    # the loud-warning helper prints for degraded marks and stays silent
    # otherwise (stderr, so stdout stays clean JSON)
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        ledger._warn_if_degraded(st)
    assert "MARKS DEGRADED" in err.getvalue()
    err2 = io.StringIO()
    with contextlib.redirect_stderr(err2):
        ledger._warn_if_degraded({"mark_meta": {"cost_marked_value_pct": 0.0}})
    assert err2.getvalue() == ""

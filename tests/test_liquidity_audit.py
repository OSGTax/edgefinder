"""Phase A: the READ-ONLY liquidity audit.

Two questions the desk could not previously answer from data: did any past
fill book a price the market would not have given us, and does mid-marking
overstate what the open book could actually be liquidated for?

The audit reports; it must never enforce and must never write. The last two
tests are the ones that matter — they pin the "reports, doesn't act" contract.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'audit.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def fq(bid, ask, **extra):
    d = {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4),
         "t": "2026-07-29T14:00:00+00:00", "src": "test", "session": "regular"}
    d.update(extra)
    return d


def _fill(store, symbol, side, shares, price, quote, **extra):
    from agent import ledger
    r = ledger.record_trade(store, symbol=symbol, side=side, shares=shares,
                            price=price, fill_quote=fq(*quote, **extra))
    assert r["ok"], r
    return r


def test_audit_reports_spread_at_fill_and_flags_nothing_when_tight(store):
    """A liquid book audits clean: every receipt replayed, nothing wide."""
    from agent import ledger
    _fill(store, "AAA", "BUY", 10, 100.05, (100.0, 100.1))
    out = ledger.liquidity_audit(store, quotes={"AAA": {"bid": 100.0, "ask": 100.1}})
    assert out["ok"] is True
    f = out["fills"]
    assert f["with_quote_receipt"] == 1
    assert f["wide_at_fill"] == 0
    assert f["with_overrides"] == 0
    assert f["widest"][0]["symbol"] == "AAA"
    # 0.1 / 100.1 ≈ 0.0999%
    assert f["widest"][0]["spread_pct"] == pytest.approx(0.0999, abs=1e-3)
    assert f["thresholds"]["enforced"] is False


def test_audit_flags_a_wide_equity_fill(store):
    """A 2%-wide equity fill is flagged for review — but still reported as a
    booked fill, because it WAS booked. The audit describes history."""
    from agent import ledger
    _fill(store, "THIN", "BUY", 10, 102.0, (100.0, 102.0))
    out = ledger.liquidity_audit(store, quotes={})
    assert out["fills"]["wide_at_fill"] == 1
    assert out["fills"]["flagged"][0]["symbol"] == "THIN"


def test_audit_option_flag_needs_both_fraction_and_cents(store):
    """The fraction-OR-cents shape, proven on the real case that motivated it:
    a 1c-wide penny contract is 50% 'wide' and perfectly tradeable, so it must
    NOT flag; a 30%-wide $2.00 contract must."""
    from agent import ledger
    # 0.01 / 0.02 — 50% wide, 1 cent absolute. Real: SPY260729P00725000.
    _fill(store, "SPY260729P00725000", "BUY", 5, 0.02, (0.01, 0.02))
    # 1.40 / 2.00 — 30% wide, 60 cents absolute.
    _fill(store, "AMD260821C00550000", "BUY", 1, 2.00, (1.40, 2.00))
    out = ledger.liquidity_audit(store, quotes={})
    flagged = {r["symbol"] for r in out["fills"]["flagged"]}
    assert "AMD260821C00550000" in flagged
    assert "SPY260729P00725000" not in flagged


def test_audit_surfaces_stamped_overrides_and_warnings(store):
    """An override is stamped on the receipt at fill time; the audit is how
    that stops being invisible."""
    from agent import ledger
    _fill(store, "GAPY", "BUY", 10, 100.05, (100.0, 100.1),
          warnings=["price deviation 44.0% vs close 90 allowed by override"])
    out = ledger.liquidity_audit(store, quotes={})
    assert out["fills"]["with_overrides"] == 1
    assert out["fills"]["with_warnings"] == 1
    assert "override" in out["fills"]["flagged"][0]["warnings"][0]


def test_audit_prices_the_liquidation_touch_not_the_mid(store):
    """The core number: a long exits at the BID, a short buys back at the ASK.
    Mid-marking flatters both, and the audit puts a dollar figure on it."""
    from agent import ledger
    _fill(store, "LONGY", "BUY", 100, 10.0, (9.9, 10.0))
    # short leg: a cash-secured put (the ledger rejects anything undefined)
    _fill(store, "SPY260821P00050000", "SELL", 5, 4.0, (4.0, 4.1))
    out = ledger.liquidity_audit(store, quotes={
        "LONGY": {"bid": 9.0, "ask": 11.0},                    # mid 10.0
        "SPY260821P00050000": {"bid": 4.0, "ask": 6.0},        # mid 5.0
    })
    by = {p["symbol"]: p for p in out["marks"]["positions"]}
    # long 100 @ mid 10.0 = 1000, at the bid 9.0 = 900 → 100 of phantom P&L
    assert by["LONGY"]["touch"] == 9.0
    assert by["LONGY"]["phantom_pnl"] == pytest.approx(100.0)
    # short 5 contracts: mid -5.0*500 = -2500, at the ask -6.0*500 = -3000
    # → the mid understates the cost to close by 500
    assert by["SPY260821P00050000"]["touch"] == 6.0
    assert by["SPY260821P00050000"]["phantom_pnl"] == pytest.approx(500.0)
    assert out["marks"]["phantom_pnl"] == pytest.approx(600.0)
    assert (out["marks"]["equity_mid_marked"]
            - out["marks"]["equity_touch_marked"]) == pytest.approx(600.0)


def test_audit_reports_unquoted_rather_than_guessing(store):
    """No quote → the position is reported as unquoted. An audit that silently
    fell back to the mid would hide exactly what it exists to measure."""
    from agent import ledger
    _fill(store, "DARK", "BUY", 10, 100.05, (100.0, 100.1))
    out = ledger.liquidity_audit(store, quotes={})
    assert out["marks"]["unquoted"] == ["DARK"]
    assert out["marks"]["positions"][0]["phantom_pnl"] is None
    assert out["marks"]["phantom_pnl"] == 0.0


def test_audit_writes_nothing(store):
    """THE contract: the audit must not move the thing it audits. Notably it
    must not call rebuild_positions (which reconciles the projection) or mark
    (which inserts an equity row)."""
    from agent import ledger
    _fill(store, "AAA", "BUY", 10, 100.05, (100.0, 100.1))
    before = {t: store.select(t) for t in
              ("desk_trades", "desk_positions", "desk_equity")}
    ledger.liquidity_audit(store, quotes={"AAA": {"bid": 99.0, "ask": 101.0}})
    after = {t: store.select(t) for t in
             ("desk_trades", "desk_positions", "desk_equity")}
    for t in before:
        assert before[t] == after[t], f"{t} changed — the audit wrote"


def test_audit_does_not_enforce(store, monkeypatch):
    """A fill the audit would flag still books. Reporting thresholds must stay
    strictly separate from the live gates."""
    from agent import ledger
    r = ledger.record_trade(store, symbol="THIN", side="BUY", shares=10,
                            price=102.0, fill_quote=fq(100.0, 102.0))
    assert r["ok"] is True
    out = ledger.liquidity_audit(store, quotes={})
    assert out["fills"]["wide_at_fill"] == 1     # flagged...
    assert len(store.select("desk_trades")) == 1  # ...and still booked

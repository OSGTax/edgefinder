"""Alpha-vs-SPY benchmarking + rejected-candidate registry (v8.15).

A long-only book's raw P&L is mostly market beta; outcomes() benchmarks every
window against SPY closes from daily_bars (index_daily froze at the cutover)
so reflection grades alpha. Decisions also record the candidates that LOST
the slot (rejected) so Friday can grade the road not taken.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

TODAY = date.today()
D_ENTRY = TODAY - timedelta(days=4)  # decision + fills booked here


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'alpha.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401 — daily_bars for the SPY series

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


def seed_spy(store, closes: dict[date, float]) -> None:
    store.insert("daily_bars", [
        {"symbol": "SPY", "date": d, "open": c, "high": c, "low": c,
         "close": c, "volume": 1e6, "source": "test"}
        for d, c in closes.items()
    ], returning=False)


def seed_backdated_book(store) -> None:
    """One decision + one XYZ buy fill on D_ENTRY, marked at 110 today."""
    from agent import ledger
    from agent.brain import save_decision

    save_decision(store, run_id="A", summary="entry",
                  picks=[{"symbol": "XYZ", "action": "buy", "why_now": "test",
                          "rationale": "trend"}],
                  rejected=[{"symbol": "ABC", "why_not": "falling knife"}])
    entry_ts = datetime(D_ENTRY.year, D_ENTRY.month, D_ENTRY.day, 15, 30)
    store.update("desk_decisions", {"run_id": "A"}, {"ts": entry_ts},
                 returning=False)
    store.insert("desk_trades", {
        "account": "agent", "ts": entry_ts, "run_id": "A", "symbol": "XYZ",
        "side": "BUY", "shares": 10.0, "price": 100.0, "dollars": 1000.0,
        "rationale": "trend"}, returning=False)
    ledger.mark(store, prices={"XYZ": 110.0})


def test_pick_run_and_book_alpha(store):
    from agent import ledger

    # SPY: 600 on entry day → 612 latest = +2.00% over the window.
    seed_spy(store, {D_ENTRY: 600.0,
                     D_ENTRY + timedelta(days=2): 606.0,
                     TODAY: 612.0})
    seed_backdated_book(store)

    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "A")
    assert run["spy_same_window_pct"] == 2.0
    pick = run["picks"][0]
    assert pick["since_this_run_pct"] == 10.0   # 110 vs 100 entry
    assert pick["spy_same_window_pct"] == 2.0
    assert pick["alpha_pct"] == 8.0             # skill, net of the market

    # Book: equity = 100_000 - 1_000 + 10*110 = 100_100 → +0.10% vs SPY +2.00%
    book = out["book"]
    assert book["inception"] == str(D_ENTRY)
    assert book["since_inception_pct"] == 0.1
    assert book["spy_since_inception_pct"] == 2.0
    assert book["alpha_pct"] == -1.9            # made money, lost to the index


def test_baseline_uses_last_close_on_or_before_window_start(store):
    from agent import ledger

    # No SPY bar ON the entry date (weekend entry): baseline falls back to the
    # prior close, and the buffer in _spy_closes must reach it.
    seed_spy(store, {D_ENTRY - timedelta(days=3): 500.0,
                     TODAY: 510.0})
    seed_backdated_book(store)

    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "A")
    assert run["spy_same_window_pct"] == 2.0    # 510 vs 500


def test_no_spy_data_degrades_to_none(store):
    from agent import ledger

    seed_backdated_book(store)
    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "A")
    assert run["spy_same_window_pct"] is None
    assert run["picks"][0]["alpha_pct"] is None
    assert out["book"]["spy_since_inception_pct"] is None
    assert out["book"]["alpha_pct"] is None
    assert out["book"]["since_inception_pct"] == 0.1  # book math still exact


def test_rejected_round_trip(store):
    from agent import ledger

    seed_spy(store, {D_ENTRY: 600.0, TODAY: 612.0})
    seed_backdated_book(store)

    rows = store.select("desk_decisions", filters={"run_id": "A"})
    assert rows[0]["rejected"] == [{"symbol": "ABC", "why_not": "falling knife"}]
    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "A")
    assert run["rejected"][0]["symbol"] == "ABC"


def test_empty_book_has_no_book_block(store):
    from agent import ledger

    out = ledger.outcomes(store, days=30)
    assert out["book"] is None and out["runs"] == []


def test_portfolio_and_decision_endpoints(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    seed_spy(store, {D_ENTRY: 600.0, TODAY: 612.0})
    seed_backdated_book(store)

    from dashboard.app import app

    with TestClient(app) as c:
        pf = c.get("/api/desk/portfolio").json()
        assert pf["vs_spy"]["spy_return_pct"] == 2.0
        assert pf["vs_spy"]["inception"] == str(D_ENTRY)
        assert pf["vs_spy"]["alpha_pct"] == pytest.approx(
            pf["total_return_pct"] - 2.0)

        d = c.get("/api/desk/decision/latest").json()
        assert d["rejected"] == [{"symbol": "ABC", "why_not": "falling knife"}]

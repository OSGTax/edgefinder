"""Alpha-vs-SPY benchmarking + rejected-candidate registry (v8.15–v8.17.3).

Conventions under test (see outcomes()'s convention string):
- SPY baseline = last close STRICTLY BEFORE the window's ET start date (a
  close ON the start date is 16:00 ET, after the intraday entry — and on a
  same-day window it would be the endpoint itself, a confident fake 0.00).
- None = too-young-to-benchmark, never zero.
- Round trips closed in-run get closed_return_pct + an exit-bounded window.
- Options carry alpha_pct = None by design.
- Decision timestamps are naive UTC; window dates are their ET dates.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

TODAY = date.today()
D_ENTRY = TODAY - timedelta(days=4)   # decision + fills booked here
D_BASE = D_ENTRY - timedelta(days=1)  # the strictly-before baseline close


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'alpha.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401 — daily_bars for the SPY series

    Base.metadata.create_all(get_engine())

    # The desk router caches the /portfolio body ~10s (plus the open-orders
    # and live-outcomes overlays); a stale entry from a sibling test's DB
    # must never serve here.
    import dashboard.routers.desk as desk_router
    desk_router._portfolio_cache = None
    desk_router._open_orders_cache = None
    desk_router._outcomes_live_cache = None

    from agent.store import get_store

    return get_store()


def seed_spy(store, closes: dict[date, float]) -> None:
    store.insert("daily_bars", [
        {"symbol": "SPY", "date": d, "open": c, "high": c, "low": c,
         "close": c, "volume": 1e6, "source": "test"}
        for d, c in closes.items()
    ], returning=False)


def ts_of(d: date, hour: int = 15, minute: int = 30) -> datetime:
    return datetime(d.year, d.month, d.day, hour, minute)  # naive UTC


def seed_trade(store, run_id: str, symbol: str, side: str, shares: float,
               price: float, ts: datetime) -> None:
    from agent import occ

    mult = 100 if occ.is_option(symbol) else 1  # OCC premium moves ×100 cash
    store.insert("desk_trades", {
        "account": "agent", "ts": ts, "run_id": run_id, "symbol": symbol,
        "side": side, "shares": shares, "price": price,
        "dollars": round(shares * price * mult, 2), "rationale": "test"},
        returning=False)


def test_et_date_rolls_back_evening_runs():
    from agent.grade import _et_date

    # 00:30 UTC is 19:30/20:30 ET the PREVIOUS calendar day.
    assert _et_date("2026-07-10T00:30:00") == "2026-07-09"
    assert _et_date(datetime(2026, 7, 10, 0, 30)) == "2026-07-09"
    assert _et_date("2026-07-10T15:30:00") == "2026-07-10"


# ── prediction registry enforcement (the write-side gate, F6) ──


def test_registry_rejects_buy_pick_missing_prediction(store):
    from agent.brain import save_decision

    r = save_decision(store, run_id="E1", summary="no registry",
                      picks=[{"symbol": "XYZ", "action": "buy",
                              "why_now": "breakout"}])
    assert not r["ok"] and "prediction registry" in r["error"]
    assert store.select("desk_decisions", filters={"run_id": "E1"}) == []
    # horizon must be an integer >= 1; kill must be non-null
    r2 = save_decision(store, run_id="E1", picks=[
        {"symbol": "XYZ", "action": "add", "prediction": "up 5%",
         "horizon_days": 0, "kill": None}])
    assert not r2["ok"] and "horizon_days" in r2["error"] and "kill" in r2["error"]
    ok = save_decision(store, run_id="E1", picks=[
        {"symbol": "XYZ", "action": "buy",
         "prediction": "XYZ +5% within 10 sessions",
         "horizon_days": 10, "kill": "closes below 90"}])
    assert ok["ok"]


def test_registry_exempts_holds_and_book_stance(store):
    from agent.brain import save_decision

    # hold/trim/exit picks manage what's already graded — nulls are fine,
    # and BOOK is the whole-book stance pseudo-symbol (hold/stance only)
    ok = save_decision(store, run_id="H1", summary="quiet cycle",
                       picks=[{"symbol": "XYZ", "action": "hold"},
                              {"symbol": "XYZ", "action": "trim"},
                              {"symbol": "BOOK", "action": "hold"}])
    assert ok["ok"], ok
    bad = save_decision(store, run_id="H2", picks=[
        {"symbol": "BOOK", "action": "buy", "prediction": "x",
         "horizon_days": 5, "kill": "y"}])
    assert not bad["ok"] and "BOOK" in bad["error"]
    assert store.select("desk_decisions", filters={"run_id": "H2"}) == []


def test_registry_rejects_unknown_action_and_empty_symbol(store):
    """F6 tightening: every pick needs a real symbol and an action from the
    skill's vocabulary (hold/buy/add/trim/exit/stance) — invented verbs and
    blank symbols are write-time errors, not grading-time surprises."""
    from agent.brain import save_decision

    r = save_decision(store, run_id="V1", picks=[
        {"symbol": "XYZ", "action": "yolo"}])
    assert not r["ok"] and "unrecognized action" in r["error"]
    r2 = save_decision(store, run_id="V1", picks=[
        {"symbol": "  ", "action": "hold"}])
    assert not r2["ok"] and "symbol" in r2["error"]
    r3 = save_decision(store, run_id="V1", picks=[{"symbol": "XYZ"}])
    assert not r3["ok"] and "unrecognized action" in r3["error"]
    assert store.select("desk_decisions", filters={"run_id": "V1"}) == []
    # the full managing vocabulary still passes without a registry
    ok = save_decision(store, run_id="V1", picks=[
        {"symbol": "XYZ", "action": "hold"},
        {"symbol": "XYZ", "action": "trim"},
        {"symbol": "XYZ", "action": "exit"}])
    assert ok["ok"], ok


# ── endpoint fixtures: era-2 fills + a canned Alpaca account ─────────────


def _seed_era2_fill(store, run_id: str, symbol: str, qty: float, price: float,
                    ts: datetime) -> None:
    """One era-2 entry fill in the desk_orders mirror (the V4 fills source —
    also what /portfolio derives its all-time inception from pre-era-1)."""
    store.insert("desk_orders", {
        "account": "agent", "run_id": run_id, "seq": 1,
        "client_order_id": f"{run_id}:01", "alpaca_order_id": f"{run_id}-1",
        "symbol": symbol, "asset_class": "us_equity", "side": "buy",
        "kind": "entry", "order_type": "market", "tif": "day",
        "order_class": "simple", "qty": qty, "status": "filled",
        "filled_qty": qty, "filled_avg_price": price,
        "submitted_at": ts.isoformat() + "+00:00",
        "filled_at": ts.isoformat() + "+00:00"}, returning=False)


def _canned_desk_trade(monkeypatch, *, equity=100100.0, cash=99100.0,
                       positions=None):
    class _FakeTrade:
        def __init__(self, *a, **k):
            pass

        def state(self):
            return {"account": "agent", "paper": True, "cash": cash,
                    "equity": equity, "buying_power": cash,
                    "starting_capital": 100000.0,
                    "total_pnl": round(equity - 100000.0, 2),
                    "total_return_pct": round(
                        (equity - 100000.0) / 100000.0 * 100, 4),
                    "positions": list(positions or []),
                    "positions_value": round(sum(
                        p.get("market_value") or 0.0
                        for p in (positions or [])), 2)}

        def orders(self, **k):
            return []

    monkeypatch.setattr("agent.trade.Trade", _FakeTrade)


def test_portfolio_and_decision_endpoints(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    seed_spy(store, {D_BASE: 600.0, TODAY: 612.0})
    from agent.brain import save_decision

    save_decision(store, run_id="A", summary="entry",
                  picks=[{"symbol": "XYZ", "action": "buy", "why_now": "test",
                          "rationale": "trend",
                          "prediction": "XYZ +5% within 10 sessions",
                          "horizon_days": 10, "kill": "closes below 90"}],
                  rejected=[{"symbol": "ABC", "why_not": "falling knife"}])
    _seed_era2_fill(store, "A", "XYZ", 10.0, 100.0, ts_of(D_ENTRY))
    _canned_desk_trade(monkeypatch, equity=100100.0)

    from dashboard.app import app

    with TestClient(app) as c:
        pf = c.get("/api/desk/portfolio").json()
        # PRICE return (charter V4): 600 baseline → 612 endpoint = +2.00%
        assert pf["vs_spy"]["spy_return_pct"] == 2.0
        assert pf["vs_spy"]["basis"] == "price_return"
        assert pf["vs_spy"]["inception"] == str(D_ENTRY)
        assert pf["vs_spy"]["alpha_pct"] == pytest.approx(
            pf["total_return_pct"] - 2.0)

        d = c.get("/api/desk/decision/latest").json()
        assert d["rejected"] == [{"symbol": "ABC", "why_not": "falling knife"}]


def test_portfolio_vs_spy_is_price_return(store, monkeypatch):
    """Charter V4 flips M2: the paper broker credits no dividends into the
    book, so the benchmark must NOT be dividend back-adjusted either — a SPY
    dividend inside the window leaves spy_return_pct at the raw price move
    (+2.00%), never the total-return +3.03%."""
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    ex = D_ENTRY + timedelta(days=1)
    seed_spy(store, {D_BASE: 600.0, ex: 600.0, TODAY: 612.0})
    store.insert("dividends", {"symbol": "SPY", "ex_date": ex,
                               "cash_amount": 6.0}, returning=False)
    _seed_era2_fill(store, "A", "XYZ", 10.0, 100.0, ts_of(D_ENTRY))
    _canned_desk_trade(monkeypatch, equity=100100.0)

    from dashboard.app import app

    with TestClient(app) as c:
        pf = c.get("/api/desk/portfolio").json()
        assert pf["vs_spy"]["spy_return_pct"] == pytest.approx(2.0)
        assert pf["vs_spy"]["basis"] == "price_return"
        assert pf["vs_spy"]["alpha_pct"] == pytest.approx(
            pf["total_return_pct"] - 2.0)


def test_portfolio_passes_alpaca_position_marks_through(store, monkeypatch):
    """/portfolio serves the broker's own position economics unmodified —
    an option position's market_value/unrealized_pl are Alpaca's numbers
    (multiplier already inside), never re-derived locally."""
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    occ_sym = "NVDA270116C00200000"
    _canned_desk_trade(monkeypatch, equity=100500.0, cash=99000.0, positions=[{
        "symbol": occ_sym, "asset_class": "us_option", "qty": 2.0,
        "qty_available": 2.0, "avg_entry_price": 5.0, "current_price": 7.5,
        "market_value": 1500.0, "cost_basis": 1000.0, "unrealized_pl": 500.0,
        "unrealized_plpc": 0.5, "change_today": None, "side": "long",
        "weight": 0.0149}])

    from dashboard.app import app

    with TestClient(app) as c:
        pf = c.get("/api/desk/portfolio").json()
        row = next(p for p in pf["positions"] if p["symbol"] == occ_sym)
        assert row["market_value"] == 1500.0     # 2 contracts × 7.5 × 100
        assert row["unrealized_pl"] == 500.0     # 2 × (7.5-5.0) × 100
        assert pf["equity"] == pytest.approx(100500.0)

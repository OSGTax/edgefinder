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


def seed_backdated_book(store) -> None:
    """One decision + one XYZ buy fill on D_ENTRY, marked at 110 today."""
    from agent import ledger
    from agent.brain import save_decision

    save_decision(store, run_id="A", summary="entry",
                  picks=[{"symbol": "XYZ", "action": "buy", "why_now": "test",
                          "rationale": "trend",
                          "prediction": "XYZ +5% within 10 sessions",
                          "horizon_days": 10, "kill": "closes below 90"}],
                  rejected=[{"symbol": "ABC", "why_not": "falling knife"}])
    store.update("desk_decisions", {"run_id": "A"}, {"ts": ts_of(D_ENTRY)},
                 returning=False)
    seed_trade(store, "A", "XYZ", "BUY", 10.0, 100.0, ts_of(D_ENTRY))
    ledger.mark(store, prices={"XYZ": 110.0})


def test_et_date_rolls_back_evening_runs():
    from agent.ledger import _et_date

    # 00:30 UTC is 19:30/20:30 ET the PREVIOUS calendar day.
    assert _et_date("2026-07-10T00:30:00") == "2026-07-09"
    assert _et_date(datetime(2026, 7, 10, 0, 30)) == "2026-07-09"
    assert _et_date("2026-07-10T15:30:00") == "2026-07-10"


def test_pick_run_and_book_alpha(store):
    from agent import ledger

    # Baseline 600 the day BEFORE entry; endpoint 612 today = +2.00%.
    seed_spy(store, {D_BASE: 600.0,
                     D_ENTRY + timedelta(days=1): 606.0,
                     TODAY: 612.0})
    seed_backdated_book(store)

    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "A")
    assert run["spy_same_window_pct"] == 2.0
    assert run["spy_window_sessions"] == 2  # two completed closes after base
    pick = run["picks"][0]
    assert pick["since_this_run_pct"] == 10.0   # 110 vs 100 entry
    assert pick["spy_same_window_pct"] == 2.0
    assert pick["alpha_pct"] == 8.0             # skill, net of the market
    assert pick["is_option"] is False

    # Book: equity = 100_000 - 1_000 + 10*110 = 100_100 → +0.10% vs SPY +2.00%
    book = out["book"]
    assert book["inception"] == str(D_ENTRY)
    assert book["since_inception_pct"] == 0.1
    assert book["spy_since_inception_pct"] == 2.0
    assert book["alpha_pct"] == -1.9            # made money, lost to the index


def test_baseline_is_strictly_before_window_start(store):
    from agent import ledger

    # A close ON the entry date must NOT be the baseline (it prints at 16:00,
    # after the intraday fill). Baseline = 500 three days earlier.
    seed_spy(store, {D_ENTRY - timedelta(days=3): 500.0,
                     D_ENTRY: 505.0,
                     TODAY: 510.0})
    seed_backdated_book(store)

    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "A")
    assert run["spy_same_window_pct"] == 2.0    # 510 vs 500, not vs 505


def test_degenerate_window_is_none_not_zero(store):
    from agent import ledger

    # Only one SPY row at/before the window: baseline row == endpoint row.
    # The old code reported a confident 0.00 here — alpha then silently
    # equaled the raw return. None means "too young to benchmark".
    seed_spy(store, {D_BASE: 600.0})
    seed_backdated_book(store)

    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "A")
    assert run["spy_same_window_pct"] is None
    assert run["picks"][0]["alpha_pct"] is None


def test_closed_round_trip_gets_exit_bounded_window(store):
    from agent import ledger
    from agent.brain import save_decision

    d_exit = D_ENTRY + timedelta(days=1)
    seed_spy(store, {D_BASE: 600.0, d_exit: 606.0, TODAY: 612.0})
    save_decision(store, run_id="A", summary="round trip",
                  picks=[{"symbol": "XYZ", "action": "buy",
                          "prediction": "XYZ +5% within 5 sessions",
                          "horizon_days": 5, "kill": "closes below 95"}])
    store.update("desk_decisions", {"run_id": "A"}, {"ts": ts_of(D_ENTRY)},
                 returning=False)
    seed_trade(store, "A", "XYZ", "BUY", 10.0, 100.0, ts_of(D_ENTRY))
    seed_trade(store, "A", "XYZ", "SELL", 10.0, 110.0, ts_of(d_exit))

    out = ledger.outcomes(store, days=30)
    pick = next(r for r in out["runs"] if r["run_id"] == "A")["picks"][0]
    assert pick["closed_return_pct"] == 10.0
    # SPY window stops at the EXIT day's close (606), not today's 612:
    assert pick["spy_same_window_pct"] == 1.0
    assert pick["alpha_pct"] == 9.0
    assert pick["open_now"] is None


def test_option_pick_alpha_is_null_by_design(store):
    from agent import ledger
    from agent.brain import save_decision

    occ_sym = "NVDA270116C00200000"
    seed_spy(store, {D_BASE: 600.0, TODAY: 612.0})
    save_decision(store, run_id="O", summary="call",
                  picks=[{"symbol": occ_sym, "action": "buy",
                          "prediction": "NVDA reclaims $210 pre-expiry",
                          "horizon_days": 20, "kill": "NVDA closes below $180"}])
    store.update("desk_decisions", {"run_id": "O"}, {"ts": ts_of(D_ENTRY)},
                 returning=False)
    seed_trade(store, "O", occ_sym, "BUY", 2.0, 5.0, ts_of(D_ENTRY))
    ledger.mark(store, prices={occ_sym: 7.5})

    out = ledger.outcomes(store, days=30)
    pick = next(r for r in out["runs"] if r["run_id"] == "O")["picks"][0]
    assert pick["is_option"] is True
    assert pick["since_this_run_pct"] == 50.0   # premium move still shown
    assert pick["alpha_pct"] is None            # but never called alpha


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


def test_prediction_registry_passes_through_outcomes(store):
    """v8.16: prediction/horizon_days/kill ride the pick dossier into
    outcomes so Friday grading is mechanical — predicted vs happened."""
    from agent import ledger
    from agent.brain import save_decision

    save_decision(store, run_id="P", summary="registry",
                  picks=[{"symbol": "XYZ", "action": "buy",
                          "prediction": "reclaims $410 within 10 sessions",
                          "horizon_days": 10, "kill": "closes below $385"}])
    out = ledger.outcomes(store, days=30)
    pick = next(r for r in out["runs"] if r["run_id"] == "P")["picks"][0]
    assert pick["prediction"] == "reclaims $410 within 10 sessions"
    assert pick["horizon_days"] == 10
    assert pick["kill"] == "closes below $385"


def test_rejected_round_trip_and_amend_clears(store):
    from agent import ledger
    from agent.brain import save_decision

    seed_spy(store, {D_BASE: 600.0, TODAY: 612.0})
    seed_backdated_book(store)

    rows = store.select("desk_decisions", filters={"run_id": "A"})
    assert rows[0]["rejected"] == [{"symbol": "ABC", "why_not": "falling knife"}]
    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "A")
    assert run["rejected"][0]["symbol"] == "ABC"

    # Amending the decision without --rejected-file must CLEAR the registry
    # (full-rewrite semantics) — a stale reject list must never stay paired
    # with rewritten picks.
    save_decision(store, run_id="A", summary="amended")
    rows = store.select("desk_decisions", filters={"run_id": "A"})
    assert rows[0]["rejected"] is None


def test_empty_book_has_no_book_block(store):
    from agent import ledger

    out = ledger.outcomes(store, days=30)
    assert out["book"] is None and out["runs"] == []


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


def test_outcomes_skips_book_picks(store):
    from agent import ledger
    from agent.brain import save_decision

    save_decision(store, run_id="B1", summary="stance",
                  picks=[{"symbol": "BOOK", "action": "hold"},
                         {"symbol": "XYZ", "action": "hold"}])
    out = ledger.outcomes(store, days=30)
    run = next(r for r in out["runs"] if r["run_id"] == "B1")
    assert [p["symbol"] for p in run["picks"]] == ["XYZ"]  # BOOK never graded


def test_portfolio_and_decision_endpoints(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    seed_spy(store, {D_BASE: 600.0, TODAY: 612.0})
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


def test_portfolio_applies_option_multiplier(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps
    from agent import ledger

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    occ_sym = "NVDA270116C00200000"
    seed_trade(store, "O", occ_sym, "BUY", 2.0, 5.0, ts_of(D_ENTRY))
    ledger.mark(store, prices={occ_sym: 7.5})

    from dashboard.app import app

    with TestClient(app) as c:
        pf = c.get("/api/desk/portfolio").json()
        row = next(p for p in pf["positions"] if p["symbol"] == occ_sym)
        assert row["market_value"] == 1500.0     # 2 contracts × 7.5 × 100
        assert row["unrealized_pnl"] == 500.0    # 2 × (7.5-5.0) × 100
        # equity = 100k - 1000 premium + 1500 mark
        assert pf["equity"] == pytest.approx(100500.0)

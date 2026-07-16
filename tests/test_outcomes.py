"""Outcome-attribution tests: realized P&L per run/symbol, pick joins, options
multiplier, settlement bucketing — the numbers wiki lessons must cite."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from agent import occ


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'oc.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def q(px):
    return {"bid": px * 0.999, "ask": px * 1.001, "mid": px, "t": "x", "src": "test"}


def _fill(store, run_id, symbol, side, shares, price):
    from agent import ledger
    r = ledger.record_trade(store, run_id=run_id, symbol=symbol, side=side,
                            shares=shares, price=price, fill_quote=q(price))
    assert r["ok"], r


def test_two_run_attribution_and_since_pct(store):
    from agent import ledger
    from agent.brain import save_decision

    save_decision(store, run_id="A", summary="run A",
                  picks=[{"symbol": "XYZ", "action": "buy",
                          "why_now": "breakout", "rationale": "trend",
                          "prediction": "XYZ +5% within 10 sessions",
                          "horizon_days": 10, "kill": "closes below 90"}])
    save_decision(store, run_id="B", summary="run B",
                  picks=[{"symbol": "XYZ", "action": "trim",
                          "why_now": "extended", "rationale": "take profit"}])
    _fill(store, "A", "XYZ", "BUY", 10, 100.0)
    _fill(store, "B", "XYZ", "BUY", 10, 110.0)
    _fill(store, "B", "XYZ", "SELL", 10, 120.0)   # closes vs avg 105 → +150 to B
    ledger.mark(store, prices={"XYZ": 130.0})

    out = ledger.outcomes(store, days=30)
    runs = {r["run_id"]: r for r in out["runs"]}
    assert runs["B"]["run_realized_pnl"] == 150.0
    assert runs["A"]["run_realized_pnl"] == 0.0

    pick_a = runs["A"]["picks"][0]
    assert pick_a["rationale"] == "trend"          # rationale joined
    assert pick_a["entry_avg_px"] == 100.0
    assert pick_a["since_this_run_pct"] == 30.0    # 130 vs run A's own 100 entry
    assert pick_a["open_now"]["shares"] == 10.0

    sym = next(s for s in out["symbols"] if s["symbol"] == "XYZ")
    assert sym["realized_pnl"] == 150.0            # exact per symbol
    assert sym["unrealized_pnl"] == pytest.approx(10 * (130 - 105))


def test_option_round_trip_multiplier(store):
    from agent import ledger
    sym = occ.build("NVDA", date.today() + timedelta(days=45), "C", 200)
    _fill(store, "R1", sym, "BUY", 2, 5.0)
    _fill(store, "R2", sym, "SELL", 2, 6.5)
    _, by_symbol = ledger._realized_pnl(ledger._trades(store, "agent"))
    assert by_symbol[sym] == pytest.approx(2 * (6.5 - 5.0) * 100)  # +300, ×100

    out = ledger.outcomes(store, days=30)
    row = next(s for s in out["symbols"] if s["symbol"] == sym)
    assert row["realized_pnl"] == 300.0 and row["is_option"] is True


def test_short_leg_realized_sign(store):
    """Buying back a short cheaper than sold = profit (sign −1 path)."""
    from agent import ledger
    # covered call: 100 shares + short 1 call sold at 5, bought back at 3
    _fill(store, "R1", "NVDA", "BUY", 100, 195.0)
    call = occ.build("NVDA", date.today() + timedelta(days=45), "C", 220)
    _fill(store, "R1", call, "SELL", 1, 5.0)
    _fill(store, "R2", call, "BUY", 1, 3.0)
    _, by_symbol = ledger._realized_pnl(ledger._trades(store, "agent"))
    assert by_symbol[call] == pytest.approx((5.0 - 3.0) * 100)  # +200


def test_settlement_bucketed_not_in_runs(store):
    from agent import ledger
    store.insert("desk_trades", {
        "account": "agent", "run_id": "settlement", "symbol": "XYZ",
        "side": "BUY", "shares": 10, "price": 100.0, "dollars": 1000.0,
        "ts": ledger._utcnow()}, returning=False)
    store.insert("desk_trades", {
        "account": "agent", "run_id": "settlement", "symbol": "XYZ",
        "side": "SELL", "shares": 10, "price": 90.0, "dollars": 900.0,
        "ts": ledger._utcnow()}, returning=False)
    out = ledger.outcomes(store, days=30)
    assert out["settlement"]["realized_pnl"] == -100.0
    assert all(r["run_id"] != "settlement" for r in out["runs"])


def test_pick_with_no_fills_still_gradable(store):
    from agent import ledger
    from agent.brain import save_decision
    save_decision(store, run_id="H", summary="held",
                  picks=[{"symbol": "AAPL", "action": "hold",
                          "why_now": "no edge", "rationale": "wait"}])
    out = ledger.outcomes(store, days=30)
    pick = out["runs"][0]["picks"][0]
    assert pick["fills"] == [] and pick["realized_pnl"] == 0.0
    assert pick["why_now"] == "no edge"


def test_unattributed_counted(store):
    from agent import ledger
    store.insert("desk_trades", {
        "account": "agent", "run_id": None, "symbol": "ZZZ", "side": "BUY",
        "shares": 1, "price": 10.0, "dollars": 10.0,
        "ts": ledger._utcnow()}, returning=False)
    assert ledger.outcomes(store, days=30)["unattributed_trades"] == 1


def test_hardstop_runs_bucketed_like_settlement(store):
    """L1: hard-stop exits book under run_id 'hardstop:<watch id>' — no
    decision carries that id, so they get their own bucket (like
    settlement) instead of vanishing from per-run grading."""
    from agent import ledger
    _fill(store, "A", "XYZ", "BUY", 10, 100.0)
    _fill(store, "hardstop:7", "XYZ", "SELL", 10, 90.0)  # the stop loss
    out = ledger.outcomes(store, days=30)
    assert out["hardstop"]["realized_pnl"] == -100.0
    assert out["settlement"]["realized_pnl"] == 0.0
    assert out["unattributed_trades"] == 0
    assert all(not str(r["run_id"]).startswith("hardstop:")
               for r in out["runs"])


# ── M1: splits must not corrupt grading ──


def _seed_trade(store, run_id, symbol, side, shares, price, ts,
                fill_quote=None):
    store.insert("desk_trades", {
        "account": "agent", "run_id": run_id, "symbol": symbol, "side": side,
        "shares": shares, "price": price, "dollars": round(shares * price, 2),
        "fill_quote": fill_quote, "ts": ts}, returning=False)


def _seed_split_adjustment(store, symbol, delta, exec_ts, ratio):
    """A booked 0-price split row, exactly as settle writes it."""
    store.insert("desk_trades", {
        "account": "agent", "run_id": "settlement", "symbol": symbol,
        "side": "BUY" if delta > 0 else "SELL", "shares": abs(delta),
        "price": 0.0, "dollars": 0.0,
        "fill_quote": {"src": "split_adjustment",
                       "execution_date": str(exec_ts.date()), "ratio": ratio},
        "ts": exec_ts}, returning=False)


def test_split_between_entry_and_mark_grades_flat_as_zero(store):
    """M1 repro: entry fills at pre-split prices, marks post-split — the
    old math graded a flat position −50% after a 2:1 and Friday's
    reflection learned a lie. entry_avg_px must rebase to the current
    share basis via the ledger's own split_adjustment rows."""
    from datetime import datetime, timedelta

    from agent import ledger
    from agent.brain import save_decision

    t0 = datetime.utcnow() - timedelta(days=6)
    save_decision(store, run_id="S", summary="entry",
                  picks=[{"symbol": "XYZ", "action": "buy",
                          "prediction": "XYZ +5% within 10 sessions",
                          "horizon_days": 10, "kill": "closes below 90"}])
    store.update("desk_decisions", {"run_id": "S"}, {"ts": t0}, returning=False)
    _seed_trade(store, "S", "XYZ", "BUY", 10, 100.0, t0)
    _seed_split_adjustment(store, "XYZ", +10,
                           datetime.utcnow() - timedelta(days=3), "2:1")
    ledger.mark(store, prices={"XYZ": 50.0})  # rebased tape, flat position

    out = ledger.outcomes(store, days=30)
    pick = next(r for r in out["runs"] if r["run_id"] == "S")["picks"][0]
    assert pick["entry_avg_px"] == pytest.approx(50.0)       # current basis
    assert pick["since_this_run_pct"] == pytest.approx(0.0)  # flat, not −50
    # the raw fills stay as booked (receipts)
    assert pick["fills"] == [{"side": "BUY", "shares": 10.0, "price": 100.0}]


def test_closed_round_trip_across_split_matches_buys_to_sells(store):
    """M1: buy 10 pre-split, sell 20 post-split — on the current basis
    that IS a closed round trip (20 vs 20) and its return is +10%
    (sell 55 vs rebased entry 50), not an unmatched half-position."""
    from datetime import datetime, timedelta

    from agent import ledger
    from agent.brain import save_decision

    t0 = datetime.utcnow() - timedelta(days=6)
    save_decision(store, run_id="C", summary="round trip",
                  picks=[{"symbol": "XYZ", "action": "buy",
                          "prediction": "XYZ +5% within 10 sessions",
                          "horizon_days": 10, "kill": "closes below 90"}])
    store.update("desk_decisions", {"run_id": "C"}, {"ts": t0}, returning=False)
    _seed_trade(store, "C", "XYZ", "BUY", 10, 100.0, t0)
    _seed_split_adjustment(store, "XYZ", +10,
                           datetime.utcnow() - timedelta(days=3), "2:1")
    _seed_trade(store, "C", "XYZ", "SELL", 20, 55.0,
                datetime.utcnow() - timedelta(days=1))

    out = ledger.outcomes(store, days=30)
    pick = next(r for r in out["runs"] if r["run_id"] == "C")["picks"][0]
    assert pick["closed_return_pct"] == pytest.approx(10.0)
    assert pick["open_now"] is None
    assert pick["realized_pnl"] == pytest.approx(100.0)  # 20 × (55 − 50)

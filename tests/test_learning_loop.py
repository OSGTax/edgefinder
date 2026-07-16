"""C3: the learning loop in code — `agent.ledger grade` materializes
machine facts into desk_outcomes, `agent.brain verdict` stores the weekly
reflection's judgment durably next to those facts, and `agent.brain context`
puts the cycle's whole working memory in one bounded read.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pytest

TODAY = date.today()


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'loop.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401 — daily_bars / dividends
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def q(px):
    return {"bid": px, "ask": px, "mid": px, "t": "x", "src": "test"}


def _seed_trade(store, run_id, symbol, side, shares, price, ts):
    store.insert("desk_trades", {
        "account": "agent", "run_id": run_id, "symbol": symbol, "side": side,
        "shares": shares, "price": price, "dollars": round(shares * price, 2),
        "ts": ts}, returning=False)


def _seed_close(store, symbol, day, close):
    store.insert("daily_bars", {"symbol": symbol, "date": day, "open": close,
                                "high": close, "low": close, "close": close,
                                "volume": 1e6, "source": "test"},
                 returning=False)


def _seed_spy(store, days=10, close=500.0):
    for i in range(days, 0, -1):
        _seed_close(store, "SPY", TODAY - timedelta(days=i), close)


def _decision(store, run_id, picks, ts):
    from agent.brain import save_decision
    r = save_decision(store, run_id=run_id, summary=f"run {run_id}",
                      picks=picks)
    assert r["ok"], r
    store.update("desk_decisions", {"run_id": run_id}, {"ts": ts},
                 returning=False)


def _buy_pick(symbol, kill="closes below $90", horizon=2):
    return {"symbol": symbol, "action": "buy",
            "prediction": f"{symbol} +5% within {horizon} sessions",
            "horizon_days": horizon, "kill": kill}


# ── kill parsing (free text → level, or honest null) ──


def test_parse_kill():
    from agent.ledger import _parse_kill
    assert _parse_kill("closes below 90") == 90.0
    assert _parse_kill("closes below $385") == 385.0
    # a $-prefixed level wins over other numbers in the sentence
    assert _parse_kill("closes below $385 within 10 sessions") == 385.0
    assert _parse_kill("closes below $1,050.50") == 1050.5
    assert _parse_kill(87.5) == 87.5
    # ambiguity and pure prose stay null — never a guess
    assert _parse_kill("below 90 or above 120") is None
    assert _parse_kill("thesis breaks on a guidance cut") is None
    assert _parse_kill(None) is None
    assert _parse_kill(True) is None
    assert _parse_kill(-5) is None


# ── grade: machine facts ──


def test_grade_materializes_open_pick_facts(store):
    """Entry facts, horizon-in-sessions math, kill parse + breach against
    stored daily closes — all machine, no LLM."""
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=6)
    entry_day = ledger._et_date(t0)
    _seed_spy(store, days=10)
    _decision(store, "G", [_buy_pick("XYZ", kill="closes below $90",
                                     horizon=2)], t0)
    _seed_trade(store, "G", "XYZ", "BUY", 10, 100.0, t0)
    # closes between entry and today: one dipped through the kill
    _seed_close(store, "XYZ",
                date.fromisoformat(entry_day) + timedelta(days=1), 85.0)
    _seed_close(store, "XYZ",
                date.fromisoformat(entry_day) + timedelta(days=2), 95.0)
    ledger.mark(store, prices={"XYZ": 95.0})

    out = ledger.grade(store, days=30)
    assert out["graded"] == 1
    row = store.select("desk_outcomes",
                       filters={"run_id": "G", "symbol": "XYZ"})[0]
    assert row["entry_avg_px"] == 100.0
    assert row["status"] == "open" and row["mark_basis"] == "mark"
    assert row["since_pct"] == pytest.approx(-5.0)
    assert row["mark_px"] == pytest.approx(95.0)
    assert row["spy_pct"] == pytest.approx(0.0)     # SPY flat over the window
    assert row["alpha_pct"] == pytest.approx(-5.0)
    assert row["horizon_days"] == 2
    assert bool(row["horizon_elapsed"]) is True     # ≥ 2 SPY sessions elapsed
    assert row["kill_level"] == 90.0
    assert bool(row["kill_breached"]) is True       # the 85 close touched it
    assert row["verdict"] is None                   # machine facts only


def test_grade_horizon_not_elapsed_and_kill_not_breached(store):
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=6)
    entry_day = ledger._et_date(t0)
    _seed_spy(store, days=10)
    _decision(store, "H", [_buy_pick("ABC", kill="closes below $50",
                                     horizon=100)], t0)
    _seed_trade(store, "H", "ABC", "BUY", 10, 100.0, t0)
    _seed_close(store, "ABC",
                date.fromisoformat(entry_day) + timedelta(days=1), 98.0)
    ledger.mark(store, prices={"ABC": 98.0})

    ledger.grade(store, days=30)
    row = store.select("desk_outcomes",
                       filters={"run_id": "H", "symbol": "ABC"})[0]
    assert bool(row["horizon_elapsed"]) is False    # 100 sessions not elapsed
    assert bool(row["kill_breached"]) is False      # no close touched 50
    # free-text kill that doesn't parse → both fields honestly null
    _decision(store, "H2", [_buy_pick("DEF",
                                      kill="thesis breaks on a guidance cut")],
              t0)
    _seed_trade(store, "H2", "DEF", "BUY", 10, 100.0, t0)
    ledger.grade(store, days=30)
    row2 = store.select("desk_outcomes",
                        filters={"run_id": "H2", "symbol": "DEF"})[0]
    assert row2["kill_level"] is None and row2["kill_breached"] is None


def test_grade_closed_round_trip_uses_exit_basis(store):
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=6)
    _seed_spy(store, days=10)
    _decision(store, "C", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "C", "XYZ", "BUY", 10, 100.0, t0)
    _seed_trade(store, "C", "XYZ", "SELL", 10, 110.0,
                datetime.utcnow() - timedelta(days=1))
    ledger.grade(store, days=30)
    row = store.select("desk_outcomes",
                       filters={"run_id": "C", "symbol": "XYZ"})[0]
    assert row["status"] == "closed" and row["mark_basis"] == "exit"
    assert row["since_pct"] == pytest.approx(10.0)
    assert row["mark_px"] == pytest.approx(110.0)


def test_grade_excludes_book_settlement_hardstop_and_no_fill_picks(store):
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=2)
    _seed_spy(store, days=10)
    _decision(store, "B", [{"symbol": "BOOK", "action": "hold"},
                           {"symbol": "AAPL", "action": "hold"}], t0)
    _seed_trade(store, "settlement", "XYZ", "BUY", 10, 100.0, t0)
    _seed_trade(store, "hardstop:7", "XYZ", "SELL", 10, 90.0, t0)
    out = ledger.grade(store, days=30)
    assert out["graded"] == 0
    assert store.select("desk_outcomes") == []


def test_grade_upserts_and_verdict_survives_regrade(store):
    """One row per (run_id, symbol); a re-run refreshes machine facts in
    place and NEVER touches the reflection agent's verdict columns."""
    from agent import ledger
    from agent.brain import set_verdict

    t0 = datetime.utcnow() - timedelta(days=6)
    _seed_spy(store, days=10)
    _decision(store, "U", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "U", "XYZ", "BUY", 10, 100.0, t0)
    ledger.mark(store, prices={"XYZ": 105.0})
    ledger.grade(store, days=30)
    v = set_verdict(store, run_id="U", symbol="XYZ", verdict="TRUE",
                    note="+5% inside the horizon")
    assert v["ok"], v
    # the mark moved; re-grade refreshes the facts, keeps the judgment
    ledger.mark(store, prices={"XYZ": 120.0})
    ledger.grade(store, days=30)
    rows = store.select("desk_outcomes",
                        filters={"run_id": "U", "symbol": "XYZ"})
    assert len(rows) == 1
    assert rows[0]["since_pct"] == pytest.approx(20.0)
    assert rows[0]["verdict"] == "TRUE"
    assert rows[0]["verdict_note"] == "+5% inside the horizon"


# ── verdict: the reflection agent's write path ──


def test_verdict_requires_a_graded_row_and_a_known_verdict(store):
    from agent.brain import set_verdict

    r = set_verdict(store, run_id="X", symbol="XYZ", verdict="TRUE")
    assert not r["ok"] and "grade" in r["error"]
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=2)
    _decision(store, "X", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "X", "XYZ", "BUY", 10, 100.0, t0)
    ledger.grade(store, days=30)
    bad = set_verdict(store, run_id="X", symbol="XYZ", verdict="MAYBE")
    assert not bad["ok"] and "TRUE/FALSE/NOT_YET" in bad["error"]
    ok = set_verdict(store, run_id="X", symbol="xyz", verdict="not yet")
    assert ok["ok"] and ok["verdict"] == "NOT_YET"
    row = store.select("desk_outcomes",
                       filters={"run_id": "X", "symbol": "XYZ"})[0]
    assert row["verdict"] == "NOT_YET"


# ── context: the working memory in one read ──


def test_context_aggregates_and_stays_bounded(store):
    from agent import ledger
    from agent.brain import (CONTEXT_CLIP, context, set_wiki, set_state,
                             watch_set)

    t0 = datetime.utcnow() - timedelta(days=3)
    _seed_spy(store, days=10)
    set_state(store, name="trend", thesis="ride winners " * 300)  # long thesis
    set_wiki(store, slug="lessons", body="grade alpha, not dollars")
    long_summary = "a very long story about the market " * 30  # > CONTEXT_CLIP
    from agent.brain import save_decision
    save_decision(store, run_id="R1", summary=long_summary,
                  picks=[_buy_pick("XYZ")])
    store.update("desk_decisions", {"run_id": "R1"}, {"ts": t0},
                 returning=False)
    _seed_trade(store, "R1", "XYZ", "BUY", 10, 100.0, t0)
    ledger.mark(store, prices={"XYZ": 104.0})
    ledger.grade(store, days=30)
    watch_set(store, symbol="XYZ", below=95.0, reason="kill level")
    store.update("desk_watch", {"symbol": "XYZ"},
                 {"status": "tripped", "tripped_price": 94.5}, returning=False)
    from agent.brain import wake_plan
    at = (datetime.utcnow() + timedelta(hours=2)).isoformat() + "Z"
    assert wake_plan(store, at=at, reason="pre-close check")["ok"]

    ctx = context(store, days=14)
    # every section present
    for key in ("account", "brief", "wiki", "strategy", "open_predictions",
                "outcomes", "watches", "wakes", "errors"):
        assert key in ctx, key
    assert ctx["errors"] == {}
    # account header with provenance
    assert ctx["account"]["equity"] > 0
    assert ctx["account"]["positions"][0]["symbol"] == "XYZ"
    assert ctx["account"]["mark_meta"]["sources"]["live"] == 1
    # no brief built → honest exists=False (same read as `agent.market brief`)
    assert ctx["brief"]["exists"] is False
    # the wiki rides whole (it is size-capped at the source)
    assert ctx["wiki"]["pages"][0]["slug"] == "lessons"
    # the open prediction joins its machine-graded facts
    pred = ctx["open_predictions"][0]
    assert pred["symbol"] == "XYZ" and pred["run_id"] == "R1"
    assert pred["outcome"]["status"] == "open"
    assert pred["outcome"]["since_pct"] == pytest.approx(4.0)
    # long free text is clipped, listed runs condensed
    run = ctx["outcomes"]["runs"][0]
    assert len(run["summary"]) <= CONTEXT_CLIP
    assert run["summary"].endswith("…")
    assert len(ctx["strategy"]["thesis"]) <= 2000
    # tripped wires and the planned next look surface
    assert ctx["watches"]["tripped"][0]["symbol"] == "XYZ"
    assert ctx["wakes"]["upcoming"][0]["reason"] == "pre-close check"
    # the whole payload stays a working set, not a dump
    assert len(json.dumps(ctx, default=str)) < 20_000


def test_context_survives_a_dead_section(store, monkeypatch):
    """One broken read lands in errors; the rest of the memory still loads
    (same convention as the brief builder)."""
    from agent import ledger
    from agent.brain import context

    monkeypatch.setattr(ledger, "outcomes",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ctx = context(store)
    assert "outcomes" in ctx["errors"]
    assert ctx["account"] is not None
    assert ctx["wakes"] is not None


def test_context_drops_judged_closed_predictions(store):
    """A closed AND verdicted pick is history, not working memory."""
    from agent import ledger
    from agent.brain import context, set_verdict

    t0 = datetime.utcnow() - timedelta(days=3)
    _seed_spy(store, days=10)
    _decision(store, "D", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "D", "XYZ", "BUY", 10, 100.0, t0)
    _seed_trade(store, "D", "XYZ", "SELL", 10, 110.0,
                datetime.utcnow() - timedelta(days=1))
    ledger.grade(store, days=30)
    assert len(context(store)["open_predictions"]) == 1  # closed, unjudged
    set_verdict(store, run_id="D", symbol="XYZ", verdict="TRUE", note="+10%")
    assert context(store)["open_predictions"] == []

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
    # M1 (reviewer matrix): numbers that are NOT price levels are excluded —
    # percentages are move sizes, unit-adjacent numbers are lookbacks/spans
    assert _parse_kill("drops 8% in a day") is None
    assert _parse_kill("kill if it loses 10%") is None
    assert _parse_kill("closes under 100DMA") is None
    assert _parse_kill("closes under the 200 DMA") is None
    assert _parse_kill("two closes below the 50-day average") is None
    assert _parse_kill("$475") == 475.0
    assert _parse_kill("close below 475") == 475.0
    # count WORDS never register — only digits are candidates, so the level
    # survives ("two" is prose, not a second number)
    assert _parse_kill("two closes below 190") == 190.0
    # plausibility (long-only stop): with the entry price known, a level
    # outside [0.2x, 2x] of it is a parse artifact, not a stop
    assert _parse_kill("close below 475", entry_px=100.0) is None
    assert _parse_kill("closes below 90", entry_px=100.0) == 90.0
    assert _parse_kill(475.0, entry_px=100.0) is None
    assert _parse_kill(87.5, entry_px=100.0) == 87.5


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
    # H3: same-run round trips carry the exit facts too
    assert row["exit_kind"] == "same_run"
    assert row["exit_avg_px"] == pytest.approx(110.0)
    assert row["realized_pnl"] == pytest.approx(100.0)


# ── H3: stop-outs and cross-run exits grade with numbers, not nulls ──


def test_grade_hardstop_exit_grades_with_real_numbers(store):
    """H3 repro: bought in run G, sold by the streamer under run_id
    'hardstop:5' — previously graded closed with null since/mark/alpha,
    so the learning ledger had numbers for open winners and nulls for
    stopped-out losers. The exit must be reconstructed."""
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=6)
    _seed_spy(store, days=10)
    _decision(store, "G", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "G", "XYZ", "BUY", 10, 100.0, t0)
    _seed_trade(store, "hardstop:5", "XYZ", "SELL", 10, 74.0,
                datetime.utcnow() - timedelta(days=1))
    out = ledger.grade(store, days=30)
    assert out["ok"] and out["graded"] == 1
    row = store.select("desk_outcomes",
                       filters={"run_id": "G", "symbol": "XYZ"})[0]
    assert row["status"] == "closed"
    assert row["exit_kind"] == "hardstop"
    assert row["exit_avg_px"] == pytest.approx(74.0)
    assert row["mark_basis"] == "exit"
    assert row["mark_px"] == pytest.approx(74.0)
    assert row["since_pct"] == pytest.approx(-26.0)   # the real stop-out loss
    assert row["spy_pct"] == pytest.approx(0.0)       # SPY flat entry→flat
    assert row["alpha_pct"] == pytest.approx(-26.0)
    assert row["realized_pnl"] == pytest.approx(-260.0)


def test_grade_cross_run_exit_grades_with_exit_kind(store):
    """H3: bought in run A, exited by a LATER run's sell — graded off the
    actual closing fills, tagged cross_run."""
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=6)
    _seed_spy(store, days=10)
    _decision(store, "A", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "A", "XYZ", "BUY", 10, 100.0, t0)
    _seed_trade(store, "B", "XYZ", "SELL", 10, 110.0,
                datetime.utcnow() - timedelta(days=1))
    ledger.grade(store, days=30)
    row = store.select("desk_outcomes",
                       filters={"run_id": "A", "symbol": "XYZ"})[0]
    assert row["status"] == "closed"
    assert row["exit_kind"] == "cross_run"
    assert row["exit_avg_px"] == pytest.approx(110.0)
    assert row["since_pct"] == pytest.approx(10.0)
    assert row["realized_pnl"] == pytest.approx(100.0)


# ── M2: degraded marks must not feed grade unflagged ──


def test_grade_degraded_mark_nulls_facts_and_flags(store, monkeypatch):
    """M2: when the latest snapshot marked the pick's symbol at COST BASIS
    (mark_meta.cost_marked), the fake-flat mark must not grade the pick —
    mark-derived facts write as null with degraded=true; a later clean
    re-grade overwrites both."""
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=6)
    _seed_spy(store, days=10)
    _decision(store, "DM", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "DM", "XYZ", "BUY", 10, 100.0, t0)
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    ledger.mark(store)                      # XYZ marked at cost → degraded
    ledger.grade(store, days=30)
    row = store.select("desk_outcomes",
                       filters={"run_id": "DM", "symbol": "XYZ"})[0]
    assert row["status"] == "open"
    assert row["since_pct"] is None and row["alpha_pct"] is None
    assert row["mark_px"] is None
    assert bool(row["degraded"]) is True
    assert row["entry_avg_px"] == 100.0     # entry facts are not mark-derived
    # a clean mark re-grades with real facts and clears the flag
    ledger.mark(store, prices={"XYZ": 95.0})
    ledger.grade(store, days=30)
    row = store.select("desk_outcomes",
                       filters={"run_id": "DM", "symbol": "XYZ"})[0]
    assert row["since_pct"] == pytest.approx(-5.0)
    assert row["mark_px"] == pytest.approx(95.0)
    assert bool(row["degraded"]) is False


# ── M3: option picks grade net of fees ──


def test_option_pick_grades_net_of_fees(store):
    """M3 (reviewer example): 10 contracts $0.40 → $0.45 is +12.5% gross,
    but $6.50 of fees each way nets +9.1% — option picks grade off the
    fee-inclusive dollars, not price×shares."""
    from agent import ledger, occ

    FEE = ledger.OPTION_FEE_PER_CONTRACT
    sym = occ.build("NVDA", TODAY + timedelta(days=45), "C", 200)
    t0 = datetime.utcnow() - timedelta(days=6)
    _seed_spy(store, days=10)
    _decision(store, "O", [{"symbol": sym, "action": "buy",
                            "prediction": "premium re-rates on the catalyst",
                            "horizon_days": 5, "kill": 0.2}], t0)
    fq_b = {"bid": 0.40, "ask": 0.40, "mid": 0.40, "t": "x", "src": "test"}
    buy = ledger.record_trade(store, run_id="O", symbol=sym, side="BUY",
                              shares=10, price=0.40, fill_quote=fq_b)
    assert buy["ok"] and buy["dollars"] == pytest.approx(400.0 + 10 * FEE)
    fq_s = {"bid": 0.45, "ask": 0.45, "mid": 0.45, "t": "x", "src": "test"}
    sell = ledger.record_trade(store, run_id="O", symbol=sym, side="SELL",
                               shares=10, price=0.45, fill_quote=fq_s)
    assert sell["ok"] and sell["dollars"] == pytest.approx(450.0 - 10 * FEE)

    out = ledger.outcomes(store, days=30)
    pick = out["runs"][0]["picks"][0]
    # (443.5 − 406.5) / 406.5 = +9.1%, NOT (0.45−0.40)/0.40 = +12.5%
    assert pick["closed_return_pct"] == pytest.approx(9.1, abs=0.01)
    assert pick["entry_avg_px"] == pytest.approx(0.4065)  # fee-incl. cost
    assert pick["realized_pnl"] == pytest.approx(37.0)    # 443.5 − 406.5
    assert pick["alpha_pct"] is None                      # options: by design

    ledger.grade(store, days=30)
    row = store.select("desk_outcomes", filters={"run_id": "O"})[0]
    assert row["since_pct"] == pytest.approx(9.1, abs=0.01)
    assert row["exit_kind"] == "same_run"
    assert row["realized_pnl"] == pytest.approx(37.0)
    assert row["alpha_pct"] is None


# ── L4: --days bounds closed-row re-grades only ──


def test_grade_days_bounds_closed_regrades_only(store):
    """An open pick older than the window still refreshes on every pass;
    a closed row outside the window with facts already stored is final."""
    from agent import ledger

    t_old = datetime.utcnow() - timedelta(days=40)
    _seed_spy(store, days=10)
    _decision(store, "OLD", [_buy_pick("OPN"), _buy_pick("CLS")], t_old)
    _seed_trade(store, "OLD", "OPN", "BUY", 10, 100.0, t_old)
    _seed_trade(store, "OLD", "CLS", "BUY", 10, 100.0, t_old)
    _seed_trade(store, "OLD", "CLS", "SELL", 10, 110.0,
                t_old + timedelta(days=1))
    ledger.mark(store, prices={"OPN": 105.0})
    # first pass: both graded (a never-graded closed pick writes its row
    # even outside the window — first facts are not a "re-grade")
    out = ledger.grade(store, days=30)
    assert {r["symbol"] for r in out["rows"]} == {"OPN", "CLS"}
    # second pass: the old closed row is final and skipped; the open pick
    # refreshes regardless of --days
    ledger.mark(store, prices={"OPN": 120.0})
    out2 = ledger.grade(store, days=30)
    assert {r["symbol"] for r in out2["rows"]} == {"OPN"}
    assert out2["closed_rows_outside_window"] == 1
    row = store.select("desk_outcomes", filters={"symbol": "OPN"})[0]
    assert row["since_pct"] == pytest.approx(20.0)


# ── L6: concurrent grade race on the insert path ──


def test_grade_survives_concurrent_insert_race(store):
    """Two graders racing on the same new pick: the loser's insert hits the
    (account, run_id, symbol) unique key and falls back to an update
    instead of crashing the pass."""
    from agent import ledger

    t0 = datetime.utcnow() - timedelta(days=3)
    _seed_spy(store, days=10)
    _decision(store, "RC", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "RC", "XYZ", "BUY", 10, 100.0, t0)
    ledger.mark(store, prices={"XYZ": 105.0})
    assert ledger.grade(store, days=30)["ok"]  # the row now exists

    class RaceStore:
        """Delegates everything, but the pick's existence check misses once
        — as if a sibling grader inserted between the check and the write."""

        def __init__(self, inner):
            self._inner = inner
            self._miss = True

        def select(self, table, **kw):
            f = kw.get("filters") or {}
            if table == "desk_outcomes" and f.get("symbol") and self._miss:
                self._miss = False
                return []
            return self._inner.select(table, **kw)

        def insert(self, *a, **kw):
            return self._inner.insert(*a, **kw)

        def update(self, *a, **kw):
            return self._inner.update(*a, **kw)

        def delete(self, *a, **kw):
            return self._inner.delete(*a, **kw)

    ledger.mark(store, prices={"XYZ": 110.0})
    out = ledger.grade(RaceStore(store), days=30)
    assert out["ok"] and out["graded"] == 1
    rows = store.select("desk_outcomes", filters={"run_id": "RC"})
    assert len(rows) == 1
    assert rows[0]["since_pct"] == pytest.approx(10.0)


# ── L1: pre-deploy grace ──


def test_grade_and_verdict_missing_table_exit_actionably(store):
    """A DB that predates desk_outcomes gets an actionable message, not a
    stack trace mid-reflection."""
    from agent import ledger
    from agent.brain import set_verdict
    from edgefinder.db.engine import Base, get_engine

    Base.metadata.tables["desk_outcomes"].drop(get_engine())
    out = ledger.grade(store, days=30)
    assert not out["ok"] and "not migrated" in out["error"]
    v = set_verdict(store, run_id="X", symbol="XYZ", verdict="TRUE")
    assert not v["ok"] and "not migrated" in v["error"]


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


def test_context_clips_a_fat_brief(store):
    """L5: context's boundedness must not depend on the brief builder — a
    bloated brief is clipped at the read (roster/screens/headlines caps),
    keeping the whole payload a working set."""
    from agent.brain import (CONTEXT_BRIEF_HEADLINES, CONTEXT_BRIEF_ROSTER,
                             CONTEXT_BRIEF_SCREEN, context)

    fat = {"as_of": str(TODAY),
           "trend_roster": [{"symbol": f"S{i}", "close": 100.0,
                             "ret_1m": 1.0, "ret_3m": 2.0, "rsi": 55.0}
                            for i in range(120)],
           "screens": {"note": "x",
                       "beyond_megacaps": [{"symbol": f"B{i}",
                                            "ret_3m_pct": 9.9}
                                           for i in range(60)],
                       "new_highs": [{"symbol": f"H{i}"} for i in range(60)]},
           "headlines": {f"N{i}": [{"title": "a headline " * 8}] * 3
                         for i in range(40)}}
    store.insert("desk_briefs", {"account": "agent", "brief_date": TODAY,
                                 "built_at": datetime.utcnow(),
                                 "payload": fat}, returning=False)
    ctx = context(store)
    assert ctx["brief"]["exists"] is True
    payload = ctx["brief"]["payload"]
    assert len(payload["trend_roster"]) == CONTEXT_BRIEF_ROSTER
    assert payload["trend_roster_clipped"] == 120     # honesty: it was more
    assert len(payload["screens"]["beyond_megacaps"]) == CONTEXT_BRIEF_SCREEN
    assert len(payload["screens"]["new_highs"]) == CONTEXT_BRIEF_SCREEN
    assert payload["screens"]["note"] == "x"          # non-lists untouched
    assert len(payload["headlines"]) == CONTEXT_BRIEF_HEADLINES
    assert payload["headlines_clipped"] == 40
    # the <20KB boundedness holds even against a fat brief
    assert len(json.dumps(ctx, default=str)) < 20_000


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

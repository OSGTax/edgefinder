"""C3: the learning loop in code — `agent.grade run` materializes machine
facts into desk_outcomes from the Alpaca mirror, `agent.brain verdict`
stores the weekly reflection's judgment durably next to those facts, and
`agent.brain context` puts the cycle's whole working memory in one
bounded read. Deep grade mechanics live in test_grade_alpaca.py; this
file keeps the loop-level contracts: windows, races, pre-deploy grace,
the verdict write path, and context.
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


_OID = iter(range(1, 10_000))


def _seed_trade(store, run_id, symbol, side, shares, price, ts):
    """A filled order in the Alpaca mirror — the V4 shape of 'a fill'."""
    n = next(_OID)
    store.insert("desk_orders", {
        "account": "agent", "run_id": run_id, "seq": n,
        "client_order_id": f"{run_id}:{n:02d}", "alpaca_order_id": f"lo-{n}",
        "symbol": symbol, "asset_class": "us_equity",
        "side": side.lower(), "kind": "entry" if side == "BUY" else "exit",
        "order_type": "market", "tif": "day", "qty": shares,
        "status": "filled", "filled_qty": shares, "filled_avg_price": price,
        "filled_at": ts.isoformat() + "+00:00"}, returning=False)


def _pos(symbol, qty, avg, cur):
    return {"symbol": symbol, "asset_class": "us_equity", "qty": qty,
            "qty_available": qty, "avg_entry_price": avg,
            "current_price": cur, "market_value": (cur or avg) * qty,
            "cost_basis": avg * qty,
            "unrealized_pl": round(((cur or avg) - avg) * qty, 2),
            "unrealized_plpc": None, "change_today": None, "side": "long"}


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
    from agent.grade import _parse_kill
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


# ── grade: loop-level contracts (mechanics live in test_grade_alpaca) ──


def test_grade_days_bounds_closed_regrades_only(store):
    """An open pick older than the window still refreshes on every pass;
    a closed row outside the window with facts already stored is final."""
    from agent.grade import grade

    t_old = datetime.utcnow() - timedelta(days=40)
    _seed_spy(store, days=10)
    _decision(store, "OLD", [_buy_pick("OPN"), _buy_pick("CLS")], t_old)
    _seed_trade(store, "OLD", "OPN", "BUY", 10, 100.0, t_old)
    _seed_trade(store, "OLD", "CLS", "BUY", 10, 100.0, t_old)
    _seed_trade(store, "OLD", "CLS", "SELL", 10, 110.0,
                t_old + timedelta(days=1))
    # first pass: both graded (a never-graded closed pick writes its row
    # even outside the window — first facts are not a "re-grade")
    out = grade(store, days=30, positions=[_pos("OPN", 10, 100.0, 105.0)])
    assert {r["symbol"] for r in out["rows"]} == {"OPN", "CLS"}
    # second pass: the old closed row is final and skipped; the open pick
    # refreshes regardless of --days
    out2 = grade(store, days=30, positions=[_pos("OPN", 10, 100.0, 120.0)])
    assert {r["symbol"] for r in out2["rows"]} == {"OPN"}
    assert out2["closed_rows_outside_window"] == 1
    row = store.select("desk_outcomes", filters={"symbol": "OPN"})[0]
    assert row["since_pct"] == pytest.approx(20.0)


# ── L6: concurrent grade race on the insert path ──


def test_grade_survives_concurrent_insert_race(store):
    """Two graders racing on the same new pick: the loser's insert hits the
    (account, run_id, symbol) unique key and falls back to an update
    instead of crashing the pass."""
    from agent.grade import grade

    t0 = datetime.utcnow() - timedelta(days=3)
    _seed_spy(store, days=10)
    _decision(store, "RC", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "RC", "XYZ", "BUY", 10, 100.0, t0)
    assert grade(store, days=30,
                 positions=[_pos("XYZ", 10, 100.0, 105.0)])["ok"]

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

    out = grade(RaceStore(store), days=30,
                positions=[_pos("XYZ", 10, 100.0, 110.0)])
    assert out["ok"] and out["graded"] == 1
    rows = store.select("desk_outcomes", filters={"run_id": "RC"})
    assert len(rows) == 1
    assert rows[0]["since_pct"] == pytest.approx(10.0)


# ── L1: pre-deploy grace ──


def test_grade_and_verdict_missing_table_exit_actionably(store):
    """A DB that predates desk_outcomes gets an actionable message, not a
    stack trace mid-reflection."""
    from agent.brain import set_verdict
    from agent.grade import grade
    from edgefinder.db.engine import Base, get_engine

    Base.metadata.tables["desk_outcomes"].drop(get_engine())
    out = grade(store, days=30)
    assert not out["ok"] and "not migrated" in out["error"]
    v = set_verdict(store, run_id="X", symbol="XYZ", verdict="TRUE")
    assert not v["ok"] and "not migrated" in v["error"]


def test_grade_and_verdict_transient_error_reraises(store):
    """M3: a transient connection error whose message merely MENTIONS
    desk_outcomes (SQLAlchemy embeds the SQL in str(exc)) must re-raise —
    the old string-match misdiagnosed every blip as 'schema not migrated'."""
    from agent.brain import set_verdict
    from agent.grade import grade

    class _Blip:
        def select(self, *a, **kw):
            raise RuntimeError(
                'connection reset during "SELECT * FROM desk_outcomes"')

    with pytest.raises(RuntimeError, match="connection reset"):
        grade(_Blip(), days=30)
    with pytest.raises(RuntimeError, match="connection reset"):
        set_verdict(_Blip(), run_id="X", symbol="XYZ", verdict="TRUE")


def test_grade_excludes_book_and_no_fill_picks(store):
    """BOOK stances and picks with no entry fills of their own never grade —
    and fills with a run_id that has no decision row grade nothing either."""
    from agent.grade import grade

    t0 = datetime.utcnow() - timedelta(days=2)
    _seed_spy(store, days=10)
    _decision(store, "B", [{"symbol": "BOOK", "action": "hold"},
                           {"symbol": "AAPL", "action": "hold"}], t0)
    _seed_trade(store, "ORPHAN", "XYZ", "BUY", 10, 100.0, t0)
    out = grade(store, days=30, positions=[])
    assert out["graded"] == 0
    assert store.select("desk_outcomes") == []


def test_grade_upserts_and_verdict_survives_regrade(store):
    """One row per (run_id, symbol); a re-run refreshes machine facts in
    place and NEVER touches the reflection agent's verdict columns."""
    from agent.brain import set_verdict
    from agent.grade import grade

    t0 = datetime.utcnow() - timedelta(days=6)
    _seed_spy(store, days=10)
    _decision(store, "U", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "U", "XYZ", "BUY", 10, 100.0, t0)
    grade(store, days=30, positions=[_pos("XYZ", 10, 100.0, 105.0)])
    v = set_verdict(store, run_id="U", symbol="XYZ", verdict="TRUE",
                    note="+5% inside the horizon")
    assert v["ok"], v
    # the mark moved; re-grade refreshes the facts, keeps the judgment
    grade(store, days=30, positions=[_pos("XYZ", 10, 100.0, 120.0)])
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
    from agent.grade import grade

    t0 = datetime.utcnow() - timedelta(days=2)
    _decision(store, "X", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "X", "XYZ", "BUY", 10, 100.0, t0)
    grade(store, days=30, positions=[_pos("XYZ", 10, 100.0, 101.0)])
    bad = set_verdict(store, run_id="X", symbol="XYZ", verdict="MAYBE")
    assert not bad["ok"] and "TRUE/FALSE/NOT_YET" in bad["error"]
    ok = set_verdict(store, run_id="X", symbol="xyz", verdict="not yet")
    assert ok["ok"] and ok["verdict"] == "NOT_YET"
    row = store.select("desk_outcomes",
                       filters={"run_id": "X", "symbol": "XYZ"})[0]
    assert row["verdict"] == "NOT_YET"


# ── context: the working memory in one read ──


def test_context_aggregates_and_stays_bounded(store, monkeypatch):
    from agent import trade as trade_mod
    from agent.brain import CONTEXT_CLIP, context, set_wiki, set_state
    from agent.grade import grade as grade_run

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
    # V4: the fill lives in the Alpaca mirror; the mark is the Alpaca
    # position's current_price.
    store.insert("desk_orders", {
        "account": "agent", "run_id": "R1", "seq": 1,
        "client_order_id": "R1:01", "alpaca_order_id": "ctx-o1",
        "symbol": "XYZ", "asset_class": "us_equity", "side": "buy",
        "kind": "entry", "order_type": "market", "tif": "day",
        "qty": 10.0, "status": "filled", "filled_qty": 10.0,
        "filled_avg_price": 100.0, "filled_at": t0.isoformat() + "+00:00"},
        returning=False)
    xyz_pos = {"symbol": "XYZ", "asset_class": "us_equity", "qty": 10.0,
               "qty_available": 10.0, "avg_entry_price": 100.0,
               "current_price": 104.0, "market_value": 1040.0,
               "cost_basis": 1000.0, "unrealized_pl": 40.0,
               "unrealized_plpc": 0.04, "change_today": None, "side": "long"}
    grade_run(store, days=30, positions=[xyz_pos])
    # The account header is a live Alpaca read — canned here.
    monkeypatch.setattr(trade_mod, "state", lambda: {
        "cash": 99_000.0, "equity": 100_040.0, "buying_power": 99_000.0,
        "total_pnl": 40.0, "total_return_pct": 0.04,
        "positions": [dict(xyz_pos, weight=0.0104)]})
    from agent.brain import wake_plan
    at = (datetime.utcnow() + timedelta(hours=2)).isoformat() + "Z"
    assert wake_plan(store, at=at, reason="pre-close check")["ok"]

    ctx = context(store, days=14)
    # every section present (V4: no watches — tripwires are gone; stops
    # rest on Alpaca's book and appear in the open-orders read)
    for key in ("account", "brief", "wiki", "strategy", "open_predictions",
                "outcomes", "commitments", "wakes", "errors"):
        assert key in ctx, key
    assert ctx["errors"] == {}
    # account header off the paper account
    assert ctx["account"]["equity"] > 0
    assert ctx["account"]["positions"][0]["symbol"] == "XYZ"
    assert ctx["account"]["positions"][0]["current_price"] == 104.0
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
    # the planned next look surfaces
    assert ctx["wakes"]["upcoming"][0]["reason"] == "pre-close check"
    # the whole payload stays a working set, not a dump
    assert len(json.dumps(ctx, default=str)) < 20_000


def test_context_survives_a_dead_section(store, monkeypatch):
    """One broken read lands in errors; the rest of the memory still loads
    (same convention as the brief builder)."""
    from agent import grade as grade_mod
    from agent.brain import context

    monkeypatch.setattr(grade_mod, "outcomes",
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
    from agent.brain import context, set_verdict
    from agent.grade import grade

    t0 = datetime.utcnow() - timedelta(days=3)
    _seed_spy(store, days=10)
    _decision(store, "D", [_buy_pick("XYZ")], t0)
    _seed_trade(store, "D", "XYZ", "BUY", 10, 100.0, t0)
    _seed_trade(store, "D", "XYZ", "SELL", 10, 110.0,
                datetime.utcnow() - timedelta(days=1))
    grade(store, days=30, positions=[])
    assert len(context(store)["open_predictions"]) == 1  # closed, unjudged
    set_verdict(store, run_id="D", symbol="XYZ", verdict="TRUE", note="+10%")
    assert context(store)["open_predictions"] == []

"""End-to-end proof of the agent tool layer on a seeded SQLite DB.

No network, no R2, no Postgres — a synthetic daily_bars universe stands in for
the real data asset so the ledger math, backtest tool, and brain writers can be
exercised exactly as the Routine will call them.
"""

from __future__ import annotations


import pandas as pd
import pytest


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    db_path = tmp_path / "agent_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    # ensure no R2 path is taken
    for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"):
        monkeypatch.delenv(k, raising=False)

    from edgefinder.db.engine import Base, get_engine
    import edgefinder.db.models  # noqa: F401 — register data tables
    import agent.models  # noqa: F401 — register desk_* tables
    import agent.data as agent_data

    engine = get_engine()
    Base.metadata.create_all(engine)
    # reset cached session factory to the new engine
    agent_data._session_factory = None

    # seed ~400 trading days of synthetic bars ending today
    from edgefinder.db.models import DailyBar
    cal = list(pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=420))
    sess = agent_data.session_factory()()
    try:
        specs = {"SPY": (400.0, 0.0004), "NVDA": (100.0, 0.0016), "AAPL": (150.0, 0.0006)}
        for sym, (p0, drift) in specs.items():
            px = p0
            for d in cal:
                px *= (1.0 + drift)
                o = px * 0.995
                sess.add(DailyBar(symbol=sym, date=d.date(), open=o, high=px * 1.01,
                                  low=o * 0.99, close=px, volume=5_000_000.0,
                                  source="test"))
        sess.commit()
    finally:
        sess.close()
    return agent_data


def test_market_observe(seeded):
    from agent import data
    q = data.latest_indicators(["SPY", "NVDA"])
    assert "SPY" in q and "NVDA" in q
    assert q["NVDA"]["close"] > 0
    assert q["NVDA"]["indicators"].get("ema_200")
    reg = data.regime()
    assert reg["tag"] in ("risk_on", "risk_off", "neutral")
    # strong uptrend → SPY above its long EMA
    assert reg["indices"]["SPY"]["above_200"] is True


def test_backtest_tool(seeded):
    from agent import backtest_tool
    out = backtest_tool.run(["NVDA", "AAPL", "SPY"], "momentum:2",
                            schedule="monthly", start=None, costed=True)
    assert "error" not in out
    assert out["num_trades"] >= 1
    assert isinstance(out["return_pct"], float)
    # buy-and-hold of the strongest name should be strongly positive on this drift
    bh = backtest_tool.run(["NVDA"], "buyhold:NVDA", costed=False)
    assert bh["return_pct"] > 0


def test_ledger_cash_integrity(seeded):
    from agent import ledger
    from agent.models import STARTING_CAPITAL
    from agent.store import get_store
    store = get_store()
    # buy
    r = ledger.record_trade(store, symbol="NVDA", side="BUY", shares=100,
                            price=ledger._latest_close("NVDA"),
                            rationale="test buy", run_id="R1")
    assert r["ok"], r
    st = ledger.state(store)
    assert st["positions"][0]["symbol"] == "NVDA"
    assert abs(st["cash"] + st["positions_value"] - st["equity"]) < 0.01

    # cash must equal starting - buy cost (ledger is source of truth)
    buy_cost = r["dollars"]
    assert abs(ledger.cash(store) - (STARTING_CAPITAL - buy_cost)) < 0.01

    # partial sell
    r2 = ledger.record_trade(store, symbol="NVDA", side="SELL", shares=40,
                             price=ledger._latest_close("NVDA"), run_id="R1")
    assert r2["ok"]
    st2 = ledger.state(store)
    assert st2["positions"][0]["shares"] == 60

    # mark appends an equity snapshot
    marked = ledger.mark(store)
    assert marked["equity"] > 0
    from agent.models import DeskEquity
    sess = seeded.session_factory()()
    try:
        assert sess.query(DeskEquity).count() == 1
    finally:
        sess.close()


def test_ledger_guards(seeded):
    from agent import ledger
    from agent.store import get_store
    store = get_store()
    close = ledger._latest_close("AAPL")
    # fill-sanity: a price far from the close is rejected
    bad = ledger.record_trade(store, symbol="AAPL", side="BUY", shares=10,
                              price=close * 5, run_id="R1")
    assert not bad["ok"] and "sanity" in bad["error"]
    # cannot sell what you don't hold
    nosell = ledger.record_trade(store, symbol="AAPL", side="SELL", shares=10,
                                 price=close, run_id="R1")
    assert not nosell["ok"]
    # cannot overdraw cash
    over = ledger.record_trade(store, symbol="AAPL", side="BUY",
                               shares=10_000_000, price=close, run_id="R1")
    assert not over["ok"]


def test_compute_book_equity_oversell_clamps_flat():
    """H2 regression: an equity SELL row must never flip a long lot short.
    record_trade guards upstream, but a double-writer race (streamer hard
    stop + trading cycle, or two streamers during deploy overlap) can land
    an over-sell row — the replay must degrade to FLAT, never fabricate a
    short lot. Options keep honest signed arithmetic."""
    from datetime import date, timedelta

    from agent import ledger, occ

    def t(sym, side, shares, price, ts):
        return {"symbol": sym, "side": side, "shares": shares, "price": price,
                "dollars": round(shares * price, 2), "ts": ts}

    # duplicate full exit: the second SELL lands on a flat book
    book = ledger._compute_book([
        t("AMD", "BUY", 10, 100.0, "2026-07-01T14:00:00"),
        t("AMD", "SELL", 10, 90.0, "2026-07-02T14:00:00"),
        t("AMD", "SELL", 10, 90.0, "2026-07-02T14:00:05"),
    ])
    assert book["AMD"]["shares"] == 0.0 and book["AMD"]["cost"] == 0.0

    # cross-through over-sell (sell more than held) also clamps at zero
    book = ledger._compute_book([
        t("AMD", "BUY", 10, 100.0, "2026-07-01T14:00:00"),
        t("AMD", "SELL", 15, 90.0, "2026-07-02T14:00:00"),
    ])
    assert book["AMD"]["shares"] == 0.0 and book["AMD"]["cost"] == 0.0

    # options are SIGNED: a covered short leg opens negative, as designed
    call = occ.build("NVDA", date.today() + timedelta(days=45), "C", 200)
    book = ledger._compute_book([t(call, "SELL", 1, 5.0, "2026-07-01T14:00:00")])
    assert book[call]["shares"] == -1.0


def test_brain_state_journal_decision(seeded):
    from agent import brain
    from agent.store import get_store
    store = get_store()
    assert brain.get_state(store)["version"] == 0
    brain.set_state(store, name="trend-follow", thesis="ride winners",
                    rules={"hold_above": "ema_200"}, params={"k": 5}, bump=True)
    s = brain.get_state(store)
    assert s["version"] == 1 and s["name"] == "trend-follow"

    brain.set_state(store, name="trend-follow v2", thesis="ride winners, cut losers",
                    rules={"hold_above": "ema_200"}, params={"k": 8}, bump=True,
                    no_learned_basis="test fixture pivot")
    assert brain.get_state(store)["version"] == 2

    brain.add_journal(store, kind="pivot", title="raised K to 8",
                      body="breadth widened", version_from=1, version_to=2)
    brain.think(store, run_id="R1", phase="research", text="NVDA momentum strong")
    d = brain.save_decision(store, run_id="R1", regime="risk_on",
                            summary="add NVDA", target_weights={"NVDA": 0.5},
                            picks=[{"symbol": "NVDA", "action": "buy",
                                    "why_now": "breakout",
                                    "prediction": "NVDA +5% within 10 sessions",
                                    "horizon_days": 10,
                                    "kill": "closes below 120"}],
                            watchlist=[{"symbol": "AAPL", "note": "near trigger"}],
                            strategy_version=2)
    assert d["ok"]
    # idempotent: saving the same run again updates, not duplicates
    brain.save_decision(store, run_id="R1", regime="risk_on", summary="add NVDA (v2)",
                        strategy_version=2)
    from agent.models import DeskDecision, DeskThinking, DeskJournal
    sess = seeded.session_factory()()
    try:
        assert sess.query(DeskDecision).count() == 1
        assert sess.query(DeskThinking).count() == 1
        # two journal rows: the explicit pivot note above, plus the
        # audited "ungated-change" note the no_learned_basis bump writes
        # (the owner-approval gate's escape-hatch receipt).
        assert sess.query(DeskJournal).count() == 2
        assert sess.query(DeskJournal).filter_by(
            kind="ungated-change").count() == 1
    finally:
        sess.close()

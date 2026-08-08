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

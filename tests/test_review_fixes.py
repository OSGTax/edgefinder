"""Public-panel regressions that outlived the V3 ledger.

The rest of this file's original content — settle allocation, CSP
reservation, close-band, slippage, rebuild — tested `agent.ledger`, which
REBUILD-V4 deleted (Alpaca's paper engine owns fills/settlement now). The
survivors below are dashboard-side: the request token bucket and the
public movers/holding-stats panels' split guards.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from agent.grade import _et_date, _utcnow

# The desk reasons in ET; anchor fixtures to the code's own notion of today.
TODAY = date.fromisoformat(_et_date(_utcnow()))


def test_token_bucket_prune_actually_prunes():
    from dashboard.routers.desk import _TokenBucket
    b = _TokenBucket(capacity=2.0, refill_per_sec=1000.0)
    for i in range(2100):
        b.allow(f"k{i}")
    assert len(b._buckets) <= 64   # refilled-to-full buckets got pruned


# ── public panels: split guard + coverage floor ──────────────────────


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'panel.db'}")
    monkeypatch.setenv("EDGEFINDER_SCHEDULER_ENABLED", "false")
    from edgefinder.db.engine import Base, get_engine
    import edgefinder.db.models  # noqa: F401
    import agent.models  # noqa: F401
    import agent.data as agent_data
    import dashboard.dependencies as deps
    Base.metadata.create_all(get_engine())
    agent_data._session_factory = None
    deps._engine = deps._session_factory = None
    import dashboard.routers.desk as desk_router
    desk_router._options_allow = None
    desk_router._options_bucket.reset()
    desk_router._session_cache = (0.0, None)
    desk_router._session_refreshing = False
    desk_router._portfolio_cache = None
    desk_router._open_orders_cache = None
    desk_router._outcomes_live_cache = None
    from fastapi.testclient import TestClient
    from dashboard.app import app
    with TestClient(app) as c:
        yield c


def _bar(sess, symbol, day, close, volume=1e6):
    from edgefinder.db.models import DailyBar
    sess.add(DailyBar(symbol=symbol, date=day, open=close, high=close,
                      low=close, close=close, volume=volume, source="test"))


def test_movers_split_guard_and_coverage_floor(client, monkeypatch):
    import agent.data as agent_data
    monkeypatch.setattr(agent_data, "FULL_COVERAGE_MIN", 3)
    sess = agent_data.session_factory()()
    d_prev, d_cur, d_thin = (TODAY - timedelta(days=3), TODAY - timedelta(days=2),
                             TODAY - timedelta(days=1))
    try:
        for sym, prev_c, cur_c in (("AAA", 100.0, 110.0), ("BBB", 50.0, 45.0),
                                   ("SPL", 1200.0, 120.0)):  # 10:1 split
            _bar(sess, sym, d_prev, prev_c)
            _bar(sess, sym, d_cur, cur_c)
        _bar(sess, "AAA", d_thin, 111.0)   # a thin partial session (1 symbol)
        from edgefinder.db.models import TickerSplit
        sess.add(TickerSplit(symbol="SPL", execution_date=str(d_cur),
                             split_from=1, split_to=10))
        sess.commit()
    finally:
        sess.close()
    out = client.get("/api/desk/movers").json()
    # the thin session is not "latest"; the split name is excluded
    assert out["as_of"] == str(d_cur) and out["prior"] == str(d_prev)
    syms = {r["symbol"] for r in out["gainers"] + out["losers"]}
    assert "SPL" not in syms and {"AAA", "BBB"} <= syms
    assert out.get("splits_excluded") == ["SPL"]


def test_holding_stats_rebases_split(client, monkeypatch):
    import agent.data as agent_data
    import dashboard.routers.desk as desk_router

    # held names now come from the cached Alpaca positions read
    class _FakeTrade:
        def __init__(self, *a, **k):
            pass

        def state(self):
            return {"account": "agent", "paper": True, "cash": 88000.0,
                    "equity": 100000.0, "buying_power": 88000.0,
                    "starting_capital": 100000.0, "total_pnl": 0.0,
                    "total_return_pct": 0.0, "positions_value": 12000.0,
                    "positions": [{"symbol": "SPL", "asset_class": "us_equity",
                                   "qty": 100.0, "avg_entry_price": 120.0,
                                   "current_price": 122.0,
                                   "market_value": 12200.0,
                                   "unrealized_pl": 200.0, "weight": 0.12}]}

    monkeypatch.setattr("agent.trade.Trade", _FakeTrade)
    desk_router._portfolio_cache = None
    sess = agent_data.session_factory()()
    try:
        for i, px in ((4, 1180.0), (3, 1200.0)):      # pre-split closes
            _bar(sess, "SPL", TODAY - timedelta(days=i), px)
        for i, px in ((2, 121.0), (1, 122.0)):        # post-split closes
            _bar(sess, "SPL", TODAY - timedelta(days=i), px)
        from edgefinder.db.models import TickerSplit
        sess.add(TickerSplit(symbol="SPL",
                             execution_date=str(TODAY - timedelta(days=2)),
                             split_from=1, split_to=10))
        sess.commit()
    finally:
        sess.close()
    out = client.get("/api/desk/holding-stats").json()
    row = out["symbols"]["SPL"]
    # day change is post-split close vs post-split close — not a fake -90%
    assert row["day_change_pct"] == pytest.approx((122.0 - 121.0) / 121.0 * 100,
                                                  abs=0.01)
    # the range is on the current share basis: high is the post-split 122
    # (not the raw 1200), low is the REBASED pre-split 1180/10 (not 121)
    assert row["wk52_high"] == pytest.approx(122.0)
    assert row["wk52_low"] == pytest.approx(118.0)

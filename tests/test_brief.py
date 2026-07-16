"""Nightly research brief (desk_briefs): build, upsert-in-place, read, movers.

The brief is the efficiency seam: the data-refresh routine packs the
whole-market picture once per night; the hourly trading cycle reads ONE
payload instead of re-deriving regime/universe/movers/news every hour.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

TODAY = date.today()


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'brief.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


def seed_day(store, d: date, closes: dict[str, float]) -> None:
    store.insert("daily_bars", [
        {"symbol": s, "date": d, "open": c, "high": c, "low": c, "close": c,
         "volume": 1000.0, "source": "test"} for s, c in closes.items()
    ], returning=False)


def test_movers_skip_thin_partial_session(store):
    from agent.market import _movers

    d1, d2 = TODAY - timedelta(days=3), TODAY - timedelta(days=2)
    seed_day(store, d1, {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0})
    seed_day(store, d2, {"AAA": 110.0, "BBB": 45.0, "CCC": 20.0})
    # Today: a thin intraday top-up — must NOT be one side of the comparison.
    seed_day(store, TODAY, {"AAA": 999.0})

    m = _movers(store, min_coverage=3)
    assert m["as_of"] == str(d2) and m["prior"] == str(d1)
    assert m["gainers"][0]["symbol"] == "AAA"
    assert m["gainers"][0]["change_pct"] == 10.0
    assert m["losers"][0]["symbol"] == "BBB"
    assert m["losers"][0]["change_pct"] == -10.0


def test_movers_honest_note_when_coverage_missing(store):
    from agent.market import _movers

    seed_day(store, TODAY, {"AAA": 100.0})
    m = _movers(store, min_coverage=300)
    assert m["as_of"] is None and "ingest" in m["note"]


def test_build_brief_upserts_one_row_per_date(store):
    from agent.market import _et_today, build_brief, get_brief

    for i in range(1, 4):
        seed_day(store, TODAY - timedelta(days=i),
                 {"AAA": 100.0 + i, "BBB": 50.0, "SPY": 600.0})

    r1 = build_brief(top=5)
    assert r1["ok"] and r1["brief_date"] == str(_et_today())
    r2 = build_brief(top=5)  # same night re-run: rebuild in place
    assert r2["ok"]
    rows = store.select("desk_briefs", filters={"account": "agent"})
    assert len(rows) == 1

    b = get_brief()
    assert b["exists"] is True and b["stale"] is False
    payload = b["payload"]
    assert payload["as_of"] == str(_et_today())
    assert "regime" in payload and "coverage" in payload
    assert "movers" in payload and "universe_top" in payload
    # Coverage verdict rides along so the brief self-documents its freshness.
    assert payload["coverage"]["status"] in ("green", "amber", "red")


def test_compute_screens_surfaces_midtier_leaders():
    from agent.market import compute_screens

    # 45 "megacaps": huge dollar volume, flat price action.
    rows = []
    days = [str(TODAY - timedelta(days=n)) for n in range(99, -1, -1)]
    for i in range(45):
        sym = f"MEGA{i:02d}"
        for d in days:
            rows.append({"symbol": sym, "date": d, "close": 100.0,
                         "volume": 1e9 - i * 1e6})
    # Mid-tier leader: modest volume, +50% over the window, at its high.
    for j, d in enumerate(days):
        rows.append({"symbol": "RISER", "date": d,
                     "close": 20.0 + j * 0.1, "volume": 3e6})
    # Mid-tier laggard and a sub-$5 name: must not appear.
    for d in days:
        rows.append({"symbol": "FLAT", "date": d, "close": 50.0, "volume": 2e6})
        rows.append({"symbol": "PENNY", "date": d, "close": 3.0, "volume": 4e6})

    s = compute_screens(rows, top_exclude=40)
    beyond = [e["symbol"] for e in s["beyond_megacaps"]]
    assert "RISER" in beyond
    assert all(not b.startswith("MEGA") for b in beyond)  # funnel bias excluded
    assert "PENNY" not in beyond and "FLAT" not in beyond
    assert "RISER" in [e["symbol"] for e in s["new_highs"]]  # ends at its high
    riser = next(e for e in s["beyond_megacaps"] if e["symbol"] == "RISER")
    assert riser["rank"] > 40 and riser["ret_3m_pct"] > 20


def test_compute_screens_excludes_split_symbols():
    from agent.market import compute_screens

    rows = []
    days = [str(TODAY - timedelta(days=n)) for n in range(99, -1, -1)]
    for i in range(45):
        sym = f"MEGA{i:02d}"
        for d in days:
            rows.append({"symbol": sym, "date": d, "close": 100.0,
                         "volume": 1e9 - i * 1e6})
    # REVSPL did a 1:10 reverse split mid-window: raw closes jump 8 → 80,
    # which screens as a fabricated +900% "leader" without the guard.
    for j, d in enumerate(days):
        rows.append({"symbol": "REVSPL", "date": d,
                     "close": 8.0 if j < 60 else 80.0, "volume": 3e6})

    # unguarded, the fake move tops the exact screen the trader shops from
    unguarded = compute_screens(rows, top_exclude=40)
    assert "REVSPL" in [e["symbol"] for e in unguarded["beyond_megacaps"]]

    s = compute_screens(rows, top_exclude=40, split_syms={"REVSPL"})
    assert "REVSPL" not in [e["symbol"] for e in s["beyond_megacaps"]]
    assert "REVSPL" not in [e["symbol"] for e in s["new_highs"]]
    assert s["splits_excluded"] == ["REVSPL"]


def test_screens_guard_queries_splits_inside_window(store, monkeypatch):
    import agent.market as market

    seed_day(store, TODAY - timedelta(days=1), {"AAA": 100.0})
    for sym, d in (("IN_WINDOW", TODAY - timedelta(days=30)),
                   ("ANCIENT", TODAY - timedelta(days=400)),   # before window
                   ("FUTURE", TODAY + timedelta(days=10))):    # not yet real
        store.insert("ticker_splits", {"symbol": sym,
                                       "execution_date": str(d),
                                       "split_from": 1, "split_to": 10},
                     returning=False)

    captured = {}

    def fake_compute(rows, *, split_syms=None, **kw):
        captured["split_syms"] = split_syms
        return {"pool_size": 0}

    monkeypatch.setattr(market, "compute_screens", fake_compute)
    market._screens(store)
    # only splits that already executed inside the lookback window count —
    # ancient ones don't touch these closes, future ones haven't yet
    assert captured["split_syms"] == {"IN_WINDOW"}


def test_brief_payload_carries_screens(store):
    from agent.market import build_brief

    for i in range(1, 3):
        seed_day(store, TODAY - timedelta(days=i), {"AAA": 100.0, "SPY": 600.0})
    build_brief(top=3)
    rows = store.select("desk_briefs", filters={"account": "agent"})
    assert "screens" in rows[0]["payload"]


def test_movers_exclude_split_symbols(store):
    from agent.market import _movers

    d1, d2 = TODAY - timedelta(days=3), TODAY - timedelta(days=2)
    seed_day(store, d1, {"AAA": 100.0, "SPL": 1000.0, "CCC": 20.0})
    seed_day(store, d2, {"AAA": 101.0, "SPL": 100.0, "CCC": 20.0})
    # SPL did a 10:1 split between the sessions — raw closes fabricate -90%.
    store.insert("ticker_splits", {
        "symbol": "SPL", "execution_date": str(d2),
        "split_from": 1, "split_to": 10}, returning=False)

    m = _movers(store, min_coverage=3)
    assert m["splits_excluded"] == ["SPL"]
    assert all(r["symbol"] != "SPL" for r in m["losers"])


def test_build_brief_survives_section_failure(store, monkeypatch):
    import agent.data as agent_data
    from agent.market import build_brief

    for i in range(1, 3):
        seed_day(store, TODAY - timedelta(days=i), {"AAA": 100.0, "SPY": 600.0})

    def boom(**kw):
        raise RuntimeError("regime backend down")

    monkeypatch.setattr(agent_data, "regime", boom)
    r = build_brief(top=3)
    assert r["ok"] is True  # one dead section must not kill the night's brief
    assert any(e.startswith("regime:") for e in r["errors"])
    rows = store.select("desk_briefs", filters={"account": "agent"})
    assert rows[0]["payload"]["regime"] == {}
    assert rows[0]["payload"]["errors"]


def test_brief_endpoint(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps
    from agent.market import build_brief, _et_today

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None
    seed_day(store, TODAY - timedelta(days=1), {"AAA": 100.0, "SPY": 600.0})
    build_brief(top=3)

    from dashboard.app import app

    with TestClient(app) as c:
        r = c.get("/api/desk/brief").json()
        assert r["exists"] is True and r["brief_date"] == str(_et_today())
        assert "universe_top" in r["payload"]


def test_get_brief_staleness_and_absence(store):
    from agent.market import get_brief

    assert get_brief()["exists"] is False

    old = datetime.now() - timedelta(hours=72)
    store.insert("desk_briefs", {
        "account": "agent", "brief_date": TODAY - timedelta(days=3),
        "built_at": old, "payload": {"as_of": "x"}}, returning=False)
    b = get_brief()
    assert b["exists"] is True and b["stale"] is True

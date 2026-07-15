"""Market-data coverage checks: verdict logic, store gatherer, preflight, API.

The failure mode under test: the nightly whole-market ingest dies while the
hourly top-up keeps a handful of held names fresh, so bar AGE looks fine while
the universe rots (the 2026-07-08 production outage). The verdict counts thin
sessions since the last full-coverage date instead.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from agent.data import coverage_verdict

# Relative to today: the verdict has a calendar-age anchor, so fixed dates
# would rot as the wall clock advances. day(3) == today.
D0 = date.today() - timedelta(days=3)


def day(n: int) -> str:
    return str(D0 + timedelta(days=n))


# ── coverage_verdict (pure logic) ──


def test_verdict_empty_is_red():
    v = coverage_verdict([])
    assert v["status"] == "red" and v["research_ok"] is False
    assert v["latest_date"] is None and v["last_full_date"] is None


def test_verdict_green_after_nightly():
    v = coverage_verdict([(day(0), 2000)])
    assert v["status"] == "green" and v["research_ok"] is True
    assert v["last_full_date"] == day(0) and v["thin_sessions"] == 0


def test_verdict_green_with_intraday_partial():
    # A healthy trading afternoon: last night full, today only the top-up so far.
    v = coverage_verdict([(day(0), 2000), (day(1), 25)])
    assert v["status"] == "green" and v["thin_sessions"] == 1
    assert v["latest_date"] == day(1) and v["latest_count"] == 25


def test_verdict_amber_one_missed_nightly():
    # Ingest missed one night: yesterday thin + today's partial = 2 thin.
    v = coverage_verdict([(day(0), 2000), (day(1), 25), (day(2), 19)])
    assert v["status"] == "amber"
    assert v["research_ok"] is True  # display degrades before the trader does
    assert v["thin_sessions"] == 2


def test_verdict_red_dead_ingest():
    v = coverage_verdict([(day(0), 2000), (day(1), 25), (day(2), 19), (day(3), 20)])
    assert v["status"] == "red" and v["research_ok"] is False
    assert v["thin_sessions"] == 3 and v["last_full_date"] == day(0)


def test_verdict_red_when_no_full_date_in_window():
    v = coverage_verdict([(day(0), 40), (day(1), 40)])
    assert v["status"] == "red" and v["last_full_date"] is None
    assert v["thin_sessions"] == 2


def test_verdict_threshold_is_inclusive_and_order_free():
    # Counts capped AT full_min by the gatherer still count as full, and
    # input order must not matter.
    v = coverage_verdict([(day(1), 3), (day(0), 5)], full_min=5)
    assert v["status"] == "green" and v["last_full_date"] == day(0)


def test_verdict_duplicate_dates_keep_max_count():
    v = coverage_verdict([(day(0), 3), (day(0), 8)], full_min=5)
    assert v["last_full_date"] == day(0) and v["status"] == "green"


# ── the calendar-age anchor: silence is not health ──


def test_verdict_total_silence_cannot_stay_green():
    # Every writer dead: no new dates at all. Shape says green (0 thin
    # sessions after a full ingest) — the age anchor must override.
    anchor = date(2026, 6, 1)
    v = coverage_verdict([(str(anchor), 2000)],
                         today=anchor + timedelta(days=5))
    assert v["status"] == "amber"  # past the long-weekend grace
    v = coverage_verdict([(str(anchor), 2000)],
                         today=anchor + timedelta(days=7))
    assert v["status"] == "red" and v["research_ok"] is False
    assert v["latest_age_days"] == 7


def test_verdict_age_anchor_respects_holiday_weekend():
    # Thursday full ingest, Friday holiday, checked Monday (age 4): green.
    anchor = date(2026, 6, 4)
    v = coverage_verdict([(str(anchor), 2000)],
                         today=anchor + timedelta(days=4))
    assert v["status"] == "green"


def test_verdict_age_anchor_never_upgrades():
    # A fresh-but-thin picture stays red; the anchor only caps, never lifts.
    anchor = date(2026, 6, 1)
    v = coverage_verdict([(str(anchor + timedelta(days=n)), 10)
                          for n in range(3)],
                         today=anchor + timedelta(days=2))
    assert v["status"] == "red"


# ── universe_coverage (store gatherer) + preflight ──


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'cov.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401 — desk_* tables (preflight reads them)
    import edgefinder.db.models  # noqa: F401 — daily_bars

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


def seed_bars(store, d: date, symbols: list[str]) -> None:
    store.insert("daily_bars", [
        {"symbol": s, "date": d, "open": 1.0, "high": 1.0, "low": 1.0,
         "close": 1.0, "volume": 100.0, "source": "test"}
        for s in symbols
    ], returning=False)


def test_universe_coverage_red_then_green(store):
    from agent.data import universe_coverage

    full = ["SPY", "AAA", "BBB", "CCC", "DDD"]  # 5 = full_min in this test
    thin = ["SPY", "AAA"]
    seed_bars(store, D0, full)
    seed_bars(store, D0 + timedelta(days=1), thin)
    seed_bars(store, D0 + timedelta(days=2), thin)
    seed_bars(store, D0 + timedelta(days=3), thin)

    v = universe_coverage(full_min=5)
    assert v["status"] == "red" and v["research_ok"] is False
    assert v["last_full_date"] == day(0) and v["thin_sessions"] == 3

    # The nightly comes back: yesterday full again → green.
    seed_bars(store, D0 + timedelta(days=3), ["BBB", "CCC", "DDD"])
    v = universe_coverage(full_min=5)
    assert v["status"] == "green" and v["last_full_date"] == day(3)


def test_universe_coverage_empty_table(store):
    from agent.data import universe_coverage

    v = universe_coverage(full_min=5)
    assert v["status"] == "red" and v["research_ok"] is False


def test_pg_store_select_honors_columns(store):
    # Regression: PgStore.select used to ignore `columns` and fetch full-width
    # rows — load-bearing once preflight polls coverage hourly.
    seed_bars(store, D0, ["SPY", "AAA"])
    rows = store.select("daily_bars", columns="date,symbol", limit=1)
    assert set(rows[0].keys()) == {"date", "symbol"}


def test_preflight_exposes_research_ok_and_siblings(store, monkeypatch):
    import agent.data as agent_data
    from agent import preflight

    # Pin the threshold so 5 seeded symbols count as a full ingest.
    monkeypatch.setattr(agent_data, "FULL_COVERAGE_MIN", 5)
    seed_bars(store, D0, ["SPY", "AAA", "BBB", "CCC", "DDD"])

    out = preflight.run()
    assert out["checks"]["universe_coverage"]["ok"] is True
    assert out["research_ok"] is True

    # Kill the nightly for three sessions → the flag flips, preflight stays ok
    # overall (degrade, don't block) and siblings still reports its warnings.
    for n in (1, 2, 3):
        seed_bars(store, D0 + timedelta(days=n), ["SPY", "AAA"])
    out = preflight.run()
    assert out["research_ok"] is False
    assert out["checks"]["universe_coverage"]["detail"]["status"] == "red"
    assert out["ok"] is True  # coverage is advisory, not a hard stop
    sib = out["checks"]["siblings"]["detail"]
    assert "warnings" in sib  # empty desk → both routines flagged overdue
    assert len(sib["warnings"]) == 2


# ── /api/desk/data-health ──


def test_data_health_endpoint(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps

    monkeypatch.setattr(agent_data, "FULL_COVERAGE_MIN", 5)
    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    seed_bars(store, D0, ["SPY", "AAA", "BBB", "CCC", "DDD"])
    seed_bars(store, D0 + timedelta(days=1), ["SPY", "AAA"])

    from dashboard.app import app

    with TestClient(app) as c:
        r = c.get("/api/desk/data-health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "green" and body["research_ok"] is True
        assert body["last_full_date"] == day(0)
        assert body["latest_date"] == day(1) and body["thin_sessions"] == 1

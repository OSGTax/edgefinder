"""The lab's news study (V4.2): headline classification, timing buckets,
tradable-close anchoring, excess-drift math, dedupe, and persistence."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest

from agent.lab import classify_headline, news_study, _timing_bucket


def test_classify_headline_classes():
    assert classify_headline("Acme beats earnings estimates") == "earnings"
    assert classify_headline("Analyst upgrades Acme to Buy") == "analyst_up"
    assert classify_headline("Acme price target cut at MegaBank") == "analyst_down"
    assert classify_headline("BigCo in deal to buy Acme") == "mna"
    assert classify_headline("Acme announces secondary offering") == "dilution_debt"
    assert classify_headline("Something ineffable happened") == "other"
    assert classify_headline(None) == "other"


def test_timing_buckets_et():
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    mk = lambda h, m: datetime(2026, 8, 10, h, m, tzinfo=et)  # noqa: E731
    assert _timing_bucket(mk(8, 0)) == "premarket"
    assert _timing_bucket(mk(10, 30)) == "intraday"
    assert _timing_bucket(mk(17, 5)) == "afterhours"
    assert _timing_bucket(mk(1, 0)) == "overnight"


# ── end-to-end on SQLite ─────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'news.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    from edgefinder.db.engine import Base, get_engine
    import edgefinder.db.models  # noqa: F401
    import agent.models  # noqa: F401
    import agent.data as agent_data
    import agent.store as agent_store
    Base.metadata.create_all(get_engine())
    agent_data._session_factory = None
    agent_store._store = None
    return agent_store.get_store()


def _sessions(n: int) -> list[date]:
    """The last n weekdays ending yesterday-ish (deterministic, no weekend)."""
    out: list[date] = []
    d = date.today() - timedelta(days=1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def _seed_bars(days: list[date], sym: str, closes: list[float]):
    import agent.data as agent_data
    from edgefinder.db.models import DailyBar

    sess = agent_data.session_factory()()
    try:
        for d, c in zip(days, closes):
            sess.add(DailyBar(symbol=sym, date=d, open=c, high=c, low=c,
                              close=c, volume=1e6, source="test"))
        sess.commit()
    finally:
        sess.close()


def _news(store, sym: str, title: str, pub_utc: str):
    store.insert("ticker_news", {"symbol": sym, "title": title,
                                 "published_utc": pub_utc,
                                 "article_url": "https://x/" + title[:8]},
                 returning=False)


def test_news_study_end_to_end(store, monkeypatch):
    from zoneinfo import ZoneInfo

    et = ZoneInfo("America/New_York")
    days = _sessions(10)
    # ACME rises 1%/session; SPY flat → excess == ACME's own move
    acme = [100.0 * (1.01 ** i) for i in range(10)]
    _seed_bars(days, "ACME", acme)
    _seed_bars(days, "SPY", [500.0] * 10)

    d5 = days[5]
    pre = datetime.combine(d5, time(8, 0), tzinfo=et)      # premarket → entry d5
    aft = datetime.combine(d5, time(17, 30), tzinfo=et)    # afterhours → entry d6+
    _news(store, "ACME", "ACME beats earnings estimates",
          pre.isoformat())
    _news(store, "ACME", "ACME tops estimates again (dupe class+day)",
          (pre + timedelta(minutes=30)).isoformat())       # deduped
    _news(store, "ACME", "Analyst upgrades ACME to Overweight",
          aft.isoformat())
    _news(store, "ACME", "junk row with bad timestamp", None)

    out = news_study(days=30, top_symbols=10, store=store)
    assert out["events"] == 2

    earn = out["by_class"]["earnings"]
    assert earn["n"] == 1
    # entry at d5 close, t+1 = d6: ACME +1.0%, SPY flat → excess ≈ +1.0
    assert earn["x1"] == pytest.approx(1.0, abs=0.02)

    up = out["by_class"]["analyst_up"]
    assert up["n"] == 1
    assert up["by_timing"]["afterhours"]["n"] == 1

    # persisted as the news-study desk_backtests row
    rows = store.select("desk_backtests",
                        filters={"label": "news-study"}, limit=5)
    assert len(rows) == 1
    assert rows[0]["result"]["events"] == 2
    assert "drift" in out["honesty"].lower() or "DRIFT" in out["honesty"]


def test_news_study_no_spy_degrades(store):
    days = _sessions(5)
    _seed_bars(days, "ACME", [100, 101, 102, 103, 104])
    out = news_study(days=30, top_symbols=5, store=store)
    assert out.get("error") and out["events"] == 0

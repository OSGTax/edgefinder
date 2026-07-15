"""The symbol page's Research tab — /api/symbols/{sym}/fundamentals + search.

Seeds fundamentals_pit (SEC EDGAR PIT rows) and daily_bars on SQLite and
checks the endpoint's honesty conventions: latest filing wins, price ratios
are computed against a NAMED stored close, uncovered symbols get a plain
explanation instead of a 404, and the chart page ships the Research tab.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'research.db'}")

    from edgefinder.db.engine import Base, get_engine
    from edgefinder.db.models import DailyBar, FundamentalsPit
    import agent.models  # noqa: F401 — desk_* tables share the metadata
    import agent.data as agent_data
    import dashboard.dependencies as deps

    engine = get_engine()
    Base.metadata.create_all(engine)
    agent_data._session_factory = None
    deps._engine = deps._session_factory = None

    sess = agent_data.session_factory()()
    try:
        # Two filings: a 10-Q, then a 10-K that restates upward. The
        # endpoint must serve the LATEST filing's values.
        sess.add(FundamentalsPit(
            symbol="ACME", cik=1, filed=date(2025, 10, 30),
            period_end=date(2025, 9, 30), form="10-Q", source="edgar",
            data={"earnings_per_share": 4.0, "return_on_equity": 0.20,
                  "debt_to_equity": 0.5, "current_ratio": 2.0,
                  "revenue_growth": 0.10,
                  "_revenue_ttm": 40e9, "_net_income_ttm": 8e9,
                  "_fcf_ttm": 6e9, "_ebitda_ttm": 12e9,
                  "_shares": 2e9, "_book_equity": 40e9,
                  "_total_debt": 20e9, "_cash": 10e9}))
        sess.add(FundamentalsPit(
            symbol="ACME", cik=1, filed=date(2026, 2, 25),
            period_end=date(2025, 12, 31), form="10-K", source="edgar",
            data={"earnings_per_share": 5.0, "return_on_equity": 0.25,
                  "debt_to_equity": 0.5, "current_ratio": 2.0,
                  "revenue_growth": 0.12,
                  "_revenue_ttm": 50e9, "_net_income_ttm": 10e9,
                  "_fcf_ttm": 8e9, "_ebitda_ttm": 15e9,
                  "_shares": 2e9, "_book_equity": 50e9,
                  "_total_debt": 20e9, "_cash": 10e9}))
        for sym, px in (("ACME", 100.0), ("ACORN", 20.0), ("SPY", 500.0)):
            sess.add(DailyBar(symbol=sym, date=date(2026, 7, 13),
                              open=px, high=px, low=px, close=px,
                              volume=1e6, source="alpaca_daily",
                              created_at=datetime.now(timezone.utc)))
        sess.commit()
    finally:
        sess.close()

    from dashboard.app import app
    with TestClient(app) as c:
        yield c


def test_fundamentals_latest_filing_and_named_price(client):
    body = client.get("/api/symbols/acme/fundamentals").json()
    assert body["covered"] is True
    assert body["filings"] == 2
    assert body["latest_form"] == "10-K"
    assert body["latest_filed"] == "2026-02-25"

    snap = body["snapshot"]
    # latest filing's price-free values, not the earlier 10-Q's
    assert snap["earnings_per_share"] == 5.0
    assert snap["return_on_equity"] == 0.25

    # price ratios computed against the named stored close ($100)
    assert body["price_used"] == 100.0
    assert body["price_as_of"] == "2026-07-13"
    assert snap["market_cap"] == pytest.approx(200e9)          # 2e9 sh × $100
    assert snap["price_to_earnings"] == pytest.approx(20.0)    # 100 / 5 EPS
    assert snap["price_to_sales"] == pytest.approx(4.0)        # 200e9 / 50e9
    assert snap["fcf_yield"] == pytest.approx(8e9 / 200e9)

    # per-filing series for the trend sparklines, oldest → newest
    revs = [r["_revenue_ttm"] for r in body["series"]]
    assert revs == [40e9, 50e9]


def test_fundamentals_uncovered_symbol_explains_itself(client):
    body = client.get("/api/symbols/SPY/fundamentals").json()
    assert body["covered"] is False
    assert "ETF" in body["note"] or "filings" in body["note"]


def test_symbol_search_prefix(client):
    body = client.get("/api/symbols/search?q=ac").json()
    syms = [r["symbol"] for r in body["results"]]
    assert syms == ["ACME", "ACORN"]
    # wildcard characters are stripped, not passed to LIKE
    assert client.get("/api/symbols/search?q=%25").json()["results"] == []


def test_symbol_page_ships_research_tab(client):
    html = client.get("/symbol/ACME").text
    assert 'data-tab="profile"' in html and ">Research<" in html
    assert ">AI trades<" in html


def test_symbol_page_tab_deep_link_renders(client):
    """?tab=options must serve the page (the JS reads it client-side) —
    the desk's retired options card deep-links here."""
    r = client.get("/symbol/ACME?tab=options")
    assert r.status_code == 200
    assert 'data-tab="options"' in r.text

"""Smoke the trading-desk page + /api/desk/* endpoints on a seeded SQLite DB."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'desk.db'}")
    monkeypatch.setenv("EDGEFINDER_SCHEDULER_ENABLED", "false")

    from edgefinder.db.engine import Base, get_engine
    import edgefinder.db.models  # noqa: F401
    import agent.models  # noqa: F401
    from agent.models import (
        ACCOUNT, DeskDecision, DeskEquity, DeskPosition, DeskThinking, DeskTrade,
    )
    import agent.data as agent_data
    import dashboard.dependencies as deps

    engine = get_engine()
    Base.metadata.create_all(engine)
    agent_data._session_factory = None
    deps._engine = deps._session_factory = None

    # Reset the desk router's in-process caches (options allowlist, options
    # rate-limit bucket, market-session cache + single-flight flag, portfolio
    # TTL cache) — module state must not leak a previous test's DB view into
    # this one.
    import dashboard.routers.desk as desk_router
    desk_router._options_allow = None
    desk_router._options_bucket.reset()
    desk_router._session_cache = (0.0, None)
    desk_router._session_refreshing = False
    desk_router._portfolio_cache = None

    now = datetime.now(timezone.utc)
    sess = agent_data.session_factory()()
    try:
        sess.add(DeskTrade(account=ACCOUNT, run_id="R1", symbol="NVDA", side="BUY",
                           shares=100, price=120.0, dollars=12000.0, ts=now))
        sess.add(DeskPosition(account=ACCOUNT, symbol="NVDA", shares=100,
                              avg_price=120.0, last_price=130.0, opened_at=now, marked_at=now))
        sess.add(DeskEquity(account=ACCOUNT, ts=now, cash=88000.0, positions_value=13000.0,
                            equity=101000.0, return_pct=1.0))
        sess.add(DeskThinking(account=ACCOUNT, run_id="R1", phase="research",
                              text="NVDA momentum strong", ts=now))
        sess.add(DeskDecision(account=ACCOUNT, run_id="R1", ts=now, regime="risk_on",
                              summary="added NVDA", target_weights={"NVDA": 0.13},
                              picks=[{"symbol": "NVDA", "action": "buy", "why_now": "breakout"}],
                              watchlist=[{"symbol": "AAPL", "note": "near trigger"}],
                              strategy_version=1))
        sess.commit()
    finally:
        sess.close()

    from dashboard.app import app
    with TestClient(app) as c:
        yield c


def test_desk_page_renders(client):
    r = client.get("/desk")
    assert r.status_code == 200
    assert "Trading Desk" in r.text
    assert "/static/js/pages/desk.js" in r.text


def test_desk_page_information_architecture(client):
    """v9.5 IA: two zones only (reasoning, learning) — the Markets zone is
    gone (live prices fold into the hero's index chips, movers/options/
    dividends live on the chart page). Lab and Notebook carry view toggles
    that absorbed the old backtests/journal cards."""
    import re

    html = client.get("/desk").text

    reasoning = html.index('id="zone-reasoning"')
    learning = html.index('id="zone-history"')
    assert reasoning < learning
    assert 'id="zone-markets"' not in html

    assert 'id="desk-watch"' in html            # the attention card
    assert 'id="desk-lab"' in html              # the lab leaderboard
    assert 'id="desk-hero-indices"' in html     # live SPY/QQQ/IWM chips
    assert 'id="desk-lab-seg"' in html          # board / recent-tests views
    assert 'id="desk-wiki-seg"' in html         # lessons / diary views
    assert 'data-zone="desk-hero">Overview' in html
    assert 'data-zone="zone-markets"' not in html

    # Retired standalone cards must be fully gone, not just hidden.
    for key in ("tape", "movers", "options", "dividends", "backtests",
                "journal"):
        assert f'data-collapse-key="{key}"' not in html, f"{key} card lingers"

    def card_tag(key):
        m = re.search(r'<div class="c-card desk-card[^"]*" '
                      r'data-collapse-key="%s"[^>]*>' % key, html)
        assert m, f"card {key} missing"
        return m.group(0)

    # Receipts ship collapsed; the reasoning/learning core is open.
    assert 'data-collapsed="1"' in card_tag("fills")
    for key in ("watch", "decision", "thinking", "lab", "wiki"):
        assert 'data-collapsed="1"' not in card_tag(key), f"{key} should be open"


def test_portfolio_endpoint(client):
    r = client.get("/api/desk/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["positions"][0]["symbol"] == "NVDA"
    # cash = 100k start - 12k buy; equity = cash + marked positions value
    assert abs(body["cash"] - 88000.0) < 0.01
    assert body["equity"] == pytest.approx(88000.0 + 100 * 130.0, abs=0.01)


def test_decision_and_thinking(client):
    d = client.get("/api/desk/decision/latest").json()
    assert d["exists"] and d["picks"][0]["symbol"] == "NVDA"
    t = client.get("/api/desk/thinking").json()
    assert t["run_id"] == "R1" and t["lines"]
    e = client.get("/api/desk/equity").json()
    assert e and e[-1]["equity"] == 101000.0


def test_whatsnew_empty_then_announced(client):
    # nothing shipped yet → empty feed, no spotlight badge
    empty = client.get("/api/desk/whatsnew").json()
    assert empty["entries"] == [] and empty["new_count"] == 0 and empty["latest"] is None

    # the routine announces a shipped improvement via the agent tool
    from agent.announce import announce, recent
    new_id = announce("Drawdown band on the equity curve",
                      "The equity chart now shades peak-to-trough drawdowns so "
                      "you can see how deep the book's dips ran.",
                      kind="feature", version="6.1.0")
    assert isinstance(new_id, int)
    assert recent()[0]["title"].startswith("Drawdown band")

    body = client.get("/api/desk/whatsnew").json()
    assert body["new_count"] == 1
    assert body["latest"]["title"].startswith("Drawdown band")
    assert body["latest"]["kind"] == "feature" and body["latest"]["version"] == "6.1.0"
    assert "shades peak-to-trough" in body["entries"][0]["detail"]


def test_announce_validates_kind(client):
    from agent.announce import announce
    with pytest.raises(ValueError):
        announce("bad", kind="totally-not-valid")
    with pytest.raises(ValueError):
        announce("   ")  # blank title rejected


def test_broker_health_no_keys(client, monkeypatch):
    for v in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"):
        monkeypatch.delenv(v, raising=False)
    from agent import broker as _b
    monkeypatch.setattr(_b.settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(_b.settings, "alpaca_api_secret", "", raising=False)
    body = client.get("/api/desk/broker-health").json()
    assert body["keys_present"] is False and "keys" in body["error"]


def test_desk_movers(client):
    """Movers endpoint ranks gainers/losers/most-active from daily_bars."""
    from datetime import date, datetime, timezone

    import agent.data as agent_data
    from edgefinder.db.models import DailyBar

    now = datetime.now(timezone.utc)
    d0, d1 = date(2026, 7, 6), date(2026, 7, 7)
    seed = [
        ("AAA", d0, 100.0, 1_000), ("AAA", d1, 120.0, 2_000),   # +20% gainer
        ("BBB", d0, 100.0, 5_000), ("BBB", d1, 80.0, 9_000),    # -20% loser, biggest $vol
        ("CCC", d0, 50.0, 100), ("CCC", d1, 50.0, 100),         # flat
        ("PENNY", d0, 0.9, 1), ("PENNY", d1, 0.5, 1),           # sub-$1 → filtered
    ]
    sess = agent_data.session_factory()()
    try:
        for sym, dd, close, vol in seed:
            sess.add(DailyBar(symbol=sym, date=dd, open=close, high=close, low=close,
                              close=close, volume=float(vol), source="test", created_at=now))
        sess.commit()
    finally:
        sess.close()

    d = client.get("/api/desk/movers?top=3").json()
    assert d["as_of"] == "2026-07-07" and d["prior"] == "2026-07-06"
    gain = [x["symbol"] for x in d["gainers"]]
    lose = [x["symbol"] for x in d["losers"]]
    assert gain[0] == "AAA"                      # +20% is the top gainer
    assert lose[0] == "BBB"                       # -20% is the top loser
    assert "PENNY" not in gain and "PENNY" not in lose   # sub-$1 filtered out
    assert d["most_active"][0]["symbol"] == "BBB"        # 80 * 9000 = biggest $ volume
    assert next(x for x in d["gainers"] if x["symbol"] == "AAA")["change_pct"] == 20.0


def test_desk_holding_stats(client):
    """Holding-stats returns day change, 52-week range, and a spark series."""
    from datetime import date, datetime, timedelta, timezone

    import agent.data as agent_data
    from edgefinder.db.models import DailyBar

    now = datetime.now(timezone.utc)
    sess = agent_data.session_factory()()
    try:
        # NVDA is the seeded holding; give it 10 rising sessions
        base = date(2026, 6, 24)
        for i in range(10):
            px = 100.0 + i  # 100 → 109, last-session change 108→109
            sess.add(DailyBar(symbol="NVDA", date=base + timedelta(days=i),
                              open=px, high=px, low=px, close=px, volume=1000.0,
                              source="test", created_at=now))
        sess.commit()
    finally:
        sess.close()

    d = client.get("/api/desk/holding-stats?spark_days=30").json()
    assert "NVDA" in d["symbols"]
    s = d["symbols"]["NVDA"]
    assert s["wk52_high"] == 109.0 and s["wk52_low"] == 100.0
    assert s["day_change_pct"] == round((109 - 108) / 108 * 100, 2)
    assert s["spark"][0] == 100.0 and s["spark"][-1] == 109.0


def test_desk_dividends(client):
    """Dividend calendar returns last/next ex-dates for dividend-paying holdings."""
    from datetime import date

    import agent.data as agent_data
    from edgefinder.db.models import DividendRecord

    sess = agent_data.session_factory()()
    try:  # NVDA is the seeded holding; 2099 is unambiguously "upcoming"
        for ex, amt in [(date(2026, 3, 5), 0.10), (date(2026, 6, 5), 0.10),
                        (date(2099, 9, 5), 0.12)]:
            sess.add(DividendRecord(symbol="NVDA", ex_date=ex, cash_amount=amt))
        sess.commit()
    finally:
        sess.close()

    d = client.get("/api/desk/dividends").json()
    nvda = next(x for x in d["holdings"] if x["symbol"] == "NVDA")
    assert nvda["has_dividend"] is True
    assert nvda["next_ex_date"] == "2099-09-05"          # the only future ex-date
    assert nvda["last_ex_date"] == "2026-06-05"          # most recent past
    assert nvda["ttm_amount"] == round(0.10 + 0.10 + 0.12, 4)


def test_desk_trades_include_fill_quote(client):
    """The trades endpoint surfaces the stamped live bid/ask receipt."""
    from datetime import datetime, timezone

    import agent.data as agent_data
    from agent.models import ACCOUNT, DeskTrade

    sess = agent_data.session_factory()()
    try:
        sess.add(DeskTrade(account=ACCOUNT, run_id="R2", symbol="LLY", side="BUY",
                           shares=2.0, price=1226.46, dollars=2452.92,
                           ts=datetime.now(timezone.utc),
                           fill_quote={"bid": 1224.96, "ask": 1226.34, "mid": 1225.65}))
        sess.commit()
    finally:
        sess.close()

    rows = client.get("/api/desk/trades?limit=20").json()
    lly = next(r for r in rows if r["symbol"] == "LLY")
    assert lly["fill_quote"]["bid"] == 1224.96 and lly["fill_quote"]["ask"] == 1226.34


def test_wiki_endpoint_empty_and_seeded(client):
    body = client.get("/api/desk/wiki").json()
    assert body["pages"] == []          # empty case, no 500

    from agent.brain import set_wiki
    set_wiki(slug="lessons", body="Momentum works in risk-on.\n\n- cite numbers",
             title="Lessons", reason="seed")
    set_wiki(slug="playbook", body="Trend first.", reason="seed")
    body = client.get("/api/desk/wiki").json()
    assert [p["slug"] for p in body["pages"]] == ["playbook", "lessons"]  # canonical order
    assert body["pages"][1]["revision"] == 1
    assert "Momentum works" in body["pages"][1]["body"]
    assert body["pages"][0]["updated_at"]  # ISO timestamp present


def test_wiki_order_covers_all_six_slugs(client):
    """Phase E: the router's order map matches agent.brain.WIKI_SLUGS —
    setups after playbook, postmortems after mistakes."""
    from agent.brain import WIKI_SLUGS, set_wiki

    for slug in reversed(WIKI_SLUGS):  # insert out of order on purpose
        set_wiki(slug=slug, body=f"{slug} body.", reason="seed")
    body = client.get("/api/desk/wiki").json()
    assert [p["slug"] for p in body["pages"]] == list(WIKI_SLUGS)


# ── Phase E: one book — /portfolio is ledger.state(), not a re-derivation ──


def test_portfolio_matches_ledger_state(client):
    from agent import ledger
    from agent.store import get_store

    st = ledger.state(get_store())
    pf = client.get("/api/desk/portfolio").json()
    assert pf["cash"] == st["cash"]
    assert pf["equity"] == st["equity"]
    assert pf["positions_value"] == st["positions_value"]
    assert pf["total_pnl"] == st["total_pnl"]
    assert pf["total_return_pct"] == round(st["total_return_pct"], 2)
    by_sym = {p["symbol"]: p for p in pf["positions"]}
    for row in st["positions"]:
        got = by_sym[row["symbol"]]
        for k in ("shares", "avg_price", "last_price", "market_value",
                  "cost_basis", "unrealized_pnl", "weight"):
            assert got[k] == row[k], (row["symbol"], k)
    # page-only fields still ride along; mark_meta is additive
    assert "opened_at" in pf["positions"][0] and "marked_at" in pf["positions"][0]
    assert "mark_meta" in pf
    assert pf["vs_spy"] is None  # no SPY bars seeded → too young to benchmark


# ── Phase E: /equity mark provenance + SPY overlay ──


def test_equity_degraded_points_and_spy_overlay(client):
    from datetime import date, datetime, timedelta, timezone

    import agent.data as agent_data
    from agent.ledger import _et_date
    from agent.models import ACCOUNT, DeskEquity
    from edgefinder.db.models import DailyBar

    now = datetime.now(timezone.utc)
    sess = agent_data.session_factory()()
    try:
        sess.add(DeskEquity(
            account=ACCOUNT, ts=now + timedelta(minutes=5), cash=88000.0,
            positions_value=12000.0, equity=100000.0, return_pct=0.0,
            mark_meta={"sources": {"live": 0, "close": 0, "cost": 1},
                       "cost_marked": ["NVDA"], "cost_marked_value_pct": 100.0,
                       "degraded": True}))
        # SPY closes: baseline strictly before inception, endpoint on it
        incep = date.fromisoformat(_et_date(now.replace(tzinfo=None)))
        for d, px in ((incep - timedelta(days=5), 600.0), (incep, 612.0)):
            sess.add(DailyBar(symbol="SPY", date=d, open=px, high=px, low=px,
                              close=px, volume=1e6, source="test",
                              created_at=now))
        sess.commit()
    finally:
        sess.close()

    # bare shape stays a list; the degraded point carries the flags additively
    series = client.get("/api/desk/equity").json()
    assert isinstance(series, list)
    assert "degraded" not in series[0]           # the clean fixture point
    last = series[-1]
    assert last["degraded"] is True
    assert last["cost_marked_value_pct"] == 100.0
    assert last["cost_marked"] == ["NVDA"]

    # with_spy=1: dict shape, TR SPY rebased to inception = 0
    body = client.get("/api/desk/equity?with_spy=1").json()
    assert [p["equity"] for p in body["points"]] == [p["equity"] for p in series]
    assert body["spy"] == [{"date": str(incep), "pct": 2.0}]
    assert body["spy_basis"] == "total_return"


# ── Phase E: /outcomes — the predictions scoreboard ──


def _seed_outcomes(client):
    from datetime import date, datetime, timedelta, timezone

    import agent.data as agent_data
    from agent.models import ACCOUNT, DeskDecision, DeskOutcome

    now = datetime.now(timezone.utc)
    sess = agent_data.session_factory()()
    try:
        sess.add(DeskDecision(
            account=ACCOUNT, run_id="R9", ts=now - timedelta(hours=1),
            regime="risk_on", summary="bought XYZ",
            picks=[{"symbol": "XYZ", "action": "buy",
                    "prediction": "XYZ +5% within 10 sessions",
                    "horizon_days": 10, "kill": "closes below 90"}]))
        sess.add(DeskOutcome(
            account=ACCOUNT, run_id="R9", symbol="XYZ",
            grade_date=date.today(), entry_avg_px=100.0, mark_px=104.0,
            mark_basis="mark", since_pct=4.0, spy_pct=1.0, alpha_pct=3.0,
            horizon_days=10, horizon_elapsed=False, kill_level=90.0,
            kill_breached=False, status="open", degraded=False))
        sess.add(DeskOutcome(
            account=ACCOUNT, run_id="R1", symbol="NVDA",
            grade_date=date.today(), entry_avg_px=120.0, mark_px=126.0,
            mark_basis="exit", since_pct=5.0, spy_pct=1.0, alpha_pct=4.0,
            status="closed", exit_kind="hardstop", exit_avg_px=126.0,
            realized_pnl=600.0, degraded=False,
            verdict="TRUE", verdict_note="called it"))
        sess.commit()
    finally:
        sess.close()


def test_outcomes_scoreboard(client):
    _seed_outcomes(client)
    body = client.get("/api/desk/outcomes").json()

    s = body["summary"]
    assert s["open"] == 1 and s["closed"] == 1
    assert s["verdicts"] == {"TRUE": 1, "ungraded": 1}
    assert s["closed_graded"] == 1 and s["hit_rate_pct"] == 100.0

    rows = body["rows"]
    assert [r["symbol"] for r in rows] == ["XYZ", "NVDA"]  # open first
    xyz = rows[0]
    # pick context joined from the decision row (the words next to the math)
    assert xyz["prediction"] == "XYZ +5% within 10 sessions"
    assert xyz["kill"] == "closes below 90"
    assert xyz["action"] == "buy" and xyz["decision_ts"]
    assert xyz["kill_level"] == 90.0 and xyz["kill_breached"] is False
    assert xyz["sessions_elapsed"] is None  # no SPY bars seeded
    nvda = rows[1]
    assert nvda["exit_kind"] == "hardstop" and nvda["verdict"] == "TRUE"
    assert nvda["verdict_note"] == "called it"
    assert nvda["realized_pnl"] == 600.0
    # R1's fixture pick carries no prediction — surfaced as null, not invented
    assert nvda["prediction"] is None

    only_open = client.get("/api/desk/outcomes?status=open").json()["rows"]
    assert [r["symbol"] for r in only_open] == ["XYZ"]
    only_closed = client.get("/api/desk/outcomes?status=closed").json()["rows"]
    assert [r["symbol"] for r in only_closed] == ["NVDA"]
    assert client.get("/api/desk/outcomes?limit=999").status_code == 422


# ── Phase E: /decisions — the archive with paging ──


def test_decisions_archive_paging(client):
    from datetime import datetime, timedelta, timezone

    import agent.data as agent_data
    from agent.models import ACCOUNT, DeskDecision

    now = datetime.now(timezone.utc)
    sess = agent_data.session_factory()()
    try:
        for i, rid in ((2, "OLD1"), (4, "OLD2")):
            sess.add(DeskDecision(
                account=ACCOUNT, run_id=rid, ts=now - timedelta(hours=i),
                regime="neutral", summary=f"decision {rid}",
                picks=[{"symbol": "XYZ", "action": "hold"}]))
        sess.commit()
    finally:
        sess.close()

    page1 = client.get("/api/desk/decisions?limit=2").json()
    assert [d["run_id"] for d in page1["decisions"]] == ["R1", "OLD1"]
    assert page1["next_before"] == page1["decisions"][-1]["id"]
    # full dossier shape (same as /decision/latest plus id)
    top = page1["decisions"][0]
    for k in ("id", "run_id", "ts", "regime", "summary", "target_weights",
              "picks", "watchlist", "rejected", "strategy_version"):
        assert k in top
    assert top["picks"][0]["symbol"] == "NVDA"

    page2 = client.get(
        f"/api/desk/decisions?limit=2&before={page1['next_before']}").json()
    assert [d["run_id"] for d in page2["decisions"]] == ["OLD2"]
    assert page2["next_before"] is None       # short page → archive exhausted

    assert client.get("/api/desk/decisions?before=not-a-ts").status_code == 422


# ── Phase E: /data-health carries the latest mark provenance ──


def test_data_health_marks_section(client):
    from datetime import datetime, timedelta, timezone

    import agent.data as agent_data
    from agent.models import ACCOUNT, DeskEquity

    body = client.get("/api/desk/data-health").json()
    assert body["marks"] is None            # fixture snapshot has no mark_meta

    sess = agent_data.session_factory()()
    try:
        sess.add(DeskEquity(
            account=ACCOUNT, ts=datetime.now(timezone.utc) + timedelta(minutes=5),
            cash=88000.0, positions_value=12000.0, equity=100000.0,
            return_pct=0.0,
            mark_meta={"sources": {"live": 0, "close": 0, "cost": 1},
                       "cost_marked": ["NVDA"], "cost_marked_value_pct": 100.0,
                       "degraded": True}))
        sess.commit()
    finally:
        sess.close()

    body = client.get("/api/desk/data-health").json()
    assert body["marks"]["degraded"] is True
    assert body["marks"]["cost_marked"] == ["NVDA"]
    assert body["marks"]["cost_marked_value_pct"] == 100.0


# ── Phase E: options endpoints are allowlisted + rate-limited ──


def test_options_allowlist_and_rate_limit(client, monkeypatch):
    import agent.options_data as od
    import dashboard.routers.desk as desk_router

    # no external calls even for allowed symbols: strip the Alpaca keys
    for v in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY",
              "ALPACA_API_KEY", "ALPACA_API_SECRET"):
        monkeypatch.delenv(v, raising=False)
    from agent import broker as _b
    monkeypatch.setattr(_b.settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(_b.settings, "alpaca_api_secret", "", raising=False)
    od._cache.clear()
    desk_router._options_allow = None
    desk_router._options_bucket.reset()

    # held position (NVDA) and the latest decision's watchlist (AAPL) → allowed
    r = client.get("/api/desk/options/NVDA")
    assert r.status_code == 200 and r.json()["available"] is False
    assert client.get("/api/desk/options/AAPL").status_code == 200
    # streamer seed (SPY) → allowed
    assert client.get("/api/desk/options/SPY").status_code == 200
    # arbitrary symbol → 404, and NO Alpaca call was attempted for it
    r = client.get("/api/desk/options/EVILCO")
    assert r.status_code == 404 and "EVILCO" in r.json()["detail"]

    # history is DB-only: allowlist applies, no rate limit needed
    hist = client.get("/api/desk/options/NVDA/history")
    assert hist.status_code == 200 and hist.json()["symbol"] == "NVDA"
    assert client.get("/api/desk/options/EVILCO/history").status_code == 404

    # rate limit: a tiny bucket exhausts, then 429
    old_bucket = desk_router._options_bucket
    try:
        desk_router._options_bucket = desk_router._TokenBucket(
            capacity=3, refill_per_sec=0.0)
        for _ in range(3):
            assert client.get("/api/desk/options/NVDA").status_code == 200
        assert client.get("/api/desk/options/NVDA").status_code == 429
        # the DB-only history lane stays open
        assert client.get("/api/desk/options/NVDA/history").status_code == 200
    finally:
        desk_router._options_bucket = old_bucket


def test_options_rate_limit_key_ignores_spoofed_xff_head(client, monkeypatch):
    """Rotating the FIRST X-Forwarded-For hop (attacker-appendable) must not
    mint fresh rate-limit buckets — the key is the LAST hop, the one appended
    by the nearest trusted proxy."""
    import agent.options_data as od
    import dashboard.routers.desk as desk_router

    for v in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY",
              "ALPACA_API_KEY", "ALPACA_API_SECRET"):
        monkeypatch.delenv(v, raising=False)
    from agent import broker as _b
    monkeypatch.setattr(_b.settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(_b.settings, "alpaca_api_secret", "", raising=False)
    od._cache.clear()
    desk_router._options_allow = None

    old_bucket = desk_router._options_bucket
    try:
        desk_router._options_bucket = desk_router._TokenBucket(
            capacity=3, refill_per_sec=0.0)
        # same trusted last hop, rotating fake first hops → ONE shared bucket
        for i in range(3):
            r = client.get("/api/desk/options/NVDA", headers={
                "x-forwarded-for": f"10.0.0.{i}, 8.8.8.8"})
            assert r.status_code == 200
        r = client.get("/api/desk/options/NVDA", headers={
            "x-forwarded-for": "10.0.0.99, 8.8.8.8"})
        assert r.status_code == 429              # the rotation bought nothing
        # a different LAST hop is a genuinely different client → its own bucket
        assert client.get("/api/desk/options/NVDA", headers={
            "x-forwarded-for": "10.0.0.99, 9.9.9.9"}).status_code == 200
    finally:
        desk_router._options_bucket = old_bucket


# ── E4 follow-up: the SSE session cache is single-flight + time-bounded ──


def test_market_session_single_flight(monkeypatch):
    """On TTL expiry exactly one refresh runs; a concurrent frame serves the
    stale value immediately instead of stacking a second broker call."""
    import asyncio
    import time

    import dashboard.routers.desk as desk_router

    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        time.sleep(0.05)  # long enough for the second caller to overlap
        return "regular"

    monkeypatch.setattr(desk_router, "_fetch_market_session", fake_fetch)
    monkeypatch.setattr(desk_router, "_session_cache", (0.0, None))
    monkeypatch.setattr(desk_router, "_session_refreshing", False)

    async def race():
        return await asyncio.gather(desk_router._market_session(),
                                    desk_router._market_session())

    results = asyncio.run(race())
    assert calls["n"] == 1                       # single-flight
    assert "regular" in results                  # the refresher got the value
    assert None in results                       # the other served stale/null
    # now cached: no further fetch inside the TTL
    assert asyncio.run(desk_router._market_session()) == "regular"
    assert calls["n"] == 1


def test_market_session_timeout_backs_off(monkeypatch):
    """A hung broker call is bounded by wait_for; the stale/null value keeps
    serving and the next attempt backs off instead of re-firing at once."""
    import asyncio
    import time

    import dashboard.routers.desk as desk_router

    calls = {"n": 0}

    def hung_fetch():
        calls["n"] += 1
        time.sleep(0.3)  # well past the patched timeout
        return "regular"

    monkeypatch.setattr(desk_router, "_fetch_market_session", hung_fetch)
    monkeypatch.setattr(desk_router, "_SESSION_FETCH_TIMEOUT", 0.02)
    monkeypatch.setattr(desk_router, "_session_cache", (0.0, None))
    monkeypatch.setattr(desk_router, "_session_refreshing", False)

    assert asyncio.run(desk_router._market_session()) is None  # stale kept
    assert calls["n"] == 1
    # backoff: an immediate retry serves the cache — no second broker call
    assert asyncio.run(desk_router._market_session()) is None
    assert calls["n"] == 1
    ts, _ = desk_router._session_cache           # timestamp pushed forward
    assert time.time() - ts < desk_router._SESSION_TTL


# ── /portfolio response cache: bounded full-ledger-scan frequency ──


def test_portfolio_response_is_ttl_cached(client):
    import dashboard.routers.desk as desk_router

    first = client.get("/api/desk/portfolio").json()

    # a new fill lands...
    import agent.data as agent_data
    from agent.models import ACCOUNT, DeskTrade
    sess = agent_data.session_factory()()
    try:
        sess.add(DeskTrade(account=ACCOUNT, run_id="R9", symbol="AMD",
                           side="BUY", shares=10, price=100.0, dollars=1000.0,
                           ts=datetime.now(timezone.utc)))
        sess.commit()
    finally:
        sess.close()

    # ...but inside the TTL the cached body still serves (bounded staleness)
    assert client.get("/api/desk/portfolio").json() == first
    # cache expiry (simulated) → the fresh ledger scan sees the fill
    desk_router._portfolio_cache = None
    fresh = client.get("/api/desk/portfolio").json()
    assert fresh["cash"] == pytest.approx(first["cash"] - 1000.0)

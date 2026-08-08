"""Smoke the trading-desk page + /api/desk/* endpoints on a seeded SQLite DB.

REBUILD-V4 era model: the Alpaca paper account is the book of record, so the
account-shaped endpoints (/portfolio, /open-orders, /broker-health) are
exercised BOTH ways — with a canned ``agent.trade.Trade`` double (conftest
strips creds, so nothing ever dials a real broker) and degraded (the double
raises, the endpoint must answer ``available: false``, never 500). Fills
come from the ``desk_orders`` mirror; the equity curve stitches the frozen
``era1_*`` archive to ``desk_portfolio_history``.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

NOW = datetime.now(timezone.utc)
FILL_TS = (NOW - timedelta(days=10)).isoformat()


def _canned_state():
    return {
        "account": "agent", "paper": True,
        "cash": 88000.0, "equity": 101000.0,
        "buying_power": 88000.0, "options_buying_power": 88000.0,
        "starting_capital": 100000.0,
        "total_pnl": 1000.0, "total_return_pct": 1.0,
        "positions_value": 13000.0,
        "positions": [{
            "symbol": "NVDA", "asset_class": "us_equity",
            "qty": 100.0, "qty_available": 100.0,
            "avg_entry_price": 120.0, "current_price": 130.0,
            "market_value": 13000.0, "cost_basis": 12000.0,
            "unrealized_pl": 1000.0, "unrealized_plpc": 0.0833,
            "change_today": None, "side": "long", "weight": 0.128713,
        }],
    }


def canned_trade(monkeypatch, *, state=None, open_orders=None, fail=False):
    """Patch agent.trade.Trade with a canned double; returns the state dict
    so tests can mutate it in place (TTL-cache assertions)."""
    st = state if state is not None else _canned_state()
    orders = list(open_orders or [])

    class _FakeTrade:
        def __init__(self, *a, **k):
            if fail:
                raise RuntimeError("trade creds not set on this host")

        def state(self):
            return copy.deepcopy(st)

        def account(self):
            return {"status": "ACTIVE", "cash": st["cash"],
                    "equity": st["equity"], "paper": True}

        def positions(self):
            return copy.deepcopy(st["positions"])

        def orders(self, status="open", limit=100, **k):
            return copy.deepcopy(orders)

    monkeypatch.setattr("agent.trade.Trade", _FakeTrade)
    return st


def _reset_desk_caches(desk_router):
    desk_router._options_allow = None
    desk_router._options_bucket.reset()
    desk_router._session_cache = (0.0, None)
    desk_router._session_refreshing = False
    desk_router._portfolio_cache = None
    desk_router._open_orders_cache = None
    desk_router._outcomes_live_cache = None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'desk.db'}")
    monkeypatch.setenv("EDGEFINDER_SCHEDULER_ENABLED", "false")

    # Importing the router first registers the era1_* archive table shapes in
    # the shared metadata, so create_all builds them (empty = pre-cutover).
    import dashboard.routers.desk as desk_router
    from edgefinder.db.engine import Base, get_engine
    import edgefinder.db.models  # noqa: F401
    import agent.models  # noqa: F401
    from agent.models import ACCOUNT, DeskDecision, DeskThinking
    import agent.data as agent_data
    import dashboard.dependencies as deps

    engine = get_engine()
    Base.metadata.create_all(engine)
    agent_data._session_factory = None
    deps._engine = deps._session_factory = None
    _reset_desk_caches(desk_router)

    # The canned paper account: NVDA 100 @ 120, marked 130.
    canned_trade(monkeypatch)

    from agent.store import get_store
    store = get_store()
    # Era-2 fill in the desk_orders mirror (the source for /trades, missed
    # dividends, and era-2 inception).
    store.insert("desk_orders", {
        "account": ACCOUNT, "run_id": "R1", "seq": 1,
        "client_order_id": "R1:01", "alpaca_order_id": "ord-1",
        "symbol": "NVDA", "asset_class": "us_equity", "side": "buy",
        "kind": "entry", "order_type": "market", "tif": "day",
        "order_class": "simple", "qty": 100.0, "status": "filled",
        "filled_qty": 100.0, "filled_avg_price": 120.0,
        "submitted_at": FILL_TS, "filled_at": FILL_TS}, returning=False)
    # One nightly equity snapshot (era 2).
    store.insert("desk_portfolio_history", {
        "account": ACCOUNT,
        "snap_date": (NOW - timedelta(days=1)).date().isoformat(),
        "equity": 100500.0, "cash": 88000.0, "profit_loss": 500.0,
        "base_value": 100000.0,
        "positions": {"NVDA": {"qty": 100.0, "avg_entry_price": 120.0}}},
        returning=False)

    sess = agent_data.session_factory()()
    try:
        sess.add(DeskThinking(account=ACCOUNT, run_id="R1", phase="research",
                              text="NVDA momentum strong", ts=NOW))
        sess.add(DeskDecision(account=ACCOUNT, run_id="R1", ts=NOW, regime="risk_on",
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
    """v10 IA: two zones (reasoning, learning); the tripwire "watching" card
    is gone (tripwires died with the streamer's dispatcher) and its slot now
    holds the open-orders / resting-protection card; the hero zone carries
    the V4 honesty strip (book of record, no market impact, missed
    dividends)."""
    import re

    html = client.get("/desk").text

    reasoning = html.index('id="zone-reasoning"')
    learning = html.index('id="zone-history"')
    assert reasoning < learning
    assert 'id="zone-markets"' not in html

    assert 'id="desk-watch"' not in html         # tripwire card retired
    assert 'id="desk-orders"' in html            # open orders & protection
    assert 'id="desk-honesty"' in html           # the V4 honesty strip
    assert 'id="desk-honesty-missed"' in html    # missed-dividends counter
    assert "Alpaca paper account" in html        # book-of-record disclosure
    assert 'id="desk-lab"' in html               # the lab leaderboard
    assert 'id="desk-hero-indices"' in html      # live SPY/QQQ/IWM chips
    assert 'id="desk-lab-seg"' in html           # board / recent-tests views
    assert 'id="desk-wiki-seg"' in html          # lessons / diary views
    assert 'data-zone="desk-hero">Overview' in html
    assert 'data-zone="zone-markets"' not in html
    assert 'id="topnav-indices"' not in html     # dead slot removed

    # Retired standalone cards must be fully gone, not just hidden.
    for key in ("tape", "movers", "options", "dividends", "backtests",
                "journal", "watch"):
        assert f'data-collapse-key="{key}"' not in html, f"{key} card lingers"

    def card_tag(key):
        m = re.search(r'<div class="c-card desk-card[^"]*" '
                      r'data-collapse-key="%s"[^>]*>' % key, html)
        assert m, f"card {key} missing"
        return m.group(0)

    # Receipts ship collapsed; the reasoning/learning core is open.
    assert 'data-collapsed="1"' in card_tag("fills")
    for key in ("orders", "decision", "thinking", "lab", "wiki"):
        assert 'data-collapsed="1"' not in card_tag(key), f"{key} should be open"


# ── /portfolio: the Alpaca paper account, canned + degraded ──


def test_portfolio_endpoint(client):
    r = client.get("/api/desk/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["positions"][0]["symbol"] == "NVDA"
    assert body["positions"][0]["qty"] == 100.0
    assert body["positions"][0]["avg_entry_price"] == 120.0
    assert body["positions"][0]["current_price"] == 130.0
    assert body["positions"][0]["unrealized_pl"] == 1000.0
    assert body["cash"] == 88000.0
    assert body["equity"] == 101000.0
    assert body["buying_power"] == 88000.0
    assert body["total_pnl"] == 1000.0
    assert body["total_return_pct"] == 1.0
    assert body["vs_spy"] is None  # no SPY bars seeded → too young to benchmark


def test_portfolio_degraded_without_creds(client, monkeypatch):
    """No trade creds (or a dead broker) → available:false + empty positions,
    never a 500 and never a fake book."""
    import dashboard.routers.desk as desk_router

    canned_trade(monkeypatch, fail=True)
    desk_router._portfolio_cache = None
    body = client.get("/api/desk/portfolio").json()
    assert body["available"] is False
    assert body["positions"] == []
    assert body["equity"] is None and body["cash"] is None
    assert "unreachable" in body["note"]


def test_portfolio_vs_spy_price_return(client, monkeypatch):
    """vs_spy is SYMMETRIC PRICE RETURN from the all-time inception (charter
    V4): dividends must NOT back-adjust the benchmark — the paper book never
    receives them."""
    from datetime import date

    import agent.data as agent_data
    import dashboard.routers.desk as desk_router
    from edgefinder.db.models import DailyBar, DividendRecord

    incep = date.fromisoformat(FILL_TS[:10])
    sess = agent_data.session_factory()()
    try:
        for d, px in ((incep - timedelta(days=2), 600.0),
                      (incep + timedelta(days=5), 606.0),
                      (date.today(), 612.0)):
            sess.add(DailyBar(symbol="SPY", date=d, open=px, high=px, low=px,
                              close=px, volume=1e6, source="test",
                              created_at=NOW))
        # A SPY dividend inside the window: a TOTAL-return series would
        # back-adjust the 600 baseline and report > 2% — price return must not.
        sess.add(DividendRecord(symbol="SPY", ex_date=incep + timedelta(days=1),
                                cash_amount=6.0))
        sess.commit()
    finally:
        sess.close()

    desk_router._portfolio_cache = None
    body = client.get("/api/desk/portfolio").json()
    vs = body["vs_spy"]
    assert vs is not None
    assert vs["basis"] == "price_return"
    assert vs["spy_return_pct"] == 2.0          # 600 → 612, dividend ignored
    assert vs["alpha_pct"] == pytest.approx(body["total_return_pct"] - 2.0)


def test_portfolio_response_is_ttl_cached(client, monkeypatch):
    import dashboard.routers.desk as desk_router

    st = canned_trade(monkeypatch)
    desk_router._portfolio_cache = None
    first = client.get("/api/desk/portfolio").json()

    # the broker book moves...
    st["cash"] = 50000.0
    st["equity"] = 99000.0

    # ...but inside the TTL the cached body still serves (bounded staleness)
    assert client.get("/api/desk/portfolio").json() == first
    # cache expiry (simulated) → the fresh broker read shows through
    desk_router._portfolio_cache = None
    fresh = client.get("/api/desk/portfolio").json()
    assert fresh["cash"] == 50000.0 and fresh["equity"] == 99000.0


# ── /equity: the stitched era curve ──


def test_equity_basic_shape_and_live_tip(client):
    body = client.get("/api/desk/equity").json()
    pts = body["points"]
    assert pts, "expected snapshot + live tip points"
    # era-2 nightly snapshot then the live tip off the cached account read
    assert pts[0]["era"] == 2 and pts[0]["equity"] == 100500.0
    assert pts[-1]["equity"] == 101000.0 and pts[-1].get("live") is True
    # era-2 inception is computed from the first mirrored fill
    assert body["era2_inception"] == FILL_TS[:10] or body["era2_inception"] \
        == (datetime.fromisoformat(FILL_TS) - timedelta(days=1)).date().isoformat()


def test_equity_era_stitch_and_spy_overlay(client):
    """Era-1 archive points come first (when the frozen tables exist), each
    point is era-tagged, and the SPY overlay is PRICE return rebased at the
    ALL-TIME inception (the era-1 book's first fill)."""
    from datetime import date

    import agent.data as agent_data
    from agent.models import ACCOUNT
    from agent.store import get_store
    from edgefinder.db.models import DailyBar

    store = get_store()
    d1 = NOW - timedelta(days=30)
    store.insert("era1_trades", {
        "account": ACCOUNT, "ts": d1.replace(tzinfo=None), "run_id": "E1",
        "symbol": "XYZ", "side": "BUY", "shares": 10.0, "price": 100.0,
        "dollars": 1000.0, "rationale": "era-1 entry",
        "fill_quote": {"bid": 99.9, "ask": 100.1}}, returning=False)
    for i, eq in ((30, 100000.0), (29, 100100.0)):
        store.insert("era1_equity", {
            "account": ACCOUNT, "ts": (NOW - timedelta(days=i)).replace(tzinfo=None),
            "cash": 99000.0, "positions_value": eq - 99000.0, "equity": eq,
            "return_pct": 0.0}, returning=False)

    incep = (d1.date())
    sess = agent_data.session_factory()()
    try:
        for d, px in ((incep - timedelta(days=1), 600.0),
                      (date.today(), 612.0)):
            sess.add(DailyBar(symbol="SPY", date=d, open=px, high=px, low=px,
                              close=px, volume=1e6, source="test",
                              created_at=NOW))
        sess.commit()
    finally:
        sess.close()

    body = client.get("/api/desk/equity?with_spy=1").json()
    pts = body["points"]
    eras = [p["era"] for p in pts]
    assert eras == sorted(eras), "era-1 points must precede era-2"
    assert eras[0] == 1 and eras[-1] == 2
    assert pts[0]["equity"] == 100000.0
    assert body["era2_inception"] is not None
    assert body["spy_basis"] == "price_return"
    # rebased at the ALL-TIME (era-1) inception: 600 → 612 = +2%
    assert body["spy_inception"] in (str(incep), str(incep - timedelta(days=1)),
                                     str(incep + timedelta(days=1)))
    assert body["spy"][-1]["pct"] == 2.0


def test_decision_and_thinking(client):
    d = client.get("/api/desk/decision/latest").json()
    assert d["exists"] and d["picks"][0]["symbol"] == "NVDA"
    t = client.get("/api/desk/thinking").json()
    assert t["run_id"] == "R1" and t["lines"]


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


# ── /broker-health: paper account + clock + last reconcile ──


def test_broker_health_canned_account(client, monkeypatch):
    for v in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET"):
        monkeypatch.delenv(v, raising=False)
    from agent import broker as _b
    monkeypatch.setattr(_b.settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(_b.settings, "alpaca_api_secret", "", raising=False)
    body = client.get("/api/desk/broker-health").json()
    assert body["paper_account"]["available"] is True
    assert body["paper_account"]["status"] == "ACTIVE"
    assert body["paper_account"]["equity"] == 101000.0
    assert body["clock"] is None                 # no data keys → no clock read
    assert body["last_reconcile"] is not None    # desk_orders mirror row exists


def test_broker_health_degraded(client, monkeypatch):
    import dashboard.routers.desk as desk_router  # noqa: F401

    canned_trade(monkeypatch, fail=True)
    from agent import broker as _b
    monkeypatch.setattr(_b.settings, "alpaca_api_key", "", raising=False)
    monkeypatch.setattr(_b.settings, "alpaca_api_secret", "", raising=False)
    body = client.get("/api/desk/broker-health").json()
    assert body["paper_account"]["available"] is False
    assert "RuntimeError" in body["paper_account"]["error"]


# ── /open-orders: resting protection ──


def test_open_orders_rows(client, monkeypatch):
    import dashboard.routers.desk as desk_router

    canned_trade(monkeypatch, open_orders=[
        {"alpaca_order_id": "o-stop", "symbol": "NVDA", "side": "sell",
         "order_type": "stop", "tif": "gtc", "stop_price": 100.0,
         "limit_price": None, "qty": 100.0, "filled_qty": 0.0,
         "status": "new",
         "submitted_at": (NOW - timedelta(days=85)).isoformat()},
        {"alpaca_order_id": "o-lim", "symbol": "AAPL", "side": "buy",
         "order_type": "limit", "tif": "day", "stop_price": None,
         "limit_price": 180.0, "qty": 10.0, "filled_qty": 0.0,
         "status": "new",
         "submitted_at": (NOW - timedelta(days=1)).isoformat()},
    ])
    desk_router._open_orders_cache = None
    body = client.get("/api/desk/open-orders").json()
    assert body["available"] is True
    rows = body["orders"]
    assert len(rows) == 2
    # stops sort first (the protection is the headline)
    stop = rows[0]
    assert stop["kind"] == "stop" and stop["symbol"] == "NVDA"
    assert stop["stop_price"] == 100.0 and stop["tif"] == "gtc"
    assert stop["age_days"] == 85                # ≥80 → the UI badges 90d expiry
    assert stop["alpaca_order_id"] == "o-stop"
    lim = rows[1]
    assert lim["kind"] == "limit" and lim["limit_price"] == 180.0
    assert lim["age_days"] in (0, 1)


def test_open_orders_degraded(client, monkeypatch):
    import dashboard.routers.desk as desk_router

    canned_trade(monkeypatch, fail=True)
    desk_router._open_orders_cache = None
    body = client.get("/api/desk/open-orders").json()
    assert body["available"] is False and body["orders"] == []


# ── the watch endpoint is gone (tripwires died with the dispatcher) ──


def test_watch_endpoint_removed(client):
    assert client.get("/api/desk/watch").status_code == 404


def test_desk_movers(client, monkeypatch):
    """Movers endpoint ranks gainers/losers/most-active from daily_bars
    across the last two FULL-COVERAGE sessions (v9.13.0: same coverage
    floor + split guard as the nightly brief's movers)."""
    from datetime import date, datetime, timezone

    import agent.data as agent_data
    from edgefinder.db.models import DailyBar

    monkeypatch.setattr(agent_data, "FULL_COVERAGE_MIN", 3)
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
    """Holding-stats sources held names from the cached Alpaca positions
    (NVDA in the canned account) and returns day change, 52-week range, and
    a spark series."""
    from datetime import date, datetime, timedelta, timezone

    import agent.data as agent_data
    from edgefinder.db.models import DailyBar

    now = datetime.now(timezone.utc)
    sess = agent_data.session_factory()()
    try:
        # NVDA is the canned holding; give it 10 rising sessions
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


def test_desk_holding_stats_degraded_account(client, monkeypatch):
    """Holdings unknown (no broker) must mean an EMPTY panel, not a 500."""
    import dashboard.routers.desk as desk_router

    canned_trade(monkeypatch, fail=True)
    desk_router._portfolio_cache = None
    d = client.get("/api/desk/holding-stats").json()
    assert d == {"as_of": None, "symbols": {}}


def test_desk_dividends(client):
    """Dividend calendar returns last/next ex-dates for dividend-paying
    holdings (held = the canned Alpaca positions), plus the V4
    missed-dividends counter."""
    from datetime import date

    import agent.data as agent_data
    from edgefinder.db.models import DividendRecord

    sess = agent_data.session_factory()()
    try:  # NVDA is the canned holding; 2099 is unambiguously "upcoming"
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
    # trailing = PAST ex-dates only — the future declared 0.12 must not
    # inflate a figure labelled "trailing" (v9.13.0 fix)
    assert nvda["ttm_amount"] == round(0.10 + 0.10, 4)
    # both seeded ex-dates precede the first era-2 fill → nothing missed yet
    md = d["missed_dividends"]
    assert md["total"] == 0.0 and md["by_symbol"] == {}
    assert "no dividends" in md["note"]


def test_missed_dividends_counter(client):
    """Shares held STRICTLY BEFORE an ex-date inside era 2 earn the missed
    estimate: the fixture's 100 NVDA shares filled 10 days ago × a $0.25
    dividend that went ex 5 days ago = $25.00 foregone."""
    from datetime import date, timedelta as td

    import agent.data as agent_data
    from edgefinder.db.models import DividendRecord

    sess = agent_data.session_factory()()
    try:
        sess.add(DividendRecord(symbol="NVDA",
                                ex_date=date.today() - td(days=5),
                                cash_amount=0.25))
        sess.commit()
    finally:
        sess.close()

    d = client.get("/api/desk/dividends").json()
    md = d["missed_dividends"]
    assert md["total"] == pytest.approx(25.0)
    assert md["by_symbol"] == {"NVDA": 25.0}


# ── /trades: era-tagged fills ──


def test_desk_trades_era2_from_mirror(client):
    rows = client.get("/api/desk/trades?limit=20").json()
    assert isinstance(rows, list)
    nvda = next(r for r in rows if r["symbol"] == "NVDA")
    assert nvda["era"] == 2
    assert nvda["side"] == "BUY" and nvda["shares"] == 100.0
    assert nvda["price"] == 120.0 and nvda["dollars"] == 12000.0
    assert nvda["run_id"] == "R1" and nvda["kind"] == "entry"
    assert nvda["rationale"] is None            # era-2 reasoning lives on the run
    assert nvda["fill_quote"] is None


def test_desk_trades_include_era1_receipts(client):
    """Era-1 rows (frozen archive) keep their original rationale + stamped
    live bid/ask receipt, era-tagged next to the era-2 broker fills."""
    from agent.models import ACCOUNT
    from agent.store import get_store

    get_store().insert("era1_trades", {
        "account": ACCOUNT, "run_id": "R0", "symbol": "LLY", "side": "BUY",
        "shares": 2.0, "price": 1226.46, "dollars": 2452.92,
        "rationale": "era-1 conviction add",
        "ts": (NOW - timedelta(days=40)).replace(tzinfo=None),
        "fill_quote": {"bid": 1224.96, "ask": 1226.34, "mid": 1225.65}},
        returning=False)

    rows = client.get("/api/desk/trades?limit=20").json()
    lly = next(r for r in rows if r["symbol"] == "LLY")
    assert lly["era"] == 1
    assert lly["fill_quote"]["bid"] == 1224.96 and lly["fill_quote"]["ask"] == 1226.34
    assert lly["rationale"] == "era-1 conviction add"
    # newest first: the era-2 NVDA fill (10d ago) precedes the 40d-old era-1 row
    syms = [r["symbol"] for r in rows]
    assert syms.index("NVDA") < syms.index("LLY")


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


# ── /outcomes — the predictions scoreboard ──


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
    # XYZ has no mirror fills/positions → the stored grade facts serve as-is
    assert xyz["since_pct"] == 4.0 and xyz["alpha_pct"] == 3.0
    nvda = rows[1]
    assert nvda["exit_kind"] == "hardstop" and nvda["verdict"] == "TRUE"
    assert nvda["verdict_note"] == "called it"
    assert nvda["realized_pnl"] == 600.0
    # R1's fixture pick carries no prediction — surfaced as null, not invented
    assert nvda["prediction"] is None
    # grade's own degraded bool passes through
    assert nvda["degraded"] is False

    only_open = client.get("/api/desk/outcomes?status=open").json()["rows"]
    assert [r["symbol"] for r in only_open] == ["XYZ"]
    only_closed = client.get("/api/desk/outcomes?status=closed").json()["rows"]
    assert [r["symbol"] for r in only_closed] == ["NVDA"]
    assert client.get("/api/desk/outcomes?limit=999").status_code == 422


def test_outcomes_open_rows_overlay_live_marks(client, monkeypatch):
    """An OPEN row whose pick has mirror fills + a live Alpaca position gets
    fresh since/mark facts from agent.grade.outcomes (the stored row's stale
    numbers move between grade runs); verdict-side columns never change."""
    import dashboard.routers.desk as desk_router

    _seed_outcomes(client)
    from agent.models import ACCOUNT

    # Point the R9/XYZ open row at the mirror: give XYZ a fill + position.
    from agent.store import get_store
    get_store().insert("desk_orders", {
        "account": ACCOUNT, "run_id": "R9", "seq": 1,
        "client_order_id": "R9:01", "alpaca_order_id": "ord-xyz",
        "symbol": "XYZ", "asset_class": "us_equity", "side": "buy",
        "kind": "entry", "order_type": "market", "tif": "day",
        "order_class": "simple", "qty": 10.0, "status": "filled",
        "filled_qty": 10.0, "filled_avg_price": 100.0,
        "submitted_at": FILL_TS, "filled_at": FILL_TS}, returning=False)
    st = _canned_state()
    st["positions"].append({
        "symbol": "XYZ", "asset_class": "us_equity", "qty": 10.0,
        "qty_available": 10.0, "avg_entry_price": 100.0,
        "current_price": 110.0, "market_value": 1100.0, "cost_basis": 1000.0,
        "unrealized_pl": 100.0, "unrealized_plpc": 0.1, "change_today": None,
        "side": "long", "weight": 0.01})
    canned_trade(monkeypatch, state=st)
    desk_router._portfolio_cache = None
    desk_router._outcomes_live_cache = None

    rows = client.get("/api/desk/outcomes?status=open").json()["rows"]
    xyz = next(r for r in rows if r["symbol"] == "XYZ")
    assert xyz["since_pct"] == 10.0          # live 110 vs 100 entry, not the stale 4.0
    assert xyz["mark_px"] == 110.0
    assert xyz["kill_level"] == 90.0         # stored machine facts untouched


# ── /decisions — the archive with paging ──


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


# ── /data-health: the marks block is dead (mark_meta died with the ledger) ──


def test_data_health_has_no_marks_block(client):
    body = client.get("/api/desk/data-health").json()
    assert "status" in body
    assert "marks" not in body


# ── options endpoints are allowlisted + rate-limited ──


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

    # held position (NVDA, canned Alpaca account) and the latest decision's
    # watchlist (AAPL) → allowed
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


# ── the SSE session cache is single-flight + time-bounded ──


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


def test_claims_and_proposals_endpoints(client):
    """The knowledge-layer projections: claims with tier authority visible,
    proposals with pending-first ordering. Read-only; no confidence floats."""
    from agent.knowledge import claim_add, proposal_add
    from agent.store import get_store

    store = get_store()
    est = claim_add(store, kclass="risk_rule", tier="established",
                    statement="honor fired kills same-cycle",
                    scope={"account": "paper"},
                    evidence=[{"kind": "probe", "note": "AAPL ~$500"}])
    claim_add(store, kclass="market_strategy", tier="candidate",
              statement="momentum diverges from mark",
              scope={"account": "paper", "regimes": ["risk_on"]},
              evidence=[{"kind": "probe", "note": "IWM/LLY"}],
              promotion_criteria={"min_n": 5})
    proposal_add(store, title="Raise concentration to 35",
                 body="why", change_kind="caps", claim_ids=[est["id"]])

    c = client.get("/api/desk/claims").json()
    assert c["summary"]["active"] == 2
    assert c["summary"]["by_tier"] == {"established": 1, "candidate": 1}
    # established sorts first (authority first), statements carry no
    # confidence field at all
    assert c["claims"][0]["tier"] == "established"
    assert c["claims"][0]["cite"] == f"[C-{est['id']}]"
    assert "confidence" not in c["claims"][0]
    assert c["claims"][1]["regimes"] == ["risk_on"]

    p = client.get("/api/desk/proposals").json()
    assert p["pending"] == 1
    assert p["proposals"][0]["ref"].startswith("PROPOSAL-")
    assert p["proposals"][0]["status"] == "pending"
    assert p["proposals"][0]["claim_ids"] == [est["id"]]

    # inactive claims stay out of the default view
    from agent.knowledge import claim_quarantine
    claim_quarantine(store, claim_id=est["id"], reason="test")
    c2 = client.get("/api/desk/claims").json()
    assert c2["summary"]["active"] == 1
    assert all(r["status"] == "active" for r in c2["claims"])
    c3 = client.get("/api/desk/claims?include_inactive=true").json()
    assert any(r["status"] == "quarantined" for r in c3["claims"])

"""The /trades page and its /api/desk/trade-history endpoint.

The feature's whole job is an HONEST human-readable history, so these tests
are mostly about the ways a profit number can be quietly wrong: a truncated
replay, a dividend row that looks like a $0 sale, an option multiplier applied
twice, an opening leg reported as breakeven.
"""

from __future__ import annotations

from datetime import datetime

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'trades.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401

    Base.metadata.create_all(get_engine())

    # The desk router memoises /portfolio and the dashboard caches its own
    # engine/session; a leftover from a sibling test would serve the WRONG
    # SQLite file here (conftest clears only get_store's lru_cache).
    import dashboard.routers.desk as desk_router
    desk_router._portfolio_cache = None
    import dashboard.dependencies as deps
    deps._engine = None
    deps._session_factory = None

    from agent.store import get_store
    return get_store()


@pytest.fixture()
def client(store):
    from fastapi.testclient import TestClient
    from dashboard.app import app
    return TestClient(app)


def _ts(day: int, hour: int = 15) -> datetime:
    return datetime(2026, 7, day, hour, 30)  # naive UTC, mid-session ET


def _seed(store, symbol, side, shares, price, ts, *, fill_quote=None,
          dollars=None):
    from agent import occ
    mult = 100 if occ.is_option(symbol) else 1
    store.insert("desk_trades", {
        "account": "agent", "run_id": "R1", "symbol": symbol, "side": side,
        "shares": shares, "price": price,
        "dollars": round(shares * price * mult, 2) if dollars is None else dollars,
        "fill_quote": fill_quote, "ts": ts}, returning=False)


def _hist(client, **params):
    r = client.get("/api/desk/trade-history", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _by_symbol(rows, symbol):
    return [r for r in rows if r["symbol"] == symbol]


# ── the page itself ──

def test_trades_page_renders(client):
    r = client.get("/trades")
    assert r.status_code == 200
    assert "Trade history" in r.text
    assert "/static/js/pages/trades.js" in r.text


def test_trades_link_in_both_nav_surfaces(client):
    """The link lives in base.html, so it must appear on a DIFFERENT page."""
    body = client.get("/desk").text
    assert 'href="/trades" class="topnav-tab"' in body
    assert 'href="/trades" class="tabbar-item"' in body


# ── empty state ──

def test_empty_ledger_returns_empty_history(client):
    d = _hist(client)
    assert d["rows"] == []
    assert d["realized_pnl"] == 0.0
    assert d["closing_fills"] == 0
    assert d["total"] == 0


# ── the core profit math ──

def test_round_trip_profit_and_open_legs_are_null(client, store):
    """A sell realizes against AVERAGE cost; the buys that opened realize
    nothing and must be null, never 0.0 (0.0 paints green)."""
    _seed(store, "XYZ", "BUY", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "BUY", 10, 110.0, _ts(7))
    _seed(store, "XYZ", "SELL", 20, 120.0, _ts(8))

    rows = _hist(client)["rows"]
    sells = [r for r in rows if r["side"] == "SELL"]
    buys = [r for r in rows if r["side"] == "BUY"]
    assert len(sells) == 1 and len(buys) == 2
    assert sells[0]["realized"] == pytest.approx(300.0, abs=0.01)
    assert all(b["realized"] is None for b in buys)


def test_pnl_identical_at_every_display_limit(client, store):
    """THE TRUNCATION TRAP. Average cost is path-dependent, so replaying only
    the newest rows books each SELL against a flat book and silently reports
    a different (wrong) profit. `limit` must slice the OUTPUT, never the
    replay input."""
    _seed(store, "XYZ", "BUY", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "BUY", 10, 110.0, _ts(7))
    _seed(store, "XYZ", "SELL", 20, 120.0, _ts(8))

    for limit in (1, 2, 500):
        d = _hist(client, limit=limit)
        sell = [r for r in d["rows"] if r["side"] == "SELL"]
        assert len(sell) == 1, f"limit={limit} should still show the newest row"
        assert sell[0]["realized"] == pytest.approx(300.0, abs=0.01), limit
        assert d["realized_pnl"] == pytest.approx(300.0, abs=0.01), limit
        assert d["total"] == 3


def test_option_round_trip_applies_multiplier_once(client, store):
    """x100 is applied inside the ledger replay — never again downstream."""
    sym = "NVDA270116C00200000"
    _seed(store, sym, "BUY", 2, 5.0, _ts(6))
    _seed(store, sym, "SELL", 2, 6.5, _ts(8))

    rows = _hist(client)["rows"]
    sell = [r for r in rows if r["side"] == "SELL"][0]
    assert sell["realized"] == pytest.approx(300.0, abs=0.01)  # not 3, not 30000
    assert sell["label"] == "NVDA $200C 2027-01-16"
    assert sell["underlying"] == "NVDA"


def test_sell_to_open_then_buy_to_close(client, store):
    """`side` does NOT mean open-vs-close. A SELL can open a short option leg
    and realize nothing; the later BUY carries the profit."""
    sym = "AMD260821C00550000"
    _seed(store, sym, "SELL", 1, 36.0, _ts(6))   # sell to open
    _seed(store, sym, "BUY", 1, 20.0, _ts(9))    # buy to close, cheaper

    rows = _hist(client)["rows"]
    opened = [r for r in rows if r["side"] == "SELL"][0]
    closed = [r for r in rows if r["side"] == "BUY"][0]
    assert opened["realized"] is None
    assert closed["realized"] == pytest.approx(1600.0, abs=0.01)


# ── rows that are not trades ──

def test_dividend_row_is_not_a_zero_profit_sale(client, store):
    """Dividend rows are booked SELL/shares=0.0 and fall into the reducing
    branch with closing==0 — without the EPS_SHARES guard they surface as a
    closed trade worth $0.00."""
    _seed(store, "XYZ", "BUY", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "SELL", 0.0, 0.0, _ts(8), dollars=12.30,
          fill_quote={"src": "dividend", "ex_date": "2026-07-08"})

    d = _hist(client)
    div = [r for r in d["rows"] if r["kind"] == "dividend"]
    assert len(div) == 1
    assert div[0]["realized"] is None
    assert d["closing_fills"] == 0
    assert d["realized_pnl"] == 0.0


def test_split_row_is_labelled_not_a_trade(client, store):
    """A split row has price 0 and dollars 0 — it must be tagged, not rendered
    as 'SELL 10 for $0.00', and must not book profit."""
    _seed(store, "XYZ", "BUY", 10, 100.0, _ts(6))
    store.insert("desk_trades", {
        "account": "agent", "run_id": "settlement", "symbol": "XYZ",
        "side": "BUY", "shares": 10, "price": 0.0, "dollars": 0.0,
        "fill_quote": {"src": "split_adjustment",
                       "execution_date": "2026-07-07", "ratio": 2.0},
        "ts": _ts(7)}, returning=False)
    _seed(store, "XYZ", "SELL", 20, 55.0, _ts(8))

    rows = _hist(client)["rows"]
    split = [r for r in rows if r["kind"] == "split"]
    assert len(split) == 1
    assert split[0]["realized"] is None
    assert split[0]["date"] == "2026-07-07"          # effective, not booked
    sell = [r for r in rows if r["side"] == "SELL"][0]
    assert sell["realized"] == pytest.approx(100.0, abs=0.01)


def test_duplicate_equity_exit_realizes_nothing(client, store):
    """A writer-race duplicate exit on a flat book books no P&L (the equity
    long-only clamp) — it must come back null, not a phantom short."""
    _seed(store, "XYZ", "BUY", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "SELL", 10, 110.0, _ts(8))
    _seed(store, "XYZ", "SELL", 10, 110.0, _ts(8, hour=16))

    d = _hist(client)
    sells = [r for r in d["rows"] if r["side"] == "SELL"]
    realized = sorted(r["realized"] for r in sells if r["realized"] is not None)
    assert realized == [pytest.approx(100.0, abs=0.01)]
    assert sum(1 for r in sells if r["realized"] is None) == 1
    assert d["realized_pnl"] == pytest.approx(100.0, abs=0.01)


# ── the invariant that keeps "one source of truth" honest ──

def test_history_total_equals_ledger_by_symbol(client, store):
    """The page's total must equal the ledger's own by_symbol sum — otherwise
    /trades and the rest of the desk are quoting different books."""
    from agent import ledger

    _seed(store, "XYZ", "BUY", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "SELL", 4, 130.0, _ts(7))
    _seed(store, "ABC", "BUY", 5, 50.0, _ts(7))
    _seed(store, "ABC", "SELL", 5, 40.0, _ts(9))

    d = _hist(client)
    _, by_symbol = ledger._realized_pnl(ledger._trades(store, "agent"))
    assert d["realized_pnl"] == pytest.approx(sum(by_symbol.values()), abs=0.01)

    per_row = sum(r["realized"] for r in d["rows"] if r["realized"] is not None)
    assert per_row == pytest.approx(d["realized_pnl"], abs=0.01)


def test_existing_trades_endpoint_shape_unchanged(client, store):
    """/api/desk/trades stays a BARE LIST — desk.js and symbol.js index it
    directly, so turning it into a dict would break them silently."""
    _seed(store, "XYZ", "BUY", 10, 100.0, _ts(6))
    r = client.get("/api/desk/trades", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert set(body[0]) == {"t", "symbol", "side", "shares", "price",
                            "dollars", "rationale", "run_id", "fill_quote"}

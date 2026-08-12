"""The /trades page and its /api/desk/trade-history endpoint (era model).

The feature's whole job is an HONEST human-readable history, so these tests
are mostly about the ways a profit number can be quietly wrong: a truncated
replay, an option multiplier applied twice, an opening leg reported as
breakeven, an mleg parent counted next to its legs, and — on the frozen
Era-1 archive — a dividend row that looks like a $0 sale.

Era 2 fills come from the ``desk_orders`` mirror (the Alpaca paper book);
Era 1 rows come from ``era1_trades`` when the cutover rename has produced
it (the tests create it via the router's registered table shape).
"""

from __future__ import annotations

from datetime import datetime

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'trades.db'}")
    # Importing the router registers the era1_* archive shapes so create_all
    # builds them (empty era-1 archive = pre-cutover state).
    import dashboard.routers.desk as desk_router
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401

    Base.metadata.create_all(get_engine())

    # The desk router memoises account reads and the dashboard caches its own
    # engine/session; a leftover from a sibling test would serve the WRONG
    # SQLite file here (conftest clears only get_store's lru_cache).
    desk_router._portfolio_cache = None
    desk_router._open_orders_cache = None
    desk_router._outcomes_live_cache = None
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


def _ts(day: int, hour: int = 15) -> str:
    return datetime(2026, 7, day, hour, 30).isoformat() + "+00:00"


_SEQ = {"n": 0}


def _seed(store, symbol, side, qty, price, ts, *, run_id="R1", kind=None,
          order_class="simple", parent=None, oid=None):
    """One era-2 fill in the desk_orders mirror."""
    import agent.trade as trade

    _SEQ["n"] += 1
    oid = oid or f"o-{_SEQ['n']}"
    store.insert("desk_orders", {
        "account": "agent", "run_id": run_id, "seq": _SEQ["n"],
        "client_order_id": (None if parent else f"{run_id}:{_SEQ['n'] % 99:02d}"),
        "alpaca_order_id": oid, "parent_order_id": parent,
        "symbol": symbol, "asset_class": trade.asset_class_of(symbol),
        "side": side.lower(), "kind": kind or ("entry" if side.lower() == "buy" else "exit"),
        "order_type": "stop" if kind == "stop" else "market", "tif": "day",
        "order_class": order_class, "qty": qty, "status": "filled",
        "filled_qty": qty, "filled_avg_price": price,
        "submitted_at": ts, "filled_at": ts}, returning=False)
    return oid


def _seed_era1(store, symbol, side, shares, price, ts_dt, *, fill_quote=None,
               dollars=None, run_id="E1"):
    from agent import occ
    mult = 100 if occ.is_option(symbol) else 1
    store.insert("era1_trades", {
        "account": "agent", "run_id": run_id, "symbol": symbol, "side": side,
        "shares": shares, "price": price,
        "dollars": round(shares * price * mult, 2) if dollars is None else dollars,
        "fill_quote": fill_quote, "ts": ts_dt}, returning=False)


def _hist(client, **params):
    r = client.get("/api/desk/trade-history", params=params)
    assert r.status_code == 200, r.text
    return r.json()


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

def test_empty_book_returns_empty_history(client):
    d = _hist(client)
    assert d["rows"] == []
    assert d["realized_pnl"] == 0.0
    assert d["era1_realized"] == 0.0 and d["era2_realized"] == 0.0
    assert d["closing_fills"] == 0
    assert d["total"] == 0


# ── the core profit math (era 2, desk_orders mirror) ──

def test_round_trip_profit_and_open_legs_are_null(client, store):
    """A sell realizes against AVERAGE cost; the buys that opened realize
    nothing and must be null, never 0.0 (0.0 paints green)."""
    _seed(store, "XYZ", "buy", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "buy", 10, 110.0, _ts(7))
    _seed(store, "XYZ", "sell", 20, 120.0, _ts(8))

    rows = _hist(client)["rows"]
    sells = [r for r in rows if r["side"] == "SELL"]
    buys = [r for r in rows if r["side"] == "BUY"]
    assert len(sells) == 1 and len(buys) == 2
    assert sells[0]["realized"] == pytest.approx(300.0, abs=0.01)
    assert sells[0]["era"] == 2
    assert all(b["realized"] is None for b in buys)


def test_pnl_identical_at_every_display_limit(client, store):
    """THE TRUNCATION TRAP. Average cost is path-dependent, so replaying only
    the newest rows books each SELL against a flat book and silently reports
    a different (wrong) profit. `limit` must slice the OUTPUT, never the
    replay input."""
    _seed(store, "XYZ", "buy", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "buy", 10, 110.0, _ts(7))
    _seed(store, "XYZ", "sell", 20, 120.0, _ts(8))

    for limit in (1, 2, 500):
        d = _hist(client, limit=limit)
        sell = [r for r in d["rows"] if r["side"] == "SELL"]
        assert len(sell) == 1, f"limit={limit} should still show the newest row"
        assert sell[0]["realized"] == pytest.approx(300.0, abs=0.01), limit
        assert d["realized_pnl"] == pytest.approx(300.0, abs=0.01), limit
        assert d["total"] == 3


def test_option_round_trip_applies_multiplier_once(client, store):
    """x100 is applied inside the replay — never again downstream."""
    sym = "NVDA270116C00200000"
    _seed(store, sym, "buy", 2, 5.0, _ts(6))
    _seed(store, sym, "sell", 2, 6.5, _ts(8))

    rows = _hist(client)["rows"]
    sell = [r for r in rows if r["side"] == "SELL"][0]
    assert sell["realized"] == pytest.approx(300.0, abs=0.01)  # not 3, not 30000
    assert sell["label"] == "NVDA $200C 2027-01-16"
    assert sell["underlying"] == "NVDA"


def test_sell_to_open_then_buy_to_close(client, store):
    """`side` does NOT mean open-vs-close. A SELL can open a short option leg
    and realize nothing; the later BUY carries the profit."""
    sym = "AMD260821C00550000"
    _seed(store, sym, "sell", 1, 36.0, _ts(6))   # sell to open
    _seed(store, sym, "buy", 1, 20.0, _ts(9))    # buy to close, cheaper

    rows = _hist(client)["rows"]
    opened = [r for r in rows if r["side"] == "SELL"][0]
    closed = [r for r in rows if r["side"] == "BUY"][0]
    assert opened["realized"] is None
    assert closed["realized"] == pytest.approx(1600.0, abs=0.01)


def test_duplicate_equity_exit_realizes_nothing(client, store):
    """A duplicate exit on a flat book books no P&L (the equity long-only
    clamp) — it must come back null, not a phantom short."""
    _seed(store, "XYZ", "buy", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "sell", 10, 110.0, _ts(8))
    _seed(store, "XYZ", "sell", 10, 110.0, _ts(8, hour=16))

    d = _hist(client)
    sells = [r for r in d["rows"] if r["side"] == "SELL"]
    realized = sorted(r["realized"] for r in sells if r["realized"] is not None)
    assert realized == [pytest.approx(100.0, abs=0.01)]
    assert sum(1 for r in sells if r["realized"] is None) == 1
    assert d["realized_pnl"] == pytest.approx(100.0, abs=0.01)


def test_mleg_parent_shell_not_a_row(client, store):
    """An mleg PARENT carries the aggregate fill fields but its LEGS are the
    per-contract fills — counting both would double every spread."""
    sym = "NVDA270116C00200000"
    parent = _seed(store, sym, "buy", 1, 3.10, _ts(6), order_class="mleg",
                   oid="parent-1")
    _seed(store, sym, "buy", 1, 5.0, _ts(6), order_class="mleg",
          parent=parent, oid="leg-1")

    d = _hist(client)
    assert d["total"] == 1                       # the parent shell is not a fill
    assert d["rows"][0]["dollars"] == pytest.approx(500.0)  # leg px × 100


def test_stop_fill_is_kind_stop(client, store):
    _seed(store, "XYZ", "buy", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "sell", 10, 90.0, _ts(8), kind="stop")

    rows = _hist(client)["rows"]
    stop = [r for r in rows if r["side"] == "SELL"][0]
    assert stop["kind"] == "stop"
    assert stop["realized"] == pytest.approx(-100.0, abs=0.01)


# ── the frozen Era-1 archive ──

def _dt(day: int, hour: int = 15) -> datetime:
    return datetime(2026, 5, day, hour, 30)  # naive UTC, pre-cutover era


def test_era1_rows_and_totals(client, store):
    """Era-1 rows ride the same replay with the OLD conventions intact:
    dividend rows are not $0.00 sales, split rows are unit shifts dated by
    their own effective date, and the era totals split out."""
    _seed_era1(store, "ABC", "BUY", 10, 100.0, _dt(6))
    _seed_era1(store, "ABC", "SELL", 0.0, 0.0, _dt(8), dollars=12.30,
               fill_quote={"src": "dividend", "ex_date": "2026-05-08"})
    store.insert("era1_trades", {
        "account": "agent", "run_id": "settlement", "symbol": "ABC",
        "side": "BUY", "shares": 10, "price": 0.0, "dollars": 0.0,
        "fill_quote": {"src": "split_adjustment",
                       "execution_date": "2026-05-09", "ratio": 2.0},
        "ts": _dt(9)}, returning=False)
    _seed_era1(store, "ABC", "SELL", 20, 55.0, _dt(10))
    # era-2 activity on the current book as well
    _seed(store, "XYZ", "buy", 10, 100.0, _ts(6))
    _seed(store, "XYZ", "sell", 10, 130.0, _ts(8))

    d = _hist(client)
    e1 = [r for r in d["rows"] if r["era"] == 1]
    e2 = [r for r in d["rows"] if r["era"] == 2]
    assert len(e1) == 4 and len(e2) == 2

    div = [r for r in e1 if r["kind"] == "dividend"]
    assert len(div) == 1 and div[0]["realized"] is None
    split = [r for r in e1 if r["kind"] == "split"]
    assert len(split) == 1 and split[0]["realized"] is None
    assert split[0]["date"] == "2026-05-09"       # effective, not booked
    sell = [r for r in e1 if r["side"] == "SELL" and r["kind"] == "trade"][0]
    # 20 × 55 vs the split-rebased avg cost 50 → +100
    assert sell["realized"] == pytest.approx(100.0, abs=0.01)

    assert d["era1_realized"] == pytest.approx(100.0, abs=0.01)
    assert d["era2_realized"] == pytest.approx(300.0, abs=0.01)
    assert d["realized_pnl"] == pytest.approx(400.0, abs=0.01)
    assert d["closing_fills"] == 2

    # newest-displayed-date first: the era-2 (July) rows precede era-1 (May)
    dates = [r["date"] for r in d["rows"]]
    assert dates == sorted(dates, reverse=True)


def test_era1_replay_is_independent_of_era2(client, store):
    """The two eras are separate books: an era-1 lot must never absorb an
    era-2 exit (the frozen archive closed at cutover by construction)."""
    _seed_era1(store, "XYZ", "BUY", 10, 100.0, _dt(6))
    _seed(store, "XYZ", "sell", 10, 150.0, _ts(8))  # era-2 sell, flat era-2 book

    d = _hist(client)
    e2_sell = [r for r in d["rows"] if r["era"] == 2][0]
    assert e2_sell["realized"] is None       # no era-2 lot → clamps, no P&L
    assert d["realized_pnl"] == 0.0


# ── the clock: a fill's stamp has to survive the trip to the browser ──

@pytest.mark.parametrize("raw,expected", [
    # PostgREST hands a `timestamp without time zone` back as naked text
    ("2026-07-06T01:30:00", "2026-07-06T01:30:00+00:00"),
    ("2026-07-06T01:30:00.5", "2026-07-06T01:30:00.500000+00:00"),
    # already unambiguous — leave the wire format alone
    ("2026-07-06T01:30:00+00:00", "2026-07-06T01:30:00+00:00"),
    ("2026-07-06T01:30:00Z", "2026-07-06T01:30:00Z"),
    ("2026-07-05T21:30:00-04:00", "2026-07-05T21:30:00-04:00"),
    # a calendar day is not an instant; unparseable text is not ours to guess
    ("2026-07-06", "2026-07-06"),
    ("not a timestamp", "not a timestamp"),
])
def test_naked_timestamps_are_stamped_utc(raw, expected):
    """THE ZONE TRAP. A zone-less date-time is read as LOCAL time by the
    browser (ES spec), so passing one through would render a 19:25 UTC fill
    at 19:25 on the reader's own clock — hours off the broker stamp these
    pages exist to let them check."""
    import dashboard.routers.desk as desk_router

    assert desk_router._stamp_utc(raw) == expected


def test_every_fill_timestamp_carries_an_offset(client, store):
    """Both eras, both endpoints. Era-1 rows come off a naive DateTime column
    and era-2 rows off Alpaca's own ISO text — the API has to land them in
    one unambiguous form, or the desk cannot honestly claim an ET time."""
    _seed(store, "XYZ", "buy", 10, 100.0, _ts(6))
    _seed_era1(store, "ABC", "BUY", 5, 50.0, _dt(6))

    stamps = ([r["t"] for r in _hist(client)["rows"]]
              + [r["t"] for r in client.get("/api/desk/trades").json()])
    assert len(stamps) == 4
    for t in stamps:
        parsed = datetime.fromisoformat(str(t).replace("Z", "+00:00"))
        assert parsed.tzinfo is not None, t


def test_evening_fill_is_dated_by_its_ET_session(client, store):
    """01:30 UTC is 9:30 PM the PREVIOUS evening in New York. The row dates
    to the ET session it traded in while ``t`` keeps the true instant — a
    UTC-dated row would send a reader to the wrong day's tape."""
    _seed(store, "XYZ", "buy", 10, 100.0, "2026-07-06T01:30:00+00:00")

    row = _hist(client)["rows"][0]
    assert row["date"] == "2026-07-05"
    assert row["t"] == "2026-07-06T01:30:00+00:00"


# ── /api/desk/trades: the receipts list stays a BARE LIST ──

def test_trades_endpoint_shape(client, store):
    """/api/desk/trades stays a BARE LIST — desk.js and symbol.js index it
    directly. Era-tagged rows share ONE key set across eras."""
    _seed(store, "XYZ", "buy", 10, 100.0, _ts(6))
    _seed_era1(store, "ABC", "BUY", 5, 50.0, _dt(6),
               fill_quote={"bid": 49.9, "ask": 50.1})
    r = client.get("/api/desk/trades", params={"limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    expected = {"id", "era", "t", "symbol", "side", "shares", "price",
                "dollars", "rationale", "run_id", "kind", "order_class",
                "fill_quote"}
    assert set(body[0]) == expected
    assert set(body[1]) == expected
    by_era = {row["era"]: row for row in body}
    assert by_era[2]["symbol"] == "XYZ" and by_era[2]["fill_quote"] is None
    assert by_era[1]["symbol"] == "ABC" and by_era[1]["fill_quote"]["bid"] == 49.9

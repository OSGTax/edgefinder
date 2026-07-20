"""v9.13.0 review-fix regressions.

Locks down the fixes from the full-system review: settlement coverage as a
drawn-down ALLOCATION (no double assignment), spread max-loss reservation,
buy-to-close releasing the CSP reserve, the outcomes window living in the
query (no newest-200 cap), _realized_pnl's equity clamps, the kill-breach
window ending at a closed pick's exit, the non-destructive positions
rebuild, adjusted-OCC fail-closed, slippage flooring, the split-aware
close band, dispatcher edge-trigger vs failed POSTs, and the public
movers/holding-stats split guards.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from agent import occ
from agent.ledger import _et_date, _utcnow

# The ledger reasons in ET (a US-market book); a UTC container clock in the
# 8pm-midnight ET window makes date.today() one day ahead of the code's
# notion of "today" and breaks the strict split-rebase window check. Anchor
# the tests to the SAME ET date the code uses.
TODAY = date.fromisoformat(_et_date(_utcnow()))
FUT = TODAY + timedelta(days=45)
PAST = TODAY - timedelta(days=3)


def C(und, strike, expiry=FUT):
    return occ.build(und, expiry, "C", strike)


def P(und, strike, expiry=FUT):
    return occ.build(und, expiry, "P", strike)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'rf.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401 — daily_bars/ticker_splits
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def q(bid, ask):
    return {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4),
            "t": "x", "src": "test"}


def _row(store, symbol, side, shares, price, dollars=None, run_id="T", ts=None):
    from agent import ledger
    mult = 100 if occ.is_option(symbol) else 1
    store.insert("desk_trades", {
        "account": "agent", "symbol": symbol, "side": side, "shares": shares,
        "price": price,
        "dollars": dollars if dollars is not None else shares * price * mult,
        "run_id": run_id, "ts": ts or ledger._utcnow()}, returning=False)


def _seed_close(store, symbol, day, px):
    store.insert("daily_bars", {"symbol": symbol, "date": day, "open": px,
                                "high": px, "low": px, "close": px,
                                "volume": 1e6, "source": "test"}, returning=False)


def _patch_underlying(monkeypatch, prices):
    from agent import ledger
    monkeypatch.setattr(ledger, "_live_mids",
                        lambda syms: {s: prices[s] for s in syms if s in prices})
    monkeypatch.setattr(ledger, "_latest_close", lambda s: prices.get(s))


# ── settle(): coverage is an allocation, not a per-leg check ─────────


def test_settle_covered_call_plus_vertical_assigns_once(store, monkeypatch):
    """The critical review repro: 100 shares + short C180 (covered call) +
    C200/C210 vertical. The old static per-leg checks share-assigned BOTH
    shorts against the same 100 shares — two 100-share settlement SELLs on
    a 100-share position, +$22k of fabricated cash on the append-only
    ledger. The allocation must book exactly ONE equity assignment."""
    from agent import ledger
    _row(store, "NVDA", "BUY", 100, 150.0)          # -15,000
    _row(store, C("NVDA", 180, PAST), "SELL", 1, 5.0)   # +500
    _row(store, C("NVDA", 210, PAST), "BUY", 1, 1.0)    # -100
    _row(store, C("NVDA", 200, PAST), "SELL", 1, 6.0)   # +600
    ledger.rebuild_positions(store)
    _seed_close(store, "NVDA", PAST, 220.0)         # everything ITM
    _patch_underlying(monkeypatch, {"NVDA": 220.0})
    ledger.settle(store)
    # exactly one equity settlement SELL, sized to the shares that exist
    eq_sells = [t for t in store.select(
        "desk_trades", filters={"symbol": "NVDA", "run_id": "settlement"})
        if t["side"] == "SELL"]
    assert len(eq_sells) == 1 and float(eq_sells[0]["shares"]) == 100.0
    pos = {r["symbol"]: r["shares"] for r in store.select("desk_positions")}
    assert "NVDA" not in pos and not any(occ.is_option(s) for s in pos)
    # honest cash: 100k -15k +500 -100 +600 (entries) +1000 (long C210)
    # -4000 (C180 spread-covered cash-settle) +20,000 (C200 assigned @200)
    assert ledger.cash(store) == 103_000.0


def test_settle_csp_plus_put_vertical_assigns_once(store, monkeypatch):
    """Put twin: CSP P700 + P690/P650 vertical. Pooled counting classified
    BOTH shorts cash-backed — two 100-share BUYs, cash driven deep negative
    and 200 unchosen shares. The allocation books one assignment."""
    from agent import ledger
    _row(store, P("SPY", 700, PAST), "SELL", 1, 10.0)   # +1000
    _row(store, P("SPY", 690, PAST), "SELL", 1, 9.0)    # +900
    _row(store, P("SPY", 650, PAST), "BUY", 1, 2.0)     # -200
    ledger.rebuild_positions(store)
    _seed_close(store, "SPY", PAST, 640.0)          # all ITM
    _patch_underlying(monkeypatch, {"SPY": 640.0})
    ledger.settle(store)
    pos = {r["symbol"]: r["shares"] for r in store.select("desk_positions")}
    assert pos.get("SPY") == 100.0                  # ONE assignment, not two
    eq_buys = [t for t in store.select(
        "desk_trades", filters={"symbol": "SPY", "run_id": "settlement"})
        if t["side"] == "BUY"]
    assert len(eq_buys) == 1 and float(eq_buys[0]["shares"]) == 100.0
    # 100k +1000 +900 -200 (entries) +1000 (long P650 iv 10)
    # -5000 (P690 spread-covered iv 50) -70,000 (P700 assigned)
    assert ledger.cash(store) == 27_700.0


def test_settle_expired_spread_not_misread_by_later_short(store, monkeypatch):
    """An expired put vertical plus a LATER-dated live short put: the old
    global latest-short-expiry filter discounted the expired long leg, so
    the spread-covered short was misclassified cash-backed and share-
    assigned. Eligibility is per-short now."""
    from agent import ledger
    _row(store, P("SPY", 690, PAST), "SELL", 1, 9.0)    # +900 (vertical short)
    _row(store, P("SPY", 650, PAST), "BUY", 1, 2.0)     # -200 (vertical long)
    _row(store, P("SPY", 400, FUT), "SELL", 1, 1.0)     # +100 (live CSP)
    ledger.rebuild_positions(store)
    _seed_close(store, "SPY", PAST, 640.0)
    _patch_underlying(monkeypatch, {"SPY": 640.0})
    ledger.settle(store)
    pos = {r["symbol"]: r["shares"] for r in store.select("desk_positions")}
    assert "SPY" not in pos                        # NO phantom assignment
    assert pos.get(P("SPY", 400, FUT)) == -1       # live short untouched
    # 100k +900 -200 +100 (entries) +1000 (long P650) -5000 (P690 covered)
    assert ledger.cash(store) == 96_800.0


# ── CSP reservation semantics ────────────────────────────────────────


def test_buy_to_close_short_put_releases_reservation(store):
    """The 898 fix: with cash deployed and the put ITM, buy-to-close must
    book — the fill itself releases the reservation. The old PRE-fill gate
    rejected exactly the close/roll the 5-DTE discipline demands."""
    from agent import ledger
    sym = P("SPY", 700)
    assert ledger.record_trade(store, symbol=sym, side="SELL", shares=1,
                               price=10.0, fill_quote=q(9.9, 10.0))["ok"]
    assert ledger.record_trade(store, symbol="NVDA", side="BUY", shares=125,
                               price=200.0, fill_quote=q(199.9, 200.0))["ok"]
    # free cash is now ~$6k; the buyback costs $8k — must still book
    r = ledger.record_trade(store, symbol=sym, side="BUY", shares=1,
                            price=80.0, fill_quote=q(79.9, 80.0))
    assert r["ok"], r
    pos = {p["symbol"] for p in store.select("desk_positions")}
    assert sym not in pos
    assert ledger.free_cash(store) == ledger.cash(store)  # reservation gone


def test_buying_covering_long_put_releases_reservation(store):
    from agent import ledger
    short = P("SPY", 700)
    assert ledger.record_trade(store, symbol=short, side="SELL", shares=1,
                               price=10.0, fill_quote=q(9.9, 10.0))["ok"]
    assert ledger.record_trade(store, symbol="NVDA", side="BUY", shares=125,
                               price=200.0, fill_quote=q(199.9, 200.0))["ok"]
    # $6.5k long put converts the CSP (70k reserved) to a 10-wide spread
    r = ledger.record_trade(store, symbol=P("SPY", 690), side="BUY", shares=1,
                            price=65.0, fill_quote=q(64.9, 65.0))
    assert r["ok"], r
    assert ledger.free_cash(store) == pytest.approx(
        ledger.cash(store) - (700 - 690) * 100)


def test_multi_strike_csp_reservation_not_max_strike_inflated():
    """F9: two uncovered short puts at different strikes reserve the SUM of
    their strikes, not max-strike x total contracts."""
    from agent import ledger
    positions = {P("SPY", 700): -1.0, P("SPY", 500): -1.0}
    assert ledger._csp_reserved(positions) == (700 + 500) * 100


# ── outcomes(): the window lives in the query ────────────────────────


def test_outcomes_window_survives_200_plus_decisions(store):
    from agent import ledger
    now = ledger._utcnow()
    old_ts = now - timedelta(days=5)
    store.insert("desk_decisions", {
        "account": "agent", "run_id": "OLD", "ts": old_ts,
        "picks": [{"symbol": "NVDA", "action": "buy"}]}, returning=False)
    _row(store, "NVDA", "BUY", 10, 100.0, run_id="OLD", ts=old_ts)
    rows = [{"account": "agent", "run_id": f"R{i}",
             "ts": now - timedelta(seconds=3600 - i), "picks": []}
            for i in range(210)]
    store.insert("desk_decisions", rows, returning=False)
    out = ledger.outcomes(store, days=30)
    assert any(r["run_id"] == "OLD" for r in out["runs"])
    by_run = ledger.outcomes(store, days=30, run_id="OLD")
    assert [r["run_id"] for r in by_run["runs"]] == ["OLD"]


# ── _realized_pnl equity clamps ──────────────────────────────────────


def test_realized_pnl_duplicate_exit_books_zero_and_stays_in_sync():
    from agent import ledger
    ts = datetime(2026, 7, 1, 14, 0)

    def t(side, px, rid, i):
        return {"symbol": "XYZ", "side": side, "shares": 10, "price": px,
                "dollars": 10 * px, "run_id": rid,
                "ts": ts + timedelta(minutes=i)}

    trades = [t("BUY", 100.0, "r1", 0), t("SELL", 110.0, "r2", 1),
              t("SELL", 110.0, "r2b", 2),   # the duplicate writer-race row
              t("BUY", 120.0, "r3", 3), t("SELL", 130.0, "r4", 4)]
    by_run, by_sym = ledger._realized_pnl(trades)
    assert by_run.get(("r2", "XYZ")) == pytest.approx(100.0)
    assert ("r2b", "XYZ") not in by_run      # duplicate books ZERO
    assert by_run.get(("r4", "XYZ")) == pytest.approx(100.0)
    assert by_sym["XYZ"] == pytest.approx(200.0)


# ── grade(): kill window ends at a closed pick's exit ────────────────


def test_kill_breach_window_ends_at_exit(store):
    from agent import ledger
    now = ledger._utcnow()
    run_ts = now - timedelta(days=10)
    store.insert("desk_decisions", {
        "account": "agent", "run_id": "K1", "ts": run_ts,
        "picks": [{"symbol": "WIN", "action": "buy", "horizon_days": 5,
                   "kill": "closes below $90"}]}, returning=False)
    _row(store, "WIN", "BUY", 10, 100.0, run_id="K1", ts=run_ts)
    _row(store, "WIN", "SELL", 10, 112.0, run_id="K1",
         ts=now - timedelta(days=7))
    _seed_close(store, "WIN", (now - timedelta(days=9)).date(), 104.0)
    _seed_close(store, "WIN", (now - timedelta(days=8)).date(), 108.0)
    # a post-exit dip that must NOT stamp a discipline failure
    _seed_close(store, "WIN", (now - timedelta(days=2)).date(), 85.0)
    out = ledger.grade(store, days=365)
    row = next(r for r in out["rows"] if r["run_id"] == "K1")
    assert row["status"] == "closed" and row["since_pct"] == pytest.approx(12.0)
    assert row["kill_breached"] is False


# ── rebuild_positions: non-destructive, marks survive fills ──────────


def test_rebuild_preserves_last_price_across_fills(store):
    from agent import ledger
    assert ledger.record_trade(store, symbol="NVDA", side="BUY", shares=10,
                               price=120.0, fill_quote=q(119.9, 120.0))["ok"]
    ledger.mark(store, prices={"NVDA": 130.0})
    assert ledger.record_trade(store, symbol="NVDA", side="BUY", shares=5,
                               price=121.0, fill_quote=q(120.9, 121.0))["ok"]
    row = store.select("desk_positions", filters={"symbol": "NVDA"})[0]
    assert float(row["shares"]) == 15.0
    assert row["last_price"] is not None and float(row["last_price"]) == 130.0


def test_grade_degrades_unmarked_position(store):
    """A projection row with no mark yet prices at cost — grade must flag
    it degraded (fake-flat), not write since_pct=0 as a clean fact."""
    from agent import ledger
    now = ledger._utcnow()
    store.insert("desk_decisions", {
        "account": "agent", "run_id": "D1", "ts": now - timedelta(days=1),
        "picks": [{"symbol": "NVDA", "action": "buy"}]}, returning=False)
    _row(store, "NVDA", "BUY", 10, 100.0, run_id="D1",
         ts=now - timedelta(days=1))
    ledger.rebuild_positions(store)   # projection exists, but never marked
    out = ledger.grade(store, days=30)
    row = next(r for r in out["rows"] if r["run_id"] == "D1")
    assert row["degraded"] is True and row["since_pct"] is None


# ── fail-closed guards on the fill path ──────────────────────────────


def test_adjusted_occ_symbol_fails_closed(store):
    from agent import ledger
    adj = f"AAPL1{FUT.strftime('%y%m%d')}C00150000"
    assert occ.is_adjusted_occ(adj) and not occ.is_option(adj)
    assert not occ.is_adjusted_occ(C("NVDA", 200))   # standard stays standard
    assert not occ.is_adjusted_occ("NVDA")
    r = ledger.record_trade(store, symbol=adj, side="BUY", shares=1,
                            price=5.0, fill_quote=q(4.9, 5.0))
    assert not r["ok"] and "ADJUSTED" in r["error"]


def test_negative_slippage_floored_to_zero(store, monkeypatch):
    from tests.test_live_fill import FakeBroker, _patch_broker
    from agent import ledger
    _patch_broker(monkeypatch, FakeBroker())
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1,
                         slippage_bp=-500.0)
    assert r["ok"], r
    assert r["price"] == 130.2                      # ask exactly, no improvement
    assert r["fill_quote"]["slippage_bp"] == 0.0


def test_close_soon_gate_exempts_buy_to_close(store, monkeypatch):
    from tests.test_live_fill import FakeBroker, _patch_broker
    from agent import ledger
    sym = P("SPY", 700)
    _row(store, sym, "SELL", 1, 10.0)
    ledger.rebuild_positions(store)
    _patch_broker(monkeypatch, FakeBroker(close_soon=True, bid=79.9, ask=80.0))
    r = ledger.live_fill(store, symbol=sym, side="buy", shares=1)
    assert r["ok"], r                               # an exit, not a new position
    # a plain equity BUY is still refused near the close
    r2 = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert not r2["ok"] and "close in <15m" in r2["error"]


def test_close_band_rebases_reference_across_split(store, monkeypatch):
    """Split day: the stored close is pre-split — the raw reference vetoed
    every same-day fill on the name until the nightly refresh."""
    from tests.test_live_fill import FakeBroker, _patch_broker
    from agent import ledger
    yesterday = TODAY - timedelta(days=1)
    _seed_close(store, "NVDA", yesterday, 1200.0)   # pre-split close
    store.insert("ticker_splits", {"symbol": "NVDA",
                                   "execution_date": str(TODAY),
                                   "split_from": 1, "split_to": 10},
                 returning=False)
    _patch_broker(monkeypatch, FakeBroker(bid=119.9, ask=120.1))
    r = ledger.live_fill(store, symbol="NVDA", side="buy", shares=1)
    assert r["ok"], r
    assert any("rebased" in w for w in r.get("warnings", []))


# ── dispatcher: edge-trigger vs FAILED dispatches; day cap countable ──


def test_failed_dispatch_does_not_silence_a_trip():
    from agent.streamer import dispatch_reason
    now = datetime(2026, 7, 15, 15, 0)
    watches = [{"id": 5, "status": "tripped", "symbol": "AMD",
                "kind": "below", "level": 100.0,
                "tripped_at": now - timedelta(minutes=30)}]
    failed = [{"ts": now - timedelta(minutes=20), "status": "failed"}]
    d = dispatch_reason([], watches, failed, now=now)
    assert d is not None and d["watch_ids"] == [5]
    sent = [{"ts": now - timedelta(minutes=20), "status": "sent"}]
    assert dispatch_reason([], watches, sent, now=now) is None  # edge held


def test_dispatch_day_cap_binds():
    from agent.streamer import DISPATCH_MAX_PER_DAY, dispatch_reason
    now = datetime(2026, 7, 15, 20, 0)   # 15:00-16:00 ET, same ET day
    wakes = [{"id": 1, "at": now - timedelta(hours=1),
              "honored_run_id": None, "dispatch_count": 0}]
    mk = lambda n: [{"ts": now - timedelta(minutes=6 * (i + 1)),
                     "status": "sent"} for i in range(n)]
    assert dispatch_reason(wakes, [], mk(DISPATCH_MAX_PER_DAY - 1), now=now)
    assert dispatch_reason(wakes, [], mk(DISPATCH_MAX_PER_DAY), now=now) is None


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


def test_holding_stats_rebases_split(client):
    import agent.data as agent_data
    from agent.models import ACCOUNT, DeskPosition
    sess = agent_data.session_factory()()
    try:
        sess.add(DeskPosition(account=ACCOUNT, symbol="SPL", shares=100,
                              avg_price=120.0))
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

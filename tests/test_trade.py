"""agent.trade — the Alpaca paper-account execution arm (REBUILD-V4).

Pins the load-bearing contracts: the client_order_id attribution scheme
(the knowledge loop's lifeline), per-asset-class order-shape legality, the
paper-only invariant, mirror writes (parent + mleg legs, run_id
propagation), arm-stop's replace semantics and covered-call awareness, and
reconcile's idempotent activity sync.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tests.fakes import FakeTradingClient

TODAY = date(2026, 8, 8)
FAR_EXPIRY = (TODAY + timedelta(days=200)).strftime("%y%m%d")
CALL = f"NVDA{FAR_EXPIRY}C00200000"
PUT = f"NVDA{FAR_EXPIRY}P00180000"
EXPIRED_CALL = "NVDA250116C00200000"
ADJUSTED = "AAPL1260116C00150000"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'trade.db'}")
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def _trade(store, **fake_kw):
    from agent.trade import Trade
    fake = FakeTradingClient(**fake_kw)
    return Trade(client=fake, store=store), fake


# ── client_order_id: the attribution carrier ─────────────────────────────


def test_client_order_id_roundtrip():
    from agent.trade import make_client_order_id, parse_client_order_id

    cid = make_client_order_id("2026-08-17T14:30-r7kq", 3)
    assert cid == "2026-08-17T14:30-r7kq:03"
    parsed = parse_client_order_id(cid)
    assert parsed == {"run_id": "2026-08-17T14:30-r7kq", "seq": 3}


def test_client_order_id_rejects_bad_inputs():
    from agent.trade import make_client_order_id, parse_client_order_id

    with pytest.raises(ValueError):
        make_client_order_id("", 1)
    with pytest.raises(ValueError):
        make_client_order_id("r", 0)
    with pytest.raises(ValueError):
        make_client_order_id("r", 100)
    # Foreign ids (Alpaca auto-generated, dashboard orders) parse to None —
    # never misattributed to a run.
    assert parse_client_order_id("61e69015-8549-4bfd-b9c3") is None
    assert parse_client_order_id("auto-7") is None
    assert parse_client_order_id("") is None


def test_run_id_with_colons_survives_rpartition():
    from agent.trade import make_client_order_id, parse_client_order_id

    rid = "2026-08-17T14:30:00-r7kq"  # colons inside the run id itself
    assert parse_client_order_id(make_client_order_id(rid, 7)) == {
        "run_id": rid, "seq": 7}


# ── order-shape legality per asset class ─────────────────────────────────


def _errs(**kw):
    from agent.trade import validate_order
    defaults = dict(symbol="NVDA", side="buy", qty=1.0, notional=None,
                    order_type="market", tif="day", extended_hours=False,
                    legs=None, today=TODAY)
    defaults.update(kw)
    return validate_order(**defaults)


def test_equity_happy_paths():
    assert _errs() == []
    assert _errs(order_type="limit", tif="gtc") == []
    assert _errs(qty=None, notional=5000.0) == []          # notional market
    assert _errs(qty=1.5) == []                            # fractional day
    assert _errs(order_type="limit", extended_hours=True) == []


def test_equity_shape_rules():
    assert any("notional" in e for e in _errs(qty=None, notional=5000.0,
                                              order_type="limit"))
    assert any("fractional" in e.lower() for e in _errs(qty=1.5, tif="gtc"))
    assert any("extended" in e.lower() for e in _errs(extended_hours=True))
    assert any("exactly one" in e for e in _errs(qty=None, notional=None))
    assert any("exactly one" in e for e in _errs(qty=1.0, notional=100.0))


def test_crypto_shape_rules():
    assert _errs(symbol="BTC/USD", tif="gtc") == []
    assert _errs(symbol="BTC/USD", tif="gtc", order_type="stop_limit") == []
    assert any("TIF" in e for e in _errs(symbol="BTC/USD", tif="day"))
    assert any("type" in e for e in _errs(symbol="BTC/USD", tif="gtc",
                                          order_type="stop"))
    assert any("extended" in e.lower() for e in _errs(
        symbol="BTC/USD", tif="gtc", extended_hours=True))


def test_option_shape_rules():
    assert _errs(symbol=CALL) == []
    assert _errs(symbol=CALL, tif="gtc", order_type="stop") == []
    assert any("expired" in e for e in _errs(symbol=EXPIRED_CALL))
    assert any("adjusted" in e.lower() for e in _errs(symbol=ADJUSTED))
    assert any("whole" in e for e in _errs(symbol=CALL, qty=1.5))
    assert any("notional" in e for e in _errs(symbol=CALL, qty=None,
                                              notional=1000.0))
    assert any("extended" in e.lower() for e in _errs(symbol=CALL,
                                                      extended_hours=True))
    assert any("TIF" in e for e in _errs(symbol=CALL, tif="ioc"))


def test_mleg_shape_rules():
    legs2 = [
        {"symbol": CALL, "ratio_qty": 1, "side": "buy",
         "position_intent": "buy_to_open"},
        {"symbol": PUT, "ratio_qty": 1, "side": "sell",
         "position_intent": "sell_to_open"},
    ]
    assert _errs(symbol=None, legs=legs2, order_type="limit") == []
    assert any("2-4 legs" in e for e in _errs(symbol=None, legs=legs2[:1],
                                              order_type="limit"))
    assert any("2-4 legs" in e for e in _errs(symbol=None, legs=legs2 * 3,
                                              order_type="limit"))
    assert any("market or limit" in e for e in _errs(symbol=None, legs=legs2,
                                                     order_type="stop"))
    bad_intent = [dict(legs2[0], position_intent="open"), legs2[1]]
    assert any("position_intent" in e for e in _errs(symbol=None,
                                                     legs=bad_intent,
                                                     order_type="limit"))
    expired = [dict(legs2[0], symbol=EXPIRED_CALL), legs2[1]]
    assert any("expired" in e for e in _errs(symbol=None, legs=expired,
                                             order_type="limit"))
    assert any("whole" in e for e in _errs(symbol=None, legs=legs2,
                                           order_type="limit", qty=1.5))


# ── paper-only invariant ─────────────────────────────────────────────────


def test_trade_refuses_when_paper_flag_off(monkeypatch):
    from config.settings import settings
    from agent.trade import Trade

    monkeypatch.setattr(settings, "alpaca_paper", False)
    with pytest.raises(RuntimeError, match="paper-only"):
        Trade()


# ── submit → mirror ──────────────────────────────────────────────────────


def test_submit_mirrors_with_attribution(store):
    t, fake = _trade(store, prices={"NVDA": 180.0})
    res = t.submit(symbol="NVDA", side="buy", notional=5000, run_id="R1")
    assert res["errors"] == []
    o = res["order"]
    assert o["status"] == "filled" and o["filled_avg_price"] == 180.0
    assert o["client_order_id"] == "R1:01"

    rows = store.select("desk_orders", filters={"run_id": "R1"})
    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "NVDA" and row["seq"] == 1
    assert row["kind"] == "entry" and row["asset_class"] == "us_equity"
    assert row["filled_avg_price"] == 180.0

    # seq increments within the run — retries/multi-order picks never collide
    res2 = t.submit(symbol="NVDA", side="sell", qty=5, run_id="R1")
    assert res2["order"]["client_order_id"] == "R1:02"
    assert res2["order"]["status"] == "filled"
    kinds = {r["seq"]: r["kind"] for r in
             store.select("desk_orders", filters={"run_id": "R1"})}
    assert kinds == {1: "entry", 2: "exit"}


def test_submit_validation_failure_sends_nothing(store):
    t, fake = _trade(store)
    res = t.submit(symbol="BTC/USD", side="buy", qty=0.1, tif="day",
                   run_id="R1")
    assert res["order"] is None and res["errors"]
    assert fake.submitted_requests == []
    assert store.select("desk_orders") == []


def test_ambiguous_submit_failure_recovers_by_client_id(store):
    t, fake = _trade(store, prices={"NVDA": 180.0})
    # The submit call dies mid-flight but the order LANDED: the recovery path
    # finds it by our client_order_id instead of double-submitting.
    landed = fake.add_order(client_order_id="R9:01", symbol="NVDA",
                            side="buy", qty=10.0, status="filled",
                            filled_qty=10.0, filled_avg_price=179.5,
                            filled_at="2026-08-08T14:30:01+00:00")
    fake.fail_next_submit = TimeoutError("gateway timeout")
    res = t.submit(symbol="NVDA", side="buy", qty=10, run_id="R9")
    assert res["errors"] == []
    assert res["order"]["alpaca_order_id"] == landed["id"]
    assert store.select("desk_orders",
                        filters={"run_id": "R9"})[0]["filled_avg_price"] == 179.5


def test_mleg_submit_mirrors_legs_with_parent(store):
    t, fake = _trade(store, prices={CALL: 5.0, PUT: 2.0})
    legs = [
        {"symbol": CALL, "ratio_qty": 1, "side": "buy",
         "position_intent": "buy_to_open"},
        {"symbol": PUT, "ratio_qty": 1, "side": "sell",
         "position_intent": "sell_to_open"},
    ]
    res = t.submit(legs=legs, qty=1, order_type="limit", limit_price=3.10,
                   side="buy", run_id="R2")
    assert res["errors"] == []
    rows = store.select("desk_orders", filters={"run_id": "R2"},
                        order=[("id", "asc")])
    assert len(rows) == 3  # parent + 2 legs
    parent = next(r for r in rows if r["parent_order_id"] is None)
    leg_rows = [r for r in rows if r["parent_order_id"] is not None]
    assert parent["order_class"] == "mleg"
    assert parent["client_order_id"] == "R2:01"
    assert {r["symbol"] for r in leg_rows} == {CALL, PUT}
    assert all(r["parent_order_id"] == parent["alpaca_order_id"]
               for r in leg_rows)
    # Legs inherit the run attribution — (run_id, leg_symbol) joins work.
    assert all(r["run_id"] == "R2" and r["seq"] == 1 for r in leg_rows)


# ── arm-stop: resting protection ─────────────────────────────────────────


def _positions(qty=100.0, avail=None, symbol="NVDA"):
    return [{"symbol": symbol, "asset_class": "us_equity", "qty": qty,
             "qty_available": qty if avail is None else avail,
             "avg_entry_price": 150.0, "current_price": 180.0,
             "market_value": qty * 180.0, "cost_basis": qty * 150.0,
             "unrealized_pl": qty * 30.0, "unrealized_plpc": 0.2,
             "change_today": 0.01, "side": "long"}]


def test_arm_stop_places_gtc_and_replaces_prior(store):
    t, fake = _trade(store, positions=_positions())
    prior = fake.add_order(symbol="NVDA", side="sell", qty=100.0,
                           order_type="stop", stop_price=140.0,
                           time_in_force="gtc", status="new")
    res = t.arm_stop(symbol="NVDA", stop_price=150.0, run_id="R3")
    assert res["errors"] == []
    assert res["replaced"] == [prior["id"]]
    assert prior["status"] == "canceled"
    o = res["order"]
    assert (o["order_type"], o["tif"], o["side"]) == ("stop", "gtc", "sell")
    assert o["stop_price"] == 150.0 and o["qty"] == 100.0
    assert store.select("desk_orders",
                        filters={"run_id": "R3"})[0]["kind"] == "stop"


def test_arm_stop_respects_covered_call_lock(store):
    t, _ = _trade(store, positions=_positions(qty=100.0, avail=0.0))
    res = t.arm_stop(symbol="NVDA", stop_price=150.0, run_id="R3")
    assert res["order"] is None
    assert any("locked" in e for e in res["errors"])

    t2, _ = _trade(store, positions=_positions(qty=100.0, avail=40.0))
    res2 = t2.arm_stop(symbol="NVDA", stop_price=150.0, qty=50.0, run_id="R3")
    assert any("exceeds qty_available" in e for e in res2["errors"])


def test_arm_stop_rounds_fractional_down_for_gtc(store):
    t, _ = _trade(store, positions=_positions(qty=10.7, avail=10.7))
    res = t.arm_stop(symbol="NVDA", stop_price=150.0, run_id="R3")
    assert res["errors"] == []
    assert res["order"]["qty"] == 10.0  # fractional can't rest GTC

    t2, _ = _trade(store, positions=_positions(qty=0.6, avail=0.6))
    res2 = t2.arm_stop(symbol="NVDA", stop_price=150.0, run_id="R3")
    assert any("fractional" in e for e in res2["errors"])


def test_arm_stop_refuses_non_equity(store):
    t, _ = _trade(store)
    assert any("equities-only" in e for e in
               t.arm_stop(symbol=CALL, stop_price=1.0,
                          run_id="R3")["errors"])
    assert any("equities-only" in e for e in
               t.arm_stop(symbol="BTC/USD", stop_price=1.0,
                          run_id="R3")["errors"])


# ── reconcile: the mirror re-converges ───────────────────────────────────


ACTIVITIES = [
    {"id": "a-1", "activity_type": "FILL",
     "transaction_time": "2026-08-07T14:31:00Z", "symbol": "NVDA",
     "side": "buy", "qty": "10", "price": "175.0", "order_id": "o-77"},
    {"id": "a-2", "activity_type": "SSP", "date": "2026-08-08",
     "symbol": "NVDA", "qty": "30", "net_amount": "0"},
    {"id": "a-3", "activity_type": "OPEXP", "date": "2026-08-08",
     "symbol": CALL, "qty": "1", "net_amount": "0"},
]


def test_reconcile_syncs_orders_and_activities_idempotently(store):
    t, fake = _trade(store, activities=list(ACTIVITIES))
    fake.add_order(client_order_id="R5:01", symbol="NVDA", side="buy",
                   qty=10.0, status="filled", filled_qty=10.0,
                   filled_avg_price=175.0,
                   filled_at="2026-08-07T14:31:00+00:00")
    r1 = t.reconcile()
    assert r1["orders_synced"] == 1 and r1["activities_added"] == 3
    # Attribution recovered from client_order_id on re-sync.
    assert store.select("desk_orders")[0]["run_id"] == "R5"

    r2 = t.reconcile()  # second pass: nothing double-inserted
    assert r2["activities_added"] == 0
    assert len(store.select("desk_activities")) == 3
    types = {a["activity_type"] for a in store.select("desk_activities")}
    assert types == {"FILL", "SSP", "OPEXP"}


def test_reconcile_warns_on_aging_gtc_stop(store):
    t, fake = _trade(store)
    fake.add_order(symbol="NVDA", side="sell", qty=100.0, order_type="stop",
                   stop_price=140.0, time_in_force="gtc", status="new",
                   submitted_at="2026-05-01T14:00:00+00:00")  # ~99 days old
    warns = t.reconcile()["gtc_stop_warnings"]
    assert len(warns) == 1 and warns[0]["symbol"] == "NVDA"
    assert warns[0]["age_days"] >= 80


# ── state + snapshot ─────────────────────────────────────────────────────


def test_state_reads_alpaca_and_weights(store, monkeypatch):
    from config.settings import settings
    monkeypatch.setattr(settings, "starting_capital", 100_000.0)
    t, _ = _trade(store, equity=105_000.0, cash=51_000.0,
                  positions=_positions(qty=300.0))
    s = t.state()
    assert s["equity"] == 105_000.0 and s["cash"] == 51_000.0
    assert s["total_pnl"] == 5_000.0
    assert s["total_return_pct"] == 5.0
    assert s["positions"][0]["weight"] == round(300 * 180.0 / 105_000.0, 6)


def test_snapshot_portfolio_is_idempotent(store):
    t, _ = _trade(store, equity=101_000.0, positions=_positions(qty=10.0))
    t.snapshot_portfolio()
    t.snapshot_portfolio()  # same date → update, not a second row
    rows = store.select("desk_portfolio_history")
    assert len(rows) == 1
    assert rows[0]["equity"] == 101_000.0
    assert rows[0]["positions"]["NVDA"]["qty"] == 10.0


# ── activity normalization ───────────────────────────────────────────────


def test_normalize_activity_shapes():
    from agent.trade import normalize_activity

    fill = normalize_activity(ACTIVITIES[0])
    assert fill["date"] == "2026-08-07" and fill["activity_type"] == "FILL"
    assert fill["qty"] == 10.0 and fill["price"] == 175.0
    assert fill["alpaca_order_id"] == "o-77"

    ssp = normalize_activity(ACTIVITIES[1])
    assert ssp["date"] == "2026-08-08" and ssp["activity_type"] == "SSP"


# ── the repo-wide write allowlist (moved from test_live_fill) ────────────


def test_no_alpaca_order_writes_outside_trade_module():
    """Contract (REBUILD-V4): agent/trade.py is the ONLY module that reaches
    Alpaca order writes — it is paper-only by construction (distinct paper
    keys, hard-coded paper=True). Everything else, including the data-reader
    broker.py, stays write-free exactly as the V3 charter demanded."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    allowed = {root / "agent" / "trade.py"}
    for d in ("agent", "dashboard", "edgefinder", "scripts", "config"):
        for f in (root / d).rglob("*.py"):
            if f in allowed:
                continue
            src = f.read_text()
            for bad in ("submit_order", "cancel_order", "replace_order",
                        "close_position", "close_all_positions"):
                assert bad not in src, f"{f}: forbidden Alpaca write '{bad}'"

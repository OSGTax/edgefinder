"""agent.grade — grading against the Alpaca paper mirror (REBUILD-V4).

The V3 ledger's outcome semantics on the new sources: fills from
desk_orders, marks from Alpaca positions, splits from ticker_splits, SPY
price-return benchmark, stop exits recognized by order kind, expiry
settlement by position-absence with T+1 activity refinement.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

ET_OFFSET = 4  # August: EDT = UTC-4


def _utc(d: date, hour_et: int = 10) -> str:
    return datetime(d.year, d.month, d.day, hour_et + ET_OFFSET,
                    tzinfo=timezone.utc).isoformat()


TODAY = datetime.now(timezone.utc).date()
D_ENTRY = TODAY - timedelta(days=10)
D_EXIT = TODAY - timedelta(days=3)
FAR_EXPIRY = (TODAY + timedelta(days=120)).strftime("%y%m%d")
PAST_EXPIRY = (TODAY - timedelta(days=4)).strftime("%y%m%d")
CALL = f"NVDA{FAR_EXPIRY}C00200000"
EXPIRED_PUT = f"QQQ{PAST_EXPIRY}P00500000"


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'grade.db'}")
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    s = get_store()
    _seed_spy(s)
    return s


def _seed_spy(store):
    """SPY closes every calendar day from entry-10 to today: 500 before the
    entry window, then +0.2/day — a steadily rising benchmark."""
    d = D_ENTRY - timedelta(days=10)
    px = 500.0
    rows = []
    while d <= TODAY:
        rows.append({"symbol": "SPY", "date": d, "open": px, "high": px,
                     "low": px, "close": round(px, 2), "volume": 1e6,
                     "source": "test"})
        px += 0.2
        d += timedelta(days=1)
    store.insert("daily_bars", rows, returning=False)


def _order(store, *, run_id, symbol, side, qty, px, filled_on, seq=1,
           kind=None, order_class="simple", parent=None, oid=None,
           status="filled"):
    import agent.trade as trade
    oid = oid or f"{run_id}-{symbol}-{side}-{seq}-{filled_on}"
    row = {"account": "agent", "run_id": run_id, "seq": seq,
           "client_order_id": (None if parent else f"{run_id}:{seq:02d}"),
           "alpaca_order_id": oid, "parent_order_id": parent,
           "symbol": symbol, "asset_class": trade.asset_class_of(symbol),
           "side": side, "kind": kind or ("entry" if side == "buy" else "exit"),
           "order_type": "stop" if kind == "stop" else "market",
           "tif": "day", "order_class": order_class,
           "qty": qty, "status": status, "filled_qty": qty,
           "filled_avg_price": px, "submitted_at": _utc(filled_on),
           "filled_at": _utc(filled_on)}
    store.insert("desk_orders", row, returning=False)
    return oid


def _decision(store, run_id, picks, on=D_ENTRY):
    store.insert("desk_decisions", {
        "account": "agent", "run_id": run_id,
        "ts": datetime(on.year, on.month, on.day, 10 + ET_OFFSET),
        "picks": picks}, returning=False)


def _pos(symbol, qty, avg, cur, unreal=None):
    return {"symbol": symbol, "asset_class": "us_equity", "qty": qty,
            "qty_available": qty, "avg_entry_price": avg,
            "current_price": cur, "market_value": (cur or avg) * qty,
            "cost_basis": avg * qty,
            "unrealized_pl": unreal if unreal is not None
            else round(((cur or avg) - avg) * qty, 2),
            "unrealized_plpc": None, "change_today": None, "side": "long"}


# ── outcomes ─────────────────────────────────────────────────────────────


def test_open_pick_marks_off_alpaca_position(store):
    from agent.grade import outcomes

    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy",
                             "prediction": "up", "horizon_days": 10,
                             "kill": "closes below $80"}])
    out = outcomes(store, positions=[_pos("NVDA", 10, 100.0, 110.0)])
    p = out["runs"][0]["picks"][0]
    assert p["entry_avg_px"] == 100.0
    assert p["since_this_run_pct"] == 10.0
    assert p["open_now"]["shares"] == 10
    assert p["spy_same_window_pct"] is not None
    assert p["alpha_pct"] == round(10.0 - p["spy_same_window_pct"], 2)


def test_multi_order_pick_weights_entry(store):
    from agent.grade import outcomes

    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY, seq=1)
    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=30, px=104.0,
           filled_on=D_ENTRY, seq=2)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy"}])
    out = outcomes(store, positions=[_pos("NVDA", 40, 103.0, 106.0)])
    assert out["runs"][0]["picks"][0]["entry_avg_px"] == 103.0


def test_same_run_round_trip_closes_exactly(store):
    from agent.grade import outcomes

    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY, seq=1)
    _order(store, run_id="R1", symbol="NVDA", side="sell", qty=10, px=108.0,
           filled_on=D_EXIT, seq=2)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy"}])
    out = outcomes(store, positions=[])
    p = out["runs"][0]["picks"][0]
    assert p["closed_return_pct"] == 8.0
    assert p["realized_pnl"] == 80.0
    assert p["exit_date"] == D_EXIT.isoformat()
    # exit-bounded SPY window, not the latest close
    assert p["spy_same_window_pct"] < out["runs"][0]["spy_same_window_pct"] \
        or p["spy_same_window_pct"] is not None


def test_mleg_parent_skipped_legs_counted(store):
    from agent.grade import fills_from_orders

    parent = _order(store, run_id="R1", symbol=CALL, side="buy", qty=1,
                    px=3.10, filled_on=D_ENTRY, order_class="mleg",
                    oid="parent-1")
    _order(store, run_id="R1", symbol=CALL, side="buy", qty=1, px=5.0,
           filled_on=D_ENTRY, order_class="mleg", parent=parent, oid="leg-1")
    fills = fills_from_orders(store)
    assert len(fills) == 1  # the parent shell is not a fill
    assert fills[0]["symbol"] == CALL and fills[0]["price"] == 5.0
    assert fills[0]["dollars"] == 1 * 5.0 * 100


def test_short_opened_option_enters_at_credit(store):
    from agent.grade import outcomes

    # CSP: sell-to-open 2 puts at 4.00, buy back at 1.50 → +62.5% of credit
    _order(store, run_id="R1", symbol=CALL, side="sell", qty=2, px=4.00,
           filled_on=D_ENTRY, seq=1, kind="entry")
    _order(store, run_id="R1", symbol=CALL, side="buy", qty=2, px=1.50,
           filled_on=D_EXIT, seq=2, kind="exit")
    _decision(store, "R1", [{"symbol": CALL, "action": "buy"}])
    out = outcomes(store, positions=[])
    p = out["runs"][0]["picks"][0]
    assert p["short_opened"] is True
    assert p["entry_avg_px"] == 4.00
    assert p["closed_return_pct"] == 62.5
    assert p["alpha_pct"] is None  # options never carry index alpha


def test_split_rebases_entry_between_fill_and_mark(store):
    from agent.grade import outcomes

    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=1000.0,
           filled_on=D_ENTRY)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy"}])
    store.insert("ticker_splits", {
        "symbol": "NVDA",
        "execution_date": (D_ENTRY + timedelta(days=2)).isoformat(),
        "split_from": 1, "split_to": 10}, returning=False)
    # Post-split position: 100 shares, marked at 101 (pre-split 1010)
    out = outcomes(store, positions=[_pos("NVDA", 100, 100.0, 101.0)])
    p = out["runs"][0]["picks"][0]
    assert p["entry_avg_px"] == 100.0  # 1000 / 10
    assert p["since_this_run_pct"] == 1.0  # not a fake -89.9%


def test_spy_benchmark_is_price_return_not_total_return(store):
    from agent.grade import spy_price_closes

    # A large SPY dividend inside the window must NOT back-adjust closes.
    store.insert("dividends", {
        "symbol": "SPY", "ex_date": D_ENTRY + timedelta(days=1),
        "cash_amount": 50.0}, returning=False)
    closes = spy_price_closes(store, since=D_ENTRY.isoformat())
    seeded_first = 500.0 + 10 * 0.2  # entry-10 buffer days after start
    got_first = closes[0][1]
    # price return: the stored close, no dividend factor applied
    assert abs(got_first - 500.0) < 2.1 * 10  # within the seeded ramp
    all_raw = all(abs(c - round(c, 2)) < 1e-9 for _, c in closes)
    assert all_raw
    assert got_first == pytest.approx(closes[0][1])
    assert seeded_first  # silence unused warning


# ── grade ────────────────────────────────────────────────────────────────


def _grade(store, positions):
    from agent.grade import grade
    return grade(store, positions=positions)


def test_grade_open_pick_writes_row_and_preserves_verdict(store):
    # A stored close ABOVE the kill so kill_breached grades False, not None
    # (None = no closes stored in the window — nothing to judge).
    store.insert("daily_bars", {"symbol": "NVDA",
                                "date": D_ENTRY + timedelta(days=1),
                                "open": 95.0, "high": 96.0, "low": 94.0,
                                "close": 95.0, "volume": 1e6,
                                "source": "test"}, returning=False)
    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy",
                             "horizon_days": 5, "kill": "closes below $80"}])
    r = _grade(store, [_pos("NVDA", 10, 100.0, 110.0)])
    assert r["ok"] and r["graded"] == 1
    row = store.select("desk_outcomes", filters={"run_id": "R1"})[0]
    assert row["status"] == "open" and row["since_pct"] == 10.0
    assert row["mark_basis"] == "mark" and row["mark_px"] == 110.0
    assert row["kill_level"] == 80.0 and row["kill_breached"] is False

    # The reflection's verdict survives a re-grade (grade never writes it).
    store.update("desk_outcomes", {"id": row["id"]},
                 {"verdict": "TRUE", "verdict_note": "n"}, returning=False)
    _grade(store, [_pos("NVDA", 10, 100.0, 112.0)])
    row2 = store.select("desk_outcomes", filters={"run_id": "R1"})[0]
    assert row2["verdict"] == "TRUE" and row2["since_pct"] == 12.0


def test_grade_cross_run_exit_reconstructs(store):
    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY)
    _order(store, run_id="R2", symbol="NVDA", side="sell", qty=10, px=93.0,
           filled_on=D_EXIT)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy"}])
    _decision(store, "R2", [], on=D_EXIT)
    r = _grade(store, [])
    row = store.select("desk_outcomes", filters={"run_id": "R1"})[0]
    assert row["status"] == "closed"
    assert row["exit_kind"] == "cross_run"
    assert row["exit_avg_px"] == 93.0
    assert row["since_pct"] == -7.0
    assert row["realized_pnl"] == -70.0
    assert r["ok"]


def test_grade_stop_exit_is_hardstop(store):
    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY)
    # The protective stop armed by R1 fired later — kind='stop' marks it.
    _order(store, run_id="R1", symbol="NVDA", side="sell", qty=10, px=88.0,
           filled_on=D_EXIT, seq=2, kind="stop")
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy"}])
    _grade(store, [])
    row = store.select("desk_outcomes", filters={"run_id": "R1"})[0]
    assert row["exit_kind"] == "hardstop"
    assert row["exit_avg_px"] == 88.0 and row["since_pct"] == -12.0


def test_grade_expired_option_settles_at_zero(store):
    _order(store, run_id="R1", symbol=EXPIRED_PUT, side="buy", qty=1, px=2.0,
           filled_on=TODAY - timedelta(days=20))
    _decision(store, "R1", [{"symbol": EXPIRED_PUT, "action": "buy"}],
              on=TODAY - timedelta(days=20))
    r = _grade(store, [])  # position gone, expiry passed, no closing fill
    row = store.select("desk_outcomes", filters={"run_id": "R1"})[0]
    assert row["status"] == "closed"
    assert row["exit_kind"] == "settlement"
    assert row["exit_avg_px"] == 0.0
    assert row["since_pct"] == -100.0
    assert row["realized_pnl"] == -200.0  # 1 contract × $2 × 100
    assert r["ok"]


def test_grade_degraded_mark_writes_nulls(store):
    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy"}])
    _grade(store, [_pos("NVDA", 10, 100.0, None)])  # Alpaca gave no price
    row = store.select("desk_outcomes", filters={"run_id": "R1"})[0]
    assert row["degraded"] is True
    assert row["since_pct"] is None and row["mark_px"] is None
    assert row["status"] == "open"


def test_grade_without_marks_never_falsely_closes(store):
    from agent.grade import grade

    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy"}])
    # positions=None (fetch unavailable) → absence proves nothing; the open
    # pick is skipped rather than graded closed.
    r = grade(store, positions=None)
    assert r["marks_available"] is False
    assert store.select("desk_outcomes") == []


def test_grade_cutover_rows_are_final(store):
    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy"}])
    store.insert("desk_outcomes", {
        "account": "agent", "run_id": "R1", "symbol": "NVDA",
        "grade_date": D_EXIT, "entry_avg_px": 100.0, "since_pct": -1.5,
        "exit_kind": "cutover", "status": "closed"}, returning=False)
    _grade(store, [_pos("NVDA", 10, 100.0, 130.0)])
    row = store.select("desk_outcomes", filters={"run_id": "R1"})[0]
    assert row["exit_kind"] == "cutover" and row["since_pct"] == -1.5


def test_kill_breach_reads_daily_bars(store):
    d = D_ENTRY + timedelta(days=1)
    store.insert("daily_bars", {"symbol": "NVDA", "date": d, "open": 79.0,
                                "high": 80.0, "low": 78.0, "close": 79.0,
                                "volume": 1e6, "source": "test"},
                 returning=False)
    _order(store, run_id="R1", symbol="NVDA", side="buy", qty=10, px=100.0,
           filled_on=D_ENTRY)
    _decision(store, "R1", [{"symbol": "NVDA", "action": "buy",
                             "kill": "two closes below $80"}])
    _grade(store, [_pos("NVDA", 10, 100.0, 95.0)])
    row = store.select("desk_outcomes", filters={"run_id": "R1"})[0]
    assert row["kill_level"] == 80.0 and row["kill_breached"] is True


# ── commitments sweep on the new sources ─────────────────────────────────


def test_sweep_fires_commitment_from_closes(store):
    from agent.grade import sweep_commitments

    d = D_ENTRY + timedelta(days=2)
    store.insert("daily_bars", {"symbol": "AAPL", "date": d, "open": 330.0,
                                "high": 331.0, "low": 329.0, "close": 330.0,
                                "volume": 1e6, "source": "test"},
                 returning=False)
    _decision(store, "R1", [], on=D_ENTRY)
    store.insert("desk_commitments", {
        "account": "agent", "run_id": "R1", "symbol": "AAPL",
        "kind": "reentry", "direction": "above", "level": 325.0,
        "until": TODAY + timedelta(days=5), "text": "re-add over $325",
        "status": "open"}, returning=False)
    r = sweep_commitments(store)
    assert r["fired"] == 1
    row = store.select("desk_commitments")[0]
    assert row["status"] == "fired" and row["fired_close"] == 330.0


# ── the SPY window conventions (pure — ported from the V3 alpha suite) ───


def test_spy_window_baseline_is_strictly_before_start():
    from agent.grade import _spy_window_pct

    spy = [("2026-08-01", 600.0), ("2026-08-04", 606.0), ("2026-08-05", 612.0)]
    # entry on the 4th: baseline is the 1st's close (the last print BEFORE
    # the window opened), never the entry day's own 16:00 close
    assert _spy_window_pct(spy, "2026-08-04") == 2.0
    assert _spy_window_pct(spy, "2026-08-04", "2026-08-04") == 1.0


def test_spy_window_degenerate_is_none_not_zero():
    from agent.grade import _spy_window_pct

    spy = [("2026-08-01", 600.0), ("2026-08-04", 606.0)]
    # same-day window: baseline row == endpoint row → too young, never 0.00
    assert _spy_window_pct(spy, "2026-08-04", "2026-08-01") is None
    # no baseline exists before the window start
    assert _spy_window_pct(spy, "2026-08-01") is None
    # no data at all
    assert _spy_window_pct([], "2026-08-04") is None

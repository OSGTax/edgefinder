"""The attention system: tripwires (desk_watch), planned wakes (desk_wakes),
the streamer's pure evaluator, and the desk surface.

Design under test: cheap code (the always-on streamer) watches the tape
continuously; the expensive brain grants itself extra runs only through the
wake-plan budget gate, with a stated reason, capped per ET day.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'attn.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    return get_store()


# ── watch-set / list / clear ──


def test_watch_set_validation(store):
    from agent.brain import watch_set

    assert not watch_set(store, symbol="AMD", reason="x")["ok"]          # neither
    assert not watch_set(store, symbol="AMD", above=1, below=2, reason="x")["ok"]
    assert not watch_set(store, symbol="AMD", below=540, reason="  ")["ok"]  # no reason
    assert not watch_set(store, symbol="AMD", below=-5, reason="x")["ok"]

    r = watch_set(store, symbol="amd", below=540.0, reason="kill level",
                  run_id="R1")
    assert r["ok"] and r["symbol"] == "AMD" and r["kind"] == "below"


def test_watch_list_buckets_and_lazy_expiry(store):
    from agent.brain import watch_list, watch_set

    watch_set(store, symbol="AMD", below=540, reason="kill")
    watch_set(store, symbol="SPY", above=760, reason="breakout",
              until=str(datetime.utcnow() - timedelta(hours=1)))  # already past
    store.update("desk_watch", {"symbol": "AMD"},
                 {"status": "tripped", "tripped_price": 539.2}, returning=False)

    out = watch_list(store)
    assert [w["symbol"] for w in out["tripped"]] == ["AMD"]
    assert out["armed"] == []  # SPY expired lazily, not reported as armed
    assert "done" not in out
    assert watch_list(store, include_done=True)["done"]


def test_watch_clear(store):
    from agent.brain import watch_clear, watch_set

    wid = watch_set(store, symbol="AMD", below=540, reason="kill")["id"]
    assert watch_clear(store, watch_id=wid)["ok"]
    assert store.select("desk_watch", filters={"id": wid})[0]["status"] == "disarmed"
    assert not watch_clear(store, watch_id=99999)["ok"]


# ── the streamer's pure evaluator ──


def _q(bid, ask, age_secs=0.0):
    return {"bid": bid, "ask": ask, "recv": time.time() - age_secs}


def test_evaluate_watches_trips_on_fresh_mid():
    from agent.streamer import evaluate_watches

    watches = [
        {"id": 1, "symbol": "AMD", "kind": "below", "level": 540.0},
        {"id": 2, "symbol": "SPY", "kind": "above", "level": 750.0},
        {"id": 3, "symbol": "NVDA", "kind": "below", "level": 100.0},
    ]
    quotes = {"AMD": _q(539.0, 539.4),      # mid 539.2 <= 540 → trip
              "SPY": _q(750.1, 750.3),      # mid 750.2 >= 750 → trip
              "NVDA": _q(180.0, 180.2)}     # far away → no trip
    tripped, expired = evaluate_watches(watches, quotes)
    assert sorted(w["id"] for w in tripped) == [1, 2]
    assert not expired
    amd = next(w for w in tripped if w["id"] == 1)
    assert amd["tripped_price"] == pytest.approx(539.2)


def test_evaluate_watches_never_trips_on_stale_or_bad_quotes():
    from agent.streamer import evaluate_watches

    watches = [{"id": 1, "symbol": "AMD", "kind": "below", "level": 540.0}]
    stale = {"AMD": _q(539.0, 539.4, age_secs=3600)}
    assert evaluate_watches(watches, stale) == ([], [])
    crossed = {"AMD": {"bid": 539.0, "ask": 500.0, "recv": time.time()}}
    assert evaluate_watches(watches, crossed) == ([], [])
    assert evaluate_watches(watches, {}) == ([], [])


def test_evaluate_watches_expires_past_until():
    from agent.streamer import evaluate_watches

    watches = [{"id": 1, "symbol": "AMD", "kind": "below", "level": 540.0,
                "until": str(datetime.utcnow() - timedelta(minutes=1))}]
    tripped, expired = evaluate_watches(
        watches, {"AMD": _q(500.0, 500.2)})  # would trip, but it's expired
    assert not tripped and [w["id"] for w in expired] == [1]


# ── hard stops: the ONE wire that acts (opt-in, F8) ──


class _StopBroker:
    """Live-fill fake: market state + a fresh quote around ``px``."""

    def __init__(self, px=100.0, open_=True):
        self._px, self._open = px, open_

    def is_market_open(self):
        return self._open

    def session(self, symbol=None):
        if symbol and "/" in symbol:
            return "crypto"
        return "regular" if self._open else "closed"

    def is_close_soon(self, minutes=15):
        return False

    def quotes(self, symbols):
        from datetime import timezone
        t = datetime.now(timezone.utc).isoformat()
        return {s: {"symbol": s, "bid": round(self._px * 0.999, 4),
                    "ask": round(self._px * 1.001, 4), "mid": self._px,
                    "t": t} for s in symbols}

    def option_quotes(self, symbols):
        return self.quotes(symbols)


def _patch_broker(monkeypatch, fake):
    import agent.broker as broker

    monkeypatch.setattr(broker, "Broker", lambda *a, **k: fake)
    monkeypatch.setattr(broker, "enabled", lambda: True)


def _seed_position(store, symbol, shares, price, last_price=None, days_ago=0):
    from agent import ledger

    store.insert("desk_trades", {
        "account": "agent", "run_id": "T", "symbol": symbol, "side": "BUY",
        "shares": shares, "price": price, "dollars": round(shares * price, 2),
        "ts": datetime.utcnow() - timedelta(days=days_ago)}, returning=False)
    ledger.rebuild_positions(store)
    if last_price is not None:
        store.update("desk_positions", {"symbol": symbol},
                     {"last_price": last_price}, returning=False)


def test_hard_stop_arm_time_validations(store, monkeypatch):
    import agent.broker as broker
    from agent.brain import watch_set

    monkeypatch.setattr(broker, "enabled", lambda: False)  # price = last mark
    # not held → refused
    r = watch_set(store, symbol="AMD", below=90, reason="stop", hard=True)
    assert not r["ok"] and "no long" in r["error"]
    _seed_position(store, "AMD", 10, 100.0, last_price=100.0)
    # a protective stop is a below-level
    assert not watch_set(store, symbol="AMD", above=110, reason="stop",
                         hard=True)["ok"]
    # at/above the market would fire the moment it's armed
    inst = watch_set(store, symbol="AMD", below=105, reason="stop", hard=True)
    assert not inst["ok"] and "instantly" in inst["error"]
    # option contracts are not hard-stoppable
    assert not watch_set(store, symbol="AMD270116C00100000", below=1,
                         reason="x", hard=True)["ok"]
    ok = watch_set(store, symbol="amd", below=90, reason="protect the swing",
                   hard=True, run_id="R1")
    assert ok["ok"] and ok["kind"] == "hard_stop"
    row = store.select("desk_watch", filters={"id": ok["id"]})[0]
    assert row["kind"] == "hard_stop" and row["status"] == "armed"


def test_evaluate_watches_hard_stop_trips_at_or_below():
    from agent.streamer import evaluate_watches

    watches = [{"id": 1, "symbol": "AMD", "kind": "hard_stop", "level": 90.0}]
    assert evaluate_watches(watches, {"AMD": _q(95.0, 95.2)}) == ([], [])
    tripped, _ = evaluate_watches(watches, {"AMD": _q(89.5, 89.7)})
    assert [w["id"] for w in tripped] == [1]


def test_hard_stop_executes_full_position_sell(store, monkeypatch):
    import agent.broker as broker
    from agent.brain import watch_set
    from agent.streamer import apply_sweep_results

    _seed_position(store, "AMD", 10, 100.0, last_price=100.0)
    monkeypatch.setattr(broker, "enabled", lambda: False)
    wid = watch_set(store, symbol="AMD", below=90, reason="protect",
                    hard=True)["id"]
    _patch_broker(monkeypatch, _StopBroker(px=89.0))  # gates price a REST quote
    watch = store.select("desk_watch", filters={"id": wid})[0]
    apply_sweep_results(store, [{**watch, "tripped_price": 89.0}], [])

    row = store.select("desk_watch", filters={"id": wid})[0]
    assert row["status"] == "executed"
    assert row["honored_run_id"] == f"hardstop:{wid}"
    assert row["tripped_price"] == 89.0 and row["tripped_at"] is not None
    sells = store.select("desk_trades", filters={"symbol": "AMD", "side": "SELL"})
    assert len(sells) == 1
    assert sells[0]["run_id"] == f"hardstop:{wid}"
    assert sells[0]["shares"] == pytest.approx(10.0)         # the FULL position
    assert sells[0]["fill_quote"]["session"] == "regular"    # gate was consulted
    assert store.select("desk_positions", filters={"symbol": "AMD"}) == []


def test_hard_stop_gated_rejection_exec_failed_no_retry(store, monkeypatch):
    import agent.broker as broker
    from agent.brain import watch_list, watch_set
    from agent.streamer import apply_sweep_results

    _seed_position(store, "AMD", 10, 100.0, last_price=100.0)
    monkeypatch.setattr(broker, "enabled", lambda: False)
    wid = watch_set(store, symbol="AMD", below=90, reason="protect",
                    hard=True)["id"]
    _patch_broker(monkeypatch, _StopBroker(px=89.0, open_=False))  # gate: closed
    watch = store.select("desk_watch", filters={"id": wid})[0]
    apply_sweep_results(store, [{**watch, "tripped_price": 89.0}], [])

    row = store.select("desk_watch", filters={"id": wid})[0]
    assert row["status"] == "exec_failed"
    assert "market closed" in row["result"]
    assert row["honored_run_id"] is None
    assert store.select("desk_trades",
                        filters={"symbol": "AMD", "side": "SELL"}) == []
    # single attempt: the sweep only ever reloads status='armed'
    assert store.select("desk_watch", filters={"status": "armed"}) == []
    # and the next trading cycle sees it with the tripped wires
    out = watch_list(store)
    assert [w["id"] for w in out["tripped"]] == [wid]


def test_hard_stop_position_gone_marks_stale(store):
    from agent.streamer import execute_hard_stop

    rows = store.insert("desk_watch", {
        "account": "agent", "symbol": "GONE", "kind": "hard_stop",
        "level": 50.0, "reason": "x", "armed_at": datetime.utcnow(),
        "status": "armed"})
    r = execute_hard_stop(store, rows[0], 49.0)
    assert not r["ok"]
    row = store.select("desk_watch", filters={"id": rows[0]["id"]})[0]
    assert row["status"] == "stale" and "gone" in row["result"]


def test_hard_stop_books_unbooked_split_before_selling(store, monkeypatch):
    """H1 regression (the exact reviewer repro): a 2:1 split executes at the
    open (unbooked), the tape rebases, and the stop armed below the
    PRE-split price trips instantly. The executor must run the per-symbol
    corp-actions pass FIRST and sell the POST-split share count — the old
    path sold the stale pre-split count at the post-split price and half
    the position's value silently vanished."""
    from datetime import date

    import agent.broker as broker
    from agent import ledger
    from agent.brain import watch_set
    from agent.streamer import apply_sweep_results

    _seed_position(store, "AMD", 10, 100.0, last_price=100.0, days_ago=5)
    monkeypatch.setattr(broker, "enabled", lambda: False)
    wid = watch_set(store, symbol="AMD", below=90, reason="protect",
                    hard=True)["id"]
    # the split lands after arming; no settle has booked it yet
    store.insert("ticker_splits", {
        "symbol": "AMD", "execution_date": str(date.today()),
        "split_from": 1, "split_to": 2}, returning=False)
    _patch_broker(monkeypatch, _StopBroker(px=49.0))  # rebased tape → trip
    watch = store.select("desk_watch", filters={"id": wid})[0]
    apply_sweep_results(store, [{**watch, "tripped_price": 49.0}], [])

    assert store.select("desk_watch", filters={"id": wid})[0]["status"] == "executed"
    sells = store.select("desk_trades", filters={"symbol": "AMD", "side": "SELL"})
    assert len(sells) == 1
    assert sells[0]["shares"] == pytest.approx(20.0)  # POST-split count
    # value conserved (± spread/slippage): ~20 × ~48.94 back into cash,
    # never 10 × 49 with the other half evaporated
    assert sells[0]["dollars"] == pytest.approx(20 * 49.0, rel=0.01)
    assert ledger.cash(store) == pytest.approx(
        100_000.0 - 1_000.0 + sells[0]["dollars"])
    # the split adjustment row booked exactly once, before the sale
    adj = [t for t in store.select("desk_trades", filters={"symbol": "AMD"})
           if (t.get("fill_quote") or {}).get("src") == "split_adjustment"]
    assert len(adj) == 1
    # and the later full settle books nothing more (idempotent; lot closed
    # → no phantom position at avg_price 0 either)
    out = ledger.settle(store)
    assert out["corp_actions"] == {"splits": 0, "dividends": 0, "details": []}
    assert store.select("desk_positions", filters={"symbol": "AMD"}) == []


def _seed_daily_bars(store, symbol, closes_volumes):
    """Seed (close, volume) daily bars ending today, oldest first."""
    from datetime import date

    d = date.today() - timedelta(days=len(closes_volumes))
    for close, vol in closes_volumes:
        d += timedelta(days=1)
        store.insert("daily_bars", {"symbol": symbol, "date": d, "open": close,
                                    "high": close, "low": close, "close": close,
                                    "volume": vol, "source": "test"},
                     returning=False)


def test_hard_stop_overrides_close_band_gap(store, monkeypatch):
    """H1 repro (the review's canonical scenario): a ~-26% gap vs the stored
    close is EXACTLY when an armed stop must fire — the last-close ENTRY
    band must not veto the protective exit into exec_failed (single-attempt
    semantics would then leave the position riding through the crash). The
    override is stamped on the persisted fill receipt; a same-tape BUY
    without the flag still rejects."""
    import agent.broker as broker
    from agent import ledger
    from agent.brain import watch_set
    from agent.streamer import apply_sweep_results

    _seed_position(store, "AMD", 10, 120.0, last_price=120.0)
    _seed_daily_bars(store, "AMD", [(120.0, 50_000_000)] * 20)
    monkeypatch.setattr(broker, "enabled", lambda: False)
    wid = watch_set(store, symbol="AMD", below=90, reason="earnings protect",
                    hard=True)["id"]
    _patch_broker(monkeypatch, _StopBroker(px=88.0))  # ~-27% vs stored 120
    # ENTRIES stay gated: the same tape without the flag refuses a BUY
    buy = ledger.live_fill(store, symbol="AMD", side="buy", shares=1)
    assert not buy["ok"] and "latest stored close" in buy["error"]

    watch = store.select("desk_watch", filters={"id": wid})[0]
    apply_sweep_results(store, [{**watch, "tripped_price": 88.0}], [])
    row = store.select("desk_watch", filters={"id": wid})[0]
    assert row["status"] == "executed", row.get("result")
    sells = store.select("desk_trades",
                         filters={"symbol": "AMD", "side": "SELL"})
    assert len(sells) == 1 and sells[0]["shares"] == pytest.approx(10.0)
    # the receipt shows a gated-but-overridden protective exit
    fq_warnings = sells[0]["fill_quote"]["warnings"]
    assert any("override" in w for w in fq_warnings), fq_warnings
    assert store.select("desk_positions", filters={"symbol": "AMD"}) == []


def test_hard_stop_overrides_adv_cap(store, monkeypatch):
    """H2 repro: a FULL-POSITION exit bigger than the 1% ADV entry cap must
    still sell — refusing to exit what the book already owns inverts the
    protection. Override stamped on the receipt; a BUY of the same size
    stays rejected."""
    import agent.broker as broker
    from agent import ledger
    from agent.brain import watch_set
    from agent.streamer import apply_sweep_results

    _seed_position(store, "AMD", 100, 100.0, last_price=100.0)
    # ADV = 100 x 500 = $50k/day → 1% cap $500, far under the ~$8.9k exit
    _seed_daily_bars(store, "AMD", [(100.0, 500)] * 20)
    monkeypatch.setattr(broker, "enabled", lambda: False)
    wid = watch_set(store, symbol="AMD", below=90, reason="protect",
                    hard=True)["id"]
    _patch_broker(monkeypatch, _StopBroker(px=89.0))
    buy = ledger.live_fill(store, symbol="AMD", side="buy", notional=8_900)
    assert not buy["ok"] and "average dollar volume" in buy["error"]

    watch = store.select("desk_watch", filters={"id": wid})[0]
    apply_sweep_results(store, [{**watch, "tripped_price": 89.0}], [])
    row = store.select("desk_watch", filters={"id": wid})[0]
    assert row["status"] == "executed", row.get("result")
    sells = store.select("desk_trades",
                         filters={"symbol": "AMD", "side": "SELL"})
    assert len(sells) == 1 and sells[0]["shares"] == pytest.approx(100.0)
    fq_warnings = sells[0]["fill_quote"]["warnings"]
    assert any("ADV" in w for w in fq_warnings), fq_warnings
    assert store.select("desk_positions", filters={"symbol": "AMD"}) == []


def test_hard_stop_claim_is_compare_and_swap(store, monkeypatch):
    """H2: the armed→executing transition is a conditional update — exactly
    one writer wins; a loser (second sweep, deploy-overlap streamer, trading
    cycle) books nothing."""
    import agent.broker as broker
    from agent.brain import watch_set
    from agent.streamer import claim_watch, execute_hard_stop

    _seed_position(store, "AMD", 10, 100.0, last_price=100.0)
    monkeypatch.setattr(broker, "enabled", lambda: False)
    wid = watch_set(store, symbol="AMD", below=90, reason="protect",
                    hard=True)["id"]
    assert claim_watch(store, wid, {"tripped_price": 89.0}) is True
    assert store.select("desk_watch", filters={"id": wid})[0]["status"] == "executing"
    assert claim_watch(store, wid) is False  # second claimant loses

    # a full executor racing on the same (stale-cached) armed row loses the
    # claim and must not touch the book or the wire's status
    watch = store.select("desk_watch", filters={"id": wid})[0]
    r = execute_hard_stop(store, {**watch, "status": "armed"}, 89.0)
    assert not r["ok"] and r["status"] == "claim_lost"
    assert store.select("desk_trades", filters={"side": "SELL"}) == []
    assert store.select("desk_watch", filters={"id": wid})[0]["status"] == "executing"


def test_stale_executing_claims_flagged_not_retried(store):
    """A wire stuck in 'executing' (executor crashed mid-flight) is flagged
    exec_failed after ~10 min — surfaced for the next cycle to inspect,
    NEVER auto-retried. A fresh claim is left alone."""
    from agent.streamer import WATCH_EXEC_STALE_SECS, flag_stale_executing

    old = store.insert("desk_watch", {
        "account": "agent", "symbol": "AMD", "kind": "hard_stop", "level": 90.0,
        "reason": "x", "armed_at": datetime.utcnow(), "status": "executing",
        "tripped_at": datetime.utcnow()
        - timedelta(seconds=WATCH_EXEC_STALE_SECS + 60)})[0]
    fresh = store.insert("desk_watch", {
        "account": "agent", "symbol": "NVDA", "kind": "hard_stop", "level": 90.0,
        "reason": "x", "armed_at": datetime.utcnow(), "status": "executing",
        "tripped_at": datetime.utcnow()})[0]
    assert flag_stale_executing(store) == 1
    old_row = store.select("desk_watch", filters={"id": old["id"]})[0]
    assert old_row["status"] == "exec_failed"
    assert old_row["result"] == "stale executing claim"
    assert store.select("desk_watch",
                        filters={"id": fresh["id"]})[0]["status"] == "executing"


def test_hard_stop_refuses_crypto_pairs(store):
    """M3: the sweep only sees the equity SIP tape — crypto quotes never
    enter it, so a crypto hard stop could never trip. Protection that
    cannot trip must not arm."""
    from agent.brain import watch_set

    r = watch_set(store, symbol="BTC/USD", below=50_000, reason="stop",
                  hard=True)
    assert not r["ok"] and "never" in r["error"] and "SIP" in r["error"]
    assert store.select("desk_watch") == []
    # plain advisory wires on crypto are still allowed (they don't promise
    # to act)
    assert watch_set(store, symbol="BTC/USD", below=50_000, reason="note")["ok"]


def test_hard_stop_refuses_covered_call_shares(store, monkeypatch):
    """M4: arming on shares that back short calls is dead protection — the
    execution would trip and then fail the ledger's coverage gate. Refused
    at arm time with a leg-out instruction; arms once the calls are gone."""
    from datetime import date

    import agent.broker as broker
    from agent import occ
    from agent.brain import watch_set
    from agent.ledger import record_trade

    monkeypatch.setattr(broker, "enabled", lambda: False)
    _seed_position(store, "NVDA", 100, 100.0, last_price=100.0)
    call = occ.build("NVDA", date.today() + timedelta(days=45), "C", 120)
    fq = {"bid": 4.9, "ask": 5.1, "mid": 5.0, "t": "x", "src": "test"}
    assert record_trade(store, symbol=call, side="SELL", shares=1, price=5.0,
                        fill_quote=fq)["ok"]
    r = watch_set(store, symbol="NVDA", below=90, reason="protect", hard=True)
    assert not r["ok"] and "call" in r["error"] and "leg out" in r["error"]
    # buy the call back → the same stop arms (re-mark first: rebuild wiped
    # last_price, and an unmarked position has no reference price — L4)
    assert record_trade(store, symbol=call, side="BUY", shares=1, price=5.0,
                        fill_quote=fq)["ok"]
    store.update("desk_positions", {"symbol": "NVDA"},
                 {"last_price": 100.0}, returning=False)
    assert watch_set(store, symbol="NVDA", below=90, reason="protect",
                     hard=True)["ok"]


def test_hard_stop_refuses_without_reference_price(store, monkeypatch):
    """L4: no live quote and no last mark → the level can't be proven below
    the market, so the stop refuses to arm (strict) with a retry hint."""
    import agent.broker as broker
    from agent.brain import watch_set

    monkeypatch.setattr(broker, "enabled", lambda: False)
    _seed_position(store, "AMD", 10, 100.0)  # last_price never marked
    r = watch_set(store, symbol="AMD", below=90, reason="stop", hard=True)
    assert not r["ok"] and "reference price" in r["error"]
    assert store.select("desk_watch") == []


def test_alert_wires_never_trade_regression(store, monkeypatch):
    """DEFAULT-OFF guarantee: plain above/below wires only flip status —
    they must never reach the fill path, even through the new sweep."""
    import agent.ledger as ledger_mod
    from agent.brain import watch_set
    from agent.streamer import apply_sweep_results

    _seed_position(store, "AMD", 10, 100.0, last_price=100.0)
    wid = watch_set(store, symbol="AMD", below=90, reason="alert only")["id"]

    def _boom(*a, **k):
        raise AssertionError("advisory wire must never reach the fill path")

    monkeypatch.setattr(ledger_mod, "live_fill", _boom)
    watch = store.select("desk_watch", filters={"id": wid})[0]
    apply_sweep_results(store, [{**watch, "tripped_price": 89.0}], [])
    row = store.select("desk_watch", filters={"id": wid})[0]
    assert row["status"] == "tripped" and row["tripped_price"] == 89.0
    assert store.select("desk_trades", filters={"side": "SELL"}) == []


# ── wake-plan: the budget gate ──


def test_wake_plan_records_and_reports_budget(store):
    from agent.brain import wake_plan

    at = datetime.utcnow() + timedelta(minutes=30)
    r = wake_plan(store, at=at.isoformat(), reason="NVDA near kill", run_id="R1")
    assert r["ok"] and r["budget_left_today"] == 19
    rows = store.select("desk_wakes")
    assert len(rows) == 1 and rows[0]["reason"] == "NVDA near kill"


def test_wake_plan_rejects_past_soon_and_unparseable(store):
    from agent.brain import wake_plan

    past = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    assert not wake_plan(store, at=past, reason="x")["ok"]
    soon = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
    assert "too soon" in wake_plan(store, at=soon, reason="x")["error"]
    assert not wake_plan(store, at="not-a-time", reason="x")["ok"]
    ok_at = (datetime.utcnow() + timedelta(minutes=30)).isoformat()
    assert not wake_plan(store, at=ok_at, reason="   ")["ok"]


def test_wake_plan_enforces_min_gap(store):
    from agent.brain import wake_plan

    base = datetime.utcnow() + timedelta(minutes=60)
    assert wake_plan(store, at=base.isoformat(), reason="a")["ok"]
    close = base + timedelta(minutes=10)  # < 15-min gap to the planned wake
    r = wake_plan(store, at=close.isoformat(), reason="b")
    assert not r["ok"] and "already planned" in r["error"]
    far = base + timedelta(minutes=20)
    assert wake_plan(store, at=far.isoformat(), reason="c")["ok"]


def test_wake_plan_enforces_daily_cap(store):
    from agent.brain import WAKE_MAX_PER_DAY, wake_plan

    # Anchor at tomorrow 14:00 UTC (10:00 ET): the 20 seeds span ~6.7 hours
    # and must all land on ONE ET day for the cap to bind — a "now + 30min"
    # base run in the evening spills seeds past ET midnight and the cap
    # legitimately doesn't trip (time-of-day flake, caught 2026-07-14).
    tomorrow = datetime.utcnow().date() + timedelta(days=1)
    base = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 14, 0)
    # Seed the day's budget as spent (direct inserts — cheap and exact).
    store.insert("desk_wakes", [
        {"account": "agent", "at": base + timedelta(minutes=20 * i),
         "reason": f"seed {i}", "created_at": datetime.utcnow()}
        for i in range(WAKE_MAX_PER_DAY)
    ], returning=False)
    r = wake_plan(store, at=(base + timedelta(minutes=20 * WAKE_MAX_PER_DAY))
                  .isoformat(), reason="one too many")
    assert not r["ok"] and "budget spent" in r["error"]


def test_wake_due_and_honor_loop(store):
    from agent.brain import wake_due, wake_honor

    now = datetime.utcnow()
    store.insert("desk_wakes", [
        {"account": "agent", "at": now - timedelta(minutes=20),
         "reason": "due now", "created_at": now - timedelta(hours=1)},
        {"account": "agent", "at": now + timedelta(hours=2),
         "reason": "future", "created_at": now},
        {"account": "agent", "at": now - timedelta(hours=12),
         "reason": "ancient", "created_at": now - timedelta(hours=13)},
    ], returning=False)

    d = wake_due(store)
    assert [w["reason"] for w in d["due"]] == ["due now"]
    assert [w["reason"] for w in d["missed"]] == ["ancient"]  # reported, not fresh

    wid = d["due"][0]["id"]
    assert wake_honor(store, wake_id=wid, run_id="RID-1")["ok"]
    # Honored exactly once; second honor and second due both refuse.
    assert not wake_honor(store, wake_id=wid, run_id="RID-2")["ok"]
    assert wake_due(store)["due"] == []


# ── the desk surface ──


def test_watch_endpoint(store, monkeypatch):
    from fastapi.testclient import TestClient

    import agent.data as agent_data
    import dashboard.dependencies as deps
    from agent.brain import wake_plan, watch_set

    deps._engine = deps._session_factory = None
    agent_data._session_factory = None

    watch_set(store, symbol="AMD", below=540, reason="kill level", run_id="R1")
    wake_plan(store, at=(datetime.utcnow() + timedelta(minutes=45)).isoformat(),
              reason="decide before close", run_id="R1")

    from dashboard.app import app

    with TestClient(app) as c:
        r = c.get("/api/desk/watch").json()
        assert r["watches"][0]["symbol"] == "AMD"
        assert r["watches"][0]["status"] == "armed"
        assert r["wakes"][0]["reason"] == "decide before close"

"""FOMC event-trade plumbing: the macro calendar, the short-dated expiry
override on the chain summary, and the ledger mechanics an iron butterfly
depends on (leg ordering, and a 4-leg settle that unwinds cleanly).

The ordering test is the load-bearing one: there is no combo order type, so
a butterfly is four separate fills, and selling a short leg before its wing
exists is rejected outright. That constraint belongs in a test, not a runbook.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from agent import occ

ET = ZoneInfo("America/New_York")
TODAY = date.today()
PAST = TODAY - timedelta(days=3)


def C(und, strike, expiry):
    return occ.build(und, expiry, "C", strike)


def P(und, strike, expiry):
    return occ.build(und, expiry, "P", strike)


def q(bid, ask):
    return {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4),
            "t": "x", "src": "test"}


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'fomc.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    import agent.store as store_mod
    store_mod._store = None
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401 — daily_bars for expiry-close settle
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


# ── the FOMC calendar ────────────────────────────────────────

def test_fomc_table_is_sorted_and_parseable():
    """A typo in a hand-transcribed constant is a wrong-day trade."""
    from agent.market import FOMC_RELEASE_DAYS

    days = [date.fromisoformat(d) for d in FOMC_RELEASE_DAYS]
    assert days == sorted(days), "calendar must be chronological"
    assert len(set(days)) == len(days), "duplicate release day"
    # The Fed meets 8 times a year; a year with a different count is a
    # transcription error worth failing on.
    for year in {d.year for d in days}:
        assert sum(1 for d in days if d.year == year) == 8


def test_next_fomc_on_a_release_day():
    from agent.market import next_fomc

    rel = date(2026, 7, 29)  # a real statement day
    now = datetime(2026, 7, 29, 12, 30, tzinfo=ET)
    out = next_fomc(rel, now=now)
    assert out["known"] and out["date"] == "2026-07-29"
    assert out["days_until"] == 0 and out["is_release_day"] is True
    assert out["release_et"] == "14:00"
    assert out["minutes_to_release"] == 90


def test_minutes_to_release_goes_negative_after_the_statement():
    from agent.market import next_fomc

    rel = date(2026, 7, 29)
    out = next_fomc(rel, now=datetime(2026, 7, 29, 15, 0, tzinfo=ET))
    assert out["minutes_to_release"] == -60


def test_next_fomc_rolls_to_the_following_meeting():
    from agent.market import next_fomc

    out = next_fomc(date(2026, 7, 30), now=datetime(2026, 7, 30, 9, 0, tzinfo=ET))
    assert out["date"] == "2026-09-16" and out["is_release_day"] is False
    assert out["days_until"] == 48
    assert "minutes_to_release" not in out  # only meaningful on the day


def test_lapsed_calendar_fails_loud_not_silent():
    """Running off the end must not read as 'no meeting scheduled'."""
    from agent.market import next_fomc

    out = next_fomc(date(2099, 1, 1), now=datetime(2099, 1, 1, 9, 0, tzinfo=ET))
    assert out["known"] is False
    assert "top up" in out["note"]
    assert out.get("date") is None


# ── the short-dated expiry override ──────────────────────────

E0 = TODAY.isoformat()                          # same-day — default skips it
E5 = (TODAY + timedelta(days=5)).isoformat()    # the default focus


def _row(type_, strike, expiry, bid, ask, iv=0.3, delta=0.5):
    return {"symbol": occ.build("SPY", date.fromisoformat(expiry), type_, strike),
            "type": type_, "strike": strike, "expiry": expiry,
            "dte": (date.fromisoformat(expiry) - TODAY).days,
            "bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4),
            "iv": iv, "delta": delta, "theta": None}


CHAIN = [
    _row("C", 100, E0, 1.0, 1.2), _row("P", 100, E0, 1.0, 1.2),
    _row("C", 100, E5, 2.4, 2.6), _row("P", 100, E5, 2.3, 2.5),
]


def test_default_focus_still_skips_same_day():
    from agent.options_data import summarize_chain

    s = summarize_chain(CHAIN, spot=100.0, today=TODAY)
    assert s["expiry"] == E5  # unchanged behaviour for the general read


def test_expiry_pin_selects_the_same_day_contract():
    from agent.options_data import summarize_chain

    s = summarize_chain(CHAIN, spot=100.0, today=TODAY, expiry=E0)
    assert s["expiry"] == E0 and s["dte"] == 0
    assert "expiry_requested" not in s  # it got what it asked for
    # the event read must price off the 0DTE straddle, not the 5-day one
    assert s["expected_move_dollars"] == pytest.approx(2.2)


def test_min_dte_zero_relaxes_the_floor():
    from agent.options_data import summarize_chain

    s = summarize_chain(CHAIN, spot=100.0, today=TODAY, min_dte=0)
    assert s["expiry"] == E0


def test_missing_expiry_falls_back_and_says_so():
    """A silently substituted expiry would misprice the structure."""
    from agent.options_data import summarize_chain

    absent = (TODAY + timedelta(days=99)).isoformat()
    s = summarize_chain(CHAIN, spot=100.0, today=TODAY, expiry=absent)
    assert s["expiry"] == E5 and s["expiry_requested"] == absent


def test_summary_cache_is_keyed_by_expiry(monkeypatch):
    """Keyed on symbol alone, the 0DTE read would be served the cached
    default-focus summary and quietly price the wrong contract."""
    import agent.options_data as od

    od._cache.clear()
    monkeypatch.setattr(od, "summarize_chain",
                        lambda rows, spot, **kw: {"available": True,
                                                  "expiry": kw.get("expiry") or "default"})

    class _B:
        def quotes(self, syms):
            return {"SPY": {"mid": 100.0}}

        def option_chain(self, sym, dte_max=60):
            return CHAIN

    # Patch the real module object, NOT sys.modules: get_summary does
    # `from agent import broker`, which resolves the attribute already bound
    # on the `agent` package once anything has imported it. A sys.modules
    # swap is therefore silently ignored whenever an earlier test in the
    # session imported agent.broker first — order-dependent, and it passes
    # alone while failing in a full run.
    import agent.broker as broker
    monkeypatch.setattr(broker, "enabled", lambda: True)
    monkeypatch.setattr(broker, "Broker", _B)
    monkeypatch.setattr(broker, "_today_et", lambda: TODAY)

    assert od.get_summary("SPY")["expiry"] == "default"
    assert od.get_summary("SPY", expiry=E0)["expiry"] == E0  # not the cached one


# ── iron butterfly mechanics in the ledger ───────────────────

def _fly_legs(expiry):
    """95/100/105 iron butterfly on SPY: wings long, body short."""
    return {"lp": P("SPY", 95, expiry), "sp": P("SPY", 100, expiry),
            "sc": C("SPY", 100, expiry), "lc": C("SPY", 105, expiry)}


def test_short_body_before_wings_is_rejected(store):
    """No combo order exists — a butterfly is four fills, and the wings must
    be bought FIRST or the short body is an uncovered short."""
    from agent import ledger
    legs = _fly_legs(TODAY + timedelta(days=30))

    bad = ledger.record_trade(store, symbol=legs["sc"], side="SELL", shares=1,
                              price=3.0, fill_quote=q(3.0, 3.1))
    assert not bad["ok"] and "uncovered short call" in bad["error"]


def test_wings_first_books_the_full_butterfly(store):
    from agent import ledger
    legs = _fly_legs(TODAY + timedelta(days=30))

    for leg, px in ((legs["lp"], 1.0), (legs["lc"], 1.0)):
        r = ledger.record_trade(store, symbol=leg, side="BUY", shares=1,
                                price=px, fill_quote=q(px - 0.05, px))
        assert r["ok"], r
    for leg, px in ((legs["sp"], 3.0), (legs["sc"], 3.0)):
        r = ledger.record_trade(store, symbol=leg, side="SELL", shares=1,
                                price=px, fill_quote=q(px, px + 0.05))
        assert r["ok"], r

    pos = {r["symbol"]: r["shares"] for r in store.select("desk_positions")}
    assert pos[legs["lp"]] == 1 and pos[legs["lc"]] == 1
    assert pos[legs["sp"]] == -1 and pos[legs["sc"]] == -1

    # net credit, less the $0.65/contract fee on all four legs
    fee = ledger.OPTION_FEE_PER_CONTRACT * 4
    assert ledger.cash(store) == pytest.approx(100_000.0 + 400.0 - fee)
    # the short put reserves the put wing's max loss, not the full strike
    assert ledger.free_cash(store) == pytest.approx(
        ledger.cash(store) - (100 - 95) * 100)


def test_butterfly_settles_flat_with_no_double_assignment(store, monkeypatch):
    """The whole point of holding to expiry: settle unwinds all four legs
    without paying four more spread crossings, and the spread-covered short
    body cash-settles rather than assigning shares we never held."""
    from agent import ledger
    legs = _fly_legs(PAST)

    for leg, side, px in ((legs["lp"], "BUY", 1.0), (legs["lc"], "BUY", 1.0),
                          (legs["sp"], "SELL", 3.0), (legs["sc"], "SELL", 3.0)):
        store.insert("desk_trades",
                     {"account": "agent", "symbol": leg, "side": side,
                      "shares": 1, "price": px, "dollars": px * 100,
                      "run_id": "T", "ts": ledger._utcnow()}, returning=False)
    ledger.rebuild_positions(store)
    cash_before = ledger.cash(store)

    # settles at 103: short call ITM by 3, everything else worthless
    store.insert("daily_bars", {"symbol": "SPY", "date": PAST, "open": 103.0,
                                "high": 103.0, "low": 103.0, "close": 103.0,
                                "volume": 1e6, "source": "test"}, returning=False)
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {})
    monkeypatch.setattr(ledger, "_latest_close", lambda s: None)

    out = ledger.settle(store)
    assert len(out["settled"]) == 4
    assert all(a["basis"] == "expiry_close" for a in out["settled"])

    pos = {r["symbol"]: r["shares"] for r in store.select("desk_positions")
           if abs(float(r["shares"])) > 1e-9}
    assert pos == {}, "every leg must unwind; no stranded short, no shares"
    # only the ITM short call pays out, at intrinsic — settlement is fee-free
    assert ledger.cash(store) == pytest.approx(cash_before - 300.0)

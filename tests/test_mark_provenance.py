"""C2: mark provenance — every equity snapshot records WHICH pricing tier
marked each position (live mid → daily close → cost basis), so a snapshot
written during a quote/data outage (cost marks = fake-flat P&L) is visibly
flagged instead of silently embedded in the curve forever.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'mark.db'}")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setenv("EDGEFINDER_DB_TRANSPORT", "pg")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    Base.metadata.create_all(get_engine())
    from agent.store import get_store
    return get_store()


def q(px):
    return {"bid": px, "ask": px, "mid": px, "t": "x", "src": "test"}


def _buy(store, symbol, shares, price):
    from agent import ledger
    r = ledger.record_trade(store, symbol=symbol, side="BUY", shares=shares,
                            price=price, fill_quote=q(price))
    assert r["ok"], r


def test_mark_meta_counts_each_fallback_tier(store, monkeypatch):
    """One position per tier: AAA gets a live mid, BBB a stored close, CCC
    nothing (cost basis) — the snapshot's mark_meta counts all three and
    names the cost-marked symbol."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)   # $1,000
    _buy(store, "BBB", 10, 100.0)   # $1,000
    _buy(store, "CCC", 10, 100.0)   # $1,000
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {"AAA": 110.0})
    monkeypatch.setattr(ledger, "_latest_closes",
                        lambda syms: {s: 105.0 for s in syms if s == "BBB"})
    st = ledger.mark(store)
    meta = st["mark_meta"]
    assert meta["sources"] == {"live": 1, "close": 1, "cost": 1}
    assert meta["cost_marked"] == ["CCC"]
    # $1,000 of $3,150 marked value ≈ 31.7% at cost → degraded
    assert meta["cost_marked_value_pct"] == pytest.approx(31.75, abs=0.01)
    assert meta["degraded"] is True
    # the snapshot row itself carries the same meta (durable provenance)
    row = store.select("desk_equity", filters={"account": "agent"})[0]
    assert row["mark_meta"]["sources"] == {"live": 1, "close": 1, "cost": 1}


def test_mark_meta_not_degraded_under_threshold(store, monkeypatch):
    """Cost-marked value under MARK_DEGRADED_COST_PCT: recorded, not flagged."""
    from agent import ledger
    _buy(store, "AAA", 100, 100.0)  # $10,000
    _buy(store, "CCC", 10, 100.0)   # $1,000 → ~9% at cost
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {"AAA": 100.0})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    st = ledger.mark(store)
    meta = st["mark_meta"]
    assert meta["sources"] == {"live": 1, "close": 0, "cost": 1}
    assert meta["cost_marked"] == ["CCC"]
    assert meta["cost_marked_value_pct"] < ledger.MARK_DEGRADED_COST_PCT
    assert "degraded" not in meta


def test_mark_explicit_prices_count_as_live(store):
    """Caller-supplied prices are explicit marks — the test/backfill path
    keeps a clean provenance record."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)
    st = ledger.mark(store, prices={"AAA": 101.0})
    assert st["mark_meta"]["sources"] == {"live": 1, "close": 0, "cost": 0}
    assert st["mark_meta"]["cost_marked"] == []
    assert st["mark_meta"]["cost_marked_value_pct"] == 0.0


def test_state_surfaces_latest_mark_meta(store, monkeypatch):
    """`ledger state` carries the LATEST snapshot's provenance so a degraded
    mark is visible on every subsequent read, not only at mark time."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    ledger.mark(store)                      # everything at cost → degraded
    st = ledger.state(store)
    assert st["mark_meta"]["degraded"] is True
    assert st["mark_meta"]["cost_marked_value_pct"] == 100.0
    # before any mark at all, the key is present and honest
    store.delete("desk_equity", {"account": "agent"})
    assert ledger.state(store)["mark_meta"] is None


def test_touch_observation_never_moves_the_booked_equity(store, monkeypatch):
    """THE Phase-B contract: adding the liquidation observation must not change
    a single booked number. Same book, same mids, wildly different touch
    quotes → identical equity."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {"AAA": 110.0})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})

    monkeypatch.setattr(ledger, "_live_touch_quotes", lambda syms: {})
    without = ledger.mark(store)
    monkeypatch.setattr(ledger, "_live_touch_quotes",
                        lambda syms: {"AAA": {"bid": 50.0, "ask": 170.0}})
    with_obs = ledger.mark(store)

    assert without["equity"] == with_obs["equity"]
    assert without["positions_value"] == with_obs["positions_value"]
    assert without["cash"] == with_obs["cash"]
    # the observation is absent without quotes, present with them
    assert "touch_equity" not in without["mark_meta"]
    assert with_obs["mark_meta"]["touch_equity"] < with_obs["equity"]
    assert with_obs["mark_meta"]["mark_basis"] == "mid"


def test_touch_observation_uses_the_exit_side_per_position_sign(store, monkeypatch):
    """A long exits at the bid, a short buys back at the ask — both directions
    make the mid look better than reality, so phantom_pnl is positive for
    each."""
    from agent import ledger
    _buy(store, "AAA", 100, 10.0)                       # long 100 shares
    r = ledger.record_trade(store, symbol="SPY260821P00050000", side="SELL",
                            shares=5, price=4.0, fill_quote=q(4.0))
    assert r["ok"], r                                   # cash-secured put
    monkeypatch.setattr(ledger, "_live_mids",
                        lambda syms: {"AAA": 10.0, "SPY260821P00050000": 5.0})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    monkeypatch.setattr(ledger, "_live_touch_quotes", lambda syms: {
        "AAA": {"bid": 9.0, "ask": 11.0},                     # mid 10.0
        "SPY260821P00050000": {"bid": 4.0, "ask": 6.0},        # mid 5.0
    })
    meta = ledger.mark(store)["mark_meta"]
    # long: 100*(10.0-9.0) = 100 ; short: 5*100*(6.0-5.0) = 500
    assert meta["phantom_pnl"] == pytest.approx(600.0)
    assert meta["touch_quoted"] == 2
    # both quotes are >2% wide
    assert meta["wide_marked"] == ["AAA", "SPY260821P00050000"]


def test_touch_observation_carries_unquoted_positions_at_the_booked_mark(
        store, monkeypatch):
    """A position with no two-sided quote contributes its booked mark, so
    touch_equity stays a whole-book figure instead of a misleading partial."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)
    _buy(store, "BBB", 10, 100.0)
    monkeypatch.setattr(ledger, "_live_mids",
                        lambda syms: {"AAA": 100.0, "BBB": 100.0})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    monkeypatch.setattr(ledger, "_live_touch_quotes",
                        lambda syms: {"AAA": {"bid": 99.0, "ask": 101.0}})
    st = ledger.mark(store)
    meta = st["mark_meta"]
    assert meta["touch_quoted"] == 1                  # only AAA had a quote
    # AAA 10@99 = 990 + BBB carried at its booked 1,000 = 1,990
    assert meta["touch_positions_value"] == pytest.approx(1990.0)
    assert meta["phantom_pnl"] == pytest.approx(10.0)
    assert meta["wide_marked"] == []                  # ~2% exactly, not over


def test_mark_survives_missing_mark_meta_column(store, capsys):
    """L1 pre-deploy grace: a DB that predates the mark_meta column must not
    crash mid-write — the insert retries WITHOUT provenance and prints a
    migration warning (the equity point matters more than its provenance
    for the one deploy where they disagree)."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)

    class NoMetaStore:
        """Delegates everything, but rejects a desk_equity insert that
        carries mark_meta — the error an unmigrated Postgres raises."""

        def __init__(self, inner):
            self._inner = inner
            self.rejected = 0

        def select(self, *a, **kw):
            return self._inner.select(*a, **kw)

        def insert(self, table, rows, **kw):
            if table == "desk_equity" and isinstance(rows, dict) \
                    and "mark_meta" in rows:
                self.rejected += 1
                raise RuntimeError('column "mark_meta" of relation '
                                   '"desk_equity" does not exist')
            return self._inner.insert(table, rows, **kw)

        def update(self, *a, **kw):
            return self._inner.update(*a, **kw)

        def delete(self, *a, **kw):
            return self._inner.delete(*a, **kw)

    proxy = NoMetaStore(store)
    st = ledger.mark(proxy, prices={"AAA": 101.0})
    assert proxy.rejected == 1
    assert "mark_meta" in capsys.readouterr().err
    rows = store.select("desk_equity", filters={"account": "agent"})
    assert len(rows) == 1 and rows[0]["mark_meta"] is None  # snapshot landed
    assert rows[0]["equity"] == pytest.approx(100_000.0 + 10 * 1.0)
    assert st["equity"] == rows[0]["equity"]
    # any OTHER insert failure still raises — the grace is narrow
    class BoomStore(NoMetaStore):
        def insert(self, table, rows, **kw):
            if table == "desk_equity":
                raise RuntimeError("connection reset")
            return self._inner.insert(table, rows, **kw)

    with pytest.raises(RuntimeError, match="connection reset"):
        ledger.mark(BoomStore(store), prices={"AAA": 101.0})


def test_degraded_mark_still_writes_the_snapshot(store, monkeypatch):
    """An outage must not stop the equity curve — degraded marks write,
    flagged, and the CLI warning helper fires on them."""
    from agent import ledger
    _buy(store, "AAA", 10, 100.0)
    monkeypatch.setattr(ledger, "_live_mids", lambda syms: {})
    monkeypatch.setattr(ledger, "_latest_closes", lambda syms: {})
    st = ledger.mark(store)
    assert len(store.select("desk_equity", filters={"account": "agent"})) == 1
    # the loud-warning helper prints for degraded marks and stays silent
    # otherwise (stderr, so stdout stays clean JSON)
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        ledger._warn_if_degraded(st)
    assert "MARKS DEGRADED" in err.getvalue()
    err2 = io.StringIO()
    with contextlib.redirect_stderr(err2):
        ledger._warn_if_degraded({"mark_meta": {"cost_marked_value_pct": 0.0}})
    assert err2.getvalue() == ""

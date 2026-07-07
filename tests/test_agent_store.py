"""The agent's integrity logic must behave identically on either transport.

These tests run the ledger/brain over an in-memory ``FakeStore`` that implements
the same generic CRUD contract as ``RestStore``/``PgStore`` — proving the
dict-based logic (cash recompute, position rebuild, fill guards, decision
upsert) is transport-agnostic, with no DB or network. A second test pins the
PostgREST client's filter/value encoding.
"""

from __future__ import annotations

from datetime import date, datetime


class FakeStore:
    """In-memory stand-in matching the agent.store CRUD contract."""

    transport = "fake"

    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = {}
        self._id = 0

    def _rows(self, table):
        return self.tables.setdefault(table, [])

    def select(self, table, *, filters=None, order=None, limit=None, columns="*"):
        rows = [dict(r) for r in self._rows(table)]
        for col, spec in (filters or {}).items():
            if isinstance(spec, tuple):
                op, val = spec
                ops = {"in": lambda r: r.get(col) in val,
                       "gte": lambda r: r.get(col) >= val,
                       "lte": lambda r: r.get(col) <= val,
                       "gt": lambda r: r.get(col) > val,
                       "lt": lambda r: r.get(col) < val}
                rows = [r for r in rows if ops[op](r)]
            else:
                rows = [r for r in rows if r.get(col) == spec]
        for col, dirn in reversed(order or []):
            rows.sort(key=lambda r: r.get(col), reverse=(dirn == "desc"))
        return rows[:limit] if limit is not None else rows

    def insert(self, table, rows, *, returning=True):
        rows = [rows] if isinstance(rows, dict) else list(rows)
        out = []
        for r in rows:
            self._id += 1
            row = dict(r)
            row.setdefault("id", self._id)
            self._rows(table).append(row)
            out.append(dict(row))
        return out if returning else []

    def update(self, table, filters, values, *, returning=True):
        changed = []
        for r in self._rows(table):
            if all(r.get(k) == v for k, v in filters.items()):
                r.update(values)
                changed.append(dict(r))
        return changed if returning else []

    def delete(self, table, filters):
        self.tables[table] = [
            r for r in self._rows(table)
            if not all(r.get(k) == v for k, v in filters.items())]


def test_ledger_integrity_on_fake_store():
    from agent import ledger
    from agent.models import STARTING_CAPITAL

    store = FakeStore()
    # buy 100 @ 120 (pass latest_close so no data lookup is needed)
    r = ledger.record_trade(store, symbol="NVDA", side="BUY", shares=100,
                            price=120.0, latest_close=120.0, run_id="R1")
    assert r["ok"]
    assert ledger.cash(store) == round(STARTING_CAPITAL - 12000.0, 2)
    st = ledger.state(store)
    assert st["positions"][0]["symbol"] == "NVDA" and st["positions"][0]["shares"] == 100
    assert abs(st["cash"] + st["positions_value"] - st["equity"]) < 0.01

    # partial sell caps at held; cash returns
    r2 = ledger.record_trade(store, symbol="NVDA", side="SELL", shares=40,
                             price=130.0, latest_close=130.0, run_id="R1")
    assert r2["ok"]
    assert ledger.state(store)["positions"][0]["shares"] == 60

    # mark writes exactly one equity snapshot and values the book at the mark
    marked = ledger.mark(store, prices={"NVDA": 130.0})
    assert marked["equity"] == round(ledger.cash(store) + 60 * 130.0, 2)
    assert len(store.tables["desk_equity"]) == 1


def test_ledger_guards_on_fake_store():
    from agent import ledger

    store = FakeStore()
    # fill-sanity: a price wildly off the latest close is rejected
    bad = ledger.record_trade(store, symbol="AAPL", side="BUY", shares=10,
                              price=500.0, latest_close=100.0)
    assert not bad["ok"] and "sanity" in bad["error"]
    # cannot sell what isn't held
    nosell = ledger.record_trade(store, symbol="AAPL", side="SELL", shares=10,
                                 price=100.0, latest_close=100.0)
    assert not nosell["ok"]
    # cannot overdraw cash
    over = ledger.record_trade(store, symbol="AAPL", side="BUY", shares=10_000_000,
                               price=100.0, latest_close=100.0)
    assert not over["ok"]
    # long-only: a SELL is capped at the held quantity, never goes short
    ledger.record_trade(store, symbol="AAPL", side="BUY", shares=10,
                        price=100.0, latest_close=100.0)
    capped = ledger.record_trade(store, symbol="AAPL", side="SELL", shares=999,
                                 price=100.0, latest_close=100.0)
    assert capped["ok"] and capped["shares"] == 10
    assert ledger.state(store)["positions"] == []


def test_brain_state_and_decision_upsert_on_fake_store():
    from agent import brain

    store = FakeStore()
    assert brain.get_state(store)["version"] == 0
    brain.set_state(store, name="v1", thesis="a", rules={"x": 1}, params={"k": 5}, bump=True)
    brain.set_state(store, name="v2", thesis="b", bump=True)
    s = brain.get_state(store)
    assert s["version"] == 2 and s["name"] == "v2"
    # tweak (no bump) edits the latest row in place
    brain.set_state(store, name="v2b", params={"k": 8})
    assert brain.get_state(store)["version"] == 2 and brain.get_state(store)["name"] == "v2b"

    brain.think(store, run_id="R1", phase="observe", text="hello")
    brain.save_decision(store, run_id="R1", regime="risk_on", summary="first")
    brain.save_decision(store, run_id="R1", regime="risk_on", summary="updated")
    # decision upserts on (account, run_id): one row, latest summary wins
    assert len(store.tables["desk_decisions"]) == 1
    assert store.tables["desk_decisions"][0]["summary"] == "updated"


def test_rest_filter_and_value_encoding():
    from agent.rest import Rest, _encode

    r = Rest(url="https://x.supabase.co", key="k")
    assert _encode(date(2026, 6, 18)) == "2026-06-18"
    assert _encode(datetime(2026, 6, 18, 14, 30)) == "2026-06-18T14:30:00"
    params = dict(r._filter_params({
        "account": "agent",
        "symbol": ("in", ["NVDA", "AMD"]),
        "date": ("gte", date(2026, 1, 1)),
    }))
    assert params["account"] == "eq.agent"
    assert params["symbol"] == "in.(NVDA,AMD)"
    assert params["date"] == "gte.2026-01-01"


def test_rest_select_paginates_past_server_cap(monkeypatch):
    """Regression (P2 verifier): PostgREST caps responses (Supabase: 1000);
    select must page with offset instead of silently truncating."""
    import json as _json

    from agent.rest import Rest

    client = Rest.__new__(Rest)  # skip __init__ (no env needed)
    served = [{"id": i} for i in range(2500)]
    calls = []

    def fake_do(method, table, *, params=None, body=None, prefer=None):
        p = dict(params)
        limit, offset = int(p["limit"]), int(p["offset"])
        limit = min(limit, 1000)  # the server-side cap
        calls.append((limit, offset))
        return 200, _json.dumps(served[offset:offset + limit])

    client._do = fake_do
    rows = client.select("daily_bars", filters={"symbol": "SPY"})
    assert len(rows) == 2500 and rows[-1]["id"] == 2499
    assert len(calls) >= 3  # paged, not one capped call
    # explicit limit still honored
    rows = client.select("daily_bars", limit=1500)
    assert len(rows) == 1500

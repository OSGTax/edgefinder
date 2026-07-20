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
    brain.set_state(store, name="v2", thesis="b", bump=True,
                    no_learned_basis="test fixture pivot")
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


def test_rest_filter_range_encodes_repeated_params():
    """A LIST of specs on one column becomes repeated PostgREST params —
    how a server-side [gte, lte] date range is expressed (dict keys are
    unique, so a single tuple can't carry both bounds)."""
    from agent.rest import Rest

    r = Rest(url="https://x.supabase.co", key="k")
    params = r._filter_params({
        "date": [("gte", date(2017, 12, 22)), ("lte", date(2018, 1, 1))]})
    assert params == [("date", "gte.2017-12-22"), ("date", "lte.2018-01-01")]


def test_pg_filter_range_applies_both_bounds(tmp_path, monkeypatch):
    """The pg transport honors the same list-of-specs range filter, so the
    lab's breadth probe and the PIT ranking window behave identically on
    either lane (and are testable on SQLite)."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'range.db'}")
    from edgefinder.db.engine import Base, get_engine
    import agent.models  # noqa: F401
    import edgefinder.db.models  # noqa: F401

    Base.metadata.create_all(get_engine())
    from agent.store import get_store

    store = get_store()
    for d in (date(2017, 12, 20), date(2017, 12, 28), date(2018, 1, 5)):
        store.insert("daily_bars", {"symbol": "AAA", "date": d, "open": 1.0,
                                    "high": 1.0, "low": 1.0, "close": 1.0,
                                    "volume": 1.0, "source": "test"},
                     returning=False)
    rows = store.select("daily_bars", columns="date",
                        filters={"date": [("gte", date(2017, 12, 22)),
                                          ("lte", date(2018, 1, 1))]})
    assert [str(r["date"])[:10] for r in rows] == ["2017-12-28"]


def test_rest_universe_pit_window_bounded_both_sides(monkeypatch):
    """M1 regression: a past ``as_of`` with a gte-only filter matched every
    bar AFTER it too — up to 200 paged HTTPS calls of rows the client threw
    away, every sweep night. Both bounds must reach the store query."""
    import agent.store as agent_store
    from agent.data import _rest_universe

    captured = {}

    class _Store:
        def select(self, table, *, columns="*", filters=None, order=None,
                   limit=None):
            captured["table"] = table
            captured["filters"] = filters
            return [{"symbol": "AAA", "close": 10.0, "volume": 100.0,
                     "date": "2017-12-29"}]

    def fake_get_store():
        return _Store()

    fake_get_store.cache_clear = lambda: None  # conftest teardown calls it
    monkeypatch.setattr(agent_store, "get_store", fake_get_store)
    assert _rest_universe(5, date(2018, 1, 1)) == ["AAA"]
    assert captured["table"] == "daily_bars"
    assert captured["filters"]["date"] == [("gte", "2017-12-22"),
                                           ("lte", "2018-01-01")]


def test_is_duplicate_key_error_covers_both_transports():
    """The constraint-error probe behind the CA insert fallback must
    recognize a unique violation however the active transport surfaces it —
    and nothing else."""
    from sqlalchemy.exc import IntegrityError

    from agent.rest import RestError
    from agent.store import is_duplicate_key_error

    # rest: PostgREST carries the Postgres code in the 409 body
    assert is_duplicate_key_error(RestError(
        409, '{"code":"23505","message":"duplicate key value violates '
             'unique constraint \\"uq_dividends_symbol_exdate\\""}'))
    assert not is_duplicate_key_error(RestError(400, '{"code":"22P02"}'))
    # pg: SQLAlchemy wraps the driver error (SQLite + Postgres wordings)
    assert is_duplicate_key_error(IntegrityError(
        "INSERT", {}, Exception("UNIQUE constraint failed: dividends.symbol")))
    assert is_duplicate_key_error(IntegrityError(
        "INSERT", {}, Exception('duplicate key value violates unique '
                                'constraint "uq_ticker_split"')))
    assert not is_duplicate_key_error(IntegrityError(
        "INSERT", {}, Exception("NOT NULL constraint failed: dividends.symbol")))
    assert not is_duplicate_key_error(RuntimeError("connection reset"))


def test_is_missing_table_error_classifies_by_code_not_string():
    """M3: 'schema not migrated' is diagnosed by exception TYPE + error CODE.
    SQLAlchemy embeds the failed SQL in str(exc), so ANY exception raised
    while querying desk_outcomes contains the table name — a transient
    connection blip must never be misdiagnosed as a missing table."""
    import sqlite3

    from sqlalchemy.exc import OperationalError, ProgrammingError

    from agent.rest import RestError
    from agent.store import is_missing_table_error

    # rest: PostgREST schema-cache miss (PGRST205, HTTP 404) + a proxied raw
    # Postgres 42P01 body
    assert is_missing_table_error(RestError(404, (
        '{"code":"PGRST205","message":"Could not find the table '
        "'public.desk_wiki_history' in the schema cache\"}")))
    assert is_missing_table_error(RestError(
        400, '{"code":"42P01","message":"relation \\"desk_outcomes\\" '
             'does not exist"}'))
    # rest negatives: a plain 404, and an unrelated error MENTIONING the table
    assert not is_missing_table_error(RestError(404, '{"message":"Not Found"}'))
    assert not is_missing_table_error(RestError(
        503, '{"message":"upstream timeout querying desk_outcomes"}'))

    # pg: UndefinedTable carries SQLSTATE 42P01 (psycopg2 .pgcode / psycopg3
    # .sqlstate) on the wrapped driver error
    class _Pg2Err(Exception):
        pgcode = "42P01"

    class _Pg3Err(Exception):
        sqlstate = "42P01"

    assert is_missing_table_error(ProgrammingError(
        "SELECT * FROM desk_outcomes", {},
        _Pg2Err('relation "desk_outcomes" does not exist')))
    assert is_missing_table_error(ProgrammingError(
        "SELECT * FROM desk_outcomes", {},
        _Pg3Err('relation "desk_outcomes" does not exist')))

    # sqlite: OperationalError with the driver's "no such table" message,
    # wrapped by SQLAlchemy or raw
    assert is_missing_table_error(OperationalError(
        "SELECT * FROM desk_wiki_history", {},
        sqlite3.OperationalError("no such table: desk_wiki_history")))
    assert is_missing_table_error(
        sqlite3.OperationalError("no such table: desk_wiki_history"))

    # THE misdiagnosis this replaces: a connection error whose wrapper string
    # embeds SQL naming the table is NOT a missing table
    blip = OperationalError("SELECT * FROM desk_wiki_history WHERE slug = ?",
                            {}, Exception("connection reset by peer"))
    assert "desk_wiki_history" in str(blip)  # the trap the old string-match hit
    assert not is_missing_table_error(blip)
    assert not is_missing_table_error(ProgrammingError(
        "SELECT * FROM desk_outcomes", {}, Exception("some other error")))
    assert not is_missing_table_error(RuntimeError(
        'server closed the connection during "SELECT * FROM desk_outcomes"'))


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

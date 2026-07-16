"""Transport-agnostic table access for the agent.

The agent tools (ledger, brain, announce, data) used to hold a SQLAlchemy
``Session`` and run ORM queries. That only works where the Postgres wire
protocol is reachable (Render, Codespaces, GitHub Actions). The Claude Code
*web* Routine runs in a sandbox that blocks the Postgres ports and allows only
HTTPS/443, so we add a second transport that talks to the same database over
the Supabase Data API (PostgREST).

Both backends expose the SAME small interface — ``select / insert / update /
delete`` over plain dicts — so the integrity logic (cash recompute, position
rebuild, fill guards) lives in ONE place and is unit-testable on SQLite.

Selection (``EDGEFINDER_DB_TRANSPORT``):
  pg    — SQLAlchemy Core over DATABASE_URL (default; Render/CI/Codespaces)
  rest  — Supabase PostgREST over HTTPS (the web Routine)
  auto  — rest iff SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set, else pg

Filters are ``{column: value}`` (equality) or ``{column: (op, value)}`` with
op in {in, gte, lte, gt, lt}; a LIST of (op, value) specs on one column
applies all of them — how a range is expressed
(``{"date": [("gte", lo), ("lte", hi)]}``), since dict keys are unique.
``order`` is a list of ``(column, "asc"|"desc")``.
"""

from __future__ import annotations

import os
from functools import lru_cache


def transport() -> str:
    t = os.getenv("EDGEFINDER_DB_TRANSPORT", "auto").strip().lower()
    if t in ("pg", "rest"):
        return t
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        return "rest"
    return "pg"


# ── PostgREST backend ───────────────────────────────────────


class RestStore:
    transport = "rest"

    def __init__(self) -> None:
        from agent.rest import Rest

        self._rest = Rest()

    def select(self, table, *, filters=None, order=None, limit=None, columns="*"):
        return self._rest.select(table, columns=columns, filters=filters,
                                 order=order, limit=limit)

    def insert(self, table, rows, *, returning=True):
        return self._rest.insert(table, rows, returning=returning)

    def update(self, table, filters, values, *, returning=True):
        return self._rest.update(table, filters, values, returning=returning)

    def delete(self, table, filters):
        self._rest.delete(table, filters)


# ── SQLAlchemy Core backend ─────────────────────────────────


class PgStore:
    transport = "pg"

    def __init__(self) -> None:
        from edgefinder.db.engine import get_engine

        self._engine = get_engine()

    def _table(self, name: str):
        # All ORM models (desk_* and the kept market tables) share one Base
        # metadata, so every table is registered after the models import.
        import agent.models  # noqa: F401 — registers desk_* tables
        import edgefinder.db.models  # noqa: F401 — registers market tables
        from edgefinder.db.engine import Base

        return Base.metadata.tables[name]

    @staticmethod
    def _apply_filters(stmt, table, filters):
        for col, spec in (filters or {}).items():
            c = table.c[col]
            # A LIST of specs ANDs each onto the same column (a range) —
            # mirrors the REST transport's repeated-param encoding.
            for sp in (spec if isinstance(spec, list) else [spec]):
                if isinstance(sp, tuple) and len(sp) == 2:
                    # Dispatch lazily — building every expression eagerly would
                    # call c.in_(val) on scalar values (dates, numbers) and raise.
                    op, val = sp
                    if op == "in":
                        stmt = stmt.where(c.in_(val))
                    elif op == "gte":
                        stmt = stmt.where(c >= val)
                    elif op == "lte":
                        stmt = stmt.where(c <= val)
                    elif op == "gt":
                        stmt = stmt.where(c > val)
                    elif op == "lt":
                        stmt = stmt.where(c < val)
                    else:
                        raise ValueError(f"unknown filter op {op!r}")
                else:
                    stmt = stmt.where(c == sp)
        return stmt

    def select(self, table, *, filters=None, order=None, limit=None, columns="*"):
        from sqlalchemy import select as sa_select

        t = self._table(table)
        if columns and columns != "*":
            stmt = sa_select(*(t.c[c.strip()] for c in columns.split(",")))
        else:
            stmt = sa_select(t)
        stmt = self._apply_filters(stmt, t, filters)
        for col, direction in (order or []):
            c = t.c[col]
            stmt = stmt.order_by(c.desc() if direction == "desc" else c.asc())
        if limit is not None:
            stmt = stmt.limit(limit)
        with self._engine.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings().all()]

    def insert(self, table, rows, *, returning=True):
        from sqlalchemy import insert as sa_insert

        t = self._table(table)
        rows = [rows] if isinstance(rows, dict) else list(rows)
        with self._engine.begin() as conn:
            if returning:
                out = []
                for row in rows:  # RETURNING + multi-values isn't uniform across DBs
                    res = conn.execute(sa_insert(t).returning(t), row)
                    out.append(dict(res.mappings().one()))
                return out
            conn.execute(sa_insert(t), rows)
            return []

    def update(self, table, filters, values, *, returning=True):
        from sqlalchemy import update as sa_update

        t = self._table(table)
        stmt = self._apply_filters(sa_update(t), t, filters).values(**values)
        if returning:
            stmt = stmt.returning(t)
        with self._engine.begin() as conn:
            res = conn.execute(stmt)
            return [dict(r) for r in res.mappings().all()] if returning else []

    def delete(self, table, filters):
        from sqlalchemy import delete as sa_delete

        t = self._table(table)
        stmt = self._apply_filters(sa_delete(t), t, filters)
        with self._engine.begin() as conn:
            conn.execute(stmt)


def is_duplicate_key_error(exc: Exception) -> bool:
    """True when ``exc`` is a unique-constraint (duplicate key) violation,
    however the active transport surfaces it.

    pg   — SQLAlchemy wraps the driver error in ``IntegrityError``; the
           message carries "duplicate key" (Postgres, code 23505) or
           "UNIQUE constraint failed" (SQLite).
    rest — PostgREST answers with the Postgres code in the body
           (``"code":"23505"`` / "duplicate key ... violates").
    """
    from agent.rest import RestError

    if isinstance(exc, RestError):
        body = (exc.body or "").lower()
        return "23505" in body or "duplicate key" in body
    try:
        from sqlalchemy.exc import IntegrityError
    except ImportError:  # pragma: no cover — REST-only host without SQLAlchemy
        return False
    if not isinstance(exc, IntegrityError):
        return False
    msg = str(getattr(exc, "orig", None) or exc).lower()
    return "unique" in msg or "duplicate" in msg


def is_missing_table_error(exc: Exception) -> bool:
    """True when ``exc`` says the target TABLE does not exist (schema not
    migrated), however the active transport surfaces it.

    Classified by exception TYPE + error CODE, never by searching str(exc)
    for the table name — SQLAlchemy embeds the failed SQL (and therefore the
    table name) in every error string, so a transient connection blip during
    ``SELECT ... FROM desk_outcomes`` would string-match and be misdiagnosed
    as "run the DDL". Transient errors must return False and re-raise at the
    call site.

    pg   — SQLAlchemy ``ProgrammingError`` wrapping Postgres UndefinedTable
           (SQLSTATE 42P01: psycopg2 ``pgcode`` / psycopg3 ``sqlstate``), or
           SQLite's ``OperationalError`` whose DRIVER message is
           "no such table" (SQLite has no error codes; the driver message —
           not the SQL-bearing wrapper string — is the signal).
    rest — PostgREST reports a table missing from its schema cache as
           ``PGRST205`` (HTTP 404, "Could not find the table ..."), or
           proxies the raw Postgres ``42P01`` code in the body — mirrors how
           ``is_duplicate_key_error`` handles both transports.
    """
    from agent.rest import RestError

    if isinstance(exc, RestError):
        body = (exc.body or "").lower()
        return ("pgrst205" in body or "42p01" in body
                or (exc.status == 404 and "could not find the table" in body))
    import sqlite3
    if isinstance(exc, sqlite3.OperationalError):  # raw, unwrapped driver error
        return "no such table" in str(exc).lower()
    try:
        from sqlalchemy.exc import OperationalError, ProgrammingError
    except ImportError:  # pragma: no cover — REST-only host without SQLAlchemy
        return False
    orig = getattr(exc, "orig", None)
    if isinstance(exc, ProgrammingError):
        code = getattr(orig, "pgcode", None) or getattr(orig, "sqlstate", None)
        return code == "42P01"
    if isinstance(exc, OperationalError):
        # SQLite reports a missing table here; match the driver's own message
        # (``orig``), NOT the wrapper string that embeds the SQL. A Postgres
        # connection failure is also an OperationalError but never carries
        # this phrase.
        return "no such table" in str(orig or "").lower()
    return False


@lru_cache(maxsize=1)
def get_store():
    """The process-wide store for the active transport."""
    return RestStore() if transport() == "rest" else PgStore()

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
op in {in, gte, lte, gt, lt}. ``order`` is a list of ``(column, "asc"|"desc")``.
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
            if isinstance(spec, tuple) and len(spec) == 2:
                op, val = spec
                stmt = stmt.where({
                    "in": c.in_(val), "gte": c >= val, "lte": c <= val,
                    "gt": c > val, "lt": c < val,
                }[op])
            else:
                stmt = stmt.where(c == spec)
        return stmt

    def select(self, table, *, filters=None, order=None, limit=None, columns="*"):
        from sqlalchemy import select as sa_select

        t = self._table(table)
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


@lru_cache(maxsize=1)
def get_store():
    """The process-wide store for the active transport."""
    return RestStore() if transport() == "rest" else PgStore()

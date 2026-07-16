"""A tiny PostgREST client — the HTTPS transport for the web Routine.

Claude Code on the web runs the agent in a sandbox whose egress is an
HTTP/HTTPS proxy: outbound TCP to the Supabase Postgres pooler (ports
6543/5432) is blocked, only 443 is open. SQLAlchemy/psycopg2 therefore can't
connect. This module reaches the SAME database over the Supabase Data API
(PostgREST) on 443 instead, which the sandbox allows.

It is deliberately stdlib-only (``urllib``) so it adds no dependency and no
package-allowlist surface. ``agent.store`` layers a generic table interface on
top; the agent tools never see this directly.

Auth uses the project's ``service_role`` key (bypasses RLS — the desk_* tables
have RLS enabled). Configure via env: ``SUPABASE_URL`` +
``SUPABASE_SERVICE_ROLE_KEY``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime


class RestError(RuntimeError):
    """A non-2xx PostgREST response (carries status + body for diagnostics)."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"PostgREST {status}: {body[:300]}")
        self.status = status
        self.body = body


def _encode(value) -> str:
    """Render a Python value for a PostgREST filter/body scalar."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _jsonable(value):
    """Make a row value JSON-serializable for an insert/update body."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


class Rest:
    """Minimal CRUD over a Supabase PostgREST endpoint.

    Filters are ``{column: value}`` for equality, or ``{column: ("in", [..])}``
    / ``("gte", v)`` / ``("lte", v)`` for the few non-eq needs. A LIST of
    specs on one column applies all of them — how a range is expressed
    (``{"date": [("gte", lo), ("lte", hi)]}``), since dict keys are unique.
    ``order`` is a list of ``(column, "asc"|"desc")``.
    """

    def __init__(self, url: str | None = None, key: str | None = None,
                 timeout: float = 30.0) -> None:
        self.base = (url or os.environ["SUPABASE_URL"]).rstrip("/") + "/rest/v1"
        self.key = key or os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        self.timeout = timeout

    # ── request plumbing ────────────────────────────────────
    def _headers(self, prefer: str | None = None) -> dict[str, str]:
        h = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if prefer:
            h["Prefer"] = prefer
        return h

    def _filter_params(self, filters: dict | None) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = []
        for col, spec in (filters or {}).items():
            # A LIST of specs applies each to the same column — PostgREST
            # accepts repeated params (``date=gte.A&date=lte.B``), which is
            # how a range is expressed since dict keys are unique.
            for sp in (spec if isinstance(spec, list) else [spec]):
                if isinstance(sp, tuple) and len(sp) == 2:
                    op, val = sp
                    if op == "in":
                        joined = ",".join(_encode(v) for v in val)
                        params.append((col, f"in.({joined})"))
                    else:
                        params.append((col, f"{op}.{_encode(val)}"))
                else:
                    params.append((col, f"eq.{_encode(sp)}"))
        return params

    def _do(self, method: str, table: str, *, params=None, body=None,
            prefer: str | None = None) -> tuple[int, str]:
        qs = urllib.parse.urlencode(params or [])
        url = f"{self.base}/{table}" + (f"?{qs}" if qs else "")
        data = None
        if body is not None:
            data = json.dumps(body, default=_jsonable).encode()
        req = urllib.request.Request(url, data=data, method=method,
                                     headers=self._headers(prefer))
        # GETs retry on transient transport drops (a Disk-IO-throttled
        # Supabase resets connections mid-select). Writes NEVER retry here:
        # a retried POST after an ambiguous failure could double-insert,
        # and the ledger's integrity beats a nightly job's convenience.
        attempts = 3 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.status, resp.read().decode()
            except urllib.error.HTTPError as exc:
                raise RestError(exc.code, exc.read().decode()) from None
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == attempts - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")  # pragma: no cover

    # PostgREST caps a single response (Supabase default: 1000 rows). Reads
    # page through with offset so a big select can never silently truncate —
    # the pre-fix behavior returned the FIRST 1000 by sort order, which for
    # date-ascending bar reads meant the oldest bars and quietly wrong data.
    PAGE_SIZE = 1000

    # ── CRUD ────────────────────────────────────────────────
    def select(self, table: str, *, columns: str = "*", filters: dict | None = None,
               order: list[tuple[str, str]] | None = None,
               limit: int | None = None) -> list[dict]:
        base_params = [("select", columns)]
        base_params += self._filter_params(filters)
        if order:
            base_params.append(("order", ",".join(f"{c}.{d}" for c, d in order)))

        out: list[dict] = []
        offset = 0
        while True:
            page_limit = self.PAGE_SIZE
            if limit is not None:
                page_limit = min(page_limit, limit - len(out))
                if page_limit <= 0:
                    break
            params = list(base_params) + [("limit", str(page_limit)),
                                          ("offset", str(offset))]
            _, body = self._do("GET", table, params=params)
            rows = json.loads(body) if body else []
            out.extend(rows)
            if len(rows) < page_limit:
                break  # server exhausted (or gave us its cap — loop continues only on full pages)
            offset += len(rows)
        return out

    def insert(self, table: str, rows: list[dict] | dict, *,
               returning: bool = True) -> list[dict]:
        prefer = "return=representation" if returning else "return=minimal"
        status, body = self._do("POST", table, body=rows, prefer=prefer)
        return json.loads(body) if (returning and body) else []

    def update(self, table: str, filters: dict, values: dict, *,
               returning: bool = True) -> list[dict]:
        prefer = "return=representation" if returning else "return=minimal"
        _, body = self._do("PATCH", table, params=self._filter_params(filters),
                           body=values, prefer=prefer)
        return json.loads(body) if (returning and body) else []

    def delete(self, table: str, filters: dict) -> None:
        self._do("DELETE", table, params=self._filter_params(filters),
                 prefer="return=minimal")

    def upsert(self, table: str, rows: list[dict] | dict, *, on_conflict: str,
               returning: bool = True) -> list[dict]:
        prefer = ("resolution=merge-duplicates,"
                  + ("return=representation" if returning else "return=minimal"))
        _, body = self._do("POST", table, params=[("on_conflict", on_conflict)],
                           body=rows, prefer=prefer)
        return json.loads(body) if (returning and body) else []

"""Smoke-test every dashboard GET endpoint against a RUNNING server.

Usage:
    python scripts/smoke_dashboard.py                       # localhost:8000
    python scripts/smoke_dashboard.py --base https://edgefinder-pm8h.onrender.com

Asserts HTTP 200 (or an allowed status) and the presence of shape keys.
Read-only; safe against production. Exits non-zero on any failure.
(The SSE /api/desk/stream is the one surface skipped — it never ends.)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

CHECKS: list[tuple[str, list[str], set[int]]] = [
    # (path, required_top_level_keys, allowed_statuses)
    # Every expectation tolerates an EMPTY-but-healthy payload (fresh DB, no
    # trades yet): shape keys only, never row counts. List-shaped responses
    # declare no keys (`k in data` would test list membership).
    ("/api/health", ["status", "version"], {200}),
    # pages: / is the desk now (307 redirect), /desk is the page itself
    ("/", [], {307}),
    ("/desk", [], {200}),
    # /api/desk/* — the read-only projections of the desk_* tables
    ("/api/desk/portfolio", ["cash", "equity", "positions"], {200}),
    ("/api/desk/equity", [], {200}),                       # bare list shape
    ("/api/desk/equity?with_spy=1", ["points", "spy"], {200}),
    ("/api/desk/decision/latest", ["exists"], {200}),
    ("/api/desk/decisions", ["decisions"], {200}),
    ("/api/desk/outcomes", ["summary", "rows"], {200}),
    ("/api/desk/thinking", ["lines"], {200}),
    ("/api/desk/backtests", [], {200}),                    # bare list shape
    ("/api/desk/strategy", ["current", "journal"], {200}),
    ("/api/desk/wiki", ["pages"], {200}),
    ("/api/desk/regime", ["tag"], {200}),
    ("/api/desk/movers", ["gainers", "losers", "most_active"], {200}),
    ("/api/desk/holding-stats", ["symbols"], {200}),
    ("/api/desk/dividends", ["holdings"], {200}),
    ("/api/desk/quotes", ["quotes", "connected"], {200}),
    # allowlist guard: a name the desk neither holds nor watches must 404
    # (the endpoint fans out to metered live options calls when allowed)
    ("/api/desk/options/ZZZQ", [], {404}),
    ("/api/desk/broker-health", ["keys_present"], {200}),
    ("/api/desk/data-health", ["status", "marks"], {200}),
    ("/api/desk/lab", ["top", "combos_tested"], {200}),
    ("/api/desk/brief", ["exists"], {200}),
    ("/api/desk/watch", ["watches", "wakes"], {200}),
    ("/api/desk/whatsnew", ["entries", "new_count"], {200}),
    ("/api/desk/trades?limit=5", [], {200}),               # bare list shape
    # symbol workstation API (404 acceptable when the symbol has no bars
    # in the target environment)
    ("/api/symbols/SPY/bars?range=3m", ["bars", "source"], {200, 404}),
]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D102
        return None


def hit(base: str, path: str, keys: list[str], allowed: set[int]) -> str | None:
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, headers={"User-Agent": "ef-smoke"})
    # When only a redirect status is acceptable, don't follow it — urllib
    # otherwise resolves the 307 to the target page's 200.
    opener = (urllib.request.build_opener(_NoRedirect())
              if allowed and 200 not in allowed and allowed <= {301, 302, 307, 308}
              else urllib.request.build_opener())
    try:
        with opener.open(req, timeout=30) as resp:
            status = resp.status
            body = resp.read()
    except urllib.error.HTTPError as e:
        status, body = e.code, b""
    except Exception as e:  # noqa: BLE001
        return f"EXC {type(e).__name__}: {e}"
    if status not in allowed:
        return f"HTTP {status}"
    if keys and status == 200:
        try:
            data = json.loads(body)
        except Exception:
            return "non-JSON body"
        missing = [k for k in keys if k not in data]
        if missing:
            return f"missing keys: {missing}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="http://localhost:8000")
    args = p.parse_args()

    failures = 0
    for path, keys, allowed in CHECKS:
        err = hit(args.base, path, keys, allowed)
        mark = "ok " if err is None else "FAIL"
        print(f"  [{mark}] {path}" + (f"  -> {err}" if err else ""))
        if err:
            failures += 1
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} passed against {args.base}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

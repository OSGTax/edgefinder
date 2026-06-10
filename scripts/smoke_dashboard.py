"""Smoke-test every dashboard GET endpoint against a RUNNING server.

Usage:
    python scripts/smoke_dashboard.py                       # localhost:8000
    python scripts/smoke_dashboard.py --base https://edgefinder-pm8h.onrender.com

Asserts HTTP 200 (or an allowed status) and the presence of shape keys.
Read-only; safe against production. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

CHECKS: list[tuple[str, list[str], set[int]]] = [
    # (path, required_top_level_keys, allowed_statuses)
    ("/api/health", ["status", "version"], {200}),
    ("/api/strategies", [], {200}),
    ("/api/strategies/accounts", [], {200}),
    ("/api/strategies/equity-curve?days=30", [], {200}),
    ("/api/strategies/scorecard?days=30", [], {200}),
    ("/api/strategies/validation", [], {200}),
    ("/api/strategies/scheduler", [], {200}),
    ("/api/trades?limit=5", [], {200}),
    ("/api/trades/stats", ["total_trades"], {200}),
    ("/api/trades/integrity", [], {200}),
    ("/api/market/regime?limit=1", [], {200}),
    ("/api/benchmarks/comparison?days=5", ["dates", "times", "indices"], {200}),
    ("/api/benchmarks/sectors", [], {200}),
    ("/api/research/search?q=AA&limit=3", [], {200}),
    ("/api/research/active", [], {200}),
    ("/api/ops/health", ["heartbeats"], {200}),
    ("/api/backtest/jobs", [], {200}),
    ("/api/inject", [], {200}),
    # symbol workstation API (404 acceptable when the symbol has no bars
    # in the target environment)
    ("/api/symbols/SPY/bars?range=3m", ["bars", "source"], {200, 404}),
    ("/api/symbols/SPY/bars?range=3m&indicators=true", ["bars"], {200, 404}),
    ("/api/symbols/SPY/events", ["dividends", "splits", "news"], {200}),
    ("/api/lab/runs?limit=5", ["total", "runs"], {200}),
    ("/api/lab/scoreboard", ["target", "finalists", "counts"], {200}),
    ("/api/lab/labels", ["prefixes"], {200}),
    ("/api/strategies/summary", ["arena", "v2", "all"], {200}),
    ("/api/strategies/promoted", [], {200}),
    ("/api/strategies/meta", [], {200}),
    ("/api/ops/activity?limit=10", ["items"], {200}),
    ("/api/ops/storage", ["db"], {200}),
    # pages
    ("/", [], {200}),
    ("/strategies", [], {200}),
    ("/trades", [], {200}),
    ("/research", [], {200, 307}),
    ("/screener", [], {200}),
    ("/backtest", [], {200, 307}),
    ("/lab", [], {200}),
    ("/ops", [], {200}),
]


def hit(base: str, path: str, keys: list[str], allowed: set[int]) -> str | None:
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, headers={"User-Agent": "ef-smoke"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
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

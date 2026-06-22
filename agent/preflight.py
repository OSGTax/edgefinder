"""Preflight — verify the agent can actually run before a cycle starts.

The web Routine sandbox blocks the Postgres port, so the agent reaches its DB
over the Supabase Data API (see ``agent.store``). This command checks, fast and
loud, that the active transport works and the data is fresh — so a broken
environment surfaces a one-line diagnosis in seconds instead of a two-minute
hang deep inside the cycle.

Exit code 0 = ready; non-zero = not ready (see ``checks`` in the JSON).

  python -m agent.preflight
"""

from __future__ import annotations

import json
import sys
from datetime import date


def run() -> dict:
    from agent.store import transport

    out: dict = {"ok": True, "transport": transport(), "checks": {}}

    def check(name: str, fn, critical: bool = True):
        try:
            out["checks"][name] = {"ok": True, "detail": fn()}
        except Exception as exc:  # noqa: BLE001
            out["checks"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
            if critical:
                out["ok"] = False

    # DB reachability via the active transport (a cheap desk_* read)
    def _db():
        from agent.ledger import state

        st = state()
        return {"equity": st["equity"], "positions": len(st["positions"]), "cash": st["cash"]}

    check("db", _db)

    # Market-data freshness (latest stored daily bar)
    def _bars():
        from agent.store import get_store

        rows = get_store().select("daily_bars", columns="date",
                                  order=[("date", "desc")], limit=1)
        if not rows:
            raise RuntimeError("daily_bars empty")
        latest = str(rows[0]["date"])[:10]
        age = (date.today() - date.fromisoformat(latest)).days
        return {"latest_bar": latest, "age_days": age}

    check("market_data", _bars, critical=False)

    # R2 archive (the deep-history bar lane)
    def _r2():
        from agent.data import r2_available

        return {"available": r2_available()}

    check("r2", _r2, critical=False)

    return out


def main(argv: list[str] | None = None) -> int:
    result = run()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

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

    # Universe coverage — bar AGE can look fresh while the nightly whole-market
    # ingest is dead (the hourly top-up keeps a handful of names current), so
    # this measures thin sessions since the last full-coverage ingest instead.
    # One retry: this is the chattiest check (several round-trips) and a single
    # network blip must not bench the trader for a whole cycle.
    def _universe():
        from agent.data import universe_coverage

        try:
            return universe_coverage()
        except Exception:  # noqa: BLE001 — retry once before reporting
            return universe_coverage()

    check("universe_coverage", _universe, critical=False)

    # Sibling routines — the desk is four independent schedules (trading,
    # nightly ingest, app evolver, weekly reflection) and none of them can see
    # a dead sibling on their own. Cheap artifact-age checks make every cycle
    # a watchdog for the others.
    def _siblings():
        from datetime import datetime, timezone

        from agent.store import get_store

        store = get_store()
        now = datetime.now(timezone.utc)

        def age_days(ts) -> float | None:
            if ts is None:
                return None
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return round((now - ts).total_seconds() / 86400, 1)

        detail: dict = {"warnings": []}
        ch = store.select("desk_changelog", columns="ts",
                          order=[("ts", "desc")], limit=1)
        detail["app_evolver_age_days"] = age_days(ch[0]["ts"]) if ch else None
        # Reflection: the journal is the durable low-volume trail — the weekly
        # skill always closes with a "Weekly reflection ..." note. (kind="wiki"
        # rows are NOT reflection-specific: hourly cycles journal wiki edits
        # too.) The thinking feed is only a fallback — at ~17 cycles/day,
        # Friday's reflect-* lines scroll past any sane limit within days.
        jr = store.select("desk_journal", columns="title,ts",
                          order=[("ts", "desc")], limit=100)
        refl = next((r for r in jr
                     if str(r.get("title") or "").startswith("Weekly reflection")),
                    None)
        if refl is None:
            th = store.select("desk_thinking", columns="run_id,ts",
                              order=[("ts", "desc")], limit=300)
            refl = next((r for r in th
                         if str(r.get("run_id") or "").startswith("reflect-")),
                        None)
        detail["reflection_age_days"] = age_days(refl["ts"]) if refl else None
        if detail["app_evolver_age_days"] is None or detail["app_evolver_age_days"] > 3:
            detail["warnings"].append("app-evolver has not shipped in >3 days")
        if detail["reflection_age_days"] is None or detail["reflection_age_days"] > 9:
            detail["warnings"].append("weekly reflection has not run in >9 days")
        return detail

    check("siblings", _siblings, critical=False)

    # R2 archive (the deep-history bar lane)
    def _r2():
        from agent.data import r2_available

        return {"available": r2_available()}

    check("r2", _r2, critical=False)

    # The skill's degrade gate: whole-market research is only trustworthy when
    # the universe coverage is green/amber (see agent.data.coverage_verdict).
    # A check that errored (after its retry) reads as NOT ok — conservative on
    # purpose — with the reason spelled out so the skill can tell "data is
    # stale" from "the check itself could not run".
    cov = out["checks"].get("universe_coverage", {})
    out["research_ok"] = bool(cov.get("ok")
                              and cov.get("detail", {}).get("research_ok"))
    if not out["research_ok"]:
        out["research_ok_reason"] = (
            cov.get("detail", {}).get("status", "unknown")
            if cov.get("ok") else
            f"coverage check failed twice: {cov.get('error', 'unknown')}")

    return out


def main(argv: list[str] | None = None) -> int:
    result = run()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())

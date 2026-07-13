"""The agent's evolving mind — strategy state, journal, thinking, decisions.

These are the write-side tools the charter skill calls each cycle to (a)
read/update its living strategy, (b) journal a pivot with its reasoning, (c)
stream its thinking for the live feed, and (d) save the run's decision +
watchlist. Persistence goes through ``agent.store`` (pg or rest transport).

CLI:
  python -m agent.brain state-get
  python -m agent.brain state-set --name "trend+momentum" --thesis "..." \
      --rules-file rules.json --params-file params.json --bump
  python -m agent.brain journal --kind pivot --title "..." --body "..." --to 2
  python -m agent.brain think --run-id R --phase research --text "..."
  python -m agent.brain decision --run-id R --regime risk_on --summary "..." \
      --weights-file w.json --picks-file picks.json --watchlist-file wl.json
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _store():
    from agent.store import get_store

    return get_store()


def _load_json(path: str | None):
    if not path:
        return None
    with open(path) as fh:
        return json.load(fh)


def _load_text(path: str | None):
    if not path:
        return None
    with open(path) as fh:
        return fh.read()


# ── strategy state ──────────────────────────────────────


def _latest_state_row(store, account: str):
    rows = store.select("desk_strategy_state", filters={"account": account},
                        order=[("version", "desc"), ("id", "desc")], limit=1)
    return rows[0] if rows else None


def get_state(store=None, account: str = "agent") -> dict:
    store = store or _store()
    row = _latest_state_row(store, account)
    if not row:
        return {"exists": False, "version": 0, "name": None, "thesis": None,
                "rules": {}, "params": {}}
    return {"exists": True, "version": row["version"], "name": row["name"],
            "thesis": row.get("thesis"), "rules": row.get("rules") or {},
            "params": row.get("params") or {},
            "updated_at": str(row["updated_at"]) if row.get("updated_at") else None}


def set_state(store=None, *, name: str, thesis: str | None = None,
              rules: dict | None = None, params: dict | None = None,
              bump: bool = False, account: str = "agent") -> dict:
    """Update the living strategy. ``bump`` writes a NEW version row (a pivot);
    otherwise the latest row is updated in place (a tweak)."""
    store = store or _store()
    cur = get_state(store, account)
    if bump or not cur["exists"]:
        version = (cur["version"] or 0) + 1
        store.insert("desk_strategy_state", {
            "account": account, "version": version, "name": name, "thesis": thesis,
            "rules": rules or {}, "params": params or {}, "updated_at": _utcnow()},
            returning=False)
    else:
        row = _latest_state_row(store, account)
        values: dict = {"name": name, "updated_at": _utcnow()}
        if thesis is not None:
            values["thesis"] = thesis
        if rules is not None:
            values["rules"] = rules
        if params is not None:
            values["params"] = params
        store.update("desk_strategy_state", {"id": row["id"]}, values, returning=False)
        version = row["version"]
    return {"ok": True, "version": version, "name": name}


# ── journal ─────────────────────────────────────────────


def add_journal(store=None, *, kind: str, title: str, body: str | None = None,
                version_from: int | None = None, version_to: int | None = None,
                account: str = "agent") -> dict:
    store = store or _store()
    rows = store.insert("desk_journal", {
        "account": account, "kind": kind, "title": title, "body": body,
        "version_from": version_from, "version_to": version_to, "ts": _utcnow()})
    return {"ok": True, "id": (rows[0]["id"] if rows else None)}


# ── thinking feed ───────────────────────────────────────


def think(store=None, *, run_id: str, text: str, phase: str | None = None,
          account: str = "agent") -> dict:
    store = store or _store()
    rows = store.insert("desk_thinking", {
        "account": account, "run_id": run_id, "phase": phase, "text": text,
        "ts": _utcnow()})
    return {"ok": True, "id": (rows[0]["id"] if rows else None)}


# ── decision ────────────────────────────────────────────


def save_decision(store=None, *, run_id: str, regime: str | None = None,
                  summary: str | None = None, target_weights: dict | None = None,
                  picks: list | None = None, watchlist: list | None = None,
                  rejected: list | None = None,
                  strategy_version: int | None = None,
                  decision_date: date | None = None, account: str = "agent") -> dict:
    store = store or _store()
    values = {
        "decision_date": decision_date or date.today(), "regime": regime,
        "summary": summary, "target_weights": target_weights, "picks": picks,
        "watchlist": watchlist, "strategy_version": strategy_version}
    # INSERT-path only: the key is omitted when absent so first writes keep
    # working against a DB that predates the column (render_start's
    # idempotent ALTER adds it on the next deploy). The update path above
    # always writes it — full-rewrite semantics.
    if rejected is not None:
        values["rejected"] = rejected
    existing = store.select("desk_decisions",
                            filters={"account": account, "run_id": run_id}, limit=1)
    if existing:
        # Full-rewrite semantics, same as every other field: an amend without
        # --rejected-file CLEARS the registry — a stale rejected list must
        # never stay paired with rewritten picks.
        store.update("desk_decisions", {"id": existing[0]["id"]},
                     {**values, "rejected": rejected}, returning=False)
        return {"ok": True, "run_id": run_id, "id": existing[0]["id"]}
    rows = store.insert("desk_decisions", {
        "account": account, "run_id": run_id, "ts": _utcnow(), **values})
    return {"ok": True, "run_id": run_id, "id": (rows[0]["id"] if rows else None)}


# ── attention: tripwires + planned wakes ────────────────
#
# The brain owns its own attention. On the way out of a cycle it can arm
# TRIPWIRES (price levels the always-on streamer watches on the live tape)
# and PLAN WAKES (self-scheduled one-shot check-ins with a stated reason).
# wake-plan is the budget gate: the skill may only arm the actual trigger
# after wake-plan says ok, so the cap and minimum gap are enforced and every
# planned check-in is on the record for the desk.

WAKE_MAX_PER_DAY = 20     # self-scheduled wakes per ET day (heartbeat separate)
WAKE_MIN_GAP_MIN = 15     # minutes between planned wakes


def watch_set(store=None, *, symbol: str, above: float | None = None,
              below: float | None = None, reason: str,
              until: str | None = None, hours: float = 24.0,
              run_id: str | None = None, account: str = "agent") -> dict:
    """Arm one tripwire. Exactly one of above/below; reason is mandatory —
    an unexplained tripwire is noise waiting to happen."""
    from datetime import timedelta

    store = store or _store()
    if (above is None) == (below is None):
        return {"ok": False, "error": "pass exactly one of --above / --below"}
    if not (reason or "").strip():
        return {"ok": False, "error": "--reason is required"}
    level = above if above is not None else below
    if level is None or level <= 0:
        return {"ok": False, "error": "level must be positive"}
    expiry = (datetime.fromisoformat(until) if until
              else _utcnow() + timedelta(hours=hours))
    rows = store.insert("desk_watch", {
        "account": account, "run_id": run_id, "symbol": symbol.upper(),
        "kind": "above" if above is not None else "below",
        "level": float(level), "reason": reason.strip(),
        "armed_at": _utcnow(), "until": expiry, "status": "armed"})
    return {"ok": True, "id": rows[0]["id"] if rows else None,
            "symbol": symbol.upper(),
            "kind": "above" if above is not None else "below",
            "level": float(level), "until": str(expiry)}


def watch_list(store=None, *, include_done: bool = False,
               account: str = "agent") -> dict:
    """Armed + tripped wires (the brain reads TRIPPED first at every wake)."""
    store = store or _store()
    rows = store.select("desk_watch", filters={"account": account},
                        order=[("id", "desc")], limit=100)
    out = {"tripped": [], "armed": [], "done": []}
    now = _utcnow()
    for r in rows:
        until = r.get("until")
        if isinstance(until, str):
            try:
                until = datetime.fromisoformat(until.replace("Z", "+00:00"))
                until = until.replace(tzinfo=None)
            except ValueError:
                until = None
        status = r.get("status")
        if status == "armed" and until is not None and until < now:
            status = "expired"  # lazily reported; the sweep also writes it
        entry = {k: (str(v) if isinstance(v, datetime) else v)
                 for k, v in r.items()}
        entry["status"] = status
        if status == "tripped":
            out["tripped"].append(entry)
        elif status == "armed":
            out["armed"].append(entry)
        elif include_done:
            out["done"].append(entry)
    if not include_done:
        out.pop("done")
    return out


def watch_clear(store=None, *, watch_id: int, account: str = "agent") -> dict:
    """Disarm one wire (position exited, level no longer relevant)."""
    store = store or _store()
    rows = store.select("desk_watch",
                        filters={"account": account, "id": watch_id}, limit=1)
    if not rows:
        return {"ok": False, "error": f"no watch id {watch_id}"}
    store.update("desk_watch", {"id": watch_id}, {"status": "disarmed"},
                 returning=False)
    return {"ok": True, "id": watch_id, "status": "disarmed"}


def wake_plan(store=None, *, at: str, reason: str,
              run_id: str | None = None, account: str = "agent") -> dict:
    """Validate + record one planned self-wake. THE budget gate.

    Enforces the per-ET-day cap and the minimum gap. The skill arms the
    actual one-shot trigger ONLY after this returns ok — so every extra
    run the trader grants itself is counted, reasoned, and visible."""
    from datetime import timedelta

    from agent.ledger import _et_date

    store = store or _store()
    if not (reason or "").strip():
        return {"ok": False, "error": "--reason is required"}
    try:
        when = datetime.fromisoformat(at.replace("Z", "+00:00"))
        when = (when.astimezone(timezone.utc).replace(tzinfo=None)
                if when.tzinfo else when)
    except ValueError:
        return {"ok": False, "error": f"unparseable --at {at!r} (use ISO UTC)"}
    now = _utcnow()
    if when <= now:
        return {"ok": False, "error": "--at must be in the future (UTC)"}
    gap = timedelta(minutes=WAKE_MIN_GAP_MIN)
    if when - now < gap:
        return {"ok": False,
                "error": f"too soon: wakes must be >= {WAKE_MIN_GAP_MIN} "
                         "minutes out — the tape is already being watched "
                         "by the streamer and your tripwires"}
    recent = store.select("desk_wakes", filters={"account": account},
                          order=[("at", "desc")], limit=60)
    same_day = [r for r in recent if _et_date(r["at"]) == _et_date(when)]
    if len(same_day) >= WAKE_MAX_PER_DAY:
        return {"ok": False,
                "error": f"budget spent: {WAKE_MAX_PER_DAY} planned wakes "
                         "already on that ET day — the heartbeat still runs"}
    for r in same_day:
        other = r["at"]
        if isinstance(other, str):
            other = datetime.fromisoformat(other.replace("Z", "+00:00"))
            other = other.replace(tzinfo=None)
        if abs((when - other).total_seconds()) < gap.total_seconds():
            return {"ok": False,
                    "error": f"a wake is already planned at {other} — keep "
                             f">= {WAKE_MIN_GAP_MIN} minutes between wakes"}
    rows = store.insert("desk_wakes", {
        "account": account, "run_id": run_id, "at": when,
        "reason": reason.strip(), "created_at": now})
    return {"ok": True, "id": rows[0]["id"] if rows else None,
            "at": str(when),
            "budget_left_today": WAKE_MAX_PER_DAY - len(same_day) - 1}


# ── lessons wiki (Karpathy-style system-prompt learning) ────
#
# A small, size-capped set of curated pages the agent reads at the start of
# every cycle and revises from MEASURED outcomes (see `agent.ledger outcomes`).
# Fixed slugs are the curation constraint; pages are edited in place; every
# edit writes a journal note (kind="wiki") so the audit trail can't be skipped.
# The wiki is ADVISORY context only — it can never loosen a charter guardrail.

WIKI_SLUGS = ("playbook", "lessons", "mistakes", "market-notes")
WIKI_PAGE_MAX_CHARS = 4000    # ~1k tokens per page
WIKI_TOTAL_MAX_CHARS = 12000  # ~3k tokens total — bounded prompt growth, forever


def get_wiki(store=None, *, slug: str | None = None, account: str = "agent") -> dict:
    """The wiki pages (all, or one slug), in canonical order, plus size usage."""
    store = store or _store()
    filters: dict = {"account": account}
    if slug:
        filters["slug"] = slug
    rows = store.select("desk_wiki", filters=filters)
    rows.sort(key=lambda r: (WIKI_SLUGS.index(r["slug"])
                             if r["slug"] in WIKI_SLUGS else len(WIKI_SLUGS)))
    pages = [{"slug": r["slug"], "title": r.get("title"), "body": r["body"],
              "revision": r.get("revision") or 1,
              "updated_at": str(r["updated_at"]) if r.get("updated_at") else None}
             for r in rows]
    return {"pages": pages,
            "total_chars": sum(len(p["body"] or "") for p in pages),
            "caps": {"page": WIKI_PAGE_MAX_CHARS, "total": WIKI_TOTAL_MAX_CHARS},
            "slugs": list(WIKI_SLUGS)}


def set_wiki(store=None, *, slug: str, body: str, title: str | None = None,
             reason: str | None = None, run_id: str | None = None,
             account: str = "agent") -> dict:
    """Create or rewrite one wiki page IN PLACE (curation, not accumulation).

    Rejects unknown slugs and cap breaches before any write. On success the
    page's revision increments and a desk_journal note (kind="wiki") records
    that + why — the audit trail; prior body text is deliberately not kept.
    """
    store = store or _store()
    slug = (slug or "").strip().lower()
    body = body or ""
    if slug not in WIKI_SLUGS:
        return {"ok": False, "error": f"unknown wiki slug {slug!r} — "
                f"the wiki has exactly these pages: {', '.join(WIKI_SLUGS)}"}
    if len(body) > WIKI_PAGE_MAX_CHARS:
        return {"ok": False, "error": "page over the size cap — curate, don't hoard",
                "chars": len(body), "max": WIKI_PAGE_MAX_CHARS}
    others = sum(len(r["body"] or "") for r in
                 store.select("desk_wiki", filters={"account": account})
                 if r["slug"] != slug)
    if others + len(body) > WIKI_TOTAL_MAX_CHARS:
        return {"ok": False, "error": "total wiki over the size cap — prune "
                "another page first", "total_chars": others + len(body),
                "max": WIKI_TOTAL_MAX_CHARS}

    existing = store.select("desk_wiki",
                            filters={"account": account, "slug": slug}, limit=1)
    if existing:
        revision = (existing[0].get("revision") or 1) + 1
        values: dict = {"body": body, "revision": revision,
                        "updated_at": _utcnow(), "updated_run_id": run_id}
        if title is not None:
            values["title"] = title
        store.update("desk_wiki", {"id": existing[0]["id"]}, values, returning=False)
    else:
        revision = 1
        store.insert("desk_wiki", {
            "account": account, "slug": slug, "title": title, "body": body,
            "revision": revision, "updated_at": _utcnow(),
            "updated_run_id": run_id}, returning=False)
    add_journal(store, kind="wiki",
                title=f"wiki/{slug} r{revision}: {reason or 'edited'}"[:200],
                account=account)
    total = others + len(body)
    return {"ok": True, "slug": slug, "revision": revision,
            "chars": len(body), "total_chars": total}


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("state-get")

    ss = sub.add_parser("state-set")
    ss.add_argument("--name", required=True)
    ss.add_argument("--thesis", default=None)
    ss.add_argument("--rules-file", default=None)
    ss.add_argument("--params-file", default=None)
    ss.add_argument("--bump", action="store_true")

    jr = sub.add_parser("journal")
    jr.add_argument("--kind", required=True, choices=["pivot", "tweak", "note"])
    jr.add_argument("--title", required=True)
    jr.add_argument("--body", default=None)
    jr.add_argument("--from", dest="vfrom", type=int, default=None)
    jr.add_argument("--to", dest="vto", type=int, default=None)

    th = sub.add_parser("think")
    th.add_argument("--run-id", required=True)
    th.add_argument("--phase", default=None)
    th.add_argument("--text", required=True)

    wg = sub.add_parser("wiki-get")
    wg.add_argument("--slug", default=None, choices=list(WIKI_SLUGS))

    ws = sub.add_parser("wiki-set")
    ws.add_argument("--slug", required=True, choices=list(WIKI_SLUGS))
    ws.add_argument("--body-file", default=None, help="plain-text/markdown file")
    ws.add_argument("--body", default=None, help="inline body (small edits)")
    ws.add_argument("--title", default=None)
    ws.add_argument("--reason", default=None,
                    help="why this edit — cite the measured result")
    ws.add_argument("--run-id", default=None)

    wset = sub.add_parser("watch-set", help="arm a tripwire on the live tape")
    wset.add_argument("--symbol", required=True)
    wset.add_argument("--above", type=float, default=None)
    wset.add_argument("--below", type=float, default=None)
    wset.add_argument("--reason", required=True)
    wset.add_argument("--until", default=None, help="ISO UTC expiry")
    wset.add_argument("--hours", type=float, default=24.0,
                      help="expiry horizon when --until not given")
    wset.add_argument("--run-id", default=None)

    wl = sub.add_parser("watch-list")
    wl.add_argument("--all", action="store_true", dest="include_done")

    wc = sub.add_parser("watch-clear")
    wc.add_argument("--id", type=int, required=True, dest="watch_id")

    wp = sub.add_parser("wake-plan",
                        help="budget-gate + record a self-scheduled check-in")
    wp.add_argument("--at", required=True, help="ISO UTC fire time")
    wp.add_argument("--reason", required=True)
    wp.add_argument("--run-id", default=None)

    de = sub.add_parser("decision")
    de.add_argument("--run-id", required=True)
    de.add_argument("--regime", default=None)
    de.add_argument("--summary", default=None)
    de.add_argument("--weights-file", default=None)
    de.add_argument("--picks-file", default=None)
    de.add_argument("--watchlist-file", default=None)
    de.add_argument("--rejected-file", default=None,
                    help="JSON [{symbol, why_not}] — candidates that lost the "
                         "slot this run; the weekly reflection grades them")
    de.add_argument("--strategy-version", type=int, default=None)

    args = p.parse_args(argv)
    store = _store()
    if args.cmd == "state-get":
        print(json.dumps(get_state(store), indent=2))
    elif args.cmd == "state-set":
        print(json.dumps(set_state(
            store, name=args.name, thesis=args.thesis,
            rules=_load_json(args.rules_file), params=_load_json(args.params_file),
            bump=args.bump), indent=2))
    elif args.cmd == "journal":
        print(json.dumps(add_journal(
            store, kind=args.kind, title=args.title, body=args.body,
            version_from=args.vfrom, version_to=args.vto), indent=2))
    elif args.cmd == "think":
        print(json.dumps(think(store, run_id=args.run_id, phase=args.phase,
                               text=args.text), indent=2))
    elif args.cmd == "wiki-get":
        print(json.dumps(get_wiki(store, slug=args.slug), indent=2))
    elif args.cmd == "wiki-set":
        body = _load_text(args.body_file) if args.body_file else args.body
        if body is None:
            print(json.dumps({"ok": False,
                              "error": "pass --body-file or --body"}, indent=2))
            return
        print(json.dumps(set_wiki(
            store, slug=args.slug, body=body, title=args.title,
            reason=args.reason, run_id=args.run_id), indent=2))
    elif args.cmd == "decision":
        print(json.dumps(save_decision(
            store, run_id=args.run_id, regime=args.regime, summary=args.summary,
            target_weights=_load_json(args.weights_file),
            picks=_load_json(args.picks_file),
            watchlist=_load_json(args.watchlist_file),
            rejected=_load_json(args.rejected_file),
            strategy_version=args.strategy_version), indent=2))
    elif args.cmd == "watch-set":
        print(json.dumps(watch_set(
            store, symbol=args.symbol, above=args.above, below=args.below,
            reason=args.reason, until=args.until, hours=args.hours,
            run_id=args.run_id), indent=2))
    elif args.cmd == "watch-list":
        print(json.dumps(watch_list(store, include_done=args.include_done),
                         indent=2))
    elif args.cmd == "watch-clear":
        print(json.dumps(watch_clear(store, watch_id=args.watch_id), indent=2))
    elif args.cmd == "wake-plan":
        print(json.dumps(wake_plan(store, at=args.at, reason=args.reason,
                                   run_id=args.run_id), indent=2))


if __name__ == "__main__":
    main()

"""The agent's evolving mind — strategy state, journal, thinking, decisions.

These are the write-side tools the charter skill calls each cycle to (a)
read/update its living strategy, (b) journal a pivot with its reasoning, (c)
stream its thinking for the live feed, and (d) save the run's decision +
watchlist. Persistence goes through ``agent.store`` (pg or rest transport).

CLI:
  python -m agent.brain context   # the cycle's working memory in ONE read
  python -m agent.brain state-get
  python -m agent.brain state-set --name "trend+momentum" --thesis "..." \
      --rules-file rules.json --params-file params.json --bump
  python -m agent.brain journal --kind pivot --title "..." --body "..." --to 2
  python -m agent.brain think --run-id R --phase research --text "..."
  python -m agent.brain decision --run-id R --regime risk_on --summary "..." \
      --weights-file w.json --picks-file picks.json --watchlist-file wl.json
  python -m agent.brain verdict --run-id R --symbol NVDA --verdict TRUE \
      --note "..."                # reflection agent only (desk_outcomes)
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

# The complete pick-action vocabulary the skill defines (SKILL.md step 6).
# Anything else is a typo or an invented verb — rejected at the write so
# grading never meets an action it can't classify.
PICK_ACTIONS = ("hold", "buy", "add", "trim", "exit", "stance")
# Pick actions that OPEN or ADD exposure — these require the prediction
# registry (SKILL.md step 6 calls it REQUIRED; this is the code that makes
# it true). hold/trim/exit picks manage what's already graded, so they may
# carry nulls.
OPENING_ACTIONS = ("buy", "add")
# The pseudo-symbol BOOK records a whole-book stance; it has no fills and
# nothing to grade per-name, so it is exempt from the registry — and only
# these actions make sense for it.
BOOK_ACTIONS = ("hold", "stance")


def _validate_picks(picks: list | None) -> str | None:
    """The prediction registry, enforced at the write. Every pick needs a
    non-empty symbol and an action from PICK_ACTIONS; every opening/adding
    pick must additionally carry a non-empty ``prediction``, an integer
    ``horizon_days`` >= 1, and a non-null ``kill`` — otherwise Friday's
    grading is vibes. Returns an actionable error string, or None when the
    picks pass."""
    problems = []
    for i, p in enumerate(picks or []):
        if not isinstance(p, dict):
            problems.append(f"picks[{i}] is not an object")
            continue
        sym = str(p.get("symbol") or "").strip().upper()
        if not sym:
            problems.append(f"picks[{i}]: 'symbol' must be a non-empty "
                            "ticker (or BOOK for a whole-book stance)")
            continue
        action = str(p.get("action") or "").strip().lower()
        if action not in PICK_ACTIONS:
            problems.append(f"{sym}: unrecognized action {action or 'none'!r}"
                            f" — use one of {'/'.join(PICK_ACTIONS)}")
            continue
        if sym == "BOOK":
            if action not in BOOK_ACTIONS:
                problems.append(
                    f"BOOK: the whole-book pseudo-symbol only takes a "
                    f"{'/'.join(BOOK_ACTIONS)} action, got {action or 'none'!r}"
                    " — trades go on real symbols")
            continue
        if action not in OPENING_ACTIONS:
            continue
        pred = p.get("prediction")
        if not (isinstance(pred, str) and pred.strip()):
            problems.append(f"{sym} ({action}): 'prediction' must be a "
                            "non-empty falsifiable sentence")
        h = p.get("horizon_days")
        if not (isinstance(h, (int, float)) and not isinstance(h, bool)
                and float(h) >= 1 and float(h).is_integer()):
            problems.append(f"{sym} ({action}): 'horizon_days' must be an "
                            "integer >= 1")
        if p.get("kill") in (None, ""):
            problems.append(f"{sym} ({action}): 'kill' (the exit criterion "
                            "that proves you wrong) must not be null")
    if problems:
        return ("prediction registry rejected the save: "
                + "; ".join(problems)
                + ". Every buy/add pick needs prediction + horizon_days + "
                  "kill (SKILL.md step 6) — fill them in, don't drop the pick.")
    return None


def save_decision(store=None, *, run_id: str, regime: str | None = None,
                  summary: str | None = None, target_weights: dict | None = None,
                  picks: list | None = None, watchlist: list | None = None,
                  rejected: list | None = None,
                  strategy_version: int | None = None,
                  decision_date: date | None = None, account: str = "agent") -> dict:
    store = store or _store()
    err = _validate_picks(picks)
    if err:
        return {"ok": False, "error": err}
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
              run_id: str | None = None, hard: bool = False,
              account: str = "agent") -> dict:
    """Arm one tripwire. Exactly one of above/below; reason is mandatory —
    an unexplained tripwire is noise waiting to happen.

    ``hard`` arms a HARD STOP instead of an advisory wire: when the live mid
    touches the level, the streamer itself SELLS THE WHOLE POSITION through
    the ledger's normal fill gates (see agent.streamer.execute_hard_stop).
    Opt-in per position — plain above/below wires never trade. A hard stop
    must be a --below level on a currently-held long EQUITY position, below
    the market so it can't fire the moment it's armed. Arm-time rules —
    protection that cannot fire must not arm:
    - equities only: the sweep watches the equity SIP websocket cache, and
      crypto quotes never enter it, so a crypto hard stop could never trip;
    - the shares must not be backing short calls (a covered-call exit would
      trip and then fail the ledger's coverage gate — leg out first);
    - a live/last reference price must exist to prove the level sits below
      the market (retry once a quote is up)."""
    from datetime import timedelta

    store = store or _store()
    symbol = symbol.upper()
    if (above is None) == (below is None):
        return {"ok": False, "error": "pass exactly one of --above / --below"}
    if not (reason or "").strip():
        return {"ok": False, "error": "--reason is required"}
    level = above if above is not None else below
    if level is None or level <= 0:
        return {"ok": False, "error": "level must be positive"}
    kind = "above" if above is not None else "below"
    if hard:
        from agent import broker, occ

        if below is None:
            return {"ok": False, "error": "a hard stop is a --below level on "
                                          "a long position — pass --below"}
        if occ.is_option(symbol):
            return {"ok": False, "error": "hard stops protect long equity "
                                          "share positions, not option contracts"}
        if broker.is_crypto(symbol):
            # protection that cannot trip must not arm: the sweep only sees
            # the equity SIP websocket cache — crypto quotes never enter it
            return {"ok": False, "error":
                    f"hard stops are equity-only: the streamer's sweep "
                    f"watches the equity SIP tape and {symbol} quotes never "
                    "enter it, so this stop could never trip — manage crypto "
                    "exits in-cycle (or arm an advisory wire and act on the "
                    "trip yourself)"}
        pos = store.select("desk_positions",
                           filters={"account": account, "symbol": symbol}, limit=1)
        held = float(pos[0]["shares"]) if pos else 0.0
        if held <= 0:
            return {"ok": False, "error": f"no long {symbol} position to "
                                          "protect — hard stops arm on names you hold"}
        # Shares that back short calls cannot sell: the stop would trip and
        # then fail the ledger's coverage gate — dead protection. Same math
        # record_trade runs at fill time, applied at arm time instead.
        from agent.ledger import (_check_equity_sell_keeps_calls_covered,
                                  _positions_map)

        err = _check_equity_sell_keeps_calls_covered(
            _positions_map(store, account), symbol, held)
        if err:
            return {"ok": False, "error":
                    f"cannot arm a hard stop on {symbol}: {err}. A stop on "
                    "covered-call shares would trip and then fail the same "
                    "coverage gate — leg out of the short calls before arming"}
        px = None
        try:
            if broker.enabled():
                q = broker.Broker().quotes([symbol]).get(symbol) or {}
                px = q.get("mid") or q.get("bid") or q.get("ask")
        except Exception:  # noqa: BLE001 — fall back to the last mark
            px = None
        if px is None and pos:
            px = pos[0].get("last_price")
        if px is None:
            # Strict: without a reference price we cannot prove the level
            # sits below the market — an at/above-market stop would fire
            # the moment the tape wakes up.
            return {"ok": False, "error":
                    f"no live or last reference price for {symbol} — cannot "
                    "verify the stop sits below the market; retry once a "
                    "quote is up (after the streamer warms or a mark runs)"}
        if float(level) >= float(px):
            return {"ok": False, "error":
                    f"hard stop at {float(level):g} would fire instantly — "
                    f"{symbol} trades at {float(px):g}; pick a level below "
                    "the market"}
        kind = "hard_stop"
    expiry = (datetime.fromisoformat(until) if until
              else _utcnow() + timedelta(hours=hours))
    rows = store.insert("desk_watch", {
        "account": account, "run_id": run_id, "symbol": symbol,
        "kind": kind, "level": float(level), "reason": reason.strip(),
        "armed_at": _utcnow(), "until": expiry, "status": "armed"})
    return {"ok": True, "id": rows[0]["id"] if rows else None,
            "symbol": symbol, "kind": kind,
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
        # exec_failed (a hard stop that tripped but was gated, e.g. market
        # closed) surfaces WITH the tripped wires — it is an unhandled trip
        # the next cycle must address first.
        if status in ("tripped", "exec_failed"):
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
            "budget_left_today": WAKE_MAX_PER_DAY - len(same_day) - 1,
            "note": "a plan is a promise the next heartbeat honors — the "
                    "first cycle at/after this time runs it as a FOCUSED "
                    "wake (Routine sessions have no scheduler MCP)"}


def wake_due(store=None, *, run_id: str | None = None,
             lookback_hours: float = 8.0, account: str = "agent") -> dict:
    """Unhonored wake-plans whose time has come — check at every cycle start.

    Returns plans with ``at`` <= now that no cycle has honored yet (recent
    ones only: a plan that aged past ``lookback_hours`` un-honored is
    reported separately as missed, not resurrected as fresh)."""
    from datetime import timedelta

    store = store or _store()
    now = _utcnow()
    rows = store.select("desk_wakes", filters={"account": account},
                        order=[("at", "desc")], limit=60)
    due, missed = [], []
    for r in rows:
        if r.get("honored_run_id"):
            continue
        at = r["at"]
        if isinstance(at, str):
            try:
                at = datetime.fromisoformat(at.replace("Z", "+00:00"))
                at = at.replace(tzinfo=None) if at.tzinfo is None else \
                    at.astimezone(timezone.utc).replace(tzinfo=None)
            except ValueError:
                continue
        if at > now:
            continue
        entry = {"id": r["id"], "at": str(at), "reason": r.get("reason"),
                 "planned_by_run": r.get("run_id")}
        if now - at <= timedelta(hours=lookback_hours):
            due.append(entry)
        else:
            missed.append(entry)
    return {"due": due, "missed": missed,
            "note": ("honor each due plan as a FOCUSED wake this cycle, then "
                     "wake-honor it" if due else "no due wake-plans")}


def wake_honor(store=None, *, wake_id: int, run_id: str,
               account: str = "agent") -> dict:
    """Stamp a due wake-plan as honored by this run (exactly once)."""
    store = store or _store()
    rows = store.select("desk_wakes",
                        filters={"account": account, "id": wake_id}, limit=1)
    if not rows:
        return {"ok": False, "error": f"no wake id {wake_id}"}
    if rows[0].get("honored_run_id"):
        return {"ok": False, "error": f"wake {wake_id} already honored by "
                                      f"{rows[0]['honored_run_id']}"}
    store.update("desk_wakes", {"id": wake_id},
                 {"honored_run_id": run_id}, returning=False)
    return {"ok": True, "id": wake_id, "honored_run_id": run_id}


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


# ── the learning loop: verdicts + the cycle's working memory ────
#
# `agent.ledger grade` writes MACHINE FACTS into desk_outcomes; the weekly
# reflection judges them and records its verdict here — LLM judgment stored
# durably next to the numbers it judged, instead of evaporating into prose.
# `context` is the other half of the loop: ONE read that puts the wiki,
# brief, strategy, open predictions (with their graded facts), recent
# outcomes, and tripped wires in front of every cycle — so the loop closes
# by construction, not by an LLM remembering to run six tools.

VERDICTS = ("TRUE", "FALSE", "NOT_YET")


def set_verdict(store=None, *, run_id: str, symbol: str, verdict: str,
                note: str | None = None, account: str = "agent") -> dict:
    """Record the reflection agent's judgment on one graded pick.

    The ONLY writer of ``desk_outcomes.verdict`` / ``verdict_note`` —
    ``agent.ledger grade`` writes the machine facts and never touches these
    two columns, so a verdict survives re-grading. The row must exist:
    grade first, judge second."""
    store = store or _store()
    v = (verdict or "").strip().upper().replace("-", "_").replace(" ", "_")
    if v not in VERDICTS:
        return {"ok": False,
                "error": f"verdict must be one of {'/'.join(VERDICTS)}"}
    symbol = (symbol or "").strip().upper()
    try:
        rows = store.select("desk_outcomes",
                            filters={"account": account, "run_id": run_id,
                                     "symbol": symbol}, limit=1)
    except Exception as exc:  # noqa: BLE001 — classify, re-raise others
        # Pre-deploy grace: a missing desk_outcomes table gets an actionable
        # message, not a stack trace mid-reflection.
        if "desk_outcomes" in str(exc):
            return {"ok": False, "error":
                    "desk_outcomes is unreachable — schema not migrated; "
                    "deploy (render_start runs the idempotent DDL) or run "
                    "scripts/setup_db.py", "detail": str(exc)[:200]}
        raise
    if not rows:
        return {"ok": False, "error":
                f"no graded outcome row for run {run_id!r} / {symbol} — run "
                "`python -m agent.ledger grade` first (machine facts before "
                "judgment)"}
    store.update("desk_outcomes", {"id": rows[0]["id"]},
                 {"verdict": v, "verdict_note": note}, returning=False)
    return {"ok": True, "run_id": run_id, "symbol": symbol, "verdict": v,
            "note": note}


CONTEXT_CLIP = 400           # chars per free-text field
CONTEXT_THESIS_CLIP = 2000   # the living thesis gets more room
CONTEXT_MAX_RUNS = 15        # outcome runs included
CONTEXT_MAX_PREDICTIONS = 20
# Brief sections are clipped HERE, not trusted to the brief builder —
# context's boundedness must not depend on another routine's output size.
CONTEXT_BRIEF_ROSTER = 15    # trend_roster rows
CONTEXT_BRIEF_SCREEN = 10    # rows per screens list
CONTEXT_BRIEF_HEADLINES = 10  # headline symbols


def _clip(text, n: int = CONTEXT_CLIP):
    if not isinstance(text, str) or len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _clip_brief(brief: dict) -> dict:
    """Bound the brief's list-heavy sections for the context read (L5): cap
    the trend roster, each screens list, and the headline symbols. The full
    brief stays one `agent.market brief` call away — this is working memory,
    not the archive."""
    payload = brief.get("payload")
    if not isinstance(payload, dict):
        return brief
    payload = dict(payload)
    roster = payload.get("trend_roster")
    if isinstance(roster, list) and len(roster) > CONTEXT_BRIEF_ROSTER:
        payload["trend_roster"] = roster[:CONTEXT_BRIEF_ROSTER]
        payload["trend_roster_clipped"] = len(roster)
    screens = payload.get("screens")
    if isinstance(screens, dict):
        payload["screens"] = {
            k: (v[:CONTEXT_BRIEF_SCREEN] if isinstance(v, list) else v)
            for k, v in screens.items()}
    heads = payload.get("headlines")
    if isinstance(heads, dict) and len(heads) > CONTEXT_BRIEF_HEADLINES:
        payload["headlines"] = dict(list(heads.items())[:CONTEXT_BRIEF_HEADLINES])
        payload["headlines_clipped"] = len(heads)
    return {**brief, "payload": payload}


def context(store=None, *, days: int = 14, account: str = "agent") -> dict:
    """The cycle's WORKING MEMORY in one read — the mandatory first call.

    Read-only aggregation, no new state: the account header (with mark
    provenance), last night's brief (read from desk_briefs exactly as
    ``agent.market brief`` does), the whole lessons wiki (already size-
    capped), the living strategy, every open prediction joined to its latest
    machine-graded desk_outcomes facts, a condensed outcomes summary over
    ``days``, tripped/exec_failed tripwires, and due/missed wake-plans.
    Bounded on purpose — free text is clipped and lists capped, so this
    stays a working set, not a dump; drill into any section with the
    individual tools. A dead section lands in ``errors`` instead of killing
    the read (same convention as the brief builder)."""
    from agent import ledger

    store = store or _store()
    errors: dict[str, str] = {}

    def _safe(name, fn, default):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — one dead section ≠ no context
            errors[name] = f"{type(exc).__name__}: {exc}"
            return default

    st = _safe("account", lambda: ledger.state(store, account), {})
    account_out = {
        "cash": st.get("cash"), "equity": st.get("equity"),
        "total_pnl": st.get("total_pnl"),
        "total_return_pct": st.get("total_return_pct"),
        "mark_meta": st.get("mark_meta"),
        "positions": [{k: p.get(k) for k in
                       ("symbol", "shares", "avg_price", "last_price",
                        "unrealized_pnl", "weight")}
                      for p in st.get("positions") or []]}

    def _brief():
        from agent import market

        return _clip_brief(market.get_brief())

    brief = _safe("brief", _brief, {"exists": False})

    wiki = _safe("wiki", lambda: get_wiki(store, account=account), {})
    strategy = _safe("strategy", lambda: get_state(store, account), {})
    if strategy.get("thesis"):
        strategy = {**strategy, "thesis": _clip(strategy["thesis"],
                                                CONTEXT_THESIS_CLIP)}

    def _open_predictions():
        decisions = store.select("desk_decisions", filters={"account": account},
                                 order=[("ts", "desc")], limit=40)
        graded: dict[tuple, dict] = {}
        for r in store.select("desk_outcomes", filters={"account": account},
                              order=[("id", "desc")], limit=200):
            graded.setdefault((r["run_id"], r["symbol"]), r)
        preds: list[dict] = []
        for d in decisions:
            for p in (d.get("picks") or []):
                sym = str(p.get("symbol") or "").upper()
                action = str(p.get("action") or "").lower()
                if sym == "BOOK" or action not in ("buy", "add") \
                        or not p.get("prediction"):
                    continue
                oc = graded.get((d["run_id"], sym))
                if oc and oc.get("status") == "closed" and oc.get("verdict"):
                    continue  # resolved AND judged — history, not working memory
                facts = None
                if oc:
                    facts = {k: (str(v) if hasattr(v, "isoformat") else v)
                             for k, v in oc.items()
                             if k in ("grade_date", "entry_avg_px", "mark_px",
                                      "mark_basis", "since_pct", "spy_pct",
                                      "alpha_pct", "horizon_days",
                                      "horizon_elapsed", "kill_level",
                                      "kill_breached", "status", "verdict")}
                preds.append({"run_id": d["run_id"],
                              "ts": str(d.get("ts") or ""), "symbol": sym,
                              "action": action,
                              "prediction": _clip(p.get("prediction")),
                              "horizon_days": p.get("horizon_days"),
                              "kill": _clip(p.get("kill"), 200),
                              "outcome": facts})
                if len(preds) >= CONTEXT_MAX_PREDICTIONS:
                    return preds
        return preds

    open_predictions = _safe("open_predictions", _open_predictions, [])

    def _outcomes():
        oc = ledger.outcomes(store, days=days, account=account)
        runs = [{"run_id": r["run_id"], "ts": r["ts"],
                 "summary": _clip(r.get("summary")),
                 "spy_same_window_pct": r.get("spy_same_window_pct"),
                 "spy_window_sessions": r.get("spy_window_sessions"),
                 "run_realized_pnl": r.get("run_realized_pnl"),
                 "picks": [{k: p.get(k) for k in
                            ("symbol", "action", "since_this_run_pct",
                             "closed_return_pct", "alpha_pct", "realized_pnl")}
                           for p in r.get("picks") or []]}
                for r in (oc.get("runs") or [])[:CONTEXT_MAX_RUNS]]
        return {"days": days, "book": oc.get("book"),
                "settlement": oc.get("settlement"),
                "hardstop": oc.get("hardstop"),
                "unattributed_trades": oc.get("unattributed_trades"),
                "runs": runs}

    outcomes_out = _safe("outcomes", _outcomes, {})

    def _watches():
        wl = watch_list(store, account=account)
        return {"tripped": wl.get("tripped") or [],
                "armed": [{k: w.get(k) for k in
                           ("id", "symbol", "kind", "level", "reason", "until")}
                          for w in wl.get("armed") or []]}

    watches = _safe("watches", _watches, {})

    def _wakes():
        out = wake_due(store, account=account)
        now = _utcnow()
        upcoming = []
        for r in store.select("desk_wakes", filters={"account": account},
                              order=[("at", "desc")], limit=30):
            if r.get("honored_run_id"):
                continue
            at = r["at"]
            if isinstance(at, str):
                try:
                    at = datetime.fromisoformat(at.replace("Z", "+00:00"))
                    at = (at.astimezone(timezone.utc).replace(tzinfo=None)
                          if at.tzinfo else at)
                except ValueError:
                    continue
            if at > now:
                upcoming.append({"id": r["id"], "at": str(at),
                                 "reason": _clip(r.get("reason"), 200)})
        out["upcoming"] = upcoming  # planned, not yet due — the next look
        return out

    wakes = _safe("wakes", _wakes, {})

    return {"as_of": str(_utcnow()),
            "note": "the cycle's working memory in one read — account header,"
                    " brief, wiki, strategy, open predictions with their"
                    " machine-graded facts, recent outcomes, tripped wires,"
                    " due wakes. Free text is clipped; drill in with the"
                    " individual tools.",
            "account": account_out, "brief": brief, "wiki": wiki,
            "strategy": strategy, "open_predictions": open_predictions,
            "outcomes": outcomes_out, "watches": watches, "wakes": wakes,
            "errors": errors}


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
    wset.add_argument("--hard", action="store_true",
                      help="arm a HARD STOP: the streamer sells the WHOLE "
                           "position through the normal fill gates when the "
                           "level trips (long EQUITY positions only — crypto "
                           "never reaches the sweep's tape; --below required)")

    wl = sub.add_parser("watch-list")
    wl.add_argument("--all", action="store_true", dest="include_done")

    wc = sub.add_parser("watch-clear")
    wc.add_argument("--id", type=int, required=True, dest="watch_id")

    wp = sub.add_parser("wake-plan",
                        help="budget-gate + record a self-scheduled check-in")
    wp.add_argument("--at", required=True, help="ISO UTC fire time")
    wp.add_argument("--reason", required=True)
    wp.add_argument("--run-id", default=None)

    wd = sub.add_parser("wake-due",
                        help="unhonored due wake-plans (check at cycle start)")

    wh = sub.add_parser("wake-honor")
    wh.add_argument("--id", type=int, required=True, dest="wake_id")
    wh.add_argument("--run-id", required=True)

    vd = sub.add_parser("verdict",
                        help="record the reflection agent's judgment on a "
                             "graded pick (desk_outcomes)")
    vd.add_argument("--run-id", required=True)
    vd.add_argument("--symbol", required=True)
    vd.add_argument("--verdict", required=True,
                    choices=["TRUE", "FALSE", "NOT_YET"])
    vd.add_argument("--note", default=None,
                    help="the judgment in one or two sentences, with numbers")

    cx = sub.add_parser("context",
                        help="the cycle's working memory in one read "
                             "(the MANDATORY first call of every cycle)")
    cx.add_argument("--days", type=int, default=14,
                    help="outcomes window for the summary section")

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
    elif args.cmd == "verdict":
        print(json.dumps(set_verdict(store, run_id=args.run_id,
                                     symbol=args.symbol, verdict=args.verdict,
                                     note=args.note), indent=2))
    elif args.cmd == "context":
        print(json.dumps(context(store, days=args.days), indent=2,
                         default=str))
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
            run_id=args.run_id, hard=args.hard), indent=2))
    elif args.cmd == "watch-list":
        print(json.dumps(watch_list(store, include_done=args.include_done),
                         indent=2))
    elif args.cmd == "watch-clear":
        print(json.dumps(watch_clear(store, watch_id=args.watch_id), indent=2))
    elif args.cmd == "wake-plan":
        print(json.dumps(wake_plan(store, at=args.at, reason=args.reason,
                                   run_id=args.run_id), indent=2))
    elif args.cmd == "wake-due":
        print(json.dumps(wake_due(store), indent=2, default=str))
    elif args.cmd == "wake-honor":
        print(json.dumps(wake_honor(store, wake_id=args.wake_id,
                                    run_id=args.run_id), indent=2))


if __name__ == "__main__":
    main()

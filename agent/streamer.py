"""Live SIP quote streamer — ONE Alpaca WebSocket feeding an in-memory cache.

The always-on Render process owns this: a single connection to Alpaca's SIP
stream (quotes + trades for the configured universe) writes into a process-
local ``QuoteCache``; the desk page reads it over SSE (``/api/desk/stream``)
and the tools read it over ``/api/desk/quotes``. This is the live tape that
prices the agent's fills — see REBUILD-V3.md's honesty contract.

Resilience (all mandatory — Render deploys/replacements kill the socket):
- reconnect with exponential backoff (1s → 60s cap, reset after a healthy run)
- tolerant of Alpaca's "connection limited" (406) during deploy overlap — the
  old instance holds the single allowed socket until it dies; we just retry
- boot-time REST warm so the cache is populated before the first WS tick
- staleness accounting: every entry carries ``recv`` (server epoch seconds);
  consumers treat quotes older than ``settings.stream_stale_secs`` as stale
  while the market is open, and NEVER price a fill off a stale quote.

The cache is only ever written from the single asyncio loop (WS task + warm
task), so no locking is needed; readers get point-in-time dict copies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone

from config.settings import settings

logger = logging.getLogger(__name__)

STREAM_URL = "wss://stream.data.alpaca.markets/v2/{feed}"


def stream_symbols() -> list[str]:
    """The configured seed universe, upper-cased and de-duplicated (order kept)."""
    out: list[str] = []
    for s in (settings.stream_symbols or "").split(","):
        s = s.strip().upper()
        if s and s not in out:
            out.append(s)
    return out


def _held_symbols() -> list[str]:
    """Symbols the agent currently holds — they always belong on the tape."""
    try:
        from agent.models import ACCOUNT
        from agent.store import get_store

        rows = get_store().select("desk_positions", filters={"account": ACCOUNT})
        return sorted({str(r["symbol"]).upper() for r in rows})
    except Exception:  # noqa: BLE001 — the tape must not die on a DB blip
        return []


def watch_symbols() -> list[str]:
    """Seed universe + currently-held names (the full subscription set)."""
    out = stream_symbols()
    for s in _held_symbols():
        if s not in out:
            out.append(s)
    return out


class QuoteCache:
    """Latest quote/trade per symbol + connection status, with staleness."""

    def __init__(self) -> None:
        self._q: dict[str, dict] = {}
        self.connected: bool = False
        self.started_at: float = time.time()
        self.last_msg_at: float | None = None

    # -- writes (single asyncio loop only) --
    def update_quote(self, sym: str, bid, ask, bid_size, ask_size, t: str | None) -> None:
        e = self._q.setdefault(sym, {})
        e.pop("warmed", None)  # live WS data supersedes the REST warm
        e.update(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
                 t=t, recv=time.time())
        if bid and ask:
            e["mid"] = round((bid + ask) / 2, 4)
        self.last_msg_at = time.time()

    def update_trade(self, sym: str, price, t: str | None) -> None:
        e = self._q.setdefault(sym, {})
        # recv = "last market data seen" — a trade is live data, so it counts
        e.update(last=price, last_t=t, recv=time.time())
        self.last_msg_at = time.time()

    def warm(self, quotes: dict[str, dict]) -> None:
        """Seed/refresh from REST latest-quotes WITHOUT overwriting fresh WS
        data. A warmed or stale entry is always refreshable (that's the point
        of the re-warm-while-disconnected path); only a live WS tick younger
        than the stale threshold is protected."""
        now = time.time()
        stale_after = settings.stream_stale_secs
        for sym, q in quotes.items():
            e = self._q.get(sym)
            if (e and e.get("recv") and not e.get("warmed")
                    and now - e["recv"] <= stale_after):
                continue  # fresh live data — keep it
            self._q[sym] = {"bid": q.get("bid"), "ask": q.get("ask"),
                            "mid": q.get("mid"), "bid_size": q.get("bid_size"),
                            "ask_size": q.get("ask_size"), "t": q.get("t"),
                            "recv": now, "warmed": True}

    # -- reads --
    def get(self, sym: str) -> dict | None:
        e = self._q.get(sym.upper())
        return dict(e) if e else None

    def snapshot(self) -> dict:
        now = time.time()
        stale_after = settings.stream_stale_secs
        out = {}
        for sym, e in self._q.items():
            age = round(now - e["recv"], 1) if e.get("recv") else None
            out[sym] = {**e, "age_secs": age,
                        "stale": (age is None or age > stale_after)}
        return {"quotes": out, "connected": self.connected,
                "server_ts": now, "symbols": len(out)}


# The process-wide cache (populated only when the streamer runs).
cache = QuoteCache()


async def _warm(symbols: list[str]) -> None:
    """REST latest-quotes into the cache (blocking SDK → thread)."""
    try:
        from agent import broker
        b = broker.Broker()
        quotes = await asyncio.to_thread(b.quotes, symbols)
        cache.warm(quotes)
        logger.info("Quote cache warmed: %d symbols", len(quotes))
    except Exception:
        logger.exception("Quote cache warm failed (stream will still populate)")


async def _watch_new_holdings(ws, subscribed: set[str]) -> None:
    """Every 5 min, subscribe to any newly-held names so a buy outside the
    seed universe appears on the tape without a restart."""
    while True:
        await asyncio.sleep(300)
        new = [s for s in await asyncio.to_thread(_held_symbols) if s not in subscribed]
        if new:
            await ws.send(json.dumps({"action": "subscribe",
                                      "quotes": new, "trades": new}))
            subscribed.update(new)
            logger.info("Tape subscribed to newly-held: %s", new)


async def _consume(ws, symbols: list[str], creds: dict) -> None:
    """Auth, subscribe, then pump messages into the cache until the socket dies."""
    await ws.send(json.dumps({"action": "auth", "key": creds["key"],
                              "secret": creds["secret"]}))
    await ws.send(json.dumps({"action": "subscribe",
                              "quotes": symbols, "trades": symbols}))
    holdings_task = asyncio.get_running_loop().create_task(
        _watch_new_holdings(ws, set(symbols)))
    try:
        await _pump(ws)
    finally:
        holdings_task.cancel()


async def _pump(ws) -> None:
    async for raw in ws:
        for msg in json.loads(raw):
            kind = msg.get("T")
            if kind == "q":
                cache.update_quote(msg.get("S"), msg.get("bp"), msg.get("ap"),
                                   msg.get("bs"), msg.get("as"), msg.get("t"))
            elif kind == "t":
                cache.update_trade(msg.get("S"), msg.get("p"), msg.get("t"))
            elif kind == "error":
                # 406 = connection limited (deploy overlap): raise to backoff-retry
                raise ConnectionError(f"alpaca stream error: {msg}")
            elif kind == "success" and msg.get("msg") == "authenticated":
                cache.connected = True
                logger.info("SIP stream authenticated")


async def run_stream() -> None:
    """The forever task: connect → consume → on any failure, backoff and retry."""
    from agent import broker

    if not broker.enabled():
        logger.warning("Streamer not started: no Alpaca keys in this environment")
        return
    creds = broker.resolve_creds()
    url = STREAM_URL.format(feed=creds["feed"])

    await _warm(await asyncio.to_thread(watch_symbols))  # DB read off the loop
    backoff = 1.0
    while True:
        try:
            import websockets
            # re-resolve held names on every (re)connect
            symbols = await asyncio.to_thread(watch_symbols)
            async with websockets.connect(url, ping_interval=15,
                                          ping_timeout=15, max_size=2 ** 22) as ws:
                started = time.time()
                try:
                    await _consume(ws, symbols, creds)
                finally:
                    cache.connected = False
                    if time.time() - started > 60:
                        backoff = 1.0  # healthy run → reset backoff
        except asyncio.CancelledError:
            cache.connected = False
            logger.info("Streamer task cancelled (shutdown)")
            raise
        except Exception as exc:  # noqa: BLE001 — the loop must survive anything
            cache.connected = False
            logger.warning("SIP stream dropped (%s: %s) — retry in %.0fs",
                           type(exc).__name__, exc, backoff)
            if backoff >= 8:
                # keep the tape usable while down (DB read off the loop)
                await _warm(await asyncio.to_thread(watch_symbols))
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60.0)


# ── tripwire sweep — cheap code watches so the brain doesn't have to ──
#
# The brain arms desk_watch rows on its way out of a cycle; this sweep
# evaluates them against the in-memory tape every few seconds and marks
# trips/expiries in the DB. One kind — the opt-in 'hard_stop' — goes further
# and EXECUTES a full-position sell through the ledger's normal live-fill
# gates (with the two entry-friction bands explicitly overridden and the
# override stamped on the fill — see execute_hard_stop); plain above/below
# wires only ever flip status (default-off by
# design: nothing trades unless the brain explicitly armed a hard stop on
# that position). FULLY isolated from the socket loop: it runs as its own
# task, every DB touch is wrapped, and any failure degrades to "wires get
# checked at the brain's next wake" — never to a dead tape.

WATCH_SWEEP_SECS = 5      # evaluate armed wires against the cache
WATCH_REFRESH_SECS = 60   # re-read armed wires from the DB
WATCH_EXEC_STALE_SECS = 600  # an 'executing' claim older than this is a crash


def evaluate_watches(watches: list[dict], quotes: dict[str, dict],
                     now: datetime | None = None) -> tuple[list, list]:
    """Pure: (tripped, expired). A wire trips on the live MID crossing its
    level — the same tape the ledger prices fills from. Stale quotes never
    trip a wire (a wire firing off a frozen tape is a false alarm).
    ``hard_stop`` trips exactly like ``below`` (at-or-below the level); what
    differs is what the sweep DOES with the trip, not how it's detected."""
    now = now or datetime.utcnow()
    tripped, expired = [], []
    for w in watches:
        until = w.get("until")
        if isinstance(until, str):
            try:
                until = datetime.fromisoformat(until.replace("Z", "+00:00"))
                until = until.replace(tzinfo=None)
            except ValueError:
                until = None
        if until is not None and until < now:
            expired.append(w)
            continue
        q = quotes.get(str(w.get("symbol") or "").upper()) or {}
        bid, ask = q.get("bid"), q.get("ask")
        recv = q.get("recv")
        fresh = (recv is not None
                 and time.time() - recv <= settings.stream_stale_secs * 6)
        if not (fresh and bid and ask and bid > 0 and ask >= bid):
            continue
        mid = (bid + ask) / 2
        if ((w.get("kind") == "above" and mid >= float(w["level"]))
                or (w.get("kind") in ("below", "hard_stop")
                    and mid <= float(w["level"]))):
            tripped.append({**w, "tripped_price": round(mid, 4)})
    return tripped, expired


def claim_watch(store, watch_id: int, values: dict | None = None) -> bool:
    """Atomically claim one watch: compare-and-swap armed→executing.

    The conditional update (``WHERE id=? AND status='armed'``) is a single
    statement on both transports, so exactly ONE writer can win — the
    changed-row count comes back as the returned representation on both:
    PgStore's ``UPDATE … RETURNING`` yields the updated rows (Postgres and
    SQLite ≥3.35 alike), and RestStore's PostgREST PATCH with
    ``Prefer: return=representation`` yields the rows the filter matched
    and changed. True iff exactly one row transitioned — the caller owns
    the execution; False means another writer (a sibling streamer during
    deploy overlap, a trading cycle) got there first, or the wire is no
    longer armed."""
    rows = store.update("desk_watch",
                        {"id": watch_id, "status": "armed"},
                        {**(values or {}), "status": "executing"},
                        returning=True)
    return len(rows) == 1


def flag_stale_executing(store, now: datetime | None = None) -> int:
    """Flag crashed hard-stop claims: a wire stuck in 'executing' longer
    than WATCH_EXEC_STALE_SECS means its executor died mid-flight. It is
    NEVER auto-retried (the position may or may not have sold — retrying
    blind could double-sell); it becomes 'exec_failed' with reason
    'stale executing claim' so the next trading cycle inspects the ledger
    and decides. The flip is itself a conditional update, so a
    still-running executor's terminal write wins if it lands first."""
    now = now or datetime.utcnow()
    try:
        rows = store.select("desk_watch", filters={"status": "executing"},
                            limit=100)
    except Exception:  # noqa: BLE001 — the sweep must not die on a DB blip
        return 0
    flagged = 0
    for w in rows:
        t = w.get("tripped_at") or w.get("armed_at")
        if isinstance(t, str):
            try:
                t = datetime.fromisoformat(t.replace("Z", "+00:00"))
                t = t.replace(tzinfo=None)
            except ValueError:
                t = None
        if t is not None and (now - t).total_seconds() <= WATCH_EXEC_STALE_SECS:
            continue  # a live executor may still be in flight
        store.update("desk_watch", {"id": w["id"], "status": "executing"},
                     {"status": "exec_failed",
                      "result": "stale executing claim"}, returning=False)
        flagged += 1
        logger.warning("HARD STOP %s (watch %s): stale executing claim "
                       "flagged exec_failed — next cycle must inspect",
                       w.get("symbol"), w.get("id"))
    return flagged


def execute_hard_stop(store, watch: dict, tripped_price: float,
                      now: datetime | None = None) -> dict:
    """Execute ONE tripped hard_stop: a FULL-POSITION SELL through the
    ledger's normal live-fill path — every gate stays in force (session,
    spread, quote staleness, long-only) EXCEPT the two entry-friction
    bands, which a protective exit overrides EXPLICITLY:
    ``allow_price_deviation`` (a >20% gap vs the stored close is the
    canonical stop scenario — an earnings crash must not veto its own
    stop) and ``allow_illiquid`` (a full-position exit may legitimately
    exceed the ADV entry cap; refusing to sell what we already own would
    invert the protection). Honesty is preserved by the receipt, not by
    refusing the exit: both overrides land as ``warnings`` inside the
    persisted ``fill_quote``, so the row shows a gated-but-overridden
    protective exit.

    Concurrency-safe by construction:
    1. CLAIM — an atomic compare-and-swap flips the wire armed→executing
       (``claim_watch``); exactly one writer can win, so two sweeps (or a
       sweep racing a trading cycle) can never sell the same position
       twice. A claim orphaned by a crash is flagged 'exec_failed' by
       ``flag_stale_executing`` — never silently retried.
    2. CORP ACTIONS FIRST — the idempotent per-symbol equity pass
       (``ledger.settle_corp_actions_for``) books any split/dividend that
       executed since the last settle. A split at today's open rebases the
       tape and trips a stop armed below the pre-split price instantly;
       selling the stale PRE-split share count at the POST-split price
       would silently destroy the difference.
    3. FRESH READ — the position is re-read AFTER the claim and the corp
       pass, and that EXPLICIT share count (never "all") is what sells.

    Exactly one attempt: the wire leaves 'armed' at the claim, so the sweep
    never retries — a gated rejection lands as status='exec_failed' with
    the reason in ``result`` for the next trading cycle to see in
    watch-list and decide.

    DEFAULT-OFF GUARANTEE: only a wire the brain explicitly armed with
    kind='hard_stop' ever reaches this function; plain above/below wires
    remain advisory alerts and can never trade."""
    from agent import broker, ledger, occ
    from agent.models import ACCOUNT

    now = now or datetime.utcnow()
    sym = str(watch.get("symbol") or "").upper()
    run_id = f"hardstop:{watch['id']}"
    stamp = {"tripped_at": now, "tripped_price": tripped_price}
    if not claim_watch(store, watch["id"], stamp):
        logger.info("HARD STOP %s @ %.4f — claim lost (another writer owns "
                    "it, or it is no longer armed)", sym, tripped_price)
        return {"ok": False, "status": "claim_lost",
                "error": "watch already claimed/handled by another writer"}
    if occ.is_option(sym) or broker.is_crypto(sym):
        why = "hard stops protect long EQUITY share positions only"
        store.update("desk_watch", {"id": watch["id"]},
                     {**stamp, "status": "stale", "result": why},
                     returning=False)
        logger.info("HARD STOP %s @ %.4f — stale (%s)", sym, tripped_price, why)
        return {"ok": False, "status": "stale", "error": why}
    try:
        corp = ledger.settle_corp_actions_for(store, sym)
        if corp.get("splits") or corp.get("dividends"):
            logger.warning("HARD STOP %s: corp actions booked before exit: %s",
                           sym, corp["details"])
    except Exception as exc:  # noqa: BLE001 — selling on an unverified basis
        # would be worse than not selling; the next cycle inspects.
        store.update("desk_watch", {"id": watch["id"]},
                     {**stamp, "status": "exec_failed",
                      "result": f"corp-action pass failed: {exc}"},
                     returning=False)
        logger.warning("HARD STOP FAILED %s @ %.4f — corp-action pass failed "
                       "(%s); no sale, next cycle decides", sym, tripped_price, exc)
        return {"ok": False, "status": "exec_failed", "error": str(exc)}
    rows = store.select("desk_positions",
                        filters={"account": ACCOUNT, "symbol": sym}, limit=1)
    shares = float(rows[0]["shares"]) if rows else 0.0
    if shares <= 0:
        why = "position already gone at trip time"
        store.update("desk_watch", {"id": watch["id"]},
                     {**stamp, "status": "stale", "result": why},
                     returning=False)
        logger.info("HARD STOP %s @ %.4f — stale (%s)", sym, tripped_price, why)
        return {"ok": False, "status": "stale", "error": why}
    r = ledger.live_fill(
        store, symbol=sym, side="SELL", shares=shares, run_id=run_id,
        # Protective-exit overrides (H1/H2): the close-band and ADV gates
        # veto ENTRIES; a stop that fires INTO the gap it protects against
        # must not be vetoed by it. Stamped on the fill's receipt.
        allow_price_deviation=True, allow_illiquid=True,
        rationale=(f"HARD STOP: full exit of {shares:g} {sym} — live mid "
                   f"{tripped_price:g} hit the armed stop at "
                   f"{float(watch.get('level') or 0):g} "
                   f"({watch.get('reason') or 'no reason recorded'})"))
    if r.get("ok"):
        store.update("desk_watch", {"id": watch["id"]},
                     {**stamp, "status": "executed", "honored_run_id": run_id,
                      "result": f"sold {r['shares']:g} @ {r['price']:g}"},
                     returning=False)
        logger.warning("HARD STOP EXECUTED %s: sold %s @ %s (%s)",
                       sym, r["shares"], r["price"], run_id)
    else:
        store.update("desk_watch", {"id": watch["id"]},
                     {**stamp, "status": "exec_failed",
                      "result": str(r.get("error") or "fill rejected")},
                     returning=False)
        logger.warning("HARD STOP FAILED %s @ %.4f — %s (no retry; the next "
                       "cycle decides)", sym, tripped_price, r.get("error"))
    return r


def apply_sweep_results(store, tripped: list[dict], expired: list[dict],
                        now: datetime | None = None) -> None:
    """Write one sweep's verdicts: advisory wires flip to 'tripped', expired
    wires to 'expired', and hard_stop wires execute (single attempt — their
    status leaves 'armed' either way, so no sweep ever retries them)."""
    now = now or datetime.utcnow()
    for w in tripped:
        if w.get("kind") == "hard_stop":
            execute_hard_stop(store, w, w["tripped_price"], now=now)
        else:
            # conditional on still-armed (the claim_watch idiom): the sweep's
            # cached list can be 60s stale, and an unconditional write would
            # resurrect a wire the brain just disarmed
            store.update("desk_watch", {"id": w["id"], "status": "armed"},
                         {"status": "tripped", "tripped_at": now,
                          "tripped_price": w["tripped_price"]},
                         returning=False)
    for w in expired:
        store.update("desk_watch", {"id": w["id"], "status": "armed"},
                     {"status": "expired"}, returning=False)


async def run_watch_sweep() -> None:
    """The forever sweep task (companion to run_stream, never coupled to it)."""
    armed: list[dict] = []
    last_refresh = 0.0
    while True:
        try:
            if time.time() - last_refresh > WATCH_REFRESH_SECS:
                def _load():
                    from agent.store import get_store
                    s = get_store()
                    # crashed hard-stop claims surface as exec_failed for
                    # the next cycle — never re-armed, never re-executed
                    flag_stale_executing(s)
                    return s.select(
                        "desk_watch", filters={"status": "armed"}, limit=100)
                armed = await asyncio.to_thread(_load)
                last_refresh = time.time()
            if armed:
                snap = {s: dict(e) for s, e in cache._q.items()}
                tripped, expired = evaluate_watches(armed, snap)
                if tripped or expired:
                    def _write():
                        from agent.store import get_store
                        apply_sweep_results(get_store(), tripped, expired)
                    await asyncio.to_thread(_write)
                    for w in tripped:
                        if w.get("kind") != "hard_stop":  # hard stops log inside
                            logger.info("TRIPWIRE %s %s %s @ %.4f — %s",
                                        w["symbol"], w["kind"], w["level"],
                                        w["tripped_price"], w.get("reason"))
                    done = {w["id"] for w in [*tripped, *expired]}
                    armed = [w for w in armed if w["id"] not in done]
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the sweep must never die loudly
            logger.exception("watch sweep failed (retrying in 30s)")
            await asyncio.sleep(30)
        await asyncio.sleep(WATCH_SWEEP_SECS)


# ── the autonomy dispatcher (v9.11.0) ────────────────────────────────
#
# When a planned wake comes due or a tripwire trips, POST a GitHub
# workflow_dispatch so the trading-agent workflow runs a cycle — no human
# finger required. Every decision is made over DB STATE ONLY (desk_wakes,
# desk_watch, desk_dispatches): process memory survives neither restarts
# nor the sibling-instance overlap Render creates during deploys. The
# at-most-once guarantee is the same bucket-CAS idiom as claim_watch.

DISPATCH_PERIOD_SECS = 60          # how often the dispatcher looks
DISPATCH_MIN_GAP_SECS = 300        # >=5 min between dispatches (the bucket) — tight
                                   # enough that a trip between chain cycles isn't
                                   # held at the door for long (v9.12.0)
DISPATCH_MAX_PER_DAY = 45          # per ET day — chain + trips + retry headroom
DISPATCH_MAX_PER_WAKE = 3          # then the wake is stamped missed:auto
DISPATCH_WAKE_LOOKBACK_HOURS = 8   # same "due" definition as brain.wake_due


def dispatch_reason(wakes: list[dict], watches: list[dict],
                    dispatches: list[dict],
                    now: datetime | None = None) -> dict | None:
    """Pure decision: should the streamer fire a trading cycle right now?

    ``wakes``/``watches``/``dispatches`` are plain row dicts (naive-UTC
    timestamps, both transports). Returns {"reason", "wake_ids",
    "watch_ids"} or None. Enforces: the min-gap debounce and the per-ET-day
    cap (from the dispatch ledger), the 8h due-window and per-wake attempt
    cap (immortal wakes are the classic infinite-loop cost trap), and
    EDGE-triggered trip detection — a tripped wire fires only if it tripped
    AFTER the newest SUCCESSFUL ('sent') dispatch. Level-triggered status
    would re-fire forever on one un-cleared wire; comparing against ANY
    newest attempt (the old rule) let a single FAILED POST silence a trip
    permanently."""
    now = now or datetime.utcnow()

    def _dt(v):
        if isinstance(v, datetime):
            return v.replace(tzinfo=None) if v.tzinfo else v
        try:
            d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d
        except (TypeError, ValueError):
            return None

    disp_times = sorted((t for t in (_dt(d.get("ts")) for d in dispatches) if t))
    if disp_times and (now - disp_times[-1]).total_seconds() < DISPATCH_MIN_GAP_SECS:
        return None
    # ET-day cap: naive-UTC minus 4/5h is close enough for a rate cap;
    # use a fixed -5 (EST) so the cap is never looser than intended.
    et_day = (now - timedelta(hours=5)).date()
    today = [t for t in disp_times if (t - timedelta(hours=5)).date() == et_day]
    if len(today) >= DISPATCH_MAX_PER_DAY:
        return None

    lookback = now - timedelta(hours=DISPATCH_WAKE_LOOKBACK_HOURS)
    due = [w for w in wakes
           if not w.get("honored_run_id")
           and int(w.get("dispatch_count") or 0) < DISPATCH_MAX_PER_WAKE
           and (t := _dt(w.get("at"))) is not None and lookback <= t <= now]
    sent_times = sorted(t for d in dispatches if d.get("status") == "sent"
                        and (t := _dt(d.get("ts"))) is not None)
    last_sent = sent_times[-1] if sent_times else datetime.min
    trips = [w for w in watches
             if w.get("status") in ("tripped", "exec_failed")
             and (t := _dt(w.get("tripped_at"))) is not None and t > last_sent]

    if not due and not trips:
        return None
    parts = []
    if trips:
        parts.append("tripwire " + ", ".join(
            f"{w.get('symbol')} {w.get('kind')} {w.get('level')}" for w in trips[:3]))
    if due:
        parts.append(f"{len(due)} wake-plan(s) due")
    return {"reason": " + ".join(parts),
            "wake_ids": [w["id"] for w in due],
            "watch_ids": [w["id"] for w in trips]}


def claim_dispatch_slot(store, decision: dict,
                        now: datetime | None = None) -> int | None:
    """CAS-claim this debounce window: insert the UNIQUE (account, bucket)
    row BEFORE posting. A duplicate-key loss means a sibling instance owns
    the window — stand down. Returns the row id or None."""
    from agent.store import is_duplicate_key_error

    now = now or datetime.utcnow()
    bucket = int(now.timestamp()) // DISPATCH_MIN_GAP_SECS
    try:
        rows = store.insert("desk_dispatches", {
            "account": "agent", "bucket": bucket, "ts": now,
            "reason": decision["reason"], "wake_ids": decision["wake_ids"],
            "watch_ids": decision["watch_ids"], "status": "claimed",
        }, returning=True)
        return rows[0]["id"] if rows else None
    except Exception as exc:  # noqa: BLE001
        if is_duplicate_key_error(exc):
            return None
        raise


def fire_workflow_dispatch(reason: str) -> int:
    """POST the workflow_dispatch (stdlib urllib; 204 = accepted).

    workflow_dispatch (not repository_dispatch) so the PAT needs only
    Actions:write — it cannot push code or read repo secrets. The reason
    carries ids/labels only; the workflow prompt is static and never sees
    this text."""
    import urllib.request

    url = (f"https://api.github.com/repos/{settings.github_dispatch_repo}"
           f"/actions/workflows/{settings.github_dispatch_workflow}/dispatches")
    body = json.dumps({"ref": "main",
                       "inputs": {"reason": reason[:200]}}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {settings.github_dispatch_token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status


def _run_dispatch_once(store, now: datetime | None = None) -> dict | None:
    """One dispatcher pass (sync; called via to_thread). Returns the
    decision it acted on, or None."""
    now = now or datetime.utcnow()
    wakes = store.select("desk_wakes", filters={"account": "agent"},
                         order=[("at", "desc")], limit=60)
    watches = store.select("desk_watch", limit=100,
                           filters={"status": ("in", ["tripped", "exec_failed"])})
    # The read must cover a full ET day of rows or the DISPATCH_MAX_PER_DAY
    # cap can never bind (the old limit=40 read made the 45/day cap dead
    # code). 300 > the bucket-math ceiling of 86400/DISPATCH_MIN_GAP_SECS.
    dispatches = store.select("desk_dispatches", order=[("ts", "desc")], limit=300)
    decision = dispatch_reason(wakes, watches, dispatches, now=now)

    # Terminal-resolve exhausted wakes so they can never loop the dispatcher:
    # an unhonored wake at the attempt cap is stamped missed:auto (honest —
    # it fired cycles that chose not to honor it, or the market was closed
    # and the skill's honor-before-stop didn't exist yet).
    lookback = now - timedelta(hours=DISPATCH_WAKE_LOOKBACK_HOURS)
    for w in wakes:
        if (not w.get("honored_run_id")
                and int(w.get("dispatch_count") or 0) >= DISPATCH_MAX_PER_WAKE):
            store.update("desk_wakes",
                         {"id": w["id"], "honored_run_id": None},
                         {"honored_run_id": "missed:auto"}, returning=False)

    if not decision:
        return None
    slot = claim_dispatch_slot(store, decision, now=now)
    if slot is None:
        return None
    try:
        status = fire_workflow_dispatch(decision["reason"])
        store.update("desk_dispatches", {"id": slot},
                     {"status": "sent", "http_status": status}, returning=False)
        for wid in decision["wake_ids"]:
            row = store.select("desk_wakes", filters={"id": wid}, limit=1)
            if row:
                store.update("desk_wakes", {"id": wid},
                             {"dispatch_count": int(row[0].get("dispatch_count") or 0) + 1},
                             returning=False)
        logger.info("AUTONOMY dispatch fired (%s): %s", status, decision["reason"])
        return decision
    except Exception as exc:  # noqa: BLE001 — mark failed; next bucket retries
        code = getattr(exc, "code", None)
        store.update("desk_dispatches", {"id": slot},
                     {"status": "failed", "http_status": code}, returning=False)
        if code in (401, 403):
            logger.error("AUTONOMY dispatch token rejected (%s) — the PAT is "
                         "expired or under-scoped; the loop is DEAD until the "
                         "owner rotates it", code)
            try:
                from agent.brain import journal
                journal(store, kind="note",
                        title="Autonomy dispatch token rejected",
                        body=f"GitHub returned {code} for workflow_dispatch — "
                             "the fine-grained PAT on Render is expired or "
                             "under-scoped. Machine-fired trading cycles are "
                             "OFF until it is rotated; the cron floor and "
                             "manual fires still work.")
            except Exception:  # noqa: BLE001
                logger.exception("could not journal the dead-dispatch alert")
        else:
            logger.exception("AUTONOMY dispatch POST failed (will retry next bucket)")
        return None


async def run_wake_dispatch() -> None:
    """Forever task: the autonomy dispatcher. Separate from run_watch_sweep
    on purpose — a hung GitHub POST must never stall the 5-second hard-stop
    loop — and started even without Alpaca keys (it needs only DB + PAT)."""
    if not settings.github_dispatch_token.strip():
        logger.info("Autonomy dispatcher disabled (no EDGEFINDER_GITHUB_DISPATCH_TOKEN)")
        return
    logger.info("Autonomy dispatcher up: %s/%s every %ss",
                settings.github_dispatch_repo, settings.github_dispatch_workflow,
                DISPATCH_PERIOD_SECS)
    while True:
        try:
            def _pass():
                from agent.store import get_store
                return _run_dispatch_once(get_store())
            await asyncio.to_thread(_pass)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the dispatcher must never die loudly
            logger.exception("autonomy dispatcher pass failed (retrying)")
        await asyncio.sleep(DISPATCH_PERIOD_SECS)


_sweep_task: asyncio.Task | None = None  # keep a ref so GC can't collect it
_dispatch_task: asyncio.Task | None = None


def start_in(_app=None) -> asyncio.Task | None:
    """Start the streamer as an asyncio task (called from the FastAPI lifespan).
    Returns the task, or None when keys are absent (dev/CI/tests). The
    autonomy dispatcher starts UNCONDITIONALLY (needs only DB + PAT) so
    revoked Alpaca keys can never silently kill machine-fired cycles."""
    global _sweep_task, _dispatch_task
    from agent import broker

    _dispatch_task = asyncio.get_running_loop().create_task(
        run_wake_dispatch(), name="wake-dispatch")
    if not broker.enabled():
        logger.info("Live streamer disabled (no Alpaca keys)")
        return None
    task = asyncio.get_running_loop().create_task(run_stream(), name="sip-streamer")
    _sweep_task = asyncio.get_running_loop().create_task(
        run_watch_sweep(), name="watch-sweep")
    return task

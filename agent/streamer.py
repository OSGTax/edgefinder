"""Live SIP quote streamer — ONE Alpaca WebSocket feeding an in-memory cache.

The always-on Render process owns this: a single connection to Alpaca's SIP
stream (quotes + trades for the configured universe) writes into a process-
local ``QuoteCache``; the desk page reads it over SSE (``/api/desk/stream``)
and the tools read it over ``/api/desk/quotes``. REBUILD-V4: this is the
desk page's LIVE TAPE — fills execute on Alpaca's paper engine against
its own NBBO view and protective stops rest on Alpaca's book, so the old
tripwire sweep and hard-stop executor no longer exist here — PLUS the
chain-wake dispatcher (V4.1): fired Routine sessions have no scheduler
MCP (probed 2026-07-13, re-proven 2026-08-10), so this always-on process
is the chain's clock — it polls ``desk_wakes`` and POSTs the
"EdgeFinder chain wakes" Routine's API /fire endpoint when a planned
wake comes due, with the ``desk_dispatches`` CAS ledger making every
window at-most-once.

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
    """EQUITY symbols the agent currently holds — they belong on the tape.

    Sourced from the Alpaca paper account (the book of record) with a
    legacy-projection fallback, via ``refresh._held_equity_symbols``. OCC
    option symbols are EXCLUDED there: this tape is the stock SIP feed, and
    Alpaca rejects the WHOLE batch when one symbol is not a stock — a
    single held option once killed the live tape for 13 days (2026-07-16 →
    07-29) by poisoning both the REST warm and the WS subscription.
    """
    try:
        from agent.refresh import _held_equity_symbols
        from agent.store import get_store

        return sorted(set(_held_equity_symbols(get_store())))
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
    """REST latest-quotes into the cache (blocking SDK → thread).

    Defence in depth: Alpaca fails the WHOLE batch on one unquotable symbol,
    which used to leave the cache completely EMPTY and the desk with nothing
    to mark against. On a batch failure, fall back to warming symbol by
    symbol so one bad ticker costs one quote instead of all of them, and
    NAME the offenders in the log so the next occurrence is diagnosable
    without code archaeology.
    """
    from agent import broker

    try:
        b = broker.Broker()
    except Exception:  # noqa: BLE001 — no broker → nothing to warm
        logger.exception("Quote cache warm failed to build a broker")
        return
    try:
        quotes = await asyncio.to_thread(b.quotes, symbols)
        cache.warm(quotes)
        logger.info("Quote cache warmed: %d symbols", len(quotes))
        return
    except Exception as exc:  # noqa: BLE001 — fall back to per-symbol
        logger.warning("Batch quote warm failed (%s: %s) — retrying per symbol",
                       type(exc).__name__, str(exc)[:200])

    quotes: dict = {}
    bad: list[str] = []
    for sym in symbols:
        try:
            quotes.update(await asyncio.to_thread(b.quotes, [sym]))
        except Exception:  # noqa: BLE001 — skip the offender, keep the rest
            bad.append(sym)
    if quotes:
        cache.warm(quotes)
    logger.warning("Quote cache warmed per-symbol: %d ok, %d unquotable%s",
                   len(quotes), len(bad),
                   (" — " + ", ".join(bad[:10])) if bad else "")


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



# ── the chain-wake dispatcher (V4.1) ─────────────────────────────────────
#
# Fired Routine sessions cannot create triggers (no scheduler MCP — probed
# 2026-07-13, re-proven live 2026-08-10), so a cycle's wake-plan row is a
# PROMISE something external must fire. This is that something: the same
# battle-tested V3 dispatcher loop, with the GitHub workflow_dispatch POST
# replaced by the "EdgeFinder chain wakes" Routine's API /fire endpoint
# (owner-created web-UI trigger; research-preview beta header per
# https://code.claude.com/docs/en/routines).

DISPATCH_PERIOD_SECS = 60          # how often the dispatcher looks
DISPATCH_MIN_GAP_SECS = 300        # >=5 min between fires (the CAS bucket)
DISPATCH_MAX_PER_DAY = 60          # per ET day — chain + retry headroom,
                                   # 1.5x brain.WAKE_MAX_PER_DAY (40)
DISPATCH_MAX_PER_WAKE = 3          # then the wake is stamped missed:auto
DISPATCH_WAKE_LOOKBACK_HOURS = 8   # same "due" definition as brain.wake_due
RESTART_MIN_GAP_SECS = 25 * 60     # chain-restart pacing: never stack a
                                   # restart on a fire <25 min old (the
                                   # session it spawned may still be booting)
ROUTINE_FIRE_BETA = "experimental-cc-routine-2026-04-01"


def dispatch_reason(wakes: list[dict], dispatches: list[dict],
                    now: datetime | None = None, *,
                    chain_quiet: bool = False) -> dict | None:
    """Pure decision: should the dispatcher fire a trading cycle right now?

    ``wakes``/``dispatches`` are plain row dicts (naive-UTC timestamps,
    both transports). Returns {"reason", "wake_ids"} or None. Enforces the
    min-gap debounce and the per-ET-day cap (from the dispatch ledger),
    the 8h due-window, and the per-wake attempt cap (immortal wakes are
    the classic infinite-loop cost trap). Two fire causes (V4.1.1 — the
    hourly floor Routine's job moved in here): a due wake-plan, or
    ``chain_quiet`` (desk hours, no cycle in 25 min — computed by the
    caller from ``brain.chain_health``), the latter paced by
    RESTART_MIN_GAP_SECS against the newest SENT fire so a booting
    session is never stacked on."""
    now = now or datetime.utcnow()

    def _dt(v):
        if isinstance(v, datetime):
            return v.replace(tzinfo=None) if v.tzinfo else v
        try:
            d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            return d.astimezone(timezone.utc).replace(tzinfo=None) if d.tzinfo else d
        except (TypeError, ValueError):
            return None

    disp_times = sorted(t for t in (_dt(d.get("ts")) for d in dispatches) if t)
    if disp_times and (now - disp_times[-1]).total_seconds() < DISPATCH_MIN_GAP_SECS:
        return None
    # ET-day cap: naive-UTC minus a fixed 5h (EST) — never looser than intended.
    et_day = (now - timedelta(hours=5)).date()
    today = [t for t in disp_times if (t - timedelta(hours=5)).date() == et_day]
    if len(today) >= DISPATCH_MAX_PER_DAY:
        return None

    lookback = now - timedelta(hours=DISPATCH_WAKE_LOOKBACK_HOURS)
    due = [w for w in wakes
           if not w.get("honored_run_id")
           and int(w.get("dispatch_count") or 0) < DISPATCH_MAX_PER_WAKE
           and (t := _dt(w.get("at"))) is not None and lookback <= t <= now]
    if due:
        return {"reason": f"{len(due)} wake-plan(s) due",
                "wake_ids": [w["id"] for w in due]}
    if chain_quiet:
        sent = sorted(t for d in dispatches if d.get("status") == "sent"
                      and (t := _dt(d.get("ts"))) is not None)
        if not sent or (now - sent[-1]).total_seconds() >= RESTART_MIN_GAP_SECS:
            return {"reason": "chain restart: desk hours, no cycle in 25+ min",
                    "wake_ids": []}
    return None


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
            "status": "claimed",
        }, returning=True)
        return rows[0]["id"] if rows else None
    except Exception as exc:  # noqa: BLE001
        if is_duplicate_key_error(exc):
            return None
        raise


def fire_routine(reason: str) -> int:
    """POST the chain-wakes Routine's /fire endpoint (stdlib urllib).

    200 = a session was created. The bearer token can ONLY fire this one
    routine (per-routine scope); the reason text rides the ``text`` field,
    which the platform wraps as untrusted payload — the routine's saved
    prompt is what the session acts on, so ids/labels here are display
    only and the prompt-injection surface stays closed."""
    import urllib.request

    req = urllib.request.Request(
        settings.routine_fire_url,
        data=json.dumps({"text": reason[:200]}).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {settings.routine_fire_token}",
            "anthropic-beta": ROUTINE_FIRE_BETA,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status


def _chain_quiet(store, now: datetime) -> bool:
    """Desk hours with no cycle in 25 min (brain.chain_health — the ONE
    definition, shared with what the floor Routine used to read). False on
    any error: a broken health read must never fire cycles."""
    try:
        from agent.brain import chain_health

        ch = chain_health(store, now=now)
        return bool(ch.get("should_run") and not ch.get("wakes_due"))
    except Exception:  # noqa: BLE001
        logger.exception("chain_health read failed — treating chain as active")
        return False


def _run_dispatch_once(store, now: datetime | None = None,
                       fire=fire_routine, quiet_fn=_chain_quiet) -> dict | None:
    """One dispatcher pass (sync; called via to_thread). Returns the
    decision it acted on, or None."""
    now = now or datetime.utcnow()
    wakes = store.select("desk_wakes", filters={"account": "agent"},
                         order=[("at", "desc")], limit=60)
    # The read must cover a full ET day of rows or the daily cap can never
    # bind. 300 > the bucket-math ceiling of 86400/DISPATCH_MIN_GAP_SECS.
    dispatches = store.select("desk_dispatches", order=[("ts", "desc")], limit=300)
    decision = dispatch_reason(wakes, dispatches, now=now,
                               chain_quiet=quiet_fn(store, now))

    # Terminal-resolve exhausted wakes so they can never loop the dispatcher:
    # an unhonored wake at the attempt cap is stamped missed:auto (honest —
    # it fired cycles that chose not to honor it, or the market was closed).
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
        status = fire(decision["reason"])
        store.update("desk_dispatches", {"id": slot},
                     {"status": "sent", "http_status": status}, returning=False)
        for wid in decision["wake_ids"]:
            row = store.select("desk_wakes", filters={"id": wid}, limit=1)
            if row:
                store.update("desk_wakes", {"id": wid},
                             {"dispatch_count": int(row[0].get("dispatch_count") or 0) + 1},
                             returning=False)
        logger.info("CHAIN dispatch fired (%s): %s", status, decision["reason"])
        return decision
    except Exception as exc:  # noqa: BLE001 — mark failed; next bucket retries
        code = getattr(exc, "code", None)
        store.update("desk_dispatches", {"id": slot},
                     {"status": "failed", "http_status": code}, returning=False)
        if code in (401, 403):
            logger.error("CHAIN dispatch token rejected (%s) — regenerate the "
                         "routine's API token and update Render; NOTHING "
                         "fires trading cycles until then", code)
            # Edge-triggered journal: only when the PREVIOUS attempt wasn't
            # already a 401/403 — a dead token retries every restart window
            # and must not bury the desk journal in duplicates.
            prior = dispatches[0] if dispatches else None
            already_noted = (prior is not None
                             and prior.get("status") == "failed"
                             and prior.get("http_status") in (401, 403))
            if already_noted:
                return None
            try:
                from agent.brain import add_journal

                add_journal(store, kind="note",
                            title="Chain-wake fire token rejected",
                            body=f"The Routines API returned {code} for the "
                                 "chain-wakes /fire call — the bearer token "
                                 "on Render is expired, revoked, or wrong. "
                                 "NO trading cycles can fire until the owner "
                                 "regenerates the token "
                                 "(claude.ai/code/routines → EdgeFinder chain "
                                 "wakes → API trigger → Regenerate) and "
                                 "updates EDGEFINDER_ROUTINE_FIRE_TOKEN on "
                                 "Render. Resting stops on Alpaca's book "
                                 "still protect every position meanwhile.")
            except Exception:  # noqa: BLE001
                logger.exception("could not journal the token rejection")
        else:
            logger.exception("chain dispatch POST failed (will retry)")
        return None


async def run_wake_dispatch() -> None:
    """Forever task: the chain's clock. Separate from the tape task on
    purpose — a hung POST must never stall the stream — and started even
    without Alpaca keys (it needs only DB + the fire credentials)."""
    if not (settings.routine_fire_url.strip()
            and settings.routine_fire_token.strip()):
        logger.info("Chain dispatcher disabled (no EDGEFINDER_ROUTINE_FIRE_URL"
                    "/_TOKEN) — NOTHING fires trading cycles; set them")
        return
    logger.info("Chain dispatcher up: firing %s every %ss when wakes come due",
                settings.routine_fire_url.split("/routines/")[-1],
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
            logger.exception("chain dispatcher pass failed (retrying)")
        await asyncio.sleep(DISPATCH_PERIOD_SECS)


_stream_task: asyncio.Task | None = None  # keep refs so GC can't collect them
_dispatch_task: asyncio.Task | None = None


def start_in(_app=None) -> asyncio.Task | None:
    """Start the background jobs from the FastAPI lifespan: the chain-wake
    dispatcher (UNCONDITIONALLY — it needs only DB + fire credentials, so
    revoked Alpaca keys can never silently kill machine-fired cycles) and
    the SIP tape task (None when Alpaca keys are absent — dev/CI/tests).
    Protective stops still rest on Alpaca's own book; the dispatcher fires
    cycles, never orders."""
    global _stream_task, _dispatch_task
    from agent import broker

    _dispatch_task = asyncio.get_running_loop().create_task(
        run_wake_dispatch(), name="chain-dispatch")
    if not broker.enabled():
        logger.info("Live streamer disabled (no Alpaca keys)")
        return None
    _stream_task = asyncio.get_running_loop().create_task(
        run_stream(), name="sip-streamer")
    return _stream_task

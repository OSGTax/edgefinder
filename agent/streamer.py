"""Live SIP quote streamer — ONE Alpaca WebSocket feeding an in-memory cache.

The always-on Render process owns this: a single connection to Alpaca's SIP
stream (quotes + trades for the configured universe) writes into a process-
local ``QuoteCache``; the desk page reads it over SSE (``/api/desk/stream``)
and the tools read it over ``/api/desk/quotes``. REBUILD-V4: this is the
desk page's LIVE TAPE only — fills execute on Alpaca's paper engine
against its own NBBO view, protective stops rest on Alpaca's book, and
cycles self-schedule via Routine triggers, so the old tripwire sweep,
hard-stop executor, and GitHub wake-dispatcher no longer exist here.

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



_stream_task: asyncio.Task | None = None  # keep a ref so GC can't collect it


def start_in(_app=None) -> asyncio.Task | None:
    """Start the streamer as an asyncio task (called from the FastAPI
    lifespan). Returns the task, or None when keys are absent (dev/CI/
    tests). REBUILD-V4: this is the ONLY background job — the tripwire
    sweep, the hard-stop executor, and the GitHub wake-dispatcher are gone
    (protective stops rest on Alpaca's own book; cycles schedule
    themselves via one-shot Routine triggers)."""
    global _stream_task
    from agent import broker

    if not broker.enabled():
        logger.info("Live streamer disabled (no Alpaca keys)")
        return None
    _stream_task = asyncio.get_running_loop().create_task(
        run_stream(), name="sip-streamer")
    return _stream_task

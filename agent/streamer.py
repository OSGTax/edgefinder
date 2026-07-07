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
        e.update(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size,
                 t=t, recv=time.time())
        if bid and ask:
            e["mid"] = round((bid + ask) / 2, 4)
        self.last_msg_at = time.time()

    def update_trade(self, sym: str, price, t: str | None) -> None:
        e = self._q.setdefault(sym, {})
        e.update(last=price, last_t=t)
        e.setdefault("recv", time.time())
        self.last_msg_at = time.time()

    def warm(self, quotes: dict[str, dict]) -> None:
        """Seed from REST latest-quotes WITHOUT overwriting fresher WS data."""
        now = time.time()
        for sym, q in quotes.items():
            if sym in self._q and self._q[sym].get("recv"):
                continue
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


async def _consume(ws, symbols: list[str], creds: dict) -> None:
    """Auth, subscribe, then pump messages into the cache until the socket dies."""
    await ws.send(json.dumps({"action": "auth", "key": creds["key"],
                              "secret": creds["secret"]}))
    await ws.send(json.dumps({"action": "subscribe",
                              "quotes": symbols, "trades": symbols}))
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
                logger.info("SIP stream authenticated; %d symbols", len(symbols))


async def run_stream() -> None:
    """The forever task: connect → consume → on any failure, backoff and retry."""
    from agent import broker

    if not broker.enabled():
        logger.warning("Streamer not started: no Alpaca keys in this environment")
        return
    creds = broker.resolve_creds()
    url = STREAM_URL.format(feed=creds["feed"])
    symbols = stream_symbols()

    await _warm(symbols)
    backoff = 1.0
    while True:
        try:
            import websockets
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
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60.0)


def start_in(loop_owner_app) -> asyncio.Task | None:
    """Start the streamer as an asyncio task (called from the FastAPI lifespan).
    Returns the task, or None when keys are absent (dev/CI/tests)."""
    from agent import broker

    if not broker.enabled():
        logger.info("Live streamer disabled (no Alpaca keys)")
        return None
    task = asyncio.get_running_loop().create_task(run_stream(), name="sip-streamer")
    return task

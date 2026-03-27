"""EdgeFinder v2 — Polygon.io WebSocket streaming.

Connects to Polygon's WebSocket API for real-time minute bars,
trades, and quotes. Includes auto-reconnect with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Awaitable, Callable

import websockets

from config.settings import settings
from edgefinder.core.models import BarData

logger = logging.getLogger(__name__)


class PolygonStreamProvider:
    """WebSocket streaming from Polygon.io for real-time data."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or settings.polygon_api_key
        self._ws_url = settings.polygon_ws_url
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._subscribed: set[str] = set()
        self._on_bar: Callable[[BarData], Awaitable[None]] | None = None
        self._on_trade: Callable[[dict], Awaitable[None]] | None = None
        self._on_quote: Callable[[dict], Awaitable[None]] | None = None
        self._running = False
        self._auto_reconnect = True
        self._max_reconnect_delay = 30.0

    async def connect(self) -> None:
        """Establish WebSocket connection and authenticate."""
        self._ws = await websockets.connect(self._ws_url)

        # Read connection status message
        msg = await self._ws.recv()
        logger.info("WS connected: %s", msg)

        # Authenticate
        await self._ws.send(
            json.dumps({"action": "auth", "params": self._api_key})
        )
        auth_resp = await self._ws.recv()
        logger.info("WS auth response: %s", auth_resp)
        self._running = True

    async def subscribe(
        self,
        tickers: list[str],
        on_bar: Callable[[BarData], Awaitable[None]] | None = None,
        on_trade: Callable[[dict], Awaitable[None]] | None = None,
        on_quote: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        """Subscribe to real-time data for given tickers."""
        if on_bar:
            self._on_bar = on_bar
        if on_trade:
            self._on_trade = on_trade
        if on_quote:
            self._on_quote = on_quote

        channels = []
        for t in tickers:
            if self._on_bar:
                channels.append(f"AM.{t}")
            if self._on_trade:
                channels.append(f"T.{t}")
            if self._on_quote:
                channels.append(f"Q.{t}")
            self._subscribed.add(t)

        if self._ws and channels:
            await self._ws.send(
                json.dumps({"action": "subscribe", "params": ",".join(channels)})
            )
            logger.info("WS subscribed to: %s", ", ".join(channels))

    async def unsubscribe(self, tickers: list[str]) -> None:
        """Unsubscribe from tickers."""
        channels = []
        for t in tickers:
            channels.extend([f"AM.{t}", f"T.{t}", f"Q.{t}"])
            self._subscribed.discard(t)

        if self._ws and channels:
            await self._ws.send(
                json.dumps({"action": "unsubscribe", "params": ",".join(channels)})
            )
            logger.info("WS unsubscribed from: %s", ", ".join(channels))

    async def listen(self) -> None:
        """Main loop: read messages and dispatch to callbacks."""
        if not self._ws:
            raise RuntimeError("Not connected. Call connect() first.")
        try:
            async for raw in self._ws:
                messages = json.loads(raw)
                if not isinstance(messages, list):
                    messages = [messages]
                for msg in messages:
                    await self._dispatch(msg)
        except websockets.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self._running = False
            if self._auto_reconnect:
                await self._reconnect()

    async def disconnect(self) -> None:
        """Cleanly close the WebSocket connection."""
        self._running = False
        self._auto_reconnect = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    # ── Private ──────────────────────────────────────

    async def _dispatch(self, msg: dict) -> None:
        ev = msg.get("ev")
        if ev == "AM" and self._on_bar:
            bar = BarData(
                timestamp=datetime.fromtimestamp(msg["s"] / 1000),
                open=msg["o"],
                high=msg["h"],
                low=msg["l"],
                close=msg["c"],
                volume=msg["v"],
                vwap=msg.get("vw"),
            )
            await self._on_bar(bar)
        elif ev == "T" and self._on_trade:
            await self._on_trade(msg)
        elif ev == "Q" and self._on_quote:
            await self._on_quote(msg)

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff, then resubscribe."""
        delay = 1.0
        while self._auto_reconnect:
            try:
                logger.info("Attempting WS reconnect in %.1fs...", delay)
                await asyncio.sleep(delay)
                await self.connect()
                if self._subscribed:
                    await self.subscribe(list(self._subscribed))
                logger.info("WS reconnected and resubscribed")
                await self.listen()
                return
            except Exception as e:
                logger.warning("WS reconnect failed: %s", e)
                delay = min(delay * 2, self._max_reconnect_delay)

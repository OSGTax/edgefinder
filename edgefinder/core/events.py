"""EdgeFinder v2 — Simple in-process event bus (pub/sub).

Lightweight decoupling between modules. Not Kafka, not Redis — just an
in-process callback registry so modules can react to events without
circular imports.

Planned event types:
    trade.opened, trade.closed, trade.cancelled
    signal.generated, signal.rejected
    scan.completed
    market.snapshot
    sentiment.update
    strategy.paused, strategy.resumed
    injection.added
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventBus:
    """Simple synchronous pub/sub event bus.

    Usage:
        bus = EventBus()
        bus.subscribe("trade.opened", my_handler)
        bus.publish("trade.opened", trade_data)
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[..., Any]) -> None:
        self._subscribers[event_type] = [
            h for h in self._subscribers[event_type] if h is not handler
        ]

    def publish(self, event_type: str, data: Any = None) -> None:
        for handler in self._subscribers.get(event_type, []):
            try:
                handler(data)
            except Exception:
                logger.exception("Error in handler for event '%s'", event_type)

    def clear(self) -> None:
        """Remove all subscribers. Useful for testing."""
        self._subscribers.clear()


# Module-level singleton — import this everywhere
event_bus = EventBus()

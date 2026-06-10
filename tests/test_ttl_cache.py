"""TTLCache — expiry, LRU eviction, thread-safety surface."""

import time

from dashboard.ttl_cache import TTLCache


def test_set_get_roundtrip():
    c = TTLCache(maxsize=4, ttl_seconds=60)
    c.set("a", 1)
    assert c.get("a") == 1
    assert c.get("missing") is None


def test_expiry(monkeypatch):
    c = TTLCache(maxsize=4, ttl_seconds=10)
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base)
    c.set("a", 1)
    monkeypatch.setattr(time, "monotonic", lambda: base + 11)
    assert c.get("a") is None
    assert len(c) == 0


def test_lru_eviction():
    c = TTLCache(maxsize=2, ttl_seconds=60)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")            # refresh a's recency
    c.set("c", 3)         # evicts b (least recently used)
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_clear():
    c = TTLCache()
    c.set("a", 1)
    c.clear()
    assert c.get("a") is None

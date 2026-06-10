"""Tiny thread-safe TTL + LRU cache for the dashboard's hot data.

Used for R2 bar frames (a GET is 100-300ms/symbol; a cached full-history
frame is ~5,400 rows — trivial memory) and the R2 manifest summary. Not a
general-purpose cache: values are returned by reference, so callers must
treat them as immutable.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict


class TTLCache:
    def __init__(self, maxsize: int = 128, ttl_seconds: float = 900.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            expires_at, value = hit
            if time.monotonic() >= expires_at:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key, value) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

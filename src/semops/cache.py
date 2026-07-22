"""A tiny content-addressed cache.

Serves the three purposes the KB calls out (Part III S3.3 / S5.1):
  - cost:          skip repeat model calls
  - latency:       ditto
  - reproducibility: a cached (model, prompt, input) -> output is stable by
                     construction, which is how we buy determinism in a DB context.

Default is in-process (dict). Swap `store` for a KV/Couchbase-backed dict-like
to share across processes.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Any, MutableMapping, Optional


def make_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, store: Optional[MutableMapping[str, Any]] = None, enabled: bool = True):
        self._store: MutableMapping[str, Any] = store if store is not None else {}
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()  # shared across worker threads

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        if self.enabled:
            with self._lock:
                self._store[key] = value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

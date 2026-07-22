"""In-memory engine — the portable "other engines" baseline.

Brute-force cosine over an in-process list. Zero dependencies, so the whole
library (and the cascade's correctness) is exercisable on a laptop with no
cluster and no network. Also the reference the Couchbase native mode is
benchmarked against.
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..types import Row
from ..vectormath import cosine_similarity
from .base import BaseEngine, EngineCaps


def _matches(doc: dict, filters: Optional[dict]) -> bool:
    if not filters:
        return True
    return all(doc.get(k) == v for k, v in filters.items())


class InMemoryEngine(BaseEngine):
    name = "inmemory"
    caps = EngineCaps(ann=True, server_side_scoring=False, pushdown_filter=True, writeback=True)

    def __init__(self) -> None:
        self._collections: dict[str, dict[str, Row]] = {}

    # --- ingestion helpers (not part of the Engine contract) ---
    def add(self, source: str, rows: Sequence[Row]) -> None:
        coll = self._collections.setdefault(source, {})
        for r in rows:
            coll[r.id] = r

    # --- Engine contract ---
    def scan(self, source, filters=None, limit=None, with_vectors=True):
        rows = [r for r in self._collections.get(source, {}).values() if _matches(r.doc, filters)]
        return rows[:limit] if limit is not None else rows

    def ann_candidates(self, source, query_vector, k, filters=None, *, with_vectors=True):
        # with_vectors is accepted for contract parity; in-memory rows already hold
        # their embeddings, so there is nothing to avoid shipping.
        scored = [
            (r, cosine_similarity(r.embedding, query_vector) if r.embedding else 0.0)
            for r in self._collections.get(source, {}).values()
            if _matches(r.doc, filters)
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def count(self, source):
        return len(self._collections.get(source, {}))

    def upsert(self, source, rows):
        self.add(source, rows)

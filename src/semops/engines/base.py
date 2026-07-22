"""Engine modes — the seam between operators and where data/vectors live.

An Engine says, via `caps`, which fast paths it can do natively; operators take
the native path when offered and fall back to a portable in-service path
otherwise. Couchbase is a first-class native mode (pushdown + writeback), not an
adapter conforming down to a lowest common denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ..types import Row
from ..vectormath import cosine_similarity


@dataclass(frozen=True)
class EngineCaps:
    ann: bool = False                        # can retrieve vector top-k
    server_side_scoring: bool = False        # can compute proxy scores near the data
    pushdown_filter: bool = False            # can apply structured predicates in the scan
    writeback: bool = False                  # can UPSERT/MERGE results back
    native_ops: frozenset[str] = field(default_factory=frozenset)  # operators executable in-engine


class BaseEngine:
    """Common defaults. Concrete engines override the parts they do natively."""

    name: str = "base"
    caps: EngineCaps = EngineCaps()

    def scan(self, source: str, filters: Optional[dict] = None,
             limit: Optional[int] = None, with_vectors: bool = True) -> list[Row]:
        raise NotImplementedError

    def fetch_vectors(self, source: str, keys: Sequence[str]) -> dict[str, list[float]]:
        """Embeddings for specific keys. Portable default: read them off the rows."""
        want = set(keys)
        return {r.id: r.embedding for r in self.scan(source) if r.id in want and r.embedding}

    def ann_candidates(self, source: str, query_vector: Sequence[float], k: int,
                       filters: Optional[dict] = None, *,
                       with_vectors: bool = True) -> list[tuple[Row, float]]:
        """Return up to k (row, similarity) pairs, highest similarity first.

        with_vectors=False lets an engine skip shipping embeddings back (and use a
        covering index if it has one); the returned Rows then have embedding=None.
        """
        raise NotImplementedError

    def ann_batch(self, source: str, query_vectors: Sequence[Sequence[float]], k: int,
                  filters: Optional[dict] = None, *,
                  with_vectors: bool = False) -> list[list[tuple[Row, float]]]:
        """N probes at once, aligned to query_vectors.

        Portable default: loop. Engines that can put several probes in one round
        trip override this — on Couchbase it is a UNION ALL and measured 2.4-2.5x.
        """
        return [self.ann_candidates(source, qv, k, filters, with_vectors=with_vectors)
                for qv in query_vectors]

    def ann_above(self, source: str, weights: Sequence[float], tau: float,
                  filters: Optional[dict] = None, *, est_k: int = 256,
                  max_k: Optional[int] = None,
                  with_vectors: bool = False) -> tuple[list[tuple[Row, float]], bool]:
        """Rows whose linear-proxy score (w.x) is >= tau. See CouchbaseEngine for
        why an ANN index can serve this. Portable default: score everything.
        Returns (rows_with_scores, exhaustive)."""
        rows = self.scan(source, filters)
        scored = [(r, sum(a * b for a, b in zip(r.embedding, weights)) if r.embedding else 0.0)
                  for r in rows]
        return [(r, s) for r, s in scored if s >= tau], True

    def proxy_scores(self, rows: Sequence[Row], query_vector: Sequence[float]) -> list[float]:
        """Portable default: in-service cosine over each row's embedding.

        Engines that can push this down override it and set
        caps.server_side_scoring=True. Correctness is identical either way.
        """
        return [
            cosine_similarity(r.embedding, query_vector) if r.embedding else 0.0
            for r in rows
        ]

    def count(self, source: str) -> int:
        """Number of docs in a source (for the nested-loop savings baseline)."""
        return len(self.scan(source))

    def upsert(self, source: str, rows: Sequence[Row]) -> None:
        raise NotImplementedError

    def native_operator(self, name: str) -> Optional[Callable[..., Any]]:
        """Graduation hook: when an operator ships in-engine, return its callable
        here and advertise it in caps.native_ops. Empty for the pure companion."""
        return None

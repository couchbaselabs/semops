"""Session + Pipeline: LOTUS-style surface over any engine mode.

    sess = semops.connect(engine=CouchbaseEngine(...), model=CapellaAIClient(...))
    (sess.search("reviews", "comfortable running shoes", k=200)
         .sem_filter("is about battery life")
         .collect())

"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from . import operators
from .backends import ModelClient
from .budget import Budget
from .cache import Cache, make_key
from .engines.base import BaseEngine
from .telemetry import Telemetry
from .types import DedupResult, FilterResult, GroupByResult, JoinResult, Row


class Session:
    def __init__(
        self,
        engine: BaseEngine,
        model: ModelClient,
        *,
        budget_usd: Optional[float] = None,
        cache: bool = True,
        telemetry: Optional[Telemetry] = None,
        workers: int = 8,
    ):
        self.engine = engine
        self.model = model
        self.cache = Cache(enabled=cache)
        self.budget = Budget(budget_usd)
        self.telemetry = telemetry or Telemetry()
        self.workers = workers  # parallel LLM calls / ANN queries (I/O-bound)

    # --- entry points that start a pipeline ---
    def search(self, source: str, query: str, k: int = 100,
               filters: Optional[dict] = None) -> "Pipeline":
        """Vector top-k via the engine (native ANN pushdown on Couchbase)."""
        key = make_key("embed", getattr(self.model, "embed_model",
                       self.model.__class__.__name__), query)
        qvec = self.cache.get(key)
        if qvec is None:
            qvec = self.model.embed([query])[0]
            self.cache.put(key, qvec)
        scored = self.engine.ann_candidates(source, qvec, k, filters)
        rows = [r for r, _ in scored]
        for r, s in scored:
            r.doc.setdefault("_score", s)
        return Pipeline(self, rows)

    def scan(self, source: str, filters: Optional[dict] = None,
             limit: Optional[int] = None, with_vectors: Optional[bool] = None) -> "Pipeline":
        """Materialise a collection as a Pipeline.

        with_vectors defaults to *leaving embeddings in the store* whenever the
        engine can score server-side. This is the difference between the pushdown
        being real and being decorative: scanning with vectors ships every
        embedding up front, so a later `VECTOR_DISTANCE(...,'dot')` pushdown saves
        nothing (measured: 0%). Left in the store, only the calibration sample's
        vectors are ever fetched.

        Pass with_vectors=True to force them out (e.g. for an engine-independent
        proxy that scores in-process).
        """
        if with_vectors is None:
            with_vectors = not getattr(self.engine.caps, "server_side_scoring", False)
        return Pipeline(self, self.engine.scan(source, filters, limit,
                                               with_vectors=with_vectors))

    def sem_filter(self, source: str, predicate: str, **kw) -> FilterResult:
        """Filter a whole collection, letting the vector index serve the proxy.

        Unlike `sess.scan(...).sem_filter(...)`, this never materialises or scores
        the rows that fall below tau_minus — the index is asked for the survivors
        directly. Needs a linear proxy_model and an ANN engine.
        """
        return operators.sem_filter_source(
            self.engine, self.model, source, predicate, cache=self.cache,
            budget=self.budget, telemetry=self.telemetry,
            **{"workers": self.workers, **kw})

    def sem_dedup(self, source: str, predicate: str = None, **kw) -> DedupResult:
        """Deduplicate a collection into entity clusters."""
        p = {"predicate": predicate} if predicate else {}
        return operators.sem_dedup(
            self.engine, self.model, source, cache=self.cache, budget=self.budget,
            telemetry=self.telemetry, **p, **{"workers": self.workers, **kw})

    def sem_group_by(self, source: str, k: int, **kw) -> GroupByResult:
        """Cluster a collection into k semantic groups."""
        return operators.sem_group_by(
            self.engine, self.model, source, k=k, cache=self.cache, budget=self.budget,
            telemetry=self.telemetry, **{"workers": self.workers, **kw})

    def rows(self, rows: Sequence[Row]) -> "Pipeline":
        return Pipeline(self, list(rows))


class Pipeline:
    def __init__(self, session: Session, rows: list[Row]):
        self.session = session
        self.rows_: list[Row] = rows
        self.last_stats: Optional[dict[str, Any]] = None

    def sem_filter(self, predicate: str, **kw) -> "Pipeline":
        result: FilterResult = operators.sem_filter(
            self.session.engine, self.session.model, self.rows_, predicate,
            cache=self.session.cache, budget=self.session.budget,
            telemetry=self.session.telemetry, **{"workers": self.session.workers, **kw},
        )
        self.last_stats = result.stats.as_dict()
        nxt = Pipeline(self.session, result.rows)
        nxt.last_stats = result.stats.as_dict()  # stats of the op that produced nxt
        return nxt

    def sem_join(self, right_source: str, predicate: str, **kw) -> JoinResult:
        """Join this pipeline's rows (left) to right_source on an NL predicate."""
        return operators.sem_join(
            self.session.engine, self.session.model, self.rows_, right_source, predicate,
            cache=self.session.cache, budget=self.session.budget,
            telemetry=self.session.telemetry, **{"workers": self.session.workers, **kw})

    # --- terminals ---
    def collect(self) -> list[dict]:
        return [r.doc if r.doc else {"id": r.id, "text": r.text} for r in self.rows_]

    def count(self) -> int:
        return len(self.rows_)

    def stats(self) -> Optional[dict[str, Any]]:
        return self.last_stats

    def __iter__(self):
        return iter(self.rows_)

    def __len__(self):
        return len(self.rows_)

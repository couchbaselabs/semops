"""semops — semantic operators over vector-indexed data.

Engine-agnostic core + operators; Couchbase is a first-class native mode
(NN pushdown + writeback), with an in-memory baseline for other engines and
offline use.

Quickstart (offline, zero deps):

    import semops
    from semops import InMemoryEngine, FakeModelClient, Row

    eng, model = InMemoryEngine(), FakeModelClient()
    eng.add("docs", [Row(id=str(i), text=t, embedding=model.embed([t])[0])
                     for i, t in enumerate(my_texts)])
    sess = semops.connect(engine=eng, model=model, budget_usd=1.0)
    kept = sess.scan("docs").sem_filter("is about battery life").collect()

Swap InMemoryEngine -> CouchbaseEngine(...) and FakeModelClient ->
CapellaAIClient(...) to run natively on Couchbase. Nothing else changes.
"""
from __future__ import annotations

from typing import Optional

from .backends import (
    AnthropicClient,
    CapellaAIClient,
    FakeModelClient,
    LocalHashingEmbedder,
    ModelClient,
    OpenAICompatClient,
)
from .budget import Budget, BudgetExceeded
from .cache import Cache
from .cascade import Thresholds, calibrate, empirical_precision_recall, wilson_lower_bound
from .engines import BaseEngine, CouchbaseEngine, EngineCaps, HttpQueryCluster, InMemoryEngine
from .operators import sem_dedup, sem_filter, sem_group_by, sem_join
from .session import Pipeline, Session
from .telemetry import Telemetry
from .types import (
    Band, CascadeStats, DedupResult, DedupStats, FilterResult, GroupByResult, GroupByStats,
    JoinResult, JoinStats, Row, SemGroup,
)

__version__ = "0.1.0"

__all__ = [
    "connect", "Session", "Pipeline",
    "Row", "Band", "CascadeStats", "FilterResult", "JoinResult", "JoinStats",
    "DedupResult", "DedupStats", "GroupByResult", "GroupByStats", "SemGroup",
    "InMemoryEngine", "CouchbaseEngine", "BaseEngine", "EngineCaps", "HttpQueryCluster",
    "OpenAICompatClient", "CapellaAIClient", "AnthropicClient", "LocalHashingEmbedder",
    "FakeModelClient", "ModelClient",
    "sem_filter", "sem_join", "sem_dedup", "sem_group_by",
    "calibrate", "Thresholds", "wilson_lower_bound",
    "empirical_precision_recall",
    "Cache", "Budget", "BudgetExceeded", "Telemetry",
    "__version__",
]


def connect(engine: BaseEngine, model: ModelClient, *, budget_usd: Optional[float] = None,
            cache: bool = True, telemetry: Optional[Telemetry] = None,
            workers: int = 8) -> Session:
    """Build a Session from an engine mode + a model provider.

    `workers` bounds parallel LLM calls / ANN queries (I/O-bound). Results are
    order-preserving, so parallel runs match sequential ones exactly.
    """
    return Session(engine, model, budget_usd=budget_usd, cache=cache,
                   telemetry=telemetry, workers=workers)

"""Couchbase engine — the native, first-priority mode.

Pushes the heavy work down (KB Part VII): the ANN search over the *whole*
collection runs in the cluster via `APPROX_VECTOR_DISTANCE` on a bhive / FTS
vector index, returning only the top-k candidates. The light per-candidate work
(cascade banding, oracle calls) then happens in the service on that bounded set.

The Couchbase Python SDK import is guarded so the rest of `semops` (and its test
suite on InMemoryEngine) works with no SDK installed.
"""
from __future__ import annotations

import base64
import json
import urllib.request
from typing import Optional, Sequence

from ..types import Row
from .base import BaseEngine, EngineCaps


class HttpQueryCluster:
    """Minimal N1QL driver over the Query REST API (`/query/service`).

    Works where the official SDK is awkward — notably `cluster_run` dev clusters
    on non-standard ports. Exposes a `.query(statement, **named_params)` shaped
    like the SDK's, so `CouchbaseEngine(cluster=HttpQueryCluster(...))` just works.
    Zero extra deps (stdlib urllib).
    """

    def __init__(self, query_url: str, username: str, password: str, timeout: float = 180.0):
        self.query_url = query_url.rstrip("/")
        self._auth = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        self.timeout = timeout

    def query(self, statement: str, **named_params):
        payload = {"statement": statement}
        for k, v in named_params.items():
            payload[k if k.startswith("$") else "$" + k] = v
        req = urllib.request.Request(
            self.query_url + "/query/service",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": self._auth},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8"))
        if out.get("status") != "success":
            raise RuntimeError(f"N1QL error: {out.get('errors') or out.get('status')}")
        return out.get("results", [])

try:  # optional dependency
    from couchbase.cluster import Cluster  # type: ignore
    from couchbase.options import ClusterOptions  # type: ignore
    from couchbase.auth import PasswordAuthenticator  # type: ignore
    _SDK = True
except Exception:  # pragma: no cover - exercised only without the SDK
    _SDK = False


class CouchbaseEngine(BaseEngine):
    name = "couchbase"
    # ANN retrieval, structured-filter pushdown, writeback, AND proxy scoring are
    # all native: VECTOR_DISTANCE computes scores in-cluster so embeddings never
    # leave. A linear (logistic-regression) proxy pushes down too, as a dot product.
    caps = EngineCaps(ann=True, server_side_scoring=True, pushdown_filter=True, writeback=True)

    def __init__(
        self,
        connstr: str,
        username: str,
        password: str,
        bucket: str,
        scope: str = "_default",
        *,
        vector_field: str = "embedding",
        text_field: str = "text",
        metric: str = "cosine",
        nprobes: Optional[int] = None,
        rerank: bool = True,
        top_n_scan: Optional[int] = None,   # None => auto (= requested k)
        covering_index: Optional[str] = None,  # INCLUDE index to force for covered scans
        key_chunk: int = 2000,   # max keys per USE KEYS request (KV bulk-get limit)
        max_heap: int = 8192,    # indexer.scan.vector.max_heap_size: caps LIMIT+OFFSET
        cluster=None,
    ):
        if cluster is None:
            if not _SDK:
                raise ImportError(
                    "couchbase SDK not installed. `pip install couchbase` or pass a "
                    "pre-built `cluster=` (or use InMemoryEngine for offline work)."
                )
            auth = PasswordAuthenticator(username, password)
            cluster = Cluster(connstr, ClusterOptions(auth))
        self._cluster = cluster
        self.bucket = bucket
        self.scope = scope
        self.vector_field = vector_field
        self.text_field = text_field
        self.metric = metric
        self.nprobes = nprobes
        self.rerank = rerank
        self.top_n_scan = top_n_scan
        self.covering_index = covering_index
        self.key_chunk = max(int(key_chunk), 1)
        self.max_heap = max(int(max_heap), 1)
        self._last_source = None  # set by scan()/ann_candidates(); used by proxy_scores()
        self.vectors_pulled = 0   # count of embeddings shipped out of the cluster (for demos)

    def _keyspace(self, source: str) -> str:
        return f"`{self.bucket}`.`{self.scope}`.`{source}`"

    def _where(self, filters: Optional[dict]) -> tuple[str, dict]:
        if not filters:
            return "", {}
        clauses, params = [], {}
        for i, (field, value) in enumerate(filters.items()):
            p = f"f{i}"
            clauses.append(f"d.`{field}` = ${p}")
            params[p] = value
        return " WHERE " + " AND ".join(clauses), params

    def _dist_expr(self, param: str, k: int, metric: Optional[str] = None) -> str:
        """APPROX_VECTOR_DISTANCE with the scan depth pinned.

        A bhive index caps ANN scan depth: LIMIT alone (and nprobes) will NOT
        return more than the index's default top-N, silently truncating the
        candidate set and capping blocking recall. topNScan (6th arg) is the
        lever — it must be >= k, and passing it requires nprobes and rerank too.
        Measured on 2k docs: topNScan=10 under LIMIT 200 returns 20 rows at
        recall 0.10; topNScan=200 returns 200 rows at recall 1.00.
        """
        topn = self.top_n_scan or max(int(k), 1)
        return (f"APPROX_VECTOR_DISTANCE(d.`{self.vector_field}`, ${param}, "
                f"'{metric or self.metric}', "
                f"{int(self.nprobes or 8)}, {str(bool(self.rerank)).lower()}, {int(topn)})")

    def _to_sim(self, dist, metric: Optional[str] = None):
        """Distance -> similarity, so higher == more likely TRUE (cascade convention).

        Careful: the metric passed to APPROX_VECTOR_DISTANCE does not necessarily
        change what the index reports. On a cosine bhive index, both 'cosine' and
        'l2_squared' come back as L2-squared over the normalised stored vectors
        (ranking is right, the value is not a cosine). 'dot' does return the plain
        inner product, which is why ann_above() asks for it by name.
        """
        if dist is None:
            return 0.0
        m = metric or self.metric
        if m == "dot":
            return -dist            # N1QL DOT returns -(q.x), so this is exactly q.x
        return (1.0 - dist) if m == "cosine" else -dist

    def _hint(self) -> str:
        return f" USE INDEX (`{self.covering_index}`)" if self.covering_index else ""

    def _projection(self, dist_expr: str, with_vectors: bool) -> str:
        # `d AS _doc` drags the full embedding back for every candidate — 862KB per
        # 100-row probe vs 17KB when we project only what the cascade reads. With an
        # INCLUDE (text,label) index the narrow form is index-covered (no KV Fetch).
        base = f"META(d).id AS _id, d.`{self.text_field}` AS _text, {dist_expr} AS _dist"
        return f"{base}, d AS _doc" if with_vectors else f"{base}, d.label AS _label"

    def _row_of(self, row, with_vectors: bool) -> Row:
        if with_vectors:
            doc = row.get("_doc") or {}
            if doc.get(self.vector_field):
                self.vectors_pulled += 1
            return Row(id=row.get("_id"), text=row.get("_text") or "",
                       embedding=doc.get(self.vector_field), doc=doc)
        return Row(id=row.get("_id"), text=row.get("_text") or "", embedding=None,
                   doc={"text": row.get("_text") or "", "label": row.get("_label")})

    def ann_candidates(self, source, query_vector, k, filters=None, *, with_vectors=True,
                       metric=None, raw_dist=False):
        """Native pushdown: ORDER BY APPROX_VECTOR_DISTANCE(...) LIMIT k in-cluster.

        with_vectors=False keeps embeddings in the cluster and lets an INCLUDE index
        cover the scan — 4x faster and 51x less data on the measured 100-row probe.
        """
        self._last_source = source
        dist = self._dist_expr("qvec", k, metric)
        where, wparams = self._where(filters)
        stmt = (f"SELECT {self._projection(dist, with_vectors)} "
                f"FROM {self._keyspace(source)} d{self._hint()}{where} "
                f"ORDER BY {dist} LIMIT $k")
        res = self._cluster.query(stmt, qvec=list(query_vector), k=int(k), **wparams)
        if raw_dist:  # caller converts (ann_above needs the untransformed value)
            return [(self._row_of(r, with_vectors), r.get("_dist")) for r in res]
        return [(self._row_of(r, with_vectors), self._to_sim(r.get("_dist"), metric))
                for r in res]

    def ann_batch(self, source, query_vectors, k, filters=None, *, with_vectors=False,
                  chunk: int = 32):
        """N ANN probes in one statement via UNION ALL — one round trip instead of N.

        sem_join blocking issues one probe per left row; batching measured 2.4-2.5x
        faster at 8-32 probes. Returns a list aligned to query_vectors. (`probe` is a
        reserved word in N1QL, hence `_pid`.)
        """
        self._last_source = source
        where, wparams = self._where(filters)
        out: list[list[tuple[Row, float]]] = [[] for _ in query_vectors]
        for base in range(0, len(query_vectors), chunk):
            batch = query_vectors[base:base + chunk]
            parts, params = [], dict(wparams)
            for j, qv in enumerate(batch):
                dist = self._dist_expr(f"q{j}", k)
                parts.append(f"(SELECT {base + j} AS _pid, {self._projection(dist, with_vectors)} "
                             f"FROM {self._keyspace(source)} d{self._hint()}{where} "
                             f"ORDER BY {dist} LIMIT {int(k)})")
                params[f"q{j}"] = list(qv)
            for r in self._cluster.query(" UNION ALL ".join(parts), **params):
                pid = r.get("_pid")
                if pid is not None and 0 <= pid < len(out):
                    out[pid].append((self._row_of(r, with_vectors), self._to_sim(r.get("_dist"))))
        # UNION ALL does not guarantee ordering across branches; restore it per probe.
        for lst in out:
            lst.sort(key=lambda rs: rs[1], reverse=True)
        return out

    def ann_above(self, source, weights, tau, filters=None, *, est_k=256, max_k=None,
                  with_vectors=False):
        """Every row whose LEARNED-PROXY score (w.x) is >= tau, without scoring the
        whole collection.

        Why a vector index can serve this: a logistic-regression proxy scores
        sigma(w.x + b), monotonic in w.x, so the proxy's top-k IS the
        maximum-inner-product top-k for query vector w. Ship w as the query vector
        and the index returns the proxy's own ranking.

        TWO REQUIREMENTS, both learned the hard way:

        1. The index must have been created with `similarity` matching the metric
           we query with. Couchbase only selects a vector index when they match
           (docs: "the distance metric should match the similarity setting that you
           used when you created the index"). Query 'dot' against a cosine-built
           index and the planner silently falls back to PrimaryScan3 + Fetch, i.e.
           a brute-force scan of the whole collection. Results stay correct, which
           is why this is easy to miss; it is just not an index scan, and it is
           slower than scanning yourself. Build a `similarity: 'dot'` index.

        2. When a vector index does serve the query, the distance it returns is
           L2-squared over the stored vectors regardless of the metric string. It
           is only the brute-force path that returns the true metric. So the score
           has to be recovered: for unit-norm stored vectors,
               dist = |w|^2 + |x|^2 - 2(w.x)  =>  w.x = (|w|^2 + 1 - dist) / 2
           Verified against exact w.x to 4 decimals. `vectors_normalised()` checks
           the precondition; without unit-norm vectors this recovery is invalid.

        Termination is checkable rather than assumed: fetch k, and if the WORST
        returned score is still >= tau there may be more above the line, so double
        k and retry. Stopping when the worst score falls below tau proves the
        result is complete, up to the index's ANN recall.

        Returns (rows_with_scores, exhaustive) where exhaustive=False means the
        max_k cap was hit before the boundary was crossed.
        """
        if not any(weights):
            return [], True
        w2 = sum(x * x for x in weights)
        # A vector index scan caps LIMIT+OFFSET at indexer.scan.vector.max_heap_size
        # (default 8192). Past that the request fails outright with error 5433, so
        # this path can only serve predicates selective enough to fit under the cap.
        # Anything wider has to fall back to scanning; exhaustive=False says so.
        max_k = min(max_k or self.count(source), self.max_heap)
        k = max(int(est_k), 1)
        while True:
            k = min(k, max_k)
            scored = self.ann_candidates(source, weights, k, filters,
                                         with_vectors=with_vectors, metric="dot",
                                         raw_dist=True)
            # recover w.x from whichever value came back (see requirement 2)
            out = [(r, self._dot_from_dist(d, w2)) for r, d in scored]
            above = [(r, sc) for r, sc in out if sc >= tau]
            if len(above) < len(out) or k >= max_k:
                return above, len(above) < len(out)
            k *= 2  # the whole page cleared tau, so there may be more below it

    def _dot_from_dist(self, dist, w2):
        """Recover w.x from what the server returned.

        An index scan gives L2-squared over unit-norm stored vectors; the
        brute-force path gives -(w.x) directly. They are trivially separable:
        L2-squared is >= 0 and near |w|^2, while -(w.x) for a top-ranked row is
        small. Disambiguating on the value keeps this correct on both paths
        instead of assuming which one the planner chose.
        """
        if dist is None:
            return 0.0
        if dist > 1.0:                       # L2-squared from an index scan
            return (w2 + 1.0 - dist) / 2.0
        return -dist                          # -(w.x) from the brute-force path

    def vectors_normalised(self, source, sample=64, tol=1e-3):
        """Precondition check for ann_above(): are stored vectors unit-norm?
        (BGE/OpenAI embeddings are; some local models are not.)"""
        rows = self._cluster.query(
            f"SELECT d.`{self.vector_field}` AS v FROM {self._keyspace(source)} d "
            f"LIMIT {int(sample)}")
        norms = [sum(x * x for x in r["v"]) ** 0.5 for r in rows if r.get("v")]
        return bool(norms) and all(abs(n - 1.0) < tol for n in norms)

    def scan(self, source, filters=None, limit=None, with_vectors=True):
        """Scan the collection. with_vectors=False omits the embedding field, so
        vectors stay in the cluster (scoring is then done server-side)."""
        self._last_source = source
        where, wparams = self._where(filters)
        lim = f" LIMIT {int(limit)}" if limit is not None else ""
        if with_vectors:
            stmt = f"SELECT META(d).id AS _id, d AS _doc FROM {self._keyspace(source)} d{where}{lim}"
        else:
            stmt = (f"SELECT META(d).id AS _id, d.`{self.text_field}` AS _text, d.label AS _label "
                    f"FROM {self._keyspace(source)} d{where}{lim}")
        res = self._cluster.query(stmt, **wparams)
        rows = []
        for row in res:
            if with_vectors:
                doc = row.get("_doc") or {}
                if doc.get(self.vector_field):
                    self.vectors_pulled += 1
                rows.append(Row(id=row.get("_id"), text=doc.get(self.text_field, ""),
                                embedding=doc.get(self.vector_field), doc=doc))
            else:
                rows.append(Row(id=row.get("_id"), text=row.get("_text", "") or "", embedding=None,
                                doc={"text": row.get("_text", ""), "label": row.get("_label")}))
        return rows

    def proxy_scores(self, rows, query_vector, metric=None):
        """Server-side scoring: VECTOR_DISTANCE computed in-cluster per doc key.
        Only scalar scores cross the wire — embeddings never leave Couchbase.
        Works for cosine-to-predicate (query_vector = predicate embedding) and for
        a linear learned proxy (query_vector = LR weights, metric='dot')."""
        if not rows:
            return []
        metric = metric or self.metric
        stmt = (f"SELECT META(d).id AS id, "
                f"VECTOR_DISTANCE(d.`{self.vector_field}`, $q, '{metric}') AS dist "
                f"FROM {self._keyspace(self._last_source)} d USE KEYS $keys")

        def sim(dist):  # convert distance -> higher-is-more-likely-true similarity
            return (1.0 - dist) if metric == "cosine" else -dist

        # USE KEYS becomes a single KV bulk get, and one request per key. Sending
        # the whole collection at once times out the KV client (error 12008,
        # "bulk get ... i/o timeout" — measured at 25k keys). Chunk it.
        byid: dict[str, float] = {}
        keys = [r.id for r in rows]
        for i in range(0, len(keys), self.key_chunk):
            res = self._cluster.query(stmt, q=list(query_vector),
                                      keys=keys[i:i + self.key_chunk])
            byid.update({r["id"]: sim(r["dist"]) for r in res if r.get("dist") is not None})
        return [byid.get(r.id, 0.0) for r in rows]

    def count(self, source):
        res = self._cluster.query(f"SELECT COUNT(*) AS c FROM {self._keyspace(source)}")
        return res[0]["c"] if res else 0

    def fetch_vectors(self, source, keys):
        """Pull stored embeddings for specific keys (used to fit a learned proxy on
        the small oracle sample without pulling the whole collection's vectors)."""
        stmt = (f"SELECT META(d).id AS id, d.`{self.vector_field}` AS v "
                f"FROM {self._keyspace(source)} d USE KEYS $keys")
        keys = list(keys)
        out = {}
        for i in range(0, len(keys), self.key_chunk):  # see proxy_scores: KV bulk get limit
            out.update({r["id"]: r["v"]
                        for r in self._cluster.query(stmt, keys=keys[i:i + self.key_chunk])})
        self.vectors_pulled += len(out)
        return out

    def upsert(self, source, rows: Sequence[Row]):
        coll = self._cluster.bucket(self.bucket).scope(self.scope).collection(source)
        for r in rows:
            coll.upsert(r.id, r.doc or {"text": r.text})

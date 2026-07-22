"""End-to-end sem_filter tests on both engine modes, fully offline.

Uses FakeModelClient as a deterministic oracle (ground truth = keywords present)
whose embeddings correlate with that oracle, so the cascade has real signal to
exploit. The Couchbase native mode is exercised through a faithful in-process
FakeCluster that runs the ANN query the way the cluster would.
"""
import random
import unittest

import semops
from semops import (
    Budget,
    BudgetExceeded,
    CouchbaseEngine,
    FakeModelClient,
    InMemoryEngine,
    Row,
)
from semops.cascade import empirical_precision_recall
from semops.vectormath import cosine_similarity

PREDICATE = "battery life"
FILLER = ["device", "product", "review", "quality", "design", "value", "everyday", "solid"]


def make_dataset(n=1000, seed=7):
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        roll = rng.random()
        f = " ".join(rng.sample(FILLER, 3))
        if roll < 0.35:                      # positive: contains BOTH battery + life
            texts.append(f"the battery life is great and lasts all day {f}")
        elif roll < 0.80:                    # clear negative: neither word
            texts.append(f"the screen is bright and the camera is sharp {f}")
        elif roll < 0.90:                    # ambiguous: battery, no life
            texts.append(f"the battery drains quickly while gaming {f}")
        else:                                # ambiguous: life, no battery
            texts.append(f"long product life and a sturdy build {f}")
    return texts


def build_rows(model, texts):
    vecs = model.embed(texts)
    return [Row(id=str(i), text=t, embedding=v, doc={"id": str(i), "text": t, "embedding": v})
            for i, (t, v) in enumerate(zip(texts, vecs))]


def ground_truth(model, rows):
    return [model.judge(PREDICATE, r.text) for r in rows]


class FakeCluster:
    """A faithful mini-Couchbase: runs the ANN query over an in-process doc set."""

    def __init__(self, docs):
        self.docs = docs  # list of dicts with id/text/embedding
        self.last_query = None

    def query(self, stmt, **params):
        self.last_query = (stmt, params)
        qvec = params.get("qvec")
        if qvec is not None:  # ANN path (APPROX_VECTOR_DISTANCE ... ORDER BY ... LIMIT)
            k = params.get("k", len(self.docs))
            scored = sorted(self.docs, key=lambda d: cosine_similarity(d["embedding"], qvec),
                            reverse=True)[:k]
            return [
                {"_id": d["id"], "_text": d["text"],
                 "_dist": 1.0 - cosine_similarity(d["embedding"], qvec), "_doc": d}
                for d in scored
            ]
        if "USE KEYS" in stmt and params.get("q") is not None:  # server-side proxy_scores by key
            byid = {d["id"]: d for d in self.docs}
            q, keys = params["q"], params.get("keys", [])
            return [{"id": k, "dist": 1.0 - cosine_similarity(byid[k]["embedding"], q)}
                    for k in keys if k in byid]
        return [{"_id": d["id"], "_doc": d} for d in self.docs]  # scan path


class TestInMemoryMode(unittest.TestCase):
    def setUp(self):
        self.model = FakeModelClient(dims=64)
        self.rows = build_rows(self.model, make_dataset(1000))
        self.truth = ground_truth(self.model, self.rows)
        self.eng = InMemoryEngine()
        self.eng.add("docs", self.rows)

    def test_cascade_is_correct_and_cheap(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        pipe = sess.scan("docs").sem_filter(PREDICATE, recall=0.9, precision=0.9, seed=0)
        stats = pipe.stats()
        kept_ids = {r["id"] for r in pipe.collect()}

        keep_pred = [r.id in kept_ids for r in self.rows]
        scored = [(0.0, y) for y in self.truth]  # only labels matter for P/R
        precision, recall = empirical_precision_recall(scored, keep_pred)

        # 1. correctness: honors the targets (with statistical slack)
        self.assertGreaterEqual(recall, 0.85, f"recall {recall}, stats={stats}")
        self.assertGreaterEqual(precision, 0.85, f"precision {precision}, stats={stats}")
        # 2. ROI: the proxy decided a chunk of rows without the oracle
        self.assertLess(stats["llm_calls"], stats["n_rows"])
        self.assertGreater(stats["n_accept"] + stats["n_reject"], 0)
        self.assertGreater(stats["savings_ratio"], 1.5)

    def test_oracle_policy_matches_ground_truth(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        pipe = sess.scan("docs").sem_filter(PREDICATE, policy="oracle")
        kept_ids = {r["id"] for r in pipe.collect()}
        keep_pred = [r.id in kept_ids for r in self.rows]
        scored = [(0.0, y) for y in self.truth]
        precision, recall = empirical_precision_recall(scored, keep_pred)
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(pipe.stats()["llm_calls"], len(self.rows))

    def test_cache_cuts_second_run(self):
        sess = semops.connect(engine=self.eng, model=self.model, cache=True)
        p1 = sess.scan("docs").sem_filter(PREDICATE, seed=0)
        p2 = sess.scan("docs").sem_filter(PREDICATE, seed=0)
        self.assertLess(p2.stats()["llm_calls"], p1.stats()["llm_calls"])
        self.assertGreater(p2.stats()["cache_hits"], 0)

    def test_budget_guard_stops_runaway(self):
        sess = semops.connect(engine=self.eng, model=self.model, budget_usd=0.001, cache=False)
        with self.assertRaises(BudgetExceeded):
            sess.scan("docs").sem_filter(PREDICATE, policy="oracle")


class TestCouchbaseMode(unittest.TestCase):
    def setUp(self):
        self.model = FakeModelClient(dims=64)
        self.rows = build_rows(self.model, make_dataset(800))
        self.truth = ground_truth(self.model, self.rows)
        docs = [r.doc for r in self.rows]
        self.cluster = FakeCluster(docs)
        self.eng = CouchbaseEngine("couchbases://stub", "u", "p", bucket="bench",
                                   cluster=self.cluster)

    def test_ann_pushes_down_a_wellformed_query(self):
        qvec = self.model.embed([PREDICATE])[0]
        cands = self.eng.ann_candidates("docs", qvec, k=50)
        stmt, params = self.cluster.last_query
        self.assertIn("APPROX_VECTOR_DISTANCE", stmt)
        self.assertIn("ORDER BY", stmt)
        self.assertIn("LIMIT $k", stmt)
        self.assertEqual(params["k"], 50)
        self.assertEqual(len(cands), 50)
        # returned highest-similarity first, and distance was converted back to similarity
        sims = [s for _, s in cands]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertLessEqual(sims[0], 1.0)

    def test_cascade_correct_over_couchbase_mode(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        pipe = sess.search("docs", PREDICATE, k=800).sem_filter(
            PREDICATE, recall=0.9, precision=0.9, seed=0)
        kept_ids = {r["id"] for r in pipe.collect()}
        # ground truth keyed by id (search may reorder)
        truth_by_id = {r.id: y for r, y in zip(self.rows, self.truth)}
        tp = sum(1 for i in kept_ids if truth_by_id[i])
        fp = len(kept_ids) - tp
        total_pos = sum(1 for y in self.truth if y)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / total_pos if total_pos else 1.0
        self.assertGreaterEqual(recall, 0.85, f"recall {recall}")
        self.assertGreaterEqual(precision, 0.85, f"precision {precision}")


class TestModeParity(unittest.TestCase):
    """Same query, same cascade, both modes -> comparable quality. This is the
    'Couchbase is first-class, not degraded' guarantee, and the demo hook."""

    def test_modes_agree(self):
        model = FakeModelClient(dims=64)
        rows = build_rows(model, make_dataset(600, seed=11))
        truth = {r.id: model.judge(PREDICATE, r.text) for r in rows}

        mem = InMemoryEngine(); mem.add("docs", rows)
        cb = CouchbaseEngine("x", "u", "p", bucket="b", cluster=FakeCluster([r.doc for r in rows]))

        def run(engine, entry):
            sess = semops.connect(engine=engine, model=FakeModelClient(dims=64))
            pipe = entry(sess).sem_filter(PREDICATE, recall=0.9, precision=0.9, seed=0)
            ids = {r["id"] for r in pipe.collect()}
            tp = sum(1 for i in ids if truth[i]); fp = len(ids) - tp
            tot = sum(1 for v in truth.values() if v)
            return (tp / (tp + fp) if tp + fp else 1.0, tp / tot if tot else 1.0)

        p_mem, r_mem = run(mem, lambda s: s.scan("docs"))
        p_cb, r_cb = run(cb, lambda s: s.search("docs", PREDICATE, k=600))
        # neither mode should be materially worse than the other
        self.assertLess(abs(p_mem - p_cb), 0.15)
        self.assertLess(abs(r_mem - r_cb), 0.15)


if __name__ == "__main__":
    unittest.main()

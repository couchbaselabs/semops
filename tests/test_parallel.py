"""Parallelism tests: results must be identical to sequential (pmap is
order-preserving), stats must not be lost to races, and wall-clock must actually
drop for I/O-bound work."""
import time
import unittest

import semops
from semops import FakeModelClient, InMemoryEngine, Row
from test_sem_join import build


class SlowModel:
    """FakeModelClient with an artificial network delay on judge()."""

    def __init__(self, delay=0.0, dims=32):
        self._f = FakeModelClient(dims=dims)
        self.delay = delay
        self.embed_model = "slow"
        self.chat_model = "slow"
        self.spent_usd = 0.0
        self.calls = 0

    def embed(self, texts):
        return self._f.embed(texts)

    def judge(self, predicate, text):
        if self.delay:
            time.sleep(self.delay)
        return self._f.judge(predicate, text)


def make_rows(n=40):
    m = FakeModelClient(dims=32)
    texts = [(f"battery life unit {i}" if i % 2 == 0 else f"camera screen unit {i}")
             for i in range(n)]
    vecs = m.embed(texts)
    return [Row(id=str(i), text=t, embedding=v, doc={"id": str(i)})
            for i, (t, v) in enumerate(zip(texts, vecs))]


PRED = "battery life"


class TestParallel(unittest.TestCase):
    def test_parallel_matches_sequential(self):
        rows = make_rows(40)
        eng = InMemoryEngine()
        eng.add("d", rows)
        seq = semops.connect(engine=eng, model=SlowModel(), workers=1) \
            .rows(rows).sem_filter(PRED, policy="oracle")
        par = semops.connect(engine=eng, model=SlowModel(), workers=8) \
            .rows(rows).sem_filter(PRED, policy="oracle")
        self.assertEqual([r.id for r in seq], [r.id for r in par])   # identical, in order
        self.assertEqual(seq.stats()["llm_calls"], par.stats()["llm_calls"])

    def test_no_lost_stat_increments_under_concurrency(self):
        rows = make_rows(60)
        eng = InMemoryEngine()
        eng.add("d", rows)
        p = semops.connect(engine=eng, model=SlowModel(), workers=16, cache=False) \
            .rows(rows).sem_filter(PRED, policy="oracle")
        self.assertEqual(p.stats()["llm_calls"], 60)  # every call counted exactly once

    def test_parallel_is_faster_for_io_bound_work(self):
        rows = make_rows(24)
        eng = InMemoryEngine()
        eng.add("d", rows)
        t0 = time.time()
        semops.connect(engine=eng, model=SlowModel(0.02), workers=1, cache=False) \
            .rows(rows).sem_filter(PRED, policy="oracle")
        seq = time.time() - t0
        t0 = time.time()
        semops.connect(engine=eng, model=SlowModel(0.02), workers=8, cache=False) \
            .rows(rows).sem_filter(PRED, policy="oracle")
        par = time.time() - t0
        self.assertLess(par, seq * 0.5, f"expected speedup: seq={seq:.2f}s par={par:.2f}s")

    def test_sem_join_parallel_matches_sequential(self):
        model, items, _groups = build(G=4, M=6)
        eng = InMemoryEngine()
        eng.add("items", items)
        left = [r for r in items if r.id.endswith(("u0", "u1"))]
        P = "Do ITEM A and ITEM B belong to the same group?"
        kw = dict(block_k=8, recall=0.9, precision=0.9, min_sample=30, seed=0)
        seq = semops.connect(engine=eng, model=model, workers=1).rows(left).sem_join("items", P, **kw)
        par = semops.connect(engine=eng, model=model, workers=8).rows(left).sem_join("items", P, **kw)
        self.assertEqual(sorted(seq.id_pairs()), sorted(par.id_pairs()))


if __name__ == "__main__":
    unittest.main()

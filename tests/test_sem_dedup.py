"""Offline sem_dedup tests: self-join (ANN blocking) + cascade adjudication +
connected-components clustering. Synthetic entities = groups, variants = the
duplicate rows that should be merged. No network / no cluster."""
import unittest

import semops
from semops import InMemoryEngine
from test_sem_join import JoinOracleModel, build


class TestSemDedup(unittest.TestCase):
    def setUp(self):
        # 5 entities, 4 variants each -> 20 rows; true answer = 5 clusters of 4
        self.model, self.items, self.groups = build(G=5, M=4)
        self.eng = InMemoryEngine()
        self.eng.add("items", self.items)

    def _partition(self, res):
        return sorted(sorted(r.id for r in c) for c in res.clusters)

    def test_recovers_entities_and_beats_all_pairs(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.sem_dedup("items", block_k=8, policy="oracle", seed=0)
        st = res.stats.as_dict()

        self.assertEqual(st["n_clusters"], 5)              # 5 distinct entities
        self.assertEqual(st["n_duplicate_rows"], 15)       # 20 rows - 5 entities
        for c in res.clusters:
            self.assertEqual(len({self.groups[r.id] for r in c}), 1)  # pure cluster
            self.assertEqual(len(c), 4)                    # all 4 variants merged
        # blocking made it cheaper than comparing all pairs
        self.assertLess(st["oracle_calls"], st["all_pairs"])
        self.assertEqual(len(res.canonical()), 5)          # one representative per entity

    def test_cascade_and_block_adjudicate_same_partition_fewer_calls(self):
        kw = dict(block_k=8, recall=0.99, precision=0.99, min_sample=40, seed=0)
        s1 = semops.connect(engine=self.eng, model=self.model)
        per_pair = s1.sem_dedup("items", **kw)
        s2 = semops.connect(engine=self.eng, model=self.model)
        block = s2.sem_dedup("items", block_adjudicate=True, **kw)

        self.assertEqual(self._partition(per_pair), self._partition(block))
        self.assertGreater(block.stats.block_calls, 0)
        self.assertLessEqual(block.stats.oracle_calls, per_pair.stats.oracle_calls)

    def test_no_false_merges_with_exact_oracle(self):
        # exact oracle -> precision 1.0 -> never merges two different entities
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.sem_dedup("items", block_k=8, policy="oracle", seed=0)
        for c in res.clusters:
            self.assertEqual(len({self.groups[r.id] for r in c}), 1)


if __name__ == "__main__":
    unittest.main()

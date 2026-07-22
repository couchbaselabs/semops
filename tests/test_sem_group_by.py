"""Offline sem_group_by tests: k-means over embeddings recovers latent groups;
optional LLM naming adds labels. No network / no cluster.

Data has ONE dominant latent factor (the topic) plus a unique per-row filler, so
a correct clustering must recover the topics."""
import unittest

import semops
from semops import FakeModelClient, InMemoryEngine, Row


def build_groups(G=5, M=8, dims=64, seed=0):
    model = FakeModelClient(dims=dims)
    texts = [f"topic{g} topic{g} topic{g} sample{g * M + i}"
             for g in range(G) for i in range(M)]
    vecs = model.embed(texts)
    rows, truth = [], {}
    for k, (g, i) in enumerate((g, i) for g in range(G) for i in range(M)):
        rid = f"r{k}"
        rows.append(Row(id=rid, text=texts[k], embedding=vecs[k], doc={"topic": g}))
        truth[rid] = g
    return model, rows, truth


def purity(groups, truth):
    total = correct = 0
    for g in groups:
        counts = {}
        for r in g.rows:
            counts[truth[r.id]] = counts.get(truth[r.id], 0) + 1
        total += len(g.rows)
        correct += max(counts.values()) if counts else 0
    return correct / total if total else 1.0


class TestSemGroupBy(unittest.TestCase):
    def setUp(self):
        self.model, self.rows, self.truth = build_groups(G=5, M=8)
        self.eng = InMemoryEngine()
        self.eng.add("items", self.rows)

    def test_embedding_clustering_recovers_groups_no_llm(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.sem_group_by("items", k=5, method="embedding", seed=0)
        st = res.stats.as_dict()

        self.assertEqual(st["k"], 5)
        self.assertEqual(st["llm_calls"], 0)             # pure embedding path: no LLM
        self.assertEqual(sum(res.sizes().values()), 40)  # every row assigned once
        self.assertGreaterEqual(purity(res.groups, self.truth), 0.95)

    def test_naming_adds_labels_and_costs_k_calls(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.sem_group_by("items", k=5, name_clusters=True, seed=0)
        self.assertEqual(res.stats.as_dict()["llm_calls"], 5)  # one naming call per group
        self.assertTrue(all(g.label for g in res.groups))

    def test_assignments_cover_all_rows(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.sem_group_by("items", k=5, seed=0)
        self.assertEqual(set(res.assignments().keys()), {r.id for r in self.rows})


if __name__ == "__main__":
    unittest.main()

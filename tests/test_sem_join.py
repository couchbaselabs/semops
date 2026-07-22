"""Offline sem_join tests: blocking (ANN) + cascade adjudication on a synthetic
same-group join. Embeddings cluster by group so blocking finds candidates; a
group-aware oracle decides matches. No network / no cluster."""
import re
import unittest

import semops
from semops import FakeModelClient, InMemoryEngine, Row, sem_join


class JoinOracleModel:
    """Embeds like FakeModelClient (so same-group items are near in vector space)
    and judges a pair as matching iff both halves carry the same group tag."""

    def __init__(self, dims=64, block_limit=10 ** 9):
        self._e = FakeModelClient(dims=dims)
        self.embed_model = "join-fake"
        self.chat_model = "join-oracle"
        self.spent_usd = 0.0
        self.calls = 0
        self.block_limit = block_limit  # simulate block-join overflow past this many

    def embed(self, texts):
        return self._e.embed(texts)

    def judge(self, predicate, text):
        self.calls += 1
        groups = re.findall(r"group(\d+)", text)  # one from ITEM A, one from ITEM B
        return len(groups) >= 2 and groups[0] == groups[1]

    def match_block(self, predicate, query, candidates):
        self.calls += 1
        qg = re.findall(r"group(\d+)", query)[:1]
        got = candidates[:self.block_limit]
        idxs = [i for i, c in enumerate(got) if re.findall(r"group(\d+)", c)[:1] == qg]
        return idxs, len(candidates) <= self.block_limit

    def generate(self, prompt, max_tokens=64):
        self.calls += 1
        from collections import Counter
        gs = re.findall(r"group\d+", prompt.lower())
        if gs:
            return Counter(gs).most_common(1)[0][0]  # dominant group tag = the label
        return "group"


def build(G=5, M=8):
    model = JoinOracleModel()
    items, groups = [], {}
    texts = [f"widget group{g} unit{i}" for g in range(G) for i in range(M)]
    vecs = model.embed(texts)
    k = 0
    for g in range(G):
        for i in range(M):
            rid = f"g{g}u{i}"
            items.append(Row(id=rid, text=texts[k], embedding=vecs[k],
                             doc={"id": rid, "group": g}))
            groups[rid] = g
            k += 1
    return model, items, groups


def prf(true_pairs, got_pairs):
    tp = len(true_pairs & got_pairs)
    p = tp / len(got_pairs) if got_pairs else 1.0
    r = tp / len(true_pairs) if true_pairs else 1.0
    return p, r


class TestSemJoin(unittest.TestCase):
    def setUp(self):
        self.model, self.items, self.groups = build(G=5, M=8)
        self.eng = InMemoryEngine()
        self.eng.add("items", self.items)
        self.left = [r for r in self.items if r.id.endswith(("u0", "u1", "u2"))]  # 3/group = 15
        # ground truth: same-group, non-self pairs
        self.truth = {
            (l.id, r.id)
            for l in self.left for r in self.items
            if r.id != l.id and self.groups[r.id] == self.groups[l.id]
        }
        self.PRED = "Do ITEM A and ITEM B belong to the same group?"

    def test_blocking_prunes_and_cascade_is_accurate(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.rows(self.left).sem_join(
            "items", self.PRED, block_k=12, recall=0.85, precision=0.85, min_sample=40, seed=0)
        st = res.stats.as_dict()
        got = set(res.id_pairs())
        p, r = prf(self.truth, got)

        # blocking pruned the quadratic space
        self.assertLess(st["candidate_pairs"], st["nested_loop_calls"])
        # cascade spent fewer oracle calls than a full nested-loop join
        self.assertLess(st["oracle_calls"], st["nested_loop_calls"])
        self.assertGreater(st["savings_ratio"], 1.0)
        # and the matches are accurate
        self.assertGreaterEqual(p, 0.9, f"precision {p}, stats={st}")
        self.assertGreaterEqual(r, 0.85, f"recall {r}, stats={st}")

    def test_oracle_policy_matches_ground_truth_within_blocks(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.rows(self.left).sem_join("items", self.PRED, block_k=12, policy="oracle")
        got = set(res.id_pairs())
        p, r = prf(self.truth, got)
        self.assertEqual(p, 1.0)          # oracle never accepts a cross-group pair
        self.assertGreaterEqual(r, 0.85)  # recall bounded only by blocking

    def test_blocking_policy_is_high_recall_no_llm(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.rows(self.left).sem_join("items", self.PRED, block_k=12, policy="blocking")
        st = res.stats.as_dict()
        self.assertEqual(st["oracle_calls"], 0)   # pure embedding join
        _, r = prf(self.truth, set(res.id_pairs()))
        self.assertGreaterEqual(r, 0.85)

    def test_block_adjudication_batches_and_matches_per_pair(self):
        # same cascade, same result — but block-join batches the escalate band,
        # so it makes fewer LLM calls than one-per-pair adjudication.
        kw = dict(block_k=12, recall=0.99, precision=0.99, min_sample=40, seed=0)
        s = semops.connect(engine=self.eng, model=self.model)
        per_pair = s.rows(self.left).sem_join("items", self.PRED, **kw)
        s2 = semops.connect(engine=self.eng, model=self.model)
        block = s2.rows(self.left).sem_join("items", self.PRED, block_adjudicate=True, **kw)

        self.assertEqual(set(per_pair.id_pairs()), set(block.id_pairs()))  # identical result
        self.assertGreater(block.stats.block_calls, 0)                     # batching ran
        self.assertLess(block.stats.oracle_calls, per_pair.stats.oracle_calls)  # fewer calls

    def test_block_adjudication_overflow_adapts(self):
        # tiny output budget forces overflow -> adaptive shrink; result stays correct
        model = JoinOracleModel(block_limit=3)
        sess = semops.connect(engine=self.eng, model=model)
        res = sess.rows(self.left).sem_join(
            "items", self.PRED, block_adjudicate=True, block_alpha=4,
            block_k=12, recall=0.99, precision=0.99, min_sample=40, seed=0)
        self.assertGreater(res.stats.overflows, 0)
        p, r = prf(self.truth, set(res.id_pairs()))
        self.assertGreaterEqual(p, 0.9)
        self.assertGreaterEqual(r, 0.85)

    def test_empty_left(self):
        sess = semops.connect(engine=self.eng, model=self.model)
        res = sess.rows([]).sem_join("items", self.PRED)
        self.assertEqual(res.pairs, [])


if __name__ == "__main__":
    unittest.main()

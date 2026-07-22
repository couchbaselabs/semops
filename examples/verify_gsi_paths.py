"""Verify the GSI fast paths against a live cluster.

Uses the stored `label` field as a free, perfect oracle so quality differences are
attributable to the retrieval path and not to LLM noise.

  ./.venv/bin/python examples/verify_gsi_paths.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import CouchbaseEngine, HttpQueryCluster
from cb_common import BUCKET, PW, QUERY, SCOPE, USER, FastEmbedEmbedder
from eval_classification import SklearnLRProxy

PREDICATE = "this is a negative or critical movie review"
COLL = "reviews"


class LabelOracle:
    """Perfect oracle from the stored label — isolates retrieval from LLM noise."""

    def __init__(self, emb):
        self._e = emb
        self.embed_model, self.chat_model = emb.embed_model, "label-oracle"
        self.spent_usd, self.calls = 0.0, 0

    def embed(self, t):
        return self._e.embed(t)

    def judge(self, predicate, text):
        self.calls += 1
        return bool(_LAB.get(text, False))


_LAB: dict = {}


def prf(truth_ids, got_ids):
    tp = len(truth_ids & got_ids)
    p = tp / len(got_ids) if got_ids else 1.0
    r = tp / len(truth_ids) if truth_ids else 1.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def main():
    q = HttpQueryCluster(QUERY, USER, PW)
    emb = FastEmbedEmbedder()
    ks = f"`{BUCKET}`.`{SCOPE}`.`{COLL}`"
    rows = q.query(f"SELECT META(d).id AS id, d.text AS t, d.label AS lab, d.embedding AS v FROM {ks} d")
    for r in rows:
        _LAB[r["t"]] = r["lab"]
    truth = {r["id"] for r in rows if r["lab"]}
    print(f"{len(rows)} docs, {len(truth)} truly negative ({len(truth)/len(rows):.1%})\n")

    def engine(**kw):
        return CouchbaseEngine("", USER, PW, BUCKET, SCOPE, cluster=q, nprobes=8, **kw)

    # ---- 1. covered vs uncovered ANN candidates -------------------------------
    print("=" * 72)
    print("1  ann_candidates: covered (with_vectors=False) vs today")
    print("=" * 72)
    pv = list(emb.embed([PREDICATE])[0])
    e_plain, e_cov = engine(), engine(covering_index="idx_reviews_cov")
    for name, eng, wv in [("with_vectors=True  (fetches doc+vector)", e_plain, True),
                          ("with_vectors=False (covered scan)      ", e_cov, False)]:
        t0 = time.time()
        got = eng.ann_candidates(COLL, pv, 200, with_vectors=wv)
        ms = (time.time() - t0) * 1000
        print(f"  {name}  {len(got):3} rows  {ms:6.1f}ms  "
              f"vectors_pulled={eng.vectors_pulled}")
    same = ([r.id for r, _ in e_plain.ann_candidates(COLL, pv, 50)] ==
            [r.id for r, _ in e_cov.ann_candidates(COLL, pv, 50, with_vectors=False)])
    print(f"  identical top-50 ordering both ways: {same}")

    # ---- 2. ann_batch --------------------------------------------------------
    print()
    print("=" * 72)
    print("2  ann_batch: N probes per round trip vs one query each")
    print("=" * 72)
    probes = [list(v) for v in emb.embed([
        "hilarious comedy", "boring plot", "stunning visuals", "terrible acting",
        "moving story", "predictable and dull", "brilliant direction", "waste of time"])]
    eng = engine(covering_index="idx_reviews_cov")
    t0 = time.time()
    seq = [eng.ann_candidates(COLL, v, 20, with_vectors=False) for v in probes]
    ms_seq = (time.time() - t0) * 1000
    t0 = time.time()
    bat = eng.ann_batch(COLL, probes, 20)
    ms_bat = (time.time() - t0) * 1000
    match = all([r.id for r, _ in a] == [r.id for r, _ in b] for a, b in zip(seq, bat))
    print(f"  sequential {ms_seq:7.1f}ms   batched {ms_bat:7.1f}ms   -> {ms_seq/ms_bat:.2f}x")
    print(f"  batched results identical to sequential: {match}")

    # ---- 3. ann_above correctness --------------------------------------------
    print()
    print("=" * 72)
    print("3  ann_above: index-served proxy vs exact w.x over the whole collection")
    print("=" * 72)
    proxy = SklearnLRProxy()
    proxy.fit([r["v"] for r in rows[:300]], [bool(r["lab"]) for r in rows[:300]])
    w, _b = proxy.linear_params()
    exact_scores = {r["id"]: sum(a * c for a, c in zip(w, r["v"])) for r in rows}
    eng = engine(covering_index="idx_reviews_cov")
    print(f"  vectors unit-norm (precondition): {eng.vectors_normalised(COLL)}")
    for tau in (6.0, 3.0, 0.0, -3.0):
        want = {i for i, s in exact_scores.items() if s >= tau}
        t0 = time.time()
        got_rows, exhaustive = eng.ann_above(COLL, w, tau, est_k=256)
        ms = (time.time() - t0) * 1000
        got = {r.id for r, _ in got_rows}
        miss = len(want - got)
        print(f"  tau={tau:5.1f}  exact={len(want):4}  ann_above={len(got):4}  "
              f"missed={miss:3}  recall={1-miss/max(len(want),1):.3f}  "
              f"exhaustive={exhaustive}  {ms:6.1f}ms")

    # ---- 4. end-to-end: sem_filter_source vs row-list sem_filter -------------
    print()
    print("=" * 72)
    print("4  end-to-end sem_filter: index-served vs scoring every row")
    print("=" * 72)
    model = LabelOracle(emb)
    for name, fn in [
        ("row-list  (scan all, score all)", "rows"),
        ("source    (index-served proxy) ", "source"),
    ]:
        eng = engine(covering_index="idx_reviews_cov")
        sess = semops.connect(engine=eng, model=model, workers=8)
        model.calls = 0
        t0 = time.time()
        if fn == "rows":
            res = sess.scan(COLL).sem_filter(PREDICATE, proxy_model=SklearnLRProxy(),
                                             recall=0.9, precision=0.9)
            kept, st = {r.id for r in res.rows_}, res.stats()
        else:
            res = sess.sem_filter(COLL, PREDICATE, proxy_model=SklearnLRProxy(),
                                  recall=0.9, precision=0.9)
            kept, st = {r.id for r in res.rows}, res.stats.as_dict()
        ms = (time.time() - t0) * 1000
        p, r, f = prf(truth, kept)
        print(f"  {name}  P={p:.3f} R={r:.3f} F1={f:.3f}  "
              f"oracle_calls={st['llm_calls']:4}  scored={st.get('n_scored') or st['n_rows']:4}  "
              f"vec_pulled={eng.vectors_pulled:4}  {ms:7.0f}ms")


if __name__ == "__main__":
    main()

"""Does the cascade's saving actually scale with corpus size?

At 2k rows the cascade looked unimpressive (1.3-1.5x). The suspicion is that this
is an artefact of size, not of the method: calibration costs a FIXED number of
oracle calls (min_sample..max_sample), so on a small corpus that fixed cost eats
the budget, while on a large one it amortises. This measures the curve.

Second question: the index-served path (Session.sem_filter over a source) was
SLOWER than scanning everything at 2k rows, because its threshold sweep issues
several queries. That should invert once scanning the collection is the expensive
part.

  ./.venv/bin/python examples/bench_scale.py --sizes 2000 10000 50000 120000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import CouchbaseEngine, HttpQueryCluster
from cb_common import BUCKET, PW, QUERY, SCOPE, USER, FastEmbedEmbedder
from eval_classification import SklearnLRProxy

PREDICATE = "this news story is about science or technology"
COLL = "news"


class LabelOracle:
    """Free perfect oracle from the stored label — isolates cost from LLM noise."""

    def __init__(self, emb, truth):
        self._e, self.truth = emb, truth
        self.embed_model, self.chat_model = emb.embed_model, "label-oracle"
        self.spent_usd, self.calls = 0.0, 0

    def embed(self, t):
        return self._e.embed(t)

    def judge(self, predicate, text):
        self.calls += 1
        return bool(self.truth.get(text, False))


def prf(truth_ids, got_ids):
    tp = len(truth_ids & got_ids)
    p = tp / len(got_ids) if got_ids else 1.0
    r = tp / len(truth_ids) if truth_ids else 1.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[2000, 10000, 50000, 120000])
    ap.add_argument("--recall", type=float, default=0.9)
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--index", default="idx_news_vec")
    args = ap.parse_args()

    q = HttpQueryCluster(QUERY, USER, PW, timeout=1800)
    emb = FastEmbedEmbedder()
    ks = f"`{BUCKET}`.`{SCOPE}`.`{COLL}`"
    total = q.query(f"SELECT COUNT(*) AS c FROM {ks}")[0]["c"]
    print(f"collection `{COLL}` holds {total} docs\n")

    print("=" * 92)
    print("A  savings vs corpus size (row-list path: scan n, score n)")
    print("=" * 92)
    print(f"  {'n':>7}  {'pos%':>5}  {'calls':>6}  {'savings':>8}  {'P':>5} {'R':>5} {'F1':>5}  "
          f"{'tau-':>7} {'tau+':>7}  {'wall':>8}")
    for n in args.sizes:
        if n > total:
            continue
        rows = q.query(f"SELECT META(d).id AS id, d.text AS t, d.label AS lab FROM {ks} d LIMIT {n}")
        truth_by_text = {r["t"]: r["lab"] for r in rows}
        truth_ids = {r["id"] for r in rows if r["lab"]}
        model = LabelOracle(emb, truth_by_text)
        eng = CouchbaseEngine("", USER, PW, BUCKET, SCOPE, cluster=q, nprobes=8)
        sess = semops.connect(engine=eng, model=model, workers=16)
        t0 = time.time()
        res = sess.scan(COLL, limit=n).sem_filter(
            PREDICATE, proxy_model=SklearnLRProxy(),
            recall=args.recall, precision=args.precision)
        ms = (time.time() - t0) * 1000
        st = res.stats()
        p, r, f = prf(truth_ids, {x.id for x in res.rows_})
        print(f"  {n:7}  {len(truth_ids)/len(rows):5.1%}  {st['llm_calls']:6}  "
              f"{n/max(st['llm_calls'],1):7.2f}x  {p:.3f} {r:.3f} {f:.3f}  "
              f"{st['tau_minus']:7.3f} {st['tau_plus']:7.3f}  {ms:7.0f}ms")

    # ---- B: index-served vs scan-everything, at full size ---------------------
    print()
    print("=" * 92)
    print(f"B  index-served (Session.sem_filter) vs scan-all, at n={total}")
    print("=" * 92)
    rows = q.query(f"SELECT META(d).id AS id, d.text AS t, d.label AS lab FROM {ks} d")
    truth_by_text = {r["t"]: r["lab"] for r in rows}
    truth_ids = {r["id"] for r in rows if r["lab"]}
    for name, mode in [("scan-all + score-all ", "rows"), ("index-served (ann_above)", "source")]:
        model = LabelOracle(emb, truth_by_text)
        eng = CouchbaseEngine("", USER, PW, BUCKET, SCOPE, cluster=q, nprobes=8,
                              covering_index=args.index)
        sess = semops.connect(engine=eng, model=model, workers=16)
        t0 = time.time()
        if mode == "rows":
            res = sess.scan(COLL).sem_filter(PREDICATE, proxy_model=SklearnLRProxy(),
                                             recall=args.recall, precision=args.precision)
            kept, st = {x.id for x in res.rows_}, res.stats()
        else:
            res = sess.sem_filter(COLL, PREDICATE, proxy_model=SklearnLRProxy(),
                                  recall=args.recall, precision=args.precision, est_k=4096)
            kept, st = {x.id for x in res.rows}, res.stats.as_dict()
        ms = (time.time() - t0) * 1000
        p, r, f = prf(truth_ids, kept)
        print(f"  {name}  P={p:.3f} R={r:.3f} F1={f:.3f}  calls={st['llm_calls']:6}  "
              f"scored={st.get('n_scored') or st['n_rows']:6}  "
              f"vec_pulled={eng.vectors_pulled:6}  {ms:8.0f}ms")


if __name__ == "__main__":
    main()

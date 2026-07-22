"""How far can we push our operators into a GSI vector index?

Four questions, measured against a live cluster (not the docs):

  H1  Can the LEARNED PROXY be served by ANN instead of a linear scan?
      The proxy is logistic regression: score = sigma(w.x + b), monotonic in w.x.
      For L2-normalised x, w.x = |w|*cos(w,x) -> ranking by w.x IS ranking by
      cosine(w,x). So a cosine index queried with the WEIGHT VECTOR should return
      the proxy's own top-k. If true, the accept band is a sublinear index lookup.
  H2  Does nprobes actually move blocking recall, and where does it saturate?
  H3  Does topNScan matter independently of LIMIT? (we hit this truncation before)
  H4  Can INCLUDE columns cover the scan so documents/vectors never get fetched?

  ./.venv/bin/python examples/gsi_probe.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semops import HttpQueryCluster
from semops.vectormath import cosine_similarity
from cb_common import BUCKET, PW, QUERY, SCOPE, USER, FastEmbedEmbedder

KS = f"`{BUCKET}`.`{SCOPE}`.`reviews`"
PREDICATE = "this is a negative or critical movie review"


def adv(metric, nprobes, rerank, topn):
    return (f"APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, '{metric}', "
            f"{nprobes}, {str(rerank).lower()}, {topn})")


def ann_ids(q, qvec, k, metric="cosine", nprobes=8, rerank=True, topn=None):
    topn = topn or k
    dist = adv(metric, nprobes, rerank, topn)
    stmt = (f"SELECT META(d).id AS id FROM {KS} d "
            f"ORDER BY {dist} LIMIT $k")
    t0 = time.time()
    rows = q.query(stmt, qvec=list(qvec), k=int(k))
    return [r["id"] for r in rows], (time.time() - t0) * 1000


def recall(got, truth_k):
    return len(set(got) & set(truth_k)) / max(len(truth_k), 1)


def main():
    q = HttpQueryCluster(QUERY, USER, PW)
    emb = FastEmbedEmbedder()

    print("pulling vectors + labels for ground truth (one time, local)...")
    rows = q.query(f"SELECT META(d).id AS id, d.embedding AS v, d.label AS lab FROM {KS} d")
    ids = [r["id"] for r in rows]
    vecs = [r["v"] for r in rows]
    labs = [bool(r["lab"]) for r in rows]
    norms = [sum(x * x for x in v) ** 0.5 for v in vecs]
    print(f"  {len(ids)} docs, |x| in [{min(norms):.4f}, {max(norms):.4f}] "
          f"-> {'L2-NORMALISED' if max(norms) - min(norms) < 1e-3 and abs(max(norms)-1) < 1e-3 else 'NOT normalised'}")

    # ---------------- fit the learned proxy exactly as sem_filter does -------
    from eval_classification import SklearnLRProxy  # the proxy sem_filter actually uses
    n_s = 300
    proxy = SklearnLRProxy()
    proxy.fit([vecs[i] for i in range(n_s)], [labs[i] for i in range(n_s)])
    w, b = proxy.linear_params()
    print(f"  fitted LR proxy on {n_s} samples: |w|={sum(x*x for x in w)**0.5:.3f}, b={b:.3f}\n")

    # =============== H1: is the learned proxy ANN-servable? =================
    print("=" * 74)
    print("H1  learned proxy via ANN  (query the cosine index with the LR WEIGHTS)")
    print("=" * 74)
    exact = sorted(range(len(vecs)), key=lambda i: sum(w[j] * vecs[i][j] for j in range(len(w))),
                   reverse=True)
    for k in (10, 50, 200):
        truth = [ids[i] for i in exact[:k]]
        got, ms = ann_ids(q, w, k, nprobes=8, topn=max(k, 100))
        # precision of the top-k *as a classifier* (how many are truly negative reviews)
        lab_by_id = dict(zip(ids, labs))
        prec = sum(lab_by_id[i] for i in got) / max(len(got), 1)
        print(f"  k={k:4}  ANN-vs-exact recall={recall(got, truth):.3f}  "
              f"label-precision@k={prec:.3f}  {ms:6.1f}ms  rows={len(got)}")

    # baseline: cosine to the raw PREDICATE TEXT (what a non-learned proxy does)
    pv = emb.embed([PREDICATE])[0]
    exact_p = sorted(range(len(vecs)), key=lambda i: cosine_similarity(pv, vecs[i]), reverse=True)
    lab_by_id = dict(zip(ids, labs))
    for k in (50, 200):
        got, ms = ann_ids(q, pv, k, nprobes=8, topn=max(k, 100))
        prec = sum(lab_by_id[i] for i in got) / max(len(got), 1)
        print(f"  [predicate-text proxy] k={k:4} label-precision@k={prec:.3f}  {ms:6.1f}ms")

    # =============== H2/H3: nprobes and topNScan ============================
    print()
    print("=" * 74)
    print("H2/H3  nprobes and topNScan vs ANN recall (query = predicate embedding)")
    print("=" * 74)
    K = 200
    truth = [ids[i] for i in exact_p[:K]]
    print(f"  recall of exact-cosine top-{K}:")
    for np_ in (1, 2, 4, 8, 16, 32, 64):
        got, ms = ann_ids(q, pv, K, nprobes=np_, topn=K)
        print(f"    nprobes={np_:3}  topNScan={K}   recall={recall(got, truth):.3f}  "
              f"rows={len(got):4}  {ms:6.1f}ms")
    print(f"  topNScan sweep at nprobes=8 (LIMIT stays {K}):")
    for tn in (10, 50, 200, 1000, 2000):
        got, ms = ann_ids(q, pv, K, nprobes=8, topn=tn)
        print(f"    topNScan={tn:5}          recall={recall(got, truth):.3f}  "
              f"rows={len(got):4}  {ms:6.1f}ms")


if __name__ == "__main__":
    main()

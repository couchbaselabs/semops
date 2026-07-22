"""Run sem_filter through CouchbaseEngine — native ANN pushdown on a real
cluster's bhive vector index. Assumes cb_ingest.py has loaded default._default.docs
and built idx_docs_vec.

  ./.venv/bin/python examples/eval_couchbase.py --predicate "is about sports such as baseball or hockey"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import CouchbaseEngine, HttpQueryCluster, Row
from cb_common import BUCKET, PW, QUERY, SCOPE, USER, FastEmbedEmbedder
from eval_classification import SklearnLRProxy, best_threshold, prf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="docs")
    ap.add_argument("--predicate", default="is about sports such as baseball or hockey")
    ap.add_argument("--recall", type=float, default=0.9)
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--min-sample", dest="min_sample", type=int, default=300)
    ap.add_argument("--oracle", default="label", choices=["label", "gemini"])
    ap.add_argument("--pushdown", action="store_true",
                    help="scan WITHOUT vectors; score server-side (VECTOR_DISTANCE) so embeddings stay in-cluster")
    args = ap.parse_args()

    coll = args.collection
    ks = f"`{BUCKET}`.`{SCOPE}`.`{coll}`"
    idx = f"idx_{coll}_vec"
    cluster = HttpQueryCluster(QUERY, USER, PW)
    emb = FastEmbedEmbedder()
    engine = CouchbaseEngine(
        "cluster_run", USER, PW, bucket=BUCKET, scope=SCOPE, cluster=cluster,
        vector_field="embedding", text_field="text", metric="cosine")

    # 0) prove the vector index is actually used (EXPLAIN the ANN query)
    qv = emb.embed([args.predicate])[0]
    print(f"0) EXPLAIN ANN query — is {idx} used?")
    plan = cluster.query(
        f"EXPLAIN SELECT META(d).id FROM {ks} d "
        f"ORDER BY APPROX_VECTOR_DISTANCE(d.embedding, $qvec, 'cosine') LIMIT 5", qvec=qv)
    txt = str(plan)
    used = idx in txt
    print(f"   index_used({idx})={used}  scan={'IndexScan3' if 'IndexScan3' in txt else '?'}")

    # 1) sem_search sanity via native ANN pushdown
    print("\n1) sem_search (ANN pushdown) top-5 for the predicate query:")
    cands = engine.ann_candidates(coll, qv, k=5)
    for r, sim in cands:
        print(f"   sim={sim:.3f}  label={r.doc.get('label')}  {r.text[:70].strip()}")

    # 2) sem_filter over the FULL mixed collection (a boolean filter needs
    #    positives AND negatives — ANN retrieval would hand us only positives).
    mode = "PUSHDOWN (vectors stay in-cluster)" if args.pushdown else "pull vectors to service"
    print(f"\n2) sem_filter cascade over the full collection (oracle={args.oracle}, {mode})")
    rows = engine.scan(coll, with_vectors=not args.pushdown)  # omit embeddings when pushing down
    labels = [bool(r.doc.get("label")) for r in rows]

    if args.oracle == "label":
        judge = lambda pred, text, _t={r.text: y for r, y in zip(rows, labels)}: _t.get(text, False)
    else:
        from semops import OpenAICompatClient
        key = os.environ.get("GEMINI_API_KEY", "")
        gem = OpenAICompatClient(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key=key, chat_model="gemini-flash-latest")
        judge = gem.judge

    class M:  # split model: fastembed proxy + chosen oracle
        embed_model = emb.embed_model
        chat_model = "cb-oracle"
        spent_usd = 0.0
        calls = 0
        def embed(self, texts): return emb.embed(texts)
        def judge(self, predicate, text): self.calls += 1; return judge(predicate, text)

    model = M()
    # vector-search-alone baseline: cosine of each doc's vector to the predicate
    base_scores = engine.proxy_scores(rows, qv)
    vf, vp, vr, vt = best_threshold(base_scores, labels)

    sess = semops.connect(engine=engine, model=model, budget_usd=50.0)
    pipe = sess.rows(rows).sem_filter(
        args.predicate, recall=args.recall, precision=args.precision,
        min_sample=args.min_sample, proxy_model=SklearnLRProxy())
    st = pipe.stats()
    kept = {r.id for r in pipe}  # Row objects carry META().id; doc has no 'id' field
    cp, cr, cf = prf(labels, [r.id in kept for r in rows])

    print("\n=== RESULTS (CouchbaseEngine, native ANN pushdown) ===")
    print(f"{'method':<34}{'P':>7}{'R':>7}{'F1':>7}{'oracle calls':>13}{'savings':>9}")
    print("-" * 78)
    print(f"{'vector search alone (best thr)':<34}{vp:>7.3f}{vr:>7.3f}{vf:>7.3f}{0:>13}{'--':>9}")
    print(f"{'sem_filter cascade':<34}{cp:>7.3f}{cr:>7.3f}{cf:>7.3f}"
          f"{st['llm_calls']:>13}{str(st['savings_ratio'])+'x':>9}")
    print("-" * 78)
    print(f"bands: accept={st['n_accept']} escalate={st['n_escalate']} reject={st['n_reject']}"
          f"  errors={st['errors']}")
    pulled = engine.vectors_pulled
    print(f"\nvectors shipped out of Couchbase: {pulled} / {len(rows)} "
          f"({100*(1-pulled/max(1,len(rows))):.0f}% stayed in-cluster; scored via VECTOR_DISTANCE)")
    print(f"takeaway: cascade F1 {cf:.3f} vs vector-alone {vf:.3f}, "
          f"{st['savings_ratio']}x fewer oracle calls.")


if __name__ == "__main__":
    main()

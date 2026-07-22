"""Verify sem_group_by / sem_dedup / sem_join run NATIVELY on the Couchbase
cluster (bhive ANN blocking per row via CouchbaseEngine). Uses 20NG with the
newsgroup CATEGORY as ground truth and a local category oracle (no LLM needed).

  ./.venv/bin/python examples/verify_couchbase_ops.py --n-per 60
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import CouchbaseEngine, HttpQueryCluster
from cb_common import BUCKET, PW, QUERY, SCOPE, USER, FastEmbedEmbedder, rest

CATS = ["sci.space", "rec.sport.baseball", "comp.graphics", "sci.med",
        "talk.politics.mideast", "rec.autos", "rec.sport.hockey"]
COLL = "posts"
KS = f"`{BUCKET}`.`{SCOPE}`.`{COLL}`"


def load(n_per, seed=0):
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", categories=CATS, remove=("headers", "footers", "quotes"))
    order = list(range(len(d.data)))
    random.Random(seed).shuffle(order)
    by = {}
    for i in order:
        c = d.target_names[d.target[i]]
        if d.data[i].strip() and len(by.get(c, [])) < n_per:
            by.setdefault(c, []).append(d.data[i])
    texts, cats = [], []
    for c, ts in by.items():
        texts += ts
        cats += [c] * len(ts)
    return texts, cats


def ingest(q, texts, cats, vecs):
    rest("POST", f"/pools/default/buckets/{BUCKET}/scopes/{SCOPE}/collections", {"name": COLL})
    time.sleep(2)
    q.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON {KS}")
    q.query(f"DELETE FROM {KS}")  # clean slate for idempotent re-runs
    for s in range(0, len(texts), 100):
        vals = [f'("p{i}", {json.dumps({"text": texts[i], "category": cats[i], "embedding": vecs[i]})})'
                for i in range(s, min(s + 100, len(texts)))]
        q.query(f"UPSERT INTO {KS} (KEY, VALUE) VALUES " + ",".join(vals))
    try:
        q.query(f"CREATE VECTOR INDEX idx_{COLL}_vec ON {KS}(embedding VECTOR) "
                f"WITH {{'dimension':{len(vecs[0])},'similarity':'cosine',"
                f"'description':'IVF,SQ8','train_list':{len(texts)}}}")
    except Exception as e:
        print("  (vector index:", str(e)[:80], ")")
    for _ in range(60):
        rows = q.query(f"SELECT state FROM system:indexes WHERE keyspace_id='{COLL}'")
        if rows and all(r["state"] == "online" for r in rows):
            break
        time.sleep(5)


class CategoryOracle:
    """Local oracle: two items match iff they share a newsgroup category.
    Embeds via fastembed (query-time only); knows category by exact text."""

    def __init__(self, embedder, cat_of):
        self._e = embedder
        self.cat = cat_of
        self.embed_model = embedder.embed_model
        self.chat_model = "category-oracle"
        self.spent_usd = 0.0
        self.calls = 0

    def embed(self, texts):
        return self._e.embed(texts)

    def judge(self, predicate, text):
        self.calls += 1
        try:
            lt, rt = text.split("ITEM A:\n", 1)[1].split("\n\nITEM B:\n", 1)
        except Exception:
            return False
        cl = self.cat.get(lt)
        return cl is not None and cl == self.cat.get(rt)

    def match_block(self, predicate, query, candidates):
        self.calls += 1
        qc = self.cat.get(query)
        idxs = [i for i, c in enumerate(candidates) if qc is not None and self.cat.get(c) == qc]
        return idxs, True


def purity(clusters, cat_of):
    total = correct = 0
    for rows in clusters:
        counts = {}
        for r in rows:
            c = cat_of.get(r.text)
            counts[c] = counts.get(c, 0) + 1
        total += len(rows)
        correct += max(counts.values()) if counts else 0
    return correct / total if total else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=60)
    ap.add_argument("--block-k", dest="block_k", type=int, default=80,
                    help="ANN candidates per row; set to 2-4x expected fan-out")
    ap.add_argument("--engine", default="couchbase", choices=["couchbase", "memory"],
                    help="memory = same operators/data without a live cluster")
    args = ap.parse_args()

    emb = FastEmbedEmbedder()
    print(f"loading + embedding {args.n_per} docs/category ({len(CATS)} categories)...")
    texts, cats = load(args.n_per)
    vecs = emb.embed(texts)
    cat_of = {t: c for t, c in zip(texts, cats)}
    print(f"  {len(texts)} docs, dim={len(vecs[0])}")

    model = CategoryOracle(emb, cat_of)
    if args.engine == "couchbase":
        q = HttpQueryCluster(QUERY, USER, PW)
        print("ingesting into Couchbase + building bhive index...")
        ingest(q, texts, cats, vecs)
        engine = CouchbaseEngine("cluster_run", USER, PW, bucket=BUCKET, scope=SCOPE, cluster=q,
                                 vector_field="embedding", text_field="text", metric="cosine")
    else:
        from semops import InMemoryEngine, Row
        q = None
        engine = InMemoryEngine()
        engine.add(COLL, [Row(id=f"p{i}", text=t, embedding=v,
                              doc={"text": t, "category": c, "embedding": v})
                          for i, (t, c, v) in enumerate(zip(texts, cats, vecs))])
    sess = semops.connect(engine=engine, model=model)
    n = engine.count(COLL)
    print(f"cluster has {n} docs in {COLL}\n")

    # 1) sem_group_by — k-means over Couchbase-stored vectors
    g = sess.sem_group_by(COLL, k=len(CATS), method="embedding", seed=0)
    print(f"sem_group_by  k={g.stats.k}  purity={purity([grp.rows for grp in g.groups], cat_of):.3f}  "
          f"sizes={sorted(g.sizes().values())}  llm_calls={g.stats.llm_calls}")

    # 2) sem_dedup — self-join (bhive ANN blocking per row) + union-find
    d = sess.sem_dedup(COLL, "ITEM A and ITEM B are about the same subject",
                       block_k=args.block_k, block_adjudicate=True, recall=0.9, precision=0.9, min_sample=80)
    st = d.stats.as_dict()
    print(f"sem_dedup     clusters={st['n_clusters']}  purity={purity([c for c in d.clusters], cat_of):.3f}  "
          f"candidate_pairs={st['candidate_pairs']}  oracle_calls={st['oracle_calls']}  "
          f"savings={st['savings_ratio']}x (vs all-pairs {st['all_pairs']})")

    # 3) sem_join — a left subset joined back to the collection
    left = engine.scan(COLL)[:50]
    truth = {(l.id, r.id) for l in left for r in engine.scan(COLL)
             if r.id != l.id and cat_of.get(r.text) == cat_of.get(l.text)}
    j = sess.rows(left).sem_join(COLL, "ITEM A and ITEM B are about the same subject",
                                 block_k=args.block_k, block_adjudicate=True, recall=0.9,
                                 precision=0.9, min_sample=60)
    got = set(j.id_pairs())
    tp = len(truth & got)
    P = tp / len(got) if got else 1.0
    R = tp / len(truth) if truth else 1.0
    js = j.stats.as_dict()
    print(f"sem_join      matches={js['matches']}  P={P:.2f} R={R:.2f}  "
          f"candidate_pairs={js['candidate_pairs']}  oracle_calls={js['oracle_calls']}  "
          f"nested_loop={js['nested_loop_calls']}  savings={js['savings_ratio']}x")

    if q is None:
        return
    print("\nEXPLAIN check (native ANN pushdown for blocking):")
    plan = str(q.query(f"EXPLAIN SELECT META(d).id FROM {KS} d "
                       f"ORDER BY APPROX_VECTOR_DISTANCE(d.embedding, $v, 'cosine') LIMIT 5",
                       v=vecs[0]))
    print(f"  idx_{COLL}_vec used = {('idx_' + COLL + '_vec') in plan}, IndexScan3 = {'IndexScan3' in plan}")


if __name__ == "__main__":
    main()

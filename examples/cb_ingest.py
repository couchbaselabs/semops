"""Ingest a labeled dataset into Couchbase for the semops native-mode demo.

  --dataset 20ng     -> collection 'docs'   (topic task: is-about-sports)
  --dataset rotten   -> collection 'reviews'(polarity task: is-a-negative-review)

Embeds locally (fastembed, 384-d) -> UPSERT {text,label,embedding} -> PRIMARY
INDEX -> bhive VECTOR INDEX (IVF,SQ8,cosine) -> wait online.

  ./.venv/bin/python examples/cb_ingest.py --dataset rotten --n 2000
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semops import HttpQueryCluster
from cb_common import BUCKET, PW, QUERY, SCOPE, USER, FastEmbedEmbedder, rest
from eval_classification import DEFAULT_CATEGORIES, load_20ng


def load_rotten(n, seed):
    """Movie-review polarity. label True = NEGATIVE review (matches the predicate)."""
    from datasets import load_dataset
    d = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    texts = [d[i]["text"] for i in idx]
    labels = [d[i]["label"] == 0 for i in idx]  # 0 = negative
    return texts, labels


def load_agnews(n, seed, target=("Sci/Tech",)):
    """AG News: 120k news stories, 4 topics. Big enough to test whether the
    cascade's savings actually scale (the calibration sample is a fixed cost, so
    the ratio should improve as the corpus grows)."""
    from datasets import load_dataset
    names = ["World", "Sports", "Business", "Sci/Tech"]
    d = load_dataset("fancyzhx/ag_news", split="train")
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    idx = idx[:n]
    tset = set(target)
    rows = d.select(idx)
    texts = [t for t in rows["text"]]
    labels = [names[l] in tset for l in rows["label"]]
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="20ng", choices=["20ng", "rotten", "agnews"])
    ap.add_argument("--collection", default=None)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--target", nargs="*", default=["rec.sport.baseball", "rec.sport.hockey"])
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=100)
    args = ap.parse_args()

    coll = args.collection or {"rotten": "reviews", "agnews": "news"}.get(args.dataset, "docs")
    ks = f"`{BUCKET}`.`{SCOPE}`.`{coll}`"
    q = HttpQueryCluster(QUERY, USER, PW)

    print(f"1) create collection {BUCKET}._default.{coll}")
    code, body = rest("POST", f"/pools/default/buckets/{BUCKET}/scopes/{SCOPE}/collections",
                      {"name": coll})
    print(f"   -> HTTP {code} {'(ok)' if code in (200,202) else body[:80]}")
    time.sleep(2)

    print(f"2) load {args.n} docs ({args.dataset})")
    if args.dataset == "rotten":
        texts, labels = load_rotten(args.n, args.seed)
    elif args.dataset == "agnews":
        texts, labels = load_agnews(args.n, args.seed)
    else:
        texts, labels = load_20ng(args.categories, args.target, args.n, args.seed)
    emb = FastEmbedEmbedder()
    n = len(texts)
    pos = sum(labels)
    print(f"   {n} docs, {pos} positive ({100*pos/n:.1f}%), model={emb.embed_model}")

    # Embed and write in chunks. Embedding the whole corpus up front materialises
    # every vector as Python floats before a single row is written — at 120k docs
    # that reached 8.7GB RSS and drove the box into swap, where it made no further
    # progress. Streaming keeps only one chunk live at a time.
    print(f"3) embed + UPSERT into {ks} (chunks of {args.batch})")
    dim = None
    t_start = time.time()
    for s in range(0, n, args.batch):
        e = min(s + args.batch, n)
        vecs = emb.embed(texts[s:e])
        if dim is None:
            dim = len(vecs[0])
        vals = [f'("r{i}", {json.dumps({"text": texts[i], "label": bool(labels[i]), "embedding": list(vecs[i - s])})})'
                for i in range(s, e)]
        q.query(f"UPSERT INTO {ks} (KEY, VALUE) VALUES " + ",".join(vals))
        del vecs, vals
        rate = e / max(time.time() - t_start, 1e-9)
        eta = (n - e) / max(rate, 1e-9)
        print(f"   {e}/{n}  {rate:.0f} docs/s  eta {eta/60:.1f} min      ", end="\r")
    print(f"\n   done: {n} docs")

    print("4) CREATE PRIMARY INDEX")
    q.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON {ks}")

    idx = f"idx_{coll}_vec"
    print(f"5) CREATE VECTOR INDEX {idx} (bhive: IVF,SQ8, cosine, INCLUDE text+label)")
    # INCLUDE (text,label) makes the candidate scan index-covered: no KV Fetch, and
    # embeddings stay in the cluster (measured 4x faster, 51x less data per probe).
    # scan_nprobes defaults to 1, which measured 0.685 recall against exact top-200;
    # >=2 reached 1.000 on this collection. 8 leaves headroom as the corpus grows.
    def vector_ddl(name, similarity):
        return (f"CREATE VECTOR INDEX {name} ON {ks}(embedding VECTOR) "
                f"INCLUDE (`text`, `label`) "
                f"WITH {{'dimension':{dim},'similarity':'{similarity}',"
                f"'description':'IVF,SQ8','scan_nprobes':8,"
                f"'train_list':{min(n, 10000)}}}")

    # Two indexes, because Couchbase only picks a vector index when the query's
    # distance metric matches the index's `similarity`. Blocking queries cosine;
    # ann_above() (the index-served learned proxy) queries dot. Without the dot
    # index that path silently falls back to PrimaryScan3 + Fetch: still correct,
    # but a full scan, and slower than not using it (953ms vs 62ms measured).
    for name, sim in ((idx, "cosine"), (f"idx_{coll}_dot", "dot")):
        try:
            q.query(vector_ddl(name, sim))
            print(f"   submitted {name} ({sim})")
        except Exception as e:
            print(f"   ({name} returned:", str(e)[:140], ")")

    print("6) wait for indexes online...")
    for _ in range(60):
        rows = q.query(f"SELECT name, state FROM system:indexes WHERE keyspace_id='{coll}'")
        states = {r.get("name"): r.get("state") for r in rows}
        print("   ", states, end="\r")
        if states and all(v == "online" for v in states.values()):
            print("\n   all online:", states)
            break
        time.sleep(5)
    cnt = q.query(f"SELECT COUNT(*) AS c FROM {ks}")
    print("docs in collection:", cnt[0]["c"] if cnt else "?")


if __name__ == "__main__":
    main()

"""semops quickstart: one command, one cluster, no API keys.

    python examples/quickstart.py

It creates a collection, loads a small bundled labelled dataset (1,000 movie
reviews) with local embeddings, builds the vector indexes, and runs `sem_filter`
for the predicate "this is a negative review" against a live Couchbase cluster.

The oracle judges each escalated row. It uses a real LLM when a key is in the
environment (GEMINI_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY — embeddings stay
local, so only a chat model is needed), and falls back to the stored label
otherwise so the whole thing runs with no key at all.

Point it at any cluster via the environment (defaults are cluster_run):

    CB_QUERY_URL   query service   (default http://localhost:9499)
    CB_REST_URL    management REST (default http://localhost:9000)
    CB_USER / CB_PASSWORD / CB_BUCKET

Stock Couchbase Server (Docker / self-managed) uses ports 8093 / 8091:

    CB_QUERY_URL=http://localhost:8093 CB_REST_URL=http://localhost:8091 \\
    CB_PASSWORD=yourpass python examples/quickstart.py

    python examples/quickstart.py --cleanup     # drop the collection and exit
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import CouchbaseEngine, HttpQueryCluster, OpenAICompatClient
from cb_common import BUCKET, PW, QUERY, REST, SCOPE, USER, FastEmbedEmbedder, rest
from eval_classification import SklearnLRProxy

COLL = "semops_quickstart"
PREDICATE = "this is a negative movie review"
DATA = os.path.join(os.path.dirname(__file__), "data", "reviews_labeled.tsv")


def load_labelled():
    texts, labels = [], []
    with open(DATA) as f:
        next(f)  # header
        for line in f:
            lab, text = line.rstrip("\n").split("\t", 1)
            texts.append(text)
            labels.append(lab == "1")  # 1 = negative review
    return texts, labels


class Oracle:
    """Judges the predicate. Embeddings are always local (fastembed); the *verdict*
    comes from a real LLM when a key is present, otherwise from the stored label.

    The LLM is the whole point of a semantic operator — it reads the text and
    decides. The label path is a zero-friction stand-in so the quickstart runs
    with no key at all; for this predicate the label is exactly what a good LLM
    would say, so it is a faithful stand-in, but it is a stand-in.
    """

    def __init__(self, emb, llm=None, by_text=None):
        self._e, self._llm, self._by_text = emb, llm, by_text or {}
        self.embed_model = emb.embed_model
        self.chat_model = llm.chat_model if llm else "label-oracle"
        self.spent_usd, self.calls = 0.0, 0

    def embed(self, t):
        return self._e.embed(t)

    def judge(self, predicate, text):
        self.calls += 1
        if self._llm is not None:
            return self._llm.judge(predicate, text)
        return bool(self._by_text.get(text, False))


def make_oracle(emb, by_text):
    """Real LLM if a key is in the environment, else the stored-label stand-in.

    Only a chat model is needed (embeddings are local), so any of these works:
      GEMINI_API_KEY / GOOGLE_API_KEY   -> Gemini
      OPENAI_API_KEY                    -> OpenAI
    """
    gkey = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    okey = os.environ.get("OPENAI_API_KEY")
    if gkey:
        llm = OpenAICompatClient(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            api_key=gkey, chat_model=os.environ.get("SEMOPS_CHAT_MODEL", "gemini-flash-latest"))
        print(f"  oracle: {llm.chat_model} (real LLM judging each escalated row)")
        return Oracle(emb, llm=llm)
    if okey:
        llm = OpenAICompatClient(
            base_url="https://api.openai.com/v1",
            api_key=okey, chat_model=os.environ.get("SEMOPS_CHAT_MODEL", "gpt-4o-mini"))
        print(f"  oracle: {llm.chat_model} (real LLM judging each escalated row)")
        return Oracle(emb, llm=llm)
    print("  oracle: stored label (no API key set)."
          "  export GEMINI_API_KEY or OPENAI_API_KEY for the real LLM.")
    return Oracle(emb, by_text=by_text)


def prf(truth, pred):
    tp = sum(1 for y, p in zip(truth, pred) if y and p)
    fp = sum(1 for y, p in zip(truth, pred) if p and not y)
    fn = sum(1 for y, p in zip(truth, pred) if y and not p)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def ks():
    return f"`{BUCKET}`.`{SCOPE}`.`{COLL}`"


def drop(q):
    for idx in (f"idx_{COLL}_vec", f"idx_{COLL}_dot"):
        try:
            q.query(f"DROP INDEX {idx} ON {ks()}")
        except Exception:
            pass
    try:
        q.query(f"DROP COLLECTION {ks()}")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanup", action="store_true", help="drop the quickstart collection and exit")
    args = ap.parse_args()

    q = HttpQueryCluster(QUERY, USER, PW)
    print(f"cluster: query={QUERY}  rest={REST}  bucket={BUCKET}")

    if args.cleanup:
        drop(q)
        print(f"dropped {COLL} and its indexes.")
        return

    # 1. collection (create via REST; ignore 'already exists')
    print(f"1/5  create collection {BUCKET}._default.{COLL}")
    rest("POST", f"/pools/default/buckets/{BUCKET}/scopes/{SCOPE}/collections", {"name": COLL})
    time.sleep(2)

    # 2. embed the bundled dataset locally (no API key)
    texts, labels = load_labelled()
    print(f"2/5  embed {len(texts)} bundled reviews (fastembed, local)")
    emb = FastEmbedEmbedder()
    vecs = emb.embed(texts)
    dim = len(vecs[0])

    # 3. upsert
    print(f"3/5  upsert into {COLL}")
    vals = ",".join(
        f'("r{i}", {json.dumps({"text": texts[i], "label": labels[i], "embedding": vecs[i]})})'
        for i in range(len(texts)))
    q.query(f"UPSERT INTO {ks()} (KEY, VALUE) VALUES {vals}")

    # 4. indexes: primary + cosine (blocking) + dot (index-served learned proxy)
    print("4/5  build primary + vector indexes (cosine + dot)")
    q.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON {ks()}")
    for name, sim in ((f"idx_{COLL}_vec", "cosine"), (f"idx_{COLL}_dot", "dot")):
        ddl = (f"CREATE VECTOR INDEX {name} ON {ks()}(embedding VECTOR) "
               f"INCLUDE (`text`, `label`) "
               f"WITH {{'dimension':{dim},'similarity':'{sim}','description':'IVF,SQ8',"
               f"'scan_nprobes':8,'train_list':200}}")
        try:
            q.query(ddl)
        except Exception as e:
            print(f"     ({name}: {str(e)[:100]})")
    for _ in range(60):
        rows = q.query(f"SELECT state FROM system:indexes WHERE keyspace_id='{COLL}'")
        if rows and all(r.get("state") == "online" for r in rows):
            break
        time.sleep(2)

    # 5. run sem_filter through the cascade
    print("5/5  sem_filter: \"" + PREDICATE + "\"")
    eng = CouchbaseEngine("", USER, PW, BUCKET, SCOPE, cluster=q, nprobes=8)
    model = make_oracle(emb, dict(zip(texts, labels)))
    real_llm = model.chat_model != "label-oracle"
    print()
    sess = semops.connect(engine=eng, model=model, workers=8)
    res = sess.scan(COLL).sem_filter(PREDICATE, proxy_model=SklearnLRProxy(),
                                     recall=0.9, precision=0.9)

    kept = {r.id for r in res.rows_}
    truth = [labels[i] for i in range(len(texts))]
    pred = [f"r{i}" in kept for i in range(len(texts))]
    p, r, f = prf(truth, pred)
    st = res.stats()
    n = st["n_rows"]
    savings = n / max(st["llm_calls"], 1)
    # F1 is always measured against the human labels. With the label oracle that
    # measures how faithfully the cascade reproduced the oracle; with a real LLM it
    # measures how well the LLM+cascade agrees with the humans (the LLM may differ).
    vs = "how well the LLM agrees with the human labels" if real_llm \
        else "the cascade reproduced the oracle's verdicts"
    print(f"  kept {len(kept)} of {n} rows as negative reviews.")
    print(f"  quality   P={p:.3f}  R={r:.3f}  F1={f:.3f}   — {vs}")
    print(f"  cost      {st['llm_calls']} oracle calls instead of {n}  ({savings:.2f}x fewer)")
    print(f"  bands     accept={st['n_accept']}  escalate={st['n_escalate']}  reject={st['n_reject']}")
    print("\n  a few rows it kept:")
    for rid in list(kept)[:3]:
        print(f"    - {texts[int(rid[1:])][:70]}")

    print("\n  Note: polarity is the low-savings case — about half the corpus is the")
    print("  answer, so only the reject band is free. Selective predicates reach")
    print("  5-6x; see the benchmarks in the README.")
    print("\ndone. drop the demo data with:  python examples/quickstart.py --cleanup")


if __name__ == "__main__":
    main()

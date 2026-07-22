"""Couchbase native-mode demo (requires a cluster + a bhive/FTS vector index
and an OpenAI-compatible or Capella AI endpoint).

    pip install couchbase
    export SEMOPS_LLM_KEY=...       # OpenAI-compatible key (or use Capella AI)
    python3 examples/demo_couchbase.py

Prereqs in the cluster:
  - a collection `bench._default.reviews` with a `text` field and an `embedding`
    field (dense vector, dims matching your embed model)
  - a vector index, e.g.:
      CREATE VECTOR INDEX idx_reviews_vec
        ON `bench`._default.reviews(embedding VECTOR)
        WITH {"dimension":1536,"similarity":"cosine","description":"IVF,SQ8","train_list":10000};

The ONLY difference from demo_offline.py is the engine + model constructors —
the operator code, cascade, and API are identical. That is the "Couchbase is a
mode, not a rewrite" guarantee.
"""
import os
import sys


import semops
from semops import CouchbaseEngine, OpenAICompatClient
# from semops import CapellaAIClient  # swap in to keep embeddings+LLM on Capella


def main():
    engine = CouchbaseEngine(
        connstr=os.environ.get("SEMOPS_CB_CONNSTR", "couchbases://localhost"),
        username=os.environ.get("SEMOPS_CB_USER", "Administrator"),
        password=os.environ.get("SEMOPS_CB_PASS", "password"),
        bucket="bench", scope="_default",
        vector_field="embedding", text_field="text", metric="cosine",
    )
    model = OpenAICompatClient(
        base_url=os.environ.get("SEMOPS_LLM_URL", "https://api.openai.com/v1"),
        api_key=os.environ.get("SEMOPS_LLM_KEY", ""),
        chat_model="gpt-4o-mini",
        embed_model="text-embedding-3-small",
    )
    # keep embeddings + LLM inside Capella instead:
    # model = CapellaAIClient(base_url=os.environ["SEMOPS_CAPELLA_URL"],
    #                         api_key=os.environ["SEMOPS_CAPELLA_KEY"])

    sess = semops.connect(engine=engine, model=model, budget_usd=5.0)

    # cluster-then-filter: ANN retrieves candidates in-engine (bhive), the
    # cascade runs in the service on that bounded set.
    pipe = (sess.search("reviews", "battery and charging complaints", k=500)
                .sem_filter("is a genuine complaint about battery life", recall=0.9, precision=0.9))

    print("kept:", pipe.count(), "docs")
    print("cascade stats:", pipe.stats())
    for doc in pipe.collect()[:5]:
        print(" -", doc.get("text", doc))


if __name__ == "__main__":
    main()

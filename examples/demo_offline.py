"""Offline demo — runs with zero deps, zero keys, zero cluster.

    python3 examples/demo_offline.py

Shows the sem_filter cascade: the embedding proxy accepts/rejects the easy
majority, the oracle is spent only on the uncertain middle, and the result
matches a full-oracle pass at a fraction of the calls.
"""
import os
import random
import sys


import semops
from semops import FakeModelClient, InMemoryEngine, Row
from semops.cascade import empirical_precision_recall

PREDICATE = "battery life"
FILLER = ["device", "product", "review", "quality", "design", "value"]


def dataset(n, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        roll, f = rng.random(), " ".join(rng.sample(FILLER, 3))
        if roll < 0.35:
            out.append(f"the battery life is great and lasts all day {f}")
        elif roll < 0.80:
            out.append(f"the screen is bright and the camera is sharp {f}")
        elif roll < 0.90:
            out.append(f"the battery drains quickly while gaming {f}")
        else:
            out.append(f"long product life and a sturdy build {f}")
    return out


def main():
    model = FakeModelClient(dims=64)
    texts = dataset(2000)
    vecs = model.embed(texts)
    rows = [Row(id=str(i), text=t, embedding=v, doc={"id": str(i), "text": t})
            for i, (t, v) in enumerate(zip(texts, vecs))]
    truth = [model.judge(PREDICATE, r.text) for r in rows]

    eng = InMemoryEngine()
    eng.add("reviews", rows)
    sess = semops.connect(engine=eng, model=model, budget_usd=100.0)

    pipe = sess.scan("reviews").sem_filter(PREDICATE, recall=0.9, precision=0.9)
    st = pipe.stats()
    kept = {r["id"] for r in pipe.collect()}
    keep_pred = [r.id in kept for r in rows]
    precision, recall = empirical_precision_recall([(0.0, y) for y in truth], keep_pred)

    print(f"\n  predicate     : {PREDICATE!r}")
    print(f"  rows          : {st['n_rows']}")
    print(f"  oracle sample : {st['n_sample'] or '(min_sample)'}")
    print(f"  thresholds    : reject <= {st['tau_minus']:.3f}"
          f"  |  escalate  |  {st['tau_plus']:.3f} <= accept   collapsed={st['proxy_collapsed']}")
    print(f"  bands         : accept={st['n_accept']}  escalate={st['n_escalate']}  reject={st['n_reject']}")
    print(f"  LLM calls     : {st['llm_calls']}   (naive per-row would be {st['n_rows']})")
    print(f"  savings       : {st['savings_ratio']}x fewer oracle calls")
    print(f"  quality       : precision={precision:.3f}  recall={recall:.3f}  (vs full oracle)")
    print(f"\n  -> the vector proxy decided {st['n_accept'] + st['n_reject']} rows for free; "
          f"the LLM only judged the {st['n_escalate']} uncertain ones.\n")


if __name__ == "__main__":
    main()

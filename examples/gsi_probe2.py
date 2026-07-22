"""Part 2: can we stop shipping vectors, and stop paying per-row round trips?

  H4  INCLUDE columns -> covering ANN scan. Today ann_candidates() does
      `SELECT ..., d AS _doc`, which drags the full 384-d embedding back for every
      candidate. If the index INCLUDEs the fields we need, the scan should be
      covered and the vectors stay in the cluster.
  H5  sem_join issues one ANN query per LEFT row -> n_left round trips. Can a
      batch of probes share a single query?

  ./.venv/bin/python examples/gsi_probe2.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semops import HttpQueryCluster
from cb_common import BUCKET, PW, QUERY, SCOPE, USER, FastEmbedEmbedder

KS = f"`{BUCKET}`.`{SCOPE}`.`reviews`"
PREDICATE = "this is a negative or critical movie review"


def timed(fn, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.time()
        out = fn()
        ts.append((time.time() - t0) * 1000)
    return out, sorted(ts)[len(ts) // 2]


def main():
    q = HttpQueryCluster(QUERY, USER, PW)
    emb = FastEmbedEmbedder()
    pv = list(emb.embed([PREDICATE])[0])
    K = 100
    D = (f"APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, 'cosine', 8, true, {K})")

    print("=" * 74)
    print("H4  how much data does an ANN candidate fetch cost today?")
    print("=" * 74)

    # what the engine does today: pulls the whole doc, embedding included
    def today():
        return q.query(f"SELECT META(d).id AS _id, d.`text` AS _text, {D} AS _dist, d AS _doc "
                       f"FROM {KS} d ORDER BY {D} LIMIT $k", qvec=pv, k=K)
    rows_today, ms_today = timed(today)
    bytes_today = len(json.dumps(rows_today))

    # what it could do: project only what the cascade needs (no vector)
    def lean():
        return q.query(f"SELECT META(d).id AS _id, d.`text` AS _text, {D} AS _dist "
                       f"FROM {KS} d ORDER BY {D} LIMIT $k", qvec=pv, k=K)
    rows_lean, ms_lean = timed(lean)
    bytes_lean = len(json.dumps(rows_lean))

    print(f"  SELECT d AS _doc (today) : {bytes_today/1024:8.1f} KB  {ms_today:6.1f}ms  ({len(rows_today)} rows)")
    print(f"  project text+dist only   : {bytes_lean/1024:8.1f} KB  {ms_lean:6.1f}ms  ({len(rows_lean)} rows)")
    print(f"  -> {bytes_today/max(bytes_lean,1):.1f}x less data, {ms_today/max(ms_lean,0.01):.1f}x faster per ANN call")

    # is the lean scan actually covered by the index, or still a KV fetch?
    plan = q.query(f"EXPLAIN SELECT META(d).id AS _id, {D} AS _dist FROM {KS} d "
                   f"ORDER BY {D} LIMIT {K}", qvec=pv)
    ptxt = json.dumps(plan)
    print(f"  plan contains Fetch operator: {'#operator\":\"Fetch' in ptxt}")
    print(f"  index used: {[s for s in ptxt.split('\"') if s.startswith('idx_')][:2]}")

    # can INCLUDE cover it? (bhive supports INCLUDE columns)
    print("\n  trying a bhive index with INCLUDE (text, label)...")
    try:
        q.query(f"CREATE VECTOR INDEX idx_reviews_cov ON {KS}(embedding VECTOR) "
                f"INCLUDE (`text`, `label`) "
                f"WITH {{'dimension':384,'similarity':'cosine','description':'IVF,SQ8',"
                f"'train_list':2000}}")
        print("   INCLUDE accepted -> covering ANN scans are available")
    except Exception as e:
        print(f"   INCLUDE rejected: {str(e)[:200]}")

    # =================== H5: batching probes =============================
    print()
    print("=" * 74)
    print("H5  one ANN query per left row, vs batching probes into one statement")
    print("=" * 74)
    qs = [list(v) for v in emb.embed([
        "a hilarious comedy", "a boring plot", "stunning visuals",
        "terrible acting", "a moving story", "predictable and dull",
        "brilliant direction", "a waste of time"])]
    k = 20

    def per_row():
        out = []
        for v in qs:
            d = f"APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, 'cosine', 8, true, {k})"
            out.append(q.query(f"SELECT META(d).id AS id FROM {KS} d ORDER BY {d} LIMIT {k}",
                               qvec=v))
        return out
    _, ms_seq = timed(per_row, reps=3)

    def batched():
        parts = []
        for i, v in enumerate(qs):
            d = f"APPROX_VECTOR_DISTANCE(d.`embedding`, $q{i}, 'cosine', 8, true, {k})"
            parts.append(f"(SELECT {i} AS probe, META(d).id AS id FROM {KS} d "
                         f"ORDER BY {d} LIMIT {k})")
        stmt = " UNION ALL ".join(parts)
        return q.query(stmt, **{f"q{i}": v for i, v in enumerate(qs)})
    try:
        rows_b, ms_bat = timed(batched, reps=3)
        got = len({r["probe"] for r in rows_b})
        print(f"  {len(qs)} probes sequentially : {ms_seq:7.1f}ms")
        print(f"  {len(qs)} probes UNION ALL    : {ms_bat:7.1f}ms  "
              f"({len(rows_b)} rows, {got} distinct probes)")
        print(f"  -> {ms_seq/max(ms_bat,0.01):.2f}x")
    except Exception as e:
        print(f"  UNION ALL batching failed: {str(e)[:250]}")


if __name__ == "__main__":
    main()

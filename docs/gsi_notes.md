# GSI vector index: working notes

Everything below was measured on a `cluster_run` build against a Hyperscale Vector
index (`IVF,SQ8`), over 384-dimensional BGE vectors. The README has the summary;
this file has the reproduction and the parts that did not fit.

Reproduce with:

```bash
./.venv/bin/python examples/gsi_probe.py        # topNScan, nProbes, proxy-as-MIPS
./.venv/bin/python examples/gsi_probe2.py       # covering scans, UNION ALL batching
./.venv/bin/python examples/verify_gsi_paths.py # all of it, end to end
```

## 1. `topNScan`, not `LIMIT`, controls ANN scan depth

At `LIMIT 200`:

| `topNScan` | rows returned | recall vs exact top-200 |
|---|---|---|
| 10 | 20 | 0.10 |
| 50 | 100 | 0.50 |
| 200 | 200 | 1.00 |

`LIMIT` alone does not raise it. Neither does `nProbes` or `reRank`. The failure
is silent: you get rows, they are just the wrong number of them, and blocking
recall quietly caps. `CouchbaseEngine` always passes `topNScan >= k`.

Note the returned-row count is roughly 2x `topNScan` until `LIMIT` binds, which is
worth knowing if you are sizing requests.

The docs note `topNScan` applies only to Hyperscale Vector indexes.

## 2. `nProbes` defaults to 1

On 2k docs, `nProbes=1` gave recall 0.685 against the exact top-200; anything
`>= 2` gave 1.000. On 50k the picture is different and much less forgiving: see
section 5.

The saturation point is a function of centroid count, so it moves with collection
size. Do not carry a tuned value across collections.

## 3. `INCLUDE` columns make the candidate scan covered

The cascade reads `text` and a label per candidate, nothing else.

| index | plan |
|---|---|
| plain vector index, selecting `text` | `Fetch` + `IndexScan3` |
| `INCLUDE (text, label)` index | `IndexScan3` only |

Worth 4.0x latency and 51x data on a 100-row probe: 43.9ms and 862 KB down to
10.9ms and 16.9 KB. The old path was dragging the full 384-d embedding back for
every candidate it was about to discard.

## 4. Probes batch into one round trip

```sql
(SELECT 0 AS _pid, ... ORDER BY APPROX_VECTOR_DISTANCE(..., $q0, ...) LIMIT 20)
UNION ALL
(SELECT 1 AS _pid, ... ORDER BY APPROX_VECTOR_DISTANCE(..., $q1, ...) LIMIT 20)
```

2.4x to 2.6x at 8 to 32 probes, same rows in the same order as issuing them one at
a time. `probe` is a reserved word in N1QL, hence `_pid`.

## 5. Serving the learned proxy from the index (`ann_above`)

A logistic-regression proxy scores `sigmoid(w·x + b)`, monotonic in `w·x`, so the
proxy's top-k *is* the maximum-inner-product top-k for query vector `w`.

**Three constraints, all of which we got wrong on the first attempt.**

**(a) The index must be built with `similarity` matching the metric you query
with.** Couchbase only selects a vector index when they match. Query `'dot'`
against a cosine-built index and the planner falls back to `PrimaryScan3` +
`Fetch`: a brute-force scan of the whole collection. Results stay *correct*, which
is why this is easy to miss. The only symptom is that the "index-served" path is
slower than scanning. Building a `similarity: 'dot'` index took one threshold-0
query from 953ms to 62ms.

**(b) When an index serves the query, the distance returned is L2-squared over the
stored vectors, whatever metric string you passed.** Only the brute-force path
returns the true metric, which is exactly what made (a) so easy to miss. For
unit-norm vectors the score is recoverable:

```
dist = |w|² + |x|² − 2(w·x)    ⇒    w·x = (|w|² + 1 − dist) / 2
```

Matches exact `w·x` to four decimals. `vectors_normalised()` checks the
precondition; the recovery is invalid without it.

**(c) A vector index scan caps `LIMIT + OFFSET` at 8192.** Past that the request
fails with error 5433, "Total heap size for (Limit + Offset) exceeded maximum heap
size allowed for vector index". The setting is
`indexer.scan.vector.max_heap_size` and it is mutable, but the default is what you
get. `ann_above` caps its doubling there and reports `exhaustive=False`.

Measured on 50k rows once all three are handled:

| rows above tau | nProbes | returned | recall | time |
|---|---|---|---|---|
| 3,150 | 8 | 2,246 | 0.713 | 202ms |
| 3,150 | 64 | 3,151 | 1.000 | 358ms |
| 9,008 | 64 | 8,192 | 0.909 | 710ms |
| 14,689 | 64 | 8,192 | 0.558 | 673ms |

So the path is sound inside a window: the qualifying set has to fit under 8192 and
`nProbes` has to be tuned to the collection. Inside it, 358ms against 9.1s for the
same filter by scanning, roughly 25x. Outside it, recall degrades.

**The `exhaustive` flag is not a completeness proof under ANN.** `ann_above`
doubles `k` until the worst returned score falls below tau. That is a valid
termination proof for *exact* search. Under approximate search it can be satisfied
while rows above tau were never visited: the `nProbes=8` row above reported
`exhaustive=True` at recall 0.713. Tune `nProbes` against a known-good exact result
before trusting it.

## 6. `USE KEYS` is a KV bulk get

Server-side proxy scoring uses `USE KEYS`. Sending 25k keys in one request times
the KV client out (error 12008, "Error performing bulk get operation"). Chunked at
2,000.

## Supported and documented

Everything used here is in the public docs: the six-argument
`APPROX_VECTOR_DISTANCE` (`vec, queryvec, metric, nProbes, reRank, topNScan`),
`VECTOR_DISTANCE`, `DOT` as a distance metric, and `INCLUDE` columns on vector
indexes.

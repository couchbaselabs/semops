# semops: semantic operators over vector-indexed data

**Semantic operators** are `WHERE` / `JOIN` / `GROUP BY` whose condition is written in
natural language and judged by an LLM. They are the frontier of query, and done the obvious way they are unusable: one LLM call per row, or per pair for a join, so a 10,000 × 10,000 join is 100 million calls.

This project uses vector indexes as the primitive that makes semantic operators affordable. The index does the bulk of the work, narrowing a whole collection down to the few rows the LLM must actually judge, and
the LLM is spent only where the index cannot decide on its own.  

| operator | the GSI capability it runs on |
|---|---|
| `sem_filter` | `APPROX_VECTOR_DISTANCE`, plus the `DOT` metric so the index can serve a *learned* proxy, not just similarity |
| `sem_join` | `APPROX_VECTOR_DISTANCE` blocking, with `topNScan` for recall control and `INCLUDE` for covered scans |
| `sem_dedup` | the same, run as a self-join |
| `sem_group_by` | the geometry of the stored vectors (k-means, no LLM at all) |

The discussion on [Why not only vector search](#why-not-only-vector-search): the index is the retrieval engine, it is not the decision procedure. Getting that division of labour right is the crux of the design.

---

## Quickstart

Run `sem_filter` on a Couchbase collection in one command. The default run needs
no API key.

### 1. A Couchbase cluster

You need a running cluster with vector-index support and a bucket to write into
(`default` is fine). 


### 2. Install

The package uses a `src/` layout, so install it (editable) before importing:

```bash
pip install -e .                            # the semops library, zero core deps
pip install -r examples/requirements.txt    # sklearn / fastembed / datasets, for the examples
```

### 3. Run

`examples/quickstart.py` creates a collection, loads 1,000 labelled movie reviews
with local embeddings, builds the vector indexes, and runs `sem_filter` for
*"this is a negative movie review"*.  

| variable | default (`cluster_run`) | stock Couchbase Server |
|---|---|---|
| `CB_QUERY_URL` | `http://localhost:9499` | `http://localhost:8093` |
| `CB_REST_URL` | `http://localhost:9000` | `http://localhost:8091` |
| `CB_USER` / `CB_PASSWORD` | `Administrator` / `asdasd` | your credentials |
| `CB_BUCKET` | `default` | your bucket |

```bash
# cluster_run (nothing to set):
python examples/quickstart.py

 
CB_QUERY_URL=http://localhost:8093 CB_REST_URL=http://localhost:8091 \
CB_PASSWORD=yourpassword python examples/quickstart.py
```



```
oracle: stored label (no API key set). export GEMINI_API_KEY or OPENAI_API_KEY for the real LLM.
kept 471 of 1000 rows as negative reviews.
quality   P=1.000  R=0.942  F1=0.970
cost      692 oracle calls instead of 1000  (1.45x fewer)
```

### 4. Use an LLM

By default the oracle is the stored label. This needs no key and is the same every
run, but it does not call an LLM: for this predicate the label is already the
answer. To have a real model read each row and decide, set a key. Embeddings are
still computed locally, so you only need a chat model:

```bash
GEMINI_API_KEY=...  python examples/quickstart.py    # or OPENAI_API_KEY
```

Clean up the demo data with `python examples/quickstart.py --cleanup`.

No cluster? `python examples/tour.py` runs every operator offline, with no cluster
and no keys, on the ticket data used throughout this README.

---

## Semantic operators

We all already know relational operators: `WHERE`, `JOIN`, `GROUP BY`. They need a
condition you can *compute*, like `price > 100`, `orders.user_id = users.id`.

A **semantic operator** is the same operator with the condition written in Natural
Language (plain English) and judged by a language model:

| relational | semantic |
|---|---|
| `WHERE price > 100` | `WHERE "the customer is threatening to cancel"` |
| `JOIN ON a.id = b.id` | `JOIN ON "this ticket is caused by this known issue [A]"` |
| `GROUP BY category` | `GROUP BY "the product area the ticket is about"` |

The idea and the naming come from **LOTUS** (Patel et al., *Semantic Operators*,
VLDB 2025), which formalised `sem_filter` / `sem_join` / `sem_topk` / `sem_agg` and
showed they can be *optimised* like relational operators. Related work is largely from:
**DocETL** (agentic rewrites for document pipelines), **Palimpzest/Abacus**
(cost-based optimisation for these operators), and **Trummer's semantic-join paper**
(arXiv 2510.08489), whose batching trick is borrowed directly.

**Why they need optimising:** The obvious implementation, calling the LLM once per
row (or once per *pair*, for a join), is ruinous. Naive
semantic operators are practically unusable because of the exorbitant LLM cost. 

---

## Why not only vector search

Vector search answers *"what is this text like?"*

A semantic operator answers *"is this statement true of this text?"*

Those are different questions, and similarity is only good at the first one.

**An example.** Ask for negative movie reviews, over 2,000 Rotten Tomatoes reviews of
which 49.3% are negative:

| | precision | recall | F1 |
|---|---|---|---|
| rank by cosine to the predicate, best cut | 0.499 | 0.978 | 0.661 |
| semops | 1.000 | 0.975 | 0.987 |

The corpus is 49.3% negative and the ranking gives 49.9% precision, no better than
picking rows at random. We also chose that cut-off after seeing the answers, which no
real system can do.

A glowing review and a scathing one are about the same film, the same actors, the same
plot. They are alike, and embeddings put alike things together.  

**Retrieval isn't the gap/where careful consideration is required. Deciding is.** In our contradiction benchmark,
embedding search pulled the contradicting email into the top 10 of 100 candidates every
time, so retrieval was perfect. Judging those same candidates by embedding similarity
scored 0.10 precision. It found the right rows and could not tell which ones matched.

Embeddings narrow the field well. They do not give a verdict. with semantic operators, we keeps the first
and replaces the second: the vector index picks who to ask about, the LLM answers.

 
---

## The cascade

Every operator here mostly follows the same general architecture. A **cheap proxy** (vector similarity) sorts
rows into three bands; the **expensive oracle** (the LLM) only sees the ambiguous ones.

```
                    proxy score  s(row)   (from the vector index, ~free)
   ────────────────────┬────────────────────┬───────────────────────
    s ≥ τ⁺             │  τ⁻ < s < τ⁺       │  s ≤ τ⁻
    ACCEPT (true)      │  ESCALATE → LLM    │  REJECT (false)
    no LLM call        │  the only $ spent  │  no LLM call
```

`τ⁻` and `τ⁺` are **learned per query**: label a small sample with the LLM, then set
thresholds so the accept band's precision and the kept set's recall meet your targets,
using Wilson score bounds at confidence 1−δ. If the proxy is too weak to certify a
target, that band **vanishes** and those rows escalate, and the operator degrades to
"just ask the LLM" rather than guessing. **It never invents a guarantee it can't back.**

Two things follow:

- **The LLM defines correctness; the vector index only decides who to ask.** Accuracy
  comes from the oracle, so quality tracks the LLM, not the embedding.
- **Cost scales with how badly your embeddings fit the predicate.** Easy predicate →
  proxy decides most rows → big savings. Hard predicate → nearly everything escalates
  → you pay, but you still get the right answer.

---

## Example data

One dataset, used by every operator below. Two collections of customer support data:

**`tickets`**
```
t1  "Payment failed three times but my card was charged anyway.
     Refund me or I'm cancelling my account."
t2  "Charged twice for the same order, please refund the duplicate."
t3  "The export button does nothing on Safari."
t4  "Can't export my report in Safari, the button is dead."
t5  "Love the new dashboard, just wanted to say thanks!"
```

**`known_issues`** (engineering's list)
```
k1  "Duplicate charge when a payment retries after a gateway timeout."
k2  "Export fails silently on Safari 17 due to a popup-blocker change."
```

Setup is the same regardless of engine:

```python
import semops
from semops import InMemoryEngine, Row, OpenAICompatClient

model  = OpenAICompatClient(base_url="https://api.openai.com/v1", api_key="…")
engine = InMemoryEngine()
engine.add("tickets", [Row(id=t["id"], text=t["text"],
                           embedding=model.embed([t["text"]])[0]) for t in tickets])

sess = semops.connect(engine=engine, model=model, budget_usd=5.0, workers=8)
```

---

## Operators

### `search`: vector top-k

Not a semantic operator; it's the ANN lookup that *feeds* them. Cheap, no LLM.

```python
sess.search("tickets", "problems with refunds", k=100)   # -> Pipeline
```
→ `t1, t2` (and, being similarity, probably some near-misses too).

On Couchbase this pushes down to `ORDER BY APPROX_VECTOR_DISTANCE(...) LIMIT k`
against a Hyperscale Vector index (bhive). Use it to bound the candidate set before spending LLM.

> **Not implemented:** LLM-based top-k re-ranking (`sem_topk` in LOTUS). `search`
> gives you similarity order only.

### `sem_filter`: WHERE

```python
angry = (sess.scan("tickets")
             .sem_filter("the customer is threatening to cancel", recall=0.9, precision=0.9)
             .collect())
```
→ **`t1`**. Note `t2` is also a refund complaint but isn't threatening to cancel. A
similarity threshold cannot make that distinction; the LLM can.

### `sem_join`: JOIN

```python
matches = (sess.scan("tickets")
               .sem_join("known_issues",
                         "the ticket is caused by this known issue",
                         block_k=50, block_adjudicate=True))
matches.id_pairs()
```
→ `(t1,k1) (t2,k1) (t3,k2) (t4,k2)`. `t5` matches nothing.

### `sem_dedup`: collapse duplicates

```python
res = sess.sem_dedup("tickets", "the two tickets report the same underlying problem",
                     block_k=50, block_adjudicate=True)
res.duplicate_groups()   # -> [[t3, t4]]
res.canonical()          # one representative per distinct problem
```
→ `t3` and `t4` are the same Safari bug in different words. `t1`/`t2` are both billing
but *different* problems, so they stay separate.

### `sem_group_by`: cluster into themes

```python
g = sess.sem_group_by("tickets", k=3, name_clusters=True)
for grp in g.groups:
    print(grp.label, [r.id for r in grp.rows])
```
→ `billing/refunds [t1,t2]`, `export bug [t3,t4]`, `praise [t5]`.

**Zero LLM calls** by default: pure k-means over the embeddings. `name_clusters=True`
adds one call *per cluster* to name it.

### Composing operators

The pipeline is the point. Narrow the search space with cheap operators before spending
on expensive ones:

```python
(sess.search("tickets", "billing problems", k=500)      # vector: 500 candidates
     .sem_filter("the customer is threatening to cancel") # LLM: only the uncertain ones
     .sem_join("known_issues", "the ticket is caused by this known issue",
               block_adjudicate=True))
```

---

## How each operator works

Full internals, including
calibration maths, failure modes and parameters, are in `docs/`:
[`sem_filter.md`](docs/sem_filter.md), [`sem_join.md`](docs/sem_join.md),
[`sem_dedup_and_group_by.md`](docs/sem_dedup_and_group_by.md).

### `sem_filter`

```
rows ──► sample K rows ──► oracle labels them        (K LLM calls)
     │                          │
     │                     calibrate τ⁻ / τ⁺          (Wilson bounds)
     ▼                          ▼
   proxy scores ──────────► band every row ──► accept | escalate→LLM | reject
```

Two proxies. Cosine-to-predicate is the default: zero setup, and weak. A learned
one (`proxy_model=`) fits a logistic regression on the sampled
`(embedding → label)` pairs and is what the literature's big wins use, because it
learns which directions in embedding space predict the label rather than assuming
proximity to a phrase.

A logistic-regression proxy is **linear**, so its score is a dot product with a
weight vector, which is exactly what a vector index computes. Train on the sample
in-process, ship the weight vector down, score everything with
`VECTOR_DISTANCE(embedding, $w, 'dot')`. Only scalars come back: **50,000 vectors
over the wire becomes 2,500**, identical F1 and identical oracle calls. The saving
is `1 − sample_frac` at any collection size, because the only embeddings that
leave are the calibration sample's.



### `sem_join`

```
BLOCKING (vector index)      each left row ──ANN──► its top-block_k right rows
   prunes n×m ──► candidate pairs                   (batched; native on Couchbase)
        │
CASCADE (per pair)           proxy = pair similarity → accept / reject free
        │                                            → uncertain middle escalates
        ▼
BLOCK-JOIN ADJUDICATION      group the escalate band BY LEFT ROW; one prompt per
   (Trummer, arXiv 2510.08489)  left listing all its candidates:
        "QUERY: <left>.  CANDIDATES: 1..n.  Which match? … then 'Finished'"
        └─ output truncated (no "Finished")? → overflow → shrink batch by α, retry
```

Batching makes adjudication **O(n_left)**, not O(candidate_pairs). Quadrupling
`block_k` quadruples the candidate pairs and leaves the LLM call count flat: we
measured `block_k` 39→160 lifting recall 0.56→0.94 with `block_calls` pinned at
40. Wide blocking is what makes the join accurate, and batching is what makes wide
blocking affordable.

**`block_k` is the biggest correctness knob here.** Recall lost in blocking is
unrecoverable: the pair never forms, the LLM is never asked, and nothing says so.
The right value is task-dependent and the spread is over an order of magnitude.

| join shape | example | `block_k` for good recall |
|---|---|---|
| symmetric, doc ↔ doc | 20 Newsgroups same-topic | **2 to 4x fan-out** |
| asymmetric, long doc ↔ short label | BioDEX paper ↔ reaction term | blocking barely prunes; **use the whole right side** |

Measure blocking recall separately before trusting a join.

### `sem_dedup` and `sem_group_by`

```
sem_dedup:  rows ──► sem_join(rows, rows, "same entity", self_join=True)
                     matched pairs ──► union-find ──► entity clusters

sem_group_by: rows ──► embeddings ──► k-means ──► k groups      (no LLM at all)
```

`sem_dedup` is a join, so everything above applies. The one behaviour that is interesting: connected components take the **transitive closure**, so if A~B and
B~C then A, B and C merge even if A~C was never confirmed. Prefer precision over
recall when setting targets for a dedup.

`sem_group_by` spends **zero LLM calls** by default; `name_clusters=True` adds one
per cluster to name it. `k` is yours to choose.

---


## How GSI vector indexes were leveraged

This is the heart of the project. Almost all of the heavy lifting is done by the Couchbase
**Hyperscale Vector index** (`IVF,SQ8`) and `APPROX_VECTOR_DISTANCE`;
 
### 1. Blocking is a one-line ANN pushdown

`sem_join` needs the top-`block_k` right rows for each left row. Because
`APPROX_VECTOR_DISTANCE` is a first-class N1QL function, that entire step is a
single statement the query engine runs in the cluster; the service never sees the
collection.

```sql
SELECT META(d).id, d.text,
       APPROX_VECTOR_DISTANCE(d.embedding, $qvec, 'cosine', 8, true, 200) AS dist
FROM   bucket.scope.collection d
ORDER  BY APPROX_VECTOR_DISTANCE(d.embedding, $qvec, 'cosine', 8, true, 200)
LIMIT  $k
```

`EXPLAIN` confirms the vector index is picked and the plan uses `IndexScan3`. 

### 2. `topNScan` gives direct control of the recall/cost trade

The sixth argument to `APPROX_VECTOR_DISTANCE` exposes scan depth to the caller,
which is exactly the knob a blocking stage needs. At `LIMIT 200`:

| `topNScan` | rows returned | recall vs exact top-200 |
|---|---|---|
| 10 | 20 | 0.10 |
| 50 | 100 | 0.50 |
| 200 | 200 | 1.00 |


`nProbes` is the second recall dial and defaults to 1. On 2k docs that default
gave recall 0.685 against exact top-200; anything `>= 2` gave 1.000. The
saturation point tracks centroid count, so it moves with collection size.

### 3. `INCLUDE` columns turn the candidate scan into a covering scan

The cascade reads `text` and a label per candidate, nothing else, and the index
can carry exactly those. Adding `INCLUDE (text, label)` drops the `Fetch` from the
plan entirely:

| | plan operators |
|---|---|
| plain vector index, selecting `text` | `Fetch` + `IndexScan3` |
| `INCLUDE (text, label)` index | `IndexScan3` only |

Worth 4.0x latency and 51x data on a 100-row probe (43.9ms and 862 KB down to
10.9ms and 16.9 KB): the covering scan never has to touch the document, so the
full 384-d embedding stays in the cluster instead of riding back with every
candidate we were about to discard.

### 4. `APPROX_VECTOR_DISTANCE` composes, so probes batch

Blocking issues one probe per left row. Because the distance function is ordinary
N1QL, a chunk of probes composes into one statement with `UNION ALL`, demuxed by a
synthetic `_pid`:

```sql
(SELECT 0 AS _pid, ... ORDER BY APPROX_VECTOR_DISTANCE(..., $q0, ...) LIMIT 20)
UNION ALL
(SELECT 1 AS _pid, ... ORDER BY APPROX_VECTOR_DISTANCE(..., $q1, ...) LIMIT 20)
...
```

2.4x to 2.6x faster at 8 to 32 probes, same rows in the same order as issuing them
one at a time.  

### 5. The `DOT` metric lets the index serve the learned proxy

the index
supports a dot-product metric. A logistic-regression proxy scores
`sigmoid(w·x + b)`, monotonic in `w·x`, so the proxy's top-k *is* the
maximum-inner-product top-k for query vector `w`. Ship the weight vector `w` as the
query and a `DOT` index returns the proxy's own ranking directly:

| k | ANN vs exact proxy ranking | of those, truly positive |
|---|---|---|
| 10 | 1.000 | 1.000 |
| 50 | 1.000 | 1.000 |
| 200 | 0.995 | 0.995 |

Cosine to the *predicate phrase* gets 0.620 on the same data. `ann_above()` uses
this to fetch every row above a calibrated threshold, doubling `k` until the worst
returned score falls below it. On 50k rows, against what an exact scan finds:

| rows above the threshold | nProbes | returned | recall | time |
|---|---|---|---|---|
| 3,150 | 8 | 2,246 | 0.713 | 202ms |
| 3,150 | **64** | 3,151 | **1.000** | 358ms |
| 9,008 | 64 | 8,192 | 0.909 | 710ms |
| 14,689 | 64 | 8,192 | 0.558 | 673ms |

Inside its window the index turns a whole-collection scan into a bounded lookup:
358ms against 9.1s for the same 50k filter, roughly 25x. The window has three
edges, which are properties of the index worth knowing rather than faults:

*The index is selected only when the query metric matches its `similarity`.* Query
`'dot'` against a cosine-built index and the planner correctly declines it,
falling back to `PrimaryScan3` + `Fetch`, a full scan. Results stay correct, so
the only symptom is the "index-served" path being slower than scanning. Build a
`similarity: 'dot'` index and the same threshold-0 query goes from 953ms to 62ms.

*A served query returns L2-squared over the stored vectors, whatever metric string
you passed.* Only the brute-force path returns the raw metric. For unit-norm
vectors the score is recoverable exactly:

```
dist = |w|² + |x|² - 2(w·x)      so      w·x = (|w|² + 1 - dist) / 2
```

matching exact `w·x` to four decimals. `vectors_normalised()` guards the
precondition.

*A vector index scan caps `LIMIT + OFFSET` at 8,192* (`indexer.scan.vector.max_heap_size`,
mutable). Past that the request fails with error 5433, so the index path only
suits predicates selective enough to keep the qualifying set under the cap;
`ann_above` caps its doubling there and reports `exhaustive=False`.


## Benchmarks

Embeddings are `BAAI/bge-small-en-v1.5` (384-dim). The LLM,
 is `gemini-flash-latest` 

The table below is scored against an oracle, the dataset's own label (20 Newsgroups, Rotten Tomatoes), a regex (AG News), or a
ground-truth rule (Emails, BioDEX). This isolates the machinery: it measures whether the
vector index and cascade reproduce a *perfect* judge,  

**With an LLM.** Run against `gemini-flash-latest` on the polarity predicate, 50
labelled reviews: **P 0.913, R 1.000, F1 0.955** versus the human labels.  

**How to read the `vector-only` column.** It is cosine similarity to the embedded
predicate, cut at the threshold that maximises F1 *using the labels*. You cannot tune with
labels in production, so it is a generous ceiling: the best vector search could do here,
not what it would actually do.

| benchmark | task | vector-only | **semops** | cost |
|---|---|---|---|---|
| [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) | topic filter | F1 0.914 | **0.978** | 5.39× fewer calls |
| [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | polarity filter | F1 **0.661** (P=0.499) | **0.987** | 1.30× fewer calls |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) 50k | "quotes a dollar amount" | F1 **0.348** (P=0.252) | **0.971** | 5.18× fewer calls |
| **[Emails](https://arxiv.org/abs/2510.08489)** (Trummer 2510.08489) | contradiction join | F1 **0.18** | **0.95** | 110 vs 1000 calls |
| **[BioDEX](https://huggingface.co/datasets/BioDEX/BioDEX-ICSR)** (LOTUS/Abacus corpus) | paper ↔ reaction join | F1 **0.21** | **0.91** | 350 vs 48,600 (**139×**) |
| [20NG](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) | `sem_group_by`, 7 topics | n/a | purity **0.900** | **0 LLM calls** |
| [20NG](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) | `sem_dedup` | n/a | purity **1.000** | 37.7× fewer |

Across all of them the cascade stays close to what asking the LLM about every row would
have given, and blocking recall, not the cascade, is what limits join quality.


Calibration costs a fixed number of oracle calls, so on a small collection it eats
the budget. AG News, 50k rows on a live cluster, `sem_filter` at recall/precision
0.9, label oracle:

| rows | oracle calls | savings | F1 | escalate band |
|---|---|---|---|---|
| 2,000 | 854 | 2.34× | 0.981 | 43% |
| 10,000 | 2,334 | 4.28× | 0.936 | 23% |
| 25,000 | 5,240 | 4.77× | 0.951 | 21% |
| 50,000 | 8,169 | **6.12×** | 0.933 | 16% |

Two caveats. The saving grows with corpus size but **asymptotically,
not linearly**. The escalate band converges to a roughly fixed fraction, so savings
tend toward ~1/that fraction rather than climbing forever.

Index-served vs scanning everything, same 50k collection:

| | F1 | recall | oracle calls | rows scored | wall |
|---|---|---|---|---|---|
| scan-all + score-all | **0.933** | 0.915 | 8,169 | 50,000 | 9.1s |
| `Session.sem_filter` (`ann_above`, nProbes=8) | 0.615 | 0.452 | 3,673 | 5,811 | **2.8s** |

The index-served path is 3.3x faster and asks the oracle less than half as often, and
it pays for both in recall. At this selectivity roughly 14,700 rows clear the threshold,
which is well past the 8,192 cap a vector index scan allows, so it cannot return them
all whatever `nProbes` is set to. Measured directly, `ann_above` reaches recall 1.000
when the qualifying set fits under the cap and `nProbes` is raised to 64 (see the GSI
section above).

Use it for selective predicates, with `nProbes` tuned. For a filter that keeps a quarter
of the corpus, scan.

### When the proxy is weak

The 50k rows above were filtered on "is about science or technology", which is topic-shaped
and the corpus is literally labelled by topic. Re-run the same collection on *"quotes
a specific dollar amount of money"*, a predicate the embedding was never organised
around (dollar figures appear across World, Business and Sci/Tech alike):

| | topic predicate | specific predicate |
|---|---|---|
| vector-only (oracle-tuned) | F1 0.512 (P 0.420) | **F1 0.348** (P 0.252) |
| cascade vs oracle | F1 0.933 | **F1 0.971** (P 1.000, R 0.944) |
| savings | 6.12× | 5.18× |
| bands | accept 7,846 / reject 33,985 | **accept 0** / reject 40,349 |
| τ⁺ | 1.264 | **+inf** |

First, **τ⁺ never certified, so the accept band
vanished**. The proxy could not support precision 0.9 anywhere on the positive
side, so nothing was auto-accepted and every kept row got an oracle call. That is
the safe-degradation property doing its job: precision came out at exactly 1.000,
and recall 0.944 still cleared its 0.9 target. A weak proxy costs money, not
correctness.

Second, **savings barely moved (6.12× → 5.18×) despite the proxy being far worse.**
Savings here are driven by *selectivity*, not proxy quality: at ~6% true, the reject
band is 40,349 rows, and rejecting confidently is a much easier statistical claim
than accepting confidently. Selective predicates pay off even with a poor proxy,
entirely through τ⁻.

Caveat on the predicate: it is lexical, so it is arguably harsher on embeddings than a
genuinely semantic one would be. Read it as evidence about how the cascade behaves when
the proxy is weak, not as a claim that the operator is good at regex.

The `vector-only` column across every benchmark spans **0.21 to 0.914**, and the low end
is not exotic: polarity, contradiction, doc↔label matching, and anything the embedding
was not organised around all land there. The operator stays at 0.97 to 1.00 throughout. Note
also that the two AG News figures (0.512 topical, 0.348 specific) sit far below 20NG's
0.914 on the same embedding model: which end of that range you get is a property of your
corpus, not something you can read off the predicate. That is the case for learning the
proxy from a labelled sample instead of trusting a phrase embedding, which is what `proxy_model`
does.

---

## Limitations

**The guarantee is relative to the oracle, not to truth.** Every number in this document
is about "close to what the LLM would have said", NOT "correct". If your oracle is wrong
about something, so is the result, and no threshold setting will catch it.

**`block_k` decides join recall and there is no estimator.** If set too low, candidate
pairs never form, so the LLM is never asked and recall is capped with nothing in the logs
to tell you. The right value swings by more than an order of magnitude across tasks (2 to
4x fan-out on 20 Newsgroups, roughly 83x on BioDEX). You set it, and you should measure
blocking recall separately before trusting a join.

**`ann_above` cannot tell a complete result from a lossy one.** It stops when the worst
returned score falls below the threshold, which is a valid proof for exact search but not
under ANN: at `nProbes=8` on 50k rows it reported `exhaustive=True` at recall 0.713. Tune
`nProbes` against a known-good exact result before trusting the index-served path.

---

## Not built yet

| | what is missing |
|---|---|
| `sem_agg` | LLM map-reduce over a group. The operator that composes with `sem_group_by`, and the one this project originally set out to build. |
| `sem_topk` | LLM re-ranking. `search` returns similarity order only. |
| `block_k` estimator | Nothing measures fan-out and picks a value, so the biggest correctness knob is a manual guess. |
| `ann_above` guardrails | It does not estimate the qualifying count, refuse the index path when that exceeds the 8,192 cap, or tune `nProbes`. All three are the caller's problem today. |
| Cache eviction | The in-process cache is unbounded. |
| Scale evidence | Exercised to 50k rows. Nothing here has been run at 10⁶, and the `nProbes` saturation point in particular moves with collection size. |

---

## Running it

The library uses a `src/` layout, so install it first:

```bash
pip install -e .
pip install -r examples/requirements.txt   # sklearn / fastembed / datasets, for the examples
```

Then:

```bash
python examples/tour.py               # every operator on the ticket data above
python examples/demo_offline.py       # the cascade alone, no keys
python -m unittest discover -s tests  # 32 tests, offline
```

`tour.py` is the one to start with: it runs all five operators on the exact data in
this README, with real embeddings and a scripted oracle, so the output is
deterministic and you can see what each operator decides.

Against a real cluster and a real model:

```bash
python examples/quickstart.py                            
python examples/cb_ingest.py --dataset rotten --n 2000  # load + index
python examples/eval_couchbase.py                       # measured vs labels
python examples/verify_gsi_paths.py                     # the GSI fast paths
python examples/bench_emails.py --oracle truth          # Trummer's join benchmark
python examples/bench_biodex.py --oracle truth          # LOTUS/Abacus corpus
```

Deeper notes live in `docs/`: [`sem_filter.md`](docs/sem_filter.md),
[`sem_join.md`](docs/sem_join.md),
[`sem_dedup_and_group_by.md`](docs/sem_dedup_and_group_by.md), and
[`gsi_notes.md`](docs/gsi_notes.md)

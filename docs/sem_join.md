# `sem_join` architecture

Naive is `n_left × n_right` LLM calls. Two independent cost attacks:

```
BLOCKING (vector index)   each left row ──ANN──► its top-block_k right rows
  prunes n × m ──► candidate pairs             (batched; native on Couchbase)
       │
CASCADE (per pair)        proxy = pair similarity ──► accept / reject free
       │                                         ──► uncertain middle escalates
       ▼
BLOCK-JOIN ADJUDICATION   group the escalate band BY LEFT ROW; one prompt per
  (Trummer, arXiv 2510.08489)  left, listing all its candidates:
       "QUERY: <left>.  CANDIDATES: 1..n.  Which match? ... then 'Finished'"
       └── output truncated (no "Finished")? ──► overflow ──► shrink batch by α, retry
```

## Why batching matters

Adjudication cost becomes **O(n_left)**, not O(candidate_pairs). Quadrupling
`block_k` quadruples the candidate pairs but leaves the number of LLM calls flat.
Measured: `block_k` 39 → 160 lifted recall 0.56 → 0.94 with `block_calls` pinned
at 40.

That is what makes wide blocking affordable, and wide blocking is what makes the
join accurate. The two facts are the same fact.

## The adaptive part (Trummer's Algorithm 3)

The right batch size depends on the join's selectivity, which you cannot estimate
for a natural-language predicate. So don't. Start optimistic; on overflow shrink
by α and retry. Converges without ever knowing selectivity.

Overflow is detected by a sentinel: the prompt asks the model to finish with the
literal token `Finished`. If it is missing, the response was truncated and the
batch was too big.

## Choosing `block_k`

The biggest correctness knob in the library. Recall lost in blocking is
unrecoverable downstream: the pair never forms, the LLM is never asked, and
nothing in the logs says so.

The right value is task-dependent, and the spread is more than an order of
magnitude:

| join shape | example | `block_k` |
|---|---|---|
| symmetric, doc ↔ doc | 20NG same-topic | 2 to 4x fan-out |
| asymmetric, long doc ↔ label | BioDEX paper/reaction | use the whole right side |

On BioDEX roughly 83x fan-out was needed for 0.85 recall, because every short
reaction term looks about equally close to a long clinical abstract. Since
adjudication is O(n_left), the fix is to stop pruning and let the LLM see the full
vocabulary.

Measure blocking recall separately before trusting a join. It is the ceiling on
join recall, and it is cheap to measure: take known-true pairs and check how many
survive into the candidate set at your `block_k`.

## Knobs

| knob | meaning |
|---|---|
| `block_k` | ANN candidates retrieved per left row |
| `block_adjudicate` | batch the escalate band by left row (Trummer block join) |
| `block_alpha` | shrink factor on overflow |
| `policy` | `cascade` \| `oracle` \| `blocking` (pure embedding join, no LLM) |
| `self_join` | dedupe symmetric candidate pairs (used by `sem_dedup`) |
| `probe_batch` | ANN probes per round trip |
| `max_sample` | capped at 500: `sample_frac × candidate_pairs` explodes otherwise. Capping it on BioDEX took F1 0.98 → 1.00 *and* calls 5010 → 350. |

## A note on cardinality

Nothing here estimates join selectivity, and nothing needs to. The adaptive batch
shrinking sidesteps it, and the cascade calibrates from a labelled sample rather
than from a cardinality estimate. That is deliberate: selectivity estimation for a
natural-language predicate is not a solved problem.

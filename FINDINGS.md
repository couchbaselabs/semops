# Findings

A consolidated record of experiments run on semops, what they established, and
what they did not. Written to be honest rather than flattering: several results
here overturned earlier conclusions of mine, and the errors are recorded
alongside the findings because they were instructive.

Models used: embeddings `BAAI/bge-small-en-v1.5` (384-d, local), LLM judge
`gemini-flash-lite-latest`, open model `Qwen/Qwen3-8B` for the internals work.
Datasets: Rotten Tomatoes, 20 Newsgroups, AG News, Abt-Buy, DBLP-ACM.

Confidence is marked throughout:
**[solid]** replicated or large-effect, **[tentative]** single run or small n,
**[retracted]** believed earlier, no longer supported.

---

## 1. Headline results

### Vector search alone vs the operator, real LLM judging

| operator | vector search only | semops + `gemini-flash-lite-latest` |
|---|---|---|
| `sem_filter` (Rotten Tomatoes, polarity) | 0.774 | **0.984** |
| `sem_join` (20 Newsgroups, same topic) | 0.540 | **0.667** |
| `sem_dedup` (20 Newsgroups, same topic) | 0.521 | **0.664** |

**[solid]** for filter. The gap is large and the mechanism is clear: similarity
measures what a text is *about*, not what it *concludes*. A review saying "I
expected a boring disaster, instead it was a triumph" is full of negative
vocabulary and ranks high under cosine-to-"negative review", wrongly.

**[tentative]** for join and dedup. Both are pair operators whose numbers depend
heavily on configuration (see §3), and the 20NG labels are themselves suspect
(see §5.1).

### On a live Couchbase GSI cluster

`sem_filter` over 200 labelled reviews, real index, label oracle:

- semops: **F1 0.981** (P 1.000, R 0.962), 174 oracle calls of 200
- vector-only cosine on the same index: **F1 0.73**

Three SQL++ statements, captured off the wire:

```sql
-- 1. scan, covered by INCLUDE (text, label): no embeddings leave the cluster
SELECT META(d).id AS _id, d.`text` AS _text, d.label AS _label
FROM `default`.`_default`.`polarity_gsi` d;

-- 2. pull only the calibration sample's vectors
SELECT META(d).id AS id, d.`embedding` AS v
FROM `default`.`_default`.`polarity_gsi` d USE KEYS $keys;

-- 3. score every row by the learned proxy, in-cluster ($q = LR weight vector)
SELECT META(d).id AS id, VECTOR_DISTANCE(d.`embedding`, $q, 'dot') AS dist
FROM `default`.`_default`.`polarity_gsi` d USE KEYS $keys;
```

The index-served variant (`ann_above`) ships the classifier weight vector as an
ANN query:

```sql
SELECT META(d).id AS _id, d.`text` AS _text,
       APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, 'dot', 8, true, 200) AS _dist,
       d.label AS _label
FROM `default`.`_default`.`polarity_gsi` d
ORDER BY APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, 'dot', 8, true, 200)
LIMIT $k;
```

`EXPLAIN` confirms `IndexScan3` on `idx_polarity_gsi_dot`, no `PrimaryScan3`, no
`Fetch`. It returned 60 rows above threshold, all 60 true negatives,
`exhaustive=True`. **[solid]**: this is the DOT trick working end to end on a
real index.

Note the two ANN paths differ: the list-path filter uses exact
`VECTOR_DISTANCE ... USE KEYS` (no ANN index), while `ann_above` uses
`APPROX_VECTOR_DISTANCE` and is genuinely index-served.

---

## 2. What actually improves a semantic join

Ranked by measured effect. The striking thing is that the two largest wins cost
nothing and involve no change to the LLM.

| lever | ΔF1 | cost | verdict |
|---|---|---|---|
| Align the predicate to the target, supply the label space | **+0.102** | free | **[solid]** biggest lever |
| Mutual kNN blocking (0.889 → 1.000 recall) | +0.023 | free | **[solid]** adopt |
| SELECT instead of per-pair | ~equal F1 | **~30x cheaper** | **[solid]** adopt |
| Chunking into smaller prompts | −0.10 | 7x more calls | **[solid]** reject |
| Select-then-match hybrid | +0.027 | 30x more calls | **[solid]** reject |
| `query_last` prompt layout | −0.03 | free | **[solid]** reject |

### 2.1 Predicate alignment is the biggest lever

Same candidates, same strategy, same 12 calls; only the question changed.

| predicate | P | R | F1 |
|---|---|---|---|
| "same **topic**" | 0.931 | 0.500 | 0.651 |
| "same **newsgroup**" | 0.967 | 0.537 | 0.690 |
| "same newsgroup" **+ list of group names** | 0.957 | **0.620** | **0.753** |

Recall rose 24% relative while precision held, so this is not the model simply
answering yes more often. Telling the model the label space is legitimate: in a
real join you know your schema.

### 2.2 Blocking is nearly solved and the fix is free

| k | one-directional top-k | **mutual kNN** |
|---|---|---|
| 8 | 0.417 | 0.593 |
| 16 | 0.750 | 0.898 |
| **24** | 0.889 | **1.000** |
| 48 | 0.954 | 1.000 |

Counting a pair when *either* document is in the other's top-k takes blocking
recall to a perfect 1.000 at k=24. Retrieval-side only, no API cost.

**Consequence: adjudication, not retrieval, is the bottleneck.** At k=24 the
index hands over every true pair and the LLM finds roughly half to two-thirds.
An earlier claim of mine that "index-served blocking is the lever that would move
the number most" is **[retracted]**: it had the emphasis backwards.

### 2.3 Batched adjudication (Trummer, arXiv:2510.08489)

Batching a query row with all its candidates into one prompt decouples cost from
`block_k`:

| adjudication | `block_k` | calls | F1 |
|---|---|---|---|
| per-pair | 8 | 96 | 0.40 |
| per-pair | 16 | 192 | 0.57 |
| per-pair | 24 | 288 | 0.62 |
| block-batched | 8 | 12 | 0.37 |
| block-batched | 16 | 12 | 0.56 |
| **block-batched** | **24** | **12** | **0.67** |
| block-batched | 32 | 12 | 0.59 |
| block-batched | 48 | 12 | 0.54 |
| block-batched | 69 | 12 | 0.35 |

Better *and* ~24x cheaper at the peak. Quality follows an inverted U over batch
size, peaking near 24 candidates. **[solid]**

Important: the decline at k=69 happens well inside the context window (~20k
characters, trivially within the model's limit), so the batch-size limit is not
the window. But see §4.2, my "attention" explanation for this did not survive
testing, and the mechanism remains unexplained.

### 2.4 Chunking is strictly harmful

Blocking pinned at 1.000, only chunk size varied:

| chunk | prompts/row | calls | precision | recall | F1 |
|---|---|---|---|---|---|
| **34 (all)** | 1.2 | **15** | **0.937** | **0.546** | **0.690** |
| 17 | 2.3 | 28 | 0.862 | 0.519 | 0.647 |
| 12 | 3.5 | 42 | 0.824 | 0.519 | 0.636 |
| 8 | 4.8 | 58 | 0.871 | 0.500 | 0.635 |
| 4 | 9.1 | 109 | 0.797 | 0.472 | 0.593 |

Monotonically worse on every metric while costing up to 7x more. Match density
is constant across these rows, so this is not a density effect. The reading:
**the model judges comparatively**, and a wider candidate list gives it more to
compare against. Taking that away makes it guess. **[solid]**

### 2.5 Strategy depends on match density

This reproduced in opposite directions on two datasets, which is what makes it
credible:

| dataset | density | SELECT | MATCH (per-pair) |
|---|---|---|---|
| 20NG | 9 matches/row, 34 candidates | **wins**, 30x cheaper | loses |
| Abt-Buy | 0.6 matches/record, 2.5 candidates | 0.900 | **0.945** |

On Abt-Buy: SELECT P 0.990 / R 0.825, MATCH P 0.965 / R 0.925. SELECT keeps
higher precision but loses recall, and at low density the recall loss dominates.
Union equalled MATCH exactly and intersection equalled SELECT exactly, so
SELECT's picks were a strict subset here.

**The planner rule:** estimate match density from blocking scores before any LLM
call. Dense → SELECT. Sparse → MATCH. **[solid]**, and it is a genuine
query-optimizer decision derived from measurement.

Related and consistent: at m=1 (one true match in a prompt) precision collapses
to **0.36** because the model will not answer "none". Per-pair matching *is* the
m=1 regime, which explains its known low precision.

---

## 3. Blocking quality vs the literature

On Abt-Buy, plain embedding retrieval over the full opposite table:

| k | blocking recall |
|---|---|
| 1 | 0.767 |
| 5 | 0.951 |
| 10 | **0.990** |
| 20 | 1.000 |

Sparkly, the SOTA blocker used by ComEM, reports recall@10 of 86.6% to 99.96%.
Ours sits inside that range untuned. **[solid]** The retrieval side is
competitive; we are not behind the literature here.

---

## 4. LLM internals (Qwen3-8B)

### 4.1 There is no reservoir of unspoken matches. **[retracted hypothesis]**

The question: when a SELECT adjudicator fails to name a true match, does it still
represent that match internally? If yes, the recall ceiling is a decoding
bottleneck and readable directly.

Answer: **no.**

| | F1 |
|---|---|
| generation | **0.855** (P 0.824, R 0.888) |
| best linear probe (layer 27) | 0.786 |

Generation beats the probe. The 195 "missed" matches in my first run were
produced by a bug in my own harness (see §6.1); the true number was 23. A weaker
classifier catching some of a stronger one's errors is unremarkable and is not
evidence of latent knowledge.

**This removes the main justification for reading model internals to recover
matches.** Idea dropped.

### 4.2 Position effects are real, well-powered, and practically minor

1,350 calls, paired design, McNemar exact tests, bootstrap CIs.

| k | first | middle | last |
|---|---|---|---|
| 8 | **0.433** | 0.573 | 0.587 |
| 24 | **0.373** | 0.573 | 0.367 |
| 48 | **0.160** | 0.480 | 0.300 |

- **Not lost-in-the-middle.** Liu et al. predict a U (ends good, middle bad). We
  measured an **inverted U**: the middle is the *best* position at k=24 and 48.
- **Position 1 is worst at every k**, and the effect deepens with list length.
  Discordant counts: 3 vs 24 (k=8), 3 vs 33 (k=24), **3 vs 51** (k=48), all
  p<0.00001. **[solid]**
- **`query_last` does not fix it.** It rescues position 1 (0.153 → 0.465) but
  wrecks middle (0.583 → 0.326) and last (0.444 → 0.287). Net **worse**: 0.360 vs
  0.394. The damage follows whichever end abuts the query text, which supports an
  adjacency mechanism, but relocating it is not a fix. **[solid]**
- **Practical impact is small.** The penalty affects 1 slot out of k, so under 1
  point of aggregate join recall at k=48.

Note this is known art: ComEM already reports position bias in select-prompts.
Also note the depth figure changed, see §4.3.

### 4.3 Operator anatomy

Probing every 4th layer of Qwen3-8B, three tasks varying predicate type and
domain:

| layer | match (Abt-Buy) | match (DBLP-ACM) | filter (Rotten Tomatoes) |
|---|---|---|---|
| 4 | 0.639 | 0.898 | 0.675 |
| 12 | 0.682 | 0.916 | 0.825 |
| **20** | **0.853** | **0.940** | **0.891** |
| 28 | 0.847 | 0.892 | 0.882 |
| 36 | 0.748 | 0.836 | 0.879 |

**All three peak at layer 20 (56% depth) and decline afterwards.** **[solid]**
My prediction that unary predicates would resolve early and binary ones late was
wrong. Whatever the operator, the decision is computed mid-network and later
layers do something else, presumably converting it to tokens.

This also corrects an earlier statement of mine that matching peaks at layer 27.
That came from sampling only layers 9/18/27/36; with finer sampling the peak is
20. **[retracted]**: a lesson about grid resolution.

Transfer, trained on one task and tested on another:

| train → test | F1 | in-domain | note |
|---|---|---|---|
| products → publications | 0.761 | 0.917 | same predicate, different domain |
| publications → products | 0.743 | 0.848 | same predicate, different domain |
| filter → match | **0.517** | 0.848 | different predicate type, ~chance |
| match → filter | 0.690 | 0.884 | different predicate type |

**Type matters more than domain**: crossing predicate type costs far more than
crossing domain. Filter → match at 0.517 is chance on a balanced task, so unary
and binary predicates appear to use different machinery despite being computed at
the same depth. **[tentative]**

**Serious caveat on the domain-transfer claim.** DBLP-ACM turns out to be
**97.4% solvable by naive token overlap** (Abt-Buy: 0.736). So the publication
task is largely a string-similarity task, which explains its 0.898 at layer 4
before the model has done much. A probe transferring onto it may be detecting
lexical similarity rather than a shared entity-matching circuit. The
"domain-general matching direction" reading is **[retracted]** pending a rerun
with two genuinely hard datasets (Walmart-Amazon, WDC Products).

**This section replicates published work.** [How do LLMs Understand
Relevance?](https://arxiv.org/abs/2504.07898) (TOIS 2025) used activation
patching and found a three-stage process: early layers extract query and document
information, **middle layers process relevance**, later layers use specific
attention heads to format the judgment. Our peak at 56% depth with a decline
afterwards is exactly that middle-to-late boundary, reached by a different method
(probing rather than patching). Corroboration, not discovery, and they cover
pointwise *and* pairwise, i.e. our filter-versus-join distinction.

### 4.5 Zero-shot predicate directions do not work. **[retracted: idea B's label-free form]**

The premise behind pushing predicates into the index without training: could the
query vector be read from the model by encoding the *predicate alone*?

20 Newsgroups, 7 categories as 7 binary predicates over the same rows. Rows
encoded alone give `b_r`; a supervised probe gives `w_p`; two label-free
constructions give `a_p` (direct encoding, and a contrastive
positive-minus-negative difference).

| | mean over 7 predicates |
|---|---|
| supervised probe F1 | **0.772** |
| zero-shot F1 (direct) | 0.295 |
| zero-shot F1 (contrastive) | 0.288 |
| all-positive baseline F1 | 0.250 |
| **cos(w_p, a_p)** | **+0.010 / +0.006** |

The cosine is the finding: the model's own representation of a predicate is
**orthogonal** to the direction that actually classifies rows by it. Individual
values scatter around zero and some are negative. Zero-shot F1 is barely above
the trivial baseline.

**Consistent with the literature**, which matters: steering vectors are
documented as brittle and highly variable, with many behaviours unsteerable "even
when sweeping across all layers and strengths". So this is not a weak
construction on my part, it is the known failure mode of the technique class.

**What survives:** rows encoded alone are linearly separable at **F1 0.772**, so
LLM hidden states *are* good index keys under supervision. The supervised
pushdown path (our existing DOT trick) is intact; only the label-free shortcut is
dead. Scope: one layer, two constructions, the fair claim is "the obvious
constructions fail decisively", not "no such direction exists".

### 4.6 Predicates compose linearly. **[tentative, the one positive internals result]**

Do directions compose? Is d(A ∧ B) ≈ α·d(A) + β·d(B)? Supervised directions,
AG News, with two overlapping semantic predicates so the conjunction has support:
A = World|Business, B = Business|Sci/Tech, A∧B = Business.

| | cos with best linear fit from wA, wB |
|---|---|
| **AND** | **+0.790** (α=+0.64, β=+0.66) |
| OR | +0.530 (α=+0.25, β=+0.24) |
| sanity: cos(wA,wB) | −0.101 (not collinear, so the test is not vacuous) |
| sanity: cos(wA,random) | +0.002 |

Classifier performance on A∧B:

| | F1 |
|---|---|
| all-positive baseline | 0.403 |
| w_A alone (control) | 0.680 |
| w_B alone (control) | 0.658 |
| **fitted α·wA + β·wB** | **0.844** |
| directly trained w_AND | 0.808 (ceiling) |

The composed direction **beats the directly trained one** and clears both
single-predicate controls, so the conjunction is being constructed rather than
read off A. The fitted coefficients are near-equal, which is what conjunction
should look like. AND and OR are recovered *differently* (0.790 vs 0.530), which
is the control that could have falsified the framing and did not.

Caveat: the OR arm is near-saturated (base rate 0.75, all-positive baseline
already 0.859), so the AND-vs-OR contrast is weak evidence and wants a rebalanced
disjunction.

### 4.7 Composition pays off exactly where labels are scarce. **[tentative]**

Follow-up prediction: composition should beat direct training when the
conjunction is *rare*, because it borrows strength from two well-supported
probes. Tested with a label-free composed direction, `unit(wA) + unit(wB)` with
fixed equal weights, the earlier least-squares fit needed conjunction labels, so
it could not support a practical claim; equal weights are justified by that fit
landing at +0.64/+0.66.

| conjunction | composed (0 labels) | direct probe, by n positives |
|---|---|---|
| **common** (25%) | **0.673** | n=10 → 0.619, n=25 → 0.692, n=250 → 0.779 |
| **rare** (5.2%) | **0.750** | n=10 → 0.501, n=25 → 0.577, n=50 → 0.678 |

**Crossover for the common conjunction is between 10 and 25 labels. For the rare
one, composition beat every probe we could fit**: the dataset ran out of
positives at n=96 before the direct probe caught up, with extrapolated crossover
near n≈100, more positives than the conjunction *has* in 3,000 rows.

The rule: **estimate the conjunction's base rate; if positives are scarce
relative to it, compose rather than train.** A second measurement-derived planner
rule alongside the density rule in §2.5.

Two qualifications. `w_B` alone reaches 0.731 on the rare conjunction against the
composed 0.750, so the margin over the best single component is thin, "mentions
a dollar amount" nearly determines Business in this corpus. And variance is
highest where it matters (sd 0.071 at n=10). The clean version needs several
conjunction pairs where neither predicate dominates.

### 4.8 Early exit: probing beats generation at ~3.3x lower cost. **[solid, corrected]**

If the decision is computed mid-network, the forward pass can stop there. Measured
quality *and* wall-clock on the same predicate, so the columns are comparable.

| method | F1 | ms/row | speedup |
|---|---|---|---|
| full generation | 0.889 | 224 | 1.0x |
| probe @ layer 8 (22%) | 0.867 | 24 | **9.3x** |
| **probe @ layer 12 (33%)** | **0.901** | **32** | **6.9x** |
| probe @ layer 20 (56%) | 0.838 | 49 | 4.5x |
| probe @ layer 36 (100%) | 0.914 | 83 | 2.7x |

Two savings stack: skipping autoregressive decode (2.7x even at full depth) and
truncating the remaining layers.

**CORRECTION, the table above is batch size 1, which inflates the speedup ~2x.**
Nothing is served at batch 1, where generation is memory-bandwidth bound and the
GPU is mostly idle. Re-measured as throughput (rows/sec) across batch sizes:

| batch | generation | probe L36 | probe L20 | probe L12 | L12 speedup |
|---|---|---|---|---|---|
| 1 | 4.5 | 12.1 | 20.5 | 30.9 | **6.9x** |
| 4 | 12.8 | 26.0 | 43.6 | 66.1 | 5.2x |
| 8 | 18.8 | 29.8 | 51.3 | 78.8 | 4.2x |
| 16 | 22.5 | 32.7 | 56.2 | 85.2 | 3.8x |
| 32 | 22.7 | 28.9 | 49.4 | 75.8 | **3.3x** |

**The honest number is ~3.3x, not 6.9x**, converging on the FLOP ratio (12 of 36
layers ≈ 3x) as it should. The finding is corrected, not retracted.

**Where the saving actually comes from changes with batching**, and this is the
useful part. At batch 32: skipping decode is worth only **1.3x**, while
truncating to layer 12 is worth **2.6x on top**. At batch 1 it was the reverse.
Under batching the decode cost amortises away and essentially all the durable
benefit is the layer truncation, which is a real FLOP reduction. So the
engineering advice is *early exit specifically*, not "avoid generation".

Serving note: throughput saturates near **batch 16** on an L4 (generation flat at
22.5 → 22.7; the probe actually drops 85.2 → 75.8 at batch 32, likely memory
pressure). Benchmark and serve at 16 on this hardware.

Quality is unaffected by this correction: F1 0.901 at layer 12 vs 0.889 for
generation was a quality measurement, not a timing one.

**The caveat that bounds the claim:** the probe needs labelled training data,
generation is zero-shot. This is *not* "probing beats the model" in general. It
*is* apples-to-apples for our cascade, which already trains a proxy on a labelled
sample. **Actionable version: train the semops proxy on LLM hidden states at
layer ~12 rather than `bge` embeddings, for generation-quality judgments at a
fraction of the cost.** Early-exit transformers are established (CALM, LayerSkip);
what is ours is the observation that operator decisions specifically resolve early
enough to exploit it.

**This corrects §4.3.** I claimed "all operators peak at layer 20". Here topic
classification peaks at **layer 12** and is near-flat from layer 8. The
difference is *predicate difficulty*, not operator type: 20NG topic is easy and
resolves early; Rotten Tomatoes sentiment is harder and needed layer 20. Correct
claim: **depth tracks predicate difficulty, so the exit layer is a per-predicate
tuning parameter**, not a universal constant. **[the layer-20-for-everything
claim is retracted]**

### 4.8b Distillation needs ~10x more labels than the cascade already pays for

The appealing idea: semops already spends LLM calls labelling a calibration
sample, so distil a small model onto those sunk-cost labels and stop calling the
big one. Tested as a learning curve, Qwen3-0.6B fine-tuned on Abt-Buy.

| n_train | positives | F1 | P | R |
|---|---|---|---|---|
| 100 | 13 | 0.176 | 0.124 | 0.306 |
| 250 | 33 | 0.244 | 0.221 | 0.272 |
| 500 | 60 | 0.281 | 0.333 | 0.243 |
| **1000** | **121** | **0.767** | 0.713 | 0.830 |
| 2000 | 222 | 0.823 | 0.763 | 0.893 |
| *reference* | | *0.855 (Qwen3-8B zero-shot)* | | |

**A sharp phase transition, not a smooth curve.** F1 jumps 0.281 → 0.767 between
500 and 1000 examples. The threshold is in **positives**, roughly **120**, not in
total labels, unsurprising on a task with a 10.8% positive rate.

**It never crosses over.** At 2000 labels it reaches 0.823, still below the 8B it
would be distilled from (0.855) and well below Gemini (0.900 / 0.945).

**The practical consequence:** our cascade calibrates on 100–300 labels, which
lands squarely in the dead zone (F1 0.18–0.28). Distillation is **not** a free
ride on labels we have already bought; it needs a deliberate ~1000+ call spend
and still trails the teacher. **[solid on the shape, see caveats]**

Caveats where I would not trust my own numbers. The throughput column is
discarded: I trained in `float32` with fixed 128-token padding while the probe
measurements used `bfloat16` with dynamic padding, so the 52.8 rows/s is an
artifact of my setup rather than a property of the model. And nothing was tuned,
fixed LR, 3 epochs, no early stopping, where published Abt-Buy fine-tunes (Ditto
and successors) reach ~0.89. The curve *shape* and the positive-count threshold
are the trustworthy parts; the absolute ceiling understates what is achievable.

### 4.9 Predicates are near-orthogonal, which explains why composition works

Seven predicate directions at layer 20, in 4096 dimensions:

```
effective rank = 6.59 of 7
variance spread evenly: 0.20 0.18 0.18 0.15 0.14 0.12 0.03
pairwise cosines all between -0.03 and -0.34
```

**The constrained-subspace hypothesis is dead**: seven predicates occupy
essentially seven dimensions, there is no compression to exploit, and each new
predicate needs its own direction.

But the negative result explains the positive one in §4.6. **Near-orthogonal
directions add without interfering**, which is why `unit(wA) + unit(wB)` recovers
the conjunction cleanly. Had predicates been packed into a low-rank subspace,
summing them would produce cross-talk. The composition run independently measured
`cos(wA, wB) = -0.101`, consistent with this. Of the two candidate explanations
for composition, constrained space versus orthogonality, the data picks
orthogonality.

(Mildly negative cosines are expected: one-vs-rest classifiers over mutually
exclusive categories are structurally anti-correlated. The strongest, graphics vs
Mac hardware at -0.34, is the most confusable pair.)

### 4.10 Qwen3-8B is competitive with the API model

On Abt-Buy: **P 0.824, R 0.888, F1 0.855**. Gemini flash-lite reached higher
precision (0.990) but notably lower recall (0.825). A self-hostable 8B model is
in the same class as the API model on this operator. **[tentative]**: the two
were measured on different subsamples and need a head-to-head on identical data.
Worth following up: it changes the cost model for shipping a model alongside the
database.

---

## 5. What we learned about measurement itself

### 5.1 Our 20NG join labels were wrong

Ground truth was "same newsgroup", but the prompt asked "same topic". These are
not the same predicate.

- same-newsgroup pairs average cosine 0.617, different-newsgroup 0.535
- **10.5% of same-label pairs are less similar than the average different-label pair**

Concrete pairs the ground truth calls a match:

- *"purchasing a grayscale printer for medical images"* vs *".DXF can not be changed over to .IFF format"* (both `comp.graphics`)
- *"And Dino Ciccarelli and Ray Sheppard"* vs *"Stanley Cup Playoff stats through April 25 1993"* (both `rec.sport.hockey`)

None are about the same topic. When the model rejected these it was **right and
our label was wrong**.

**Consequence:** the ~0.55 recall ceiling that resisted every intervention was
substantially a *label* ceiling. Relative comparisons between strategies survive
(they all used the same labels); absolute recall as a measure of model capability
does not. **[solid]**

### 5.2 The model is deterministic, so power comes from examples not repeats

5 distinct prompts × 10 identical repeats at temperature 0: **5/5 byte-identical
every time**. Re-running a prompt buys nothing.

This matches Liu et al.'s methodology (greedy decoding, no seeds; power from
2,655 queries and 500 examples per condition). Our n=20 pilot gave ±0.11 standard
error, which is why the powered rerun used n=150 with paired McNemar tests.

### 5.3 Benchmark sample construction matters

Our Abt-Buy evaluation stratified the sample (120 records with a match, 80
without), giving a 60% positive base rate against a natural 28%. That inflates
precision. Valid for the within-experiment strategy comparison it was built for,
**not comparable to published Abt-Buy F1**.

---

## 6. Errors made, and what they teach

Recorded deliberately. Two of them produced results that *confirmed my
hypothesis*, which is the failure mode most worth guarding against.

### 6.1 A broken baseline manufactured my headline result

The first probe run showed generation recall 0.053 versus probe recall 0.774,
apparently spectacular support for "the model knows but doesn't say". It was
entirely my bug: I fed a raw string to an instruction-tuned model without the
chat template, and Qwen3 defaults to thinking mode which consumed the 12-token
generation budget. After fixing both, generation recall went **0.053 → 0.888**
and the hypothesis died.

A recall of 0.053 from an 8B instruct model is implausible on its face, and a
comparable 0.825 from Gemini was already sitting in the conversation. **Validate
the baseline before trusting the contrast**, especially when the contrast is
flattering.

### 6.2 An easy dataset undermined a transfer claim

I reported a "domain-general same-entity representation" before checking whether
the target task was hard. DBLP-ACM is 97.4% token overlap. **Establish a trivial
baseline for every dataset before interpreting model results on it.**

### 6.3 Calling a 3-call demo "validated"

I ran three API calls showing Gemini's prompt cache returns
`cachedContentTokenCount: 2922`, then described the hash-join framing as
"validated". What was validated was that prompt caching works as documented. No
join operator, no build-side selection by cardinality, no cost comparison, no
quality check existed. **Distinguish a feasibility probe from a result.**

### 6.4 Asserting novelty without searching

I claimed "nobody in the ER literature frames batching this way" having searched
only for entity-matching prompting strategies. When I did search properly, ideas
C (per-candidate logits) and D (hidden states as embeddings) were already
published as FIRST (EMNLP 2024) and LLM2Vec. **Lit review before pitching, not
after.** Doing it earlier would have saved several experiments.

### 6.5 Coarse grids give wrong answers

Sampling layers 9/18/27/36 put the matching peak at 27. Sampling every 4 layers
put it at 20. The first was the closest sample to the truth, not the truth.

### 6.6 Underpowered pilots

n=20 per cell gave ±0.11 SE across ~15 comparisons. The pattern happened to hold
at n=150, but it could easily not have.

### 6.7 Benchmarking at batch size 1

The early-exit speedup was first measured unbatched, which inflated it from 3.3x
to 6.9x, the regime that most flatters a forward-pass-only method against
autoregressive generation. **Benchmark at the batch size you will serve at.** The
finding survived, but only because it was checked before being acted on.

### 6.8 Diagnosing from theory instead of logs

When the batched run failed twice, I asserted it was "almost certainly a hard
CUDA OOM" and redesigned around memory pressure. The logs said
`ModuleNotFoundError: No module named 'sklearn'`, I had omitted scikit-learn
from that script's image. Two wasted relaunches because I theorised a cause that
fit my expectations rather than reading the error. Same pattern as §6.1.

---

## 7. Position relative to the literature

| our idea | status |
|---|---|
| Per-candidate logits for ranking | published: [FIRST](https://arxiv.org/abs/2406.15657), EMNLP 2024 |
| LLM hidden states as retrieval embeddings | published: LLM2Vec, E5-Mistral, RepLLaMA |
| Position bias in select-prompts | published: [ComEM](https://aclanthology.org/2025.coling-main.8.pdf), COLING 2025 |
| Match/Compare/Select taxonomy and its precision/recall tradeoff | published: ComEM. We reproduced it independently |
| Blocking via ANN | mature field: Sparkly, BlockingPy, standard practice is to raise k until validation recall > 99.5% |
| Model-derived predicate direction pushed into a **database** index | not found. Closest is [HyDE](https://arxiv.org/abs/2212.10496), which generates text then embeds with a separate encoder. **But see §4.5: the label-free version does not work** |
| Layer-wise anatomy of relevance judgment | published: [2504.07898](https://arxiv.org/abs/2504.07898) (TOIS 2025), three-stage process via activation patching, covers pointwise and pairwise. Also [MechIR](https://arxiv.org/pdf/2501.10165), a mech-interp framework for IR |
| Linear composition of predicate directions (d(A∧B) from d(A), d(B)) | not found. Survives the survey and is the one positive internals result we have |

### Two directions to avoid, with evidence

**Steering vectors / CAA.** Steerability is "highly variable across inputs",
"many behaviours turn out to be unsteerable, even when sweeping across all layers
and strengths", and vectors are brittle to reasonable prompt changes. There is a
paper titled *A Sober Look at Steering Vectors for LLMs*. This is the documented
default, and it explains §4.5 as a known failure mode rather than a weak
construction on our part.

**Sparse autoencoders.** The most hyped direction and the most likely to burn a
week. A majority of SAE neurons show low interpretability *or* low steerability
*or* both; a 65k-latent dictionary failed to represent **27–45%** of concepts from
standard benchmarks; sparsity pressure fragments broad concepts into narrow
shards; features are "incomplete and dataset dependent".

**What is solid:** activation patching (causal, not merely correlational, the
method the relevance paper used) and linear probing for the narrow question
"is X decodable here".

**The gap:** both literatures treat the other side as a black box. The ER/LLM
papers use FAISS or Sparkly offline and hit the model via API. The DB papers
(LOTUS, CADENZA, Palimpzest) treat the model as an opaque UDF and optimize
*around* it. Nobody connects model internals to the query engine.

What is genuinely ours: the index-served learned proxy executing as N1QL on a GSI
vector index, with blocking and scoring pushed into the database, plus the
certified cascade (Wilson bounds). FIRST has scores without guarantees; LOTUS has
guarantees without scores.

---

## 8. Open questions

0. **Wire early exit into `sem_filter`.** The most actionable item here, and it
   needs no new research: train the cascade proxy on LLM hidden states at a
   shallow layer instead of `bge` embeddings (§4.8). Measured at generation-level
   quality for ~1/3 the cost under realistic batching. Requires picking the exit
   layer per predicate, since depth tracks predicate difficulty.
1. **Does the predicate algebra extend to negation?** Written and launched but
   **never completed**: the run was cancelled mid-encoding three times by Modal
   client disconnects. `scratchpad/modal_negation.py` is ready to rerun and tests
   whether d(A ∧ ¬B) ≈ d(A) − d(B) across all four quadrants, with the
   `w_A`-alone and `w_B`-alone controls. If subtraction works alongside the sum
   in §4.6, most of boolean predicate algebra is available in activation space.
   This is the cheapest remaining extension of the one positive result.
2. **Composition with balanced predicate pairs.** The strongest open *research*
   thread. §4.6 and §4.7 are positive but rest on one dataset and one conjunction
   pair each, and in the rare case one predicate nearly carried the result.
   Several pairs where neither half dominates would separate "composition works"
   from "one predicate happens to be sufficient". §4.9 now gives a mechanism to
   test against: composition should degrade as predicate pairs become less
   orthogonal.
3. **Rerun the transfer test** with two hard datasets (Walmart-Amazon, WDC
   Products) to see whether a domain-general matching direction actually exists.
4. **Head-to-head Qwen vs Gemini** on identical samples. If an 8B matches a
   frontier API on semantic operators, the cost model changes.
5. **Semantic selectivity estimation.** Cardinality estimation is the oldest
   problem in query optimization and every semantic-operator system needs it;
   CADENZA explicitly punts cross-operator optimization for want of it. Our own
   density rule (§2.5) needs exactly this.
6. **Why does batched adjudication decline past k≈24?** Not lost-in-the-middle,
   not an output budget, not positional adjacency. Still unexplained.
7. **Prompt cache as a join build side.** Framing is appealing and mechanically
   feasible, but nothing is built and the novelty is unverified.

---

## 9. If I had to compress this to five sentences

The two biggest wins for semantic joins were free and involved no change to the
model: **align the predicate to what you actually want** (+0.102 F1) and **use
mutual kNN blocking** (recall 0.889 → 1.000). **Batched adjudication is both
cheaper and better than per-pair** at the right batch size (~24 candidates,
24x fewer calls), but the right *strategy* depends on match density, which is
estimable before any LLM call. Most of the sophisticated interventions I tried,
chunking, hybrids, prompt-layout fixes, latent-knowledge probing, zero-shot
predicate directions, operators-as-task-vectors, distillation onto a small model
were washes or losses, and the recall ceiling I spent the most effort
attacking turned out to be substantially **a property of our labels, not of the
model**. Two internals results did survive every control: **predicate directions
compose linearly** (because they are near-orthogonal), paying off precisely where
labels are scarce, and **the decision is computed early enough that you can stop
the forward pass at roughly a third depth and get generation-quality judgments
for a third of the cost**: the one finding here that is immediately worth
shipping.

The meta-pattern, stated plainly because it recurred: **every time an internals
hypothesis excited me, it was either already published, documented as
unreliable, or an artifact of my own harness.** The engineering levers that
actually moved numbers were unglamorous and involved no model internals at all.

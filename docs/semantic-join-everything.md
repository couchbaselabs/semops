# Semantic Join: complete experimental record

Everything we ran on the semantic join operator, in one place. This is a source
document rather than a narrative: each experiment gets its motivation, setup,
full results, and what it settled. Nothing is omitted for being negative or
embarrassing, and where a result was later corrected the original is kept
alongside the correction.

Read section 1 for context and then jump to whichever stage you need.

---

## Contents

1. [What semops is](#1-what-semops-is)
2. [The semantic join operator](#2-the-semantic-join-operator)
3. [Datasets, models, protocol](#3-datasets-models-protocol)
4. [Stage 1: blocking](#4-stage-1-blocking)
5. [Stage 2: adjudication](#5-stage-2-adjudication)
6. [Stage 3: resolution](#6-stage-3-resolution)
7. [End to end results](#7-end-to-end-results)
8. [Directions that failed](#8-directions-that-failed)
9. [Bugs, corrections, retractions](#9-bugs-corrections-retractions)
10. [Open questions](#10-open-questions)
11. [Appendix: prompts, SQL, configs](#11-appendix)

---

## 1. What semops is

semops is a library for semantic operators over Couchbase: WHERE, JOIN, GROUP BY
and dedup where the predicate is written in English and an LLM does the judging,
with a vector index making it affordable. Four operators are built (`sem_filter`,
`sem_join`, `sem_dedup`, `sem_group_by`); `sem_topk`, `sem_agg` and `sem_map` are
not.

Two properties matter for what follows.

**Scoring runs inside the cluster.** A logistic proxy's score is a dot product, so
it can be evaluated as SQL++ with `VECTOR_DISTANCE`, and embeddings never cross the
wire. Measured: 50,000 vectors reduced to 2,500 on the wire, about 95% less data.

**The cascade certifies its own thresholds.** A sample is labelled by the LLM, a
cheap proxy is fitted on it, and Wilson score lower bounds at confidence 1 - delta/2
pick an accept threshold and a reject threshold such that precision and recall
targets provably hold on the sample. Rows in between escalate to the LLM. If a
target cannot be certified the threshold degrades to plus or minus infinity, so the
operator escalates rather than mislabels.

Everything below is about `sem_join` specifically.

---

## 2. The semantic join operator

### 2.1 What it does

```python
sess.rows(tickets).sem_join(
    "known_issues",
    "the ticket is caused by this known issue",
    block_k=24,
)
```

Two relations, a predicate in English, and the output is the set of pairs satisfying
it. The naive implementation is a nested loop that asks the LLM about every pair,
which is |R| x |S| calls and immediately unaffordable: on our 20NG configuration
that is 840 calls for 12 left rows against 70 right rows, and on a 1081 x 1092
product join it is 1.18 million.

### 2.2 The three stage decomposition

This framing is the organising idea of the whole investigation, and it was arrived
at late, after the stage 1 and stage 2 experiments came back mostly negative.

```
for each left row a in R:
    C(a)      = block(a, S)                          # candidate right rows
    verdicts  = { (a,b, judge(a,b)) : b in C(a) }    # per-pair LLM judgments
output        = resolve( all verdicts )              # which verdicts to keep
```

**Stage 1, blocking.** A vector index proposes candidates so the LLM never sees the
cross product. Studied heavily in the entity resolution literature.

**Stage 2, adjudication.** The LLM judges candidates. This is where essentially all
the LLM-era work goes: LOTUS proxies, ComEM's match/compare/select taxonomy, FIRST's
logit reading, Trummer's batching.

**Stage 3, resolution.** Deciding which verdicts to keep. Almost always a threshold,
and almost never discussed.

The reason stage 3 is not a formality: stages 1 and 2 both operate on a single pair
or a single left row's candidate list. Neither has a view of the join as a whole. If
right row `b` is plausible for eleven different left rows, a per-pair judge accepts
all eleven because each decision is locally reasonable, and a global threshold cannot
fix that because it is also a per-pair rule. Only resolution can use the fact that a
join has cardinality structure.

### 2.3 End to end quality, headline

Real LLM judging, `gemini-flash-lite-latest`, against labels:

| operator | vector search alone | semops |
|---|---|---|
| `sem_join` (20 Newsgroups, best config) | 0.540 | 0.667 |

That 0.667 is the number currently in the README, and sections 5 and 6 both improve
on it.

---

## 3. Datasets, models, protocol

### 3.1 Abt-Buy

`matchbench/Abt-Buy` on HuggingFace, a standard Magellan entity-matching benchmark.
Carries all the main results.

```
tableA           1081 rows  (name, description, price)
tableB           1092 rows
cross product    1,180,452 candidate pairs
ground truth     1028 matching pairs, pooled from train + valid + test
test.csv         1916 labelled pairs, 206 positive
```

Cardinality structure, measured rather than assumed:

```
left-row  match counts : {1: 1004, 2: 12}
right-row match counts : {1: 1020, 2: 4}
matched left rows      : 1016 of 1081, so 65 (6%) match nothing
fraction of matched left rows with exactly one match: 0.988
```

Token-overlap baseline (Jaccard over lowercased alphanumeric tokens, best threshold):
**F1 0.736**. So the dataset is not trivially solvable and model results on it mean
something.

### 3.2 20 Newsgroups self-join

Seven categories, self-join, predicate "these two posts are about the same topic",
ground truth "same newsgroup". Sizes vary by experiment: 70 docs (10 per category),
105, 210, 420, 700.

**Important caveat, established in section 5.4:** this ground truth is substantially
wrong for the predicate as written. 20NG numbers are valid as relative comparisons
between strategies and are not valid as absolute quality.

### 3.3 DBLP-ACM

Dropped. Token-overlap baseline reaches **F1 0.974**, so a one-line baseline nearly
solves it and it cannot support conclusions about model behaviour. Discovered only
after briefly building a transfer claim on it (section 9.2).

### 3.4 Models

```
embedding    BAAI/bge-small-en-v1.5, 384-dim, fastembed, local
LLM judge    gemini-flash-lite-latest via the OpenAI-compatible endpoint
open model   Qwen/Qwen3-8B, 36 layers, hidden 4096, bfloat16
small model  Qwen/Qwen3-0.6B for distillation
hardware     Modal, single NVIDIA L4
```

Two Qwen3 traps that corrupt results silently rather than erroring:

```python
# 1. instruction-tuned model needs the chat template, or it does raw completion
# 2. thinking mode is ON by default and eats a small max_new_tokens budget
prompt = tok.apply_chat_template(
    [{"role": "user", "content": body}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
answer = re.sub(r"<think>.*?</think>", " ", raw, flags=re.S)
```

Model choice note: Qwen3-8B over the newer Qwen3.5-9B deliberately, because 3.5 uses
`Qwen3_5ForConditionalGeneration` with a nested config and this work depends on
clean per-layer hidden state extraction.

### 3.5 Determinism

Five distinct prompts, ten identical repetitions each, temperature 0, on the batched
adjudication task:

```
5 of 5 prompts returned byte-identical output on all 10 repetitions
5 of 5 parsed match-sets identical
```

Consequence: repeating a prompt buys nothing, so all statistical power must come from
distinct examples. This set the sample sizes in section 5.5 and matches Liu et al.'s
methodology (greedy decoding, no seeds, power from 500 to 2655 examples per
condition).

### 3.6 Evaluation protocol

Three conventions, each introduced after an early version produced a number I could
not defend.

**Ground truth pooled across splits.** Scoring against test positives alone counts a
correctly recovered train-split match as a false positive.

**Thresholds tuned on validation positives only.** Never on test, never by maximising
the reported metric. An early version swept the threshold against the evaluation set,
which flatters whichever method has the most knobs.

**Scoring covers the whole join.** All 1081 left rows against pooled truth. An early
version scored only left rows that have a match, which excludes exactly the cases
where a force-pairing resolver does damage.

---

## 4. Stage 1: blocking

### 4.1 Directionality: one-directional vs mutual kNN

**Motivation.** The implementation counted `(a,b)` as a candidate when `b` was in
`a`'s top-k. This is asymmetric: if `a` is a generic record with many near
neighbours, its true match may sit at rank 30 while `a` sits at rank 2 in the true
match's own neighbourhood.

**Method.** Admit a pair when either row appears in the other's top-k (reciprocal or
mutual kNN from the blocking literature). 20NG self-join, blocking recall against the
true pair set.

**Result.**

| k | one-directional | mutual |
|---|---|---|
| 4 | 0.213 | 0.371 |
| 8 | 0.417 | 0.593 |
| 16 | 0.750 | 0.898 |
| 24 | 0.889 | **1.000** |
| 32 | 0.923 | 1.000 |

**Settled.** Candidate recall reaches 1.000 at k=24 against 0.889 one-directional, at
zero API cost. This is free and should always be on.

**Consequence.** It reversed a position I had argued for days, that index-served
blocking was the lever most likely to move join quality. With blocking at 1.000 and
the LLM still converting only half to two thirds of the pairs handed to it,
retrieval was demonstrably not the binding constraint.

### 4.2 Blocking recall vs k on Abt-Buy

**Method.** bge-small over the full opposite table, no pre-filtering, measured on
held-out test positives.

| k | blocking recall |
|---|---|
| 1 | 0.767 |
| 5 | 0.951 |
| 10 | 0.990 |
| 20 | 1.000 |
| 50 | 1.000 |

**Reference point.** Sparkly, the SOTA blocker used by ComEM, reports recall@10
between 86.6% and 99.96% across datasets. Ours sits inside that range untuned, so the
retrieval side is competitive with the literature without work.

### 4.3 Which representation makes the best join key?

**Motivation.** semops already pushes a *filter* predicate into the index, and it
works (section 11.2 has the SQL and the EXPLAIN). The natural next step is a join.
There was direct encouragement: probing Qwen3-8B on Abt-Buy pairs, "these two records
are the same entity" is linearly decodable from the layer-20 hidden state at **F1
0.853**, essentially equal to full generation (0.855). If the representation contains
the answer, index the representation.

**Method.** Encode both tables row by row with prompt `Product: {text}`, take the
last-token hidden state at layers 8, 12, 16, 20, 24, 28, 36. Blocking recall over the
full 1081 x 1092 cross product on 206 held-out test positives, no candidate
pre-filtering.

**Result.**

| representation | dim | R@1 | R@2 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|---|
| **bge-small** | 384 | **0.772** | 0.883 | 0.951 | 0.990 | 1.000 | 0.853 |
| qwen-L8 | 4096 | 0.073 | 0.092 | 0.121 | 0.184 | 0.286 | 0.112 |
| qwen-L12 | 4096 | 0.058 | 0.083 | 0.102 | 0.165 | 0.233 | 0.095 |
| qwen-L16 | 4096 | 0.044 | 0.063 | 0.087 | 0.117 | 0.184 | 0.075 |
| qwen-L20 | 4096 | 0.024 | 0.034 | 0.073 | 0.102 | 0.175 | 0.052 |
| qwen-L24 | 4096 | 0.019 | 0.024 | 0.068 | 0.078 | 0.112 | 0.045 |
| qwen-L28 | 4096 | 0.019 | 0.044 | 0.068 | 0.078 | 0.126 | 0.048 |
| qwen-L36 | 4096 | 0.073 | 0.083 | 0.141 | 0.180 | 0.218 | 0.108 |

**Settled.** A purpose-trained 384-dim embedder beats an 8B model's 4096-dim
activations by about **30x at R@1**. Every layer is unusable. Note the
non-monotonicity: layer 20, the best layer for *judging* a pair, is the worst for
*retrieving* a match. The extremes (L8, L36) do marginally better than the middle.

**Interpretation.** Consistent with why LLM2Vec, E5-Mistral and RepLLaMA exist at
all. A causal decoder's last-token state is optimised for next-token prediction, is
known to be anisotropic and dominated by a few high-magnitude directions, and is not
a metric space without substantial retraining.

### 4.4 Can an indexable match metric be learned?

**Motivation.** The raw states failing does not settle it. An index requires the
scoring function to factor as `f(a) . g(b)`, since it compares one stored vector
against one query vector. Any bilinear form qualifies: with
`score(a,b) = h(a)^T W h(b)` and `W = L L^T`, the score is
`(L^T h(a)) . (L^T h(b))`, so you store `L^T h(b)` and query with `L^T h(a)`.

**Attempt 1, naive.** Rank 128, logistic objective on labelled training pairs, hard
negatives mined as the top-20 cosine neighbours of each positive's left row, class
weighting, Adam at lr 3e-3, 300 full-batch epochs.

| representation | raw R@1 | learned R@1 |
|---|---|---|
| bge-small | 0.772 | 0.184 |
| qwen-L8 | 0.073 | 0.000 |
| qwen-L20 | 0.024 | 0.000 |
| qwen-L36 | 0.073 | 0.010 |

Learning destroyed every representation. Had I stopped here I would have written this
up as "match predicates do not factor", which was the conclusion I was already
inclined toward, supported by numbers that in fact showed something else.

**The control that caught it.** A random projection to the same rank, included on the
reasoning that a claim about a learned 128-dim metric needs comparison against an
arbitrary 128-dim metric:

```
bge-small, random projection rank 128 : R@1 0.728   (raw 0.772)
qwen-L20,  random projection rank 128 : R@1 0.024   (raw 0.024)
```

A random matrix barely dents bge (0.772 to 0.728) while my learned one collapsed it
to 0.184. Learning was destroying information a random projection preserved, which is
overfitting: 4096 x 128 is half a million parameters against a few hundred positives,
and 300 full-batch epochs at 3e-3 with no early stopping memorises them.

**Attempt 2, fair.** Three changes. Initialise `L` at the top-128 principal
directions of the pooled row representations, so training *starts* at approximately
raw cosine geometry. Learning rate 1e-4 with weight decay 1e-4. Validation recall@1
every 20 epochs with early stopping after three non-improving evaluations, reporting
the best checkpoint.

```
bge-small : validation R@1   init 0.762  ep20 0.757  ep40 0.762  ep60 0.757
                             best 0.762, early stopped
qwen-L20  : validation R@1   init 0.015  ep20 0.053  ep40 0.039  ep60 0.044
                             best 0.053, early stopped
```

Full-join confirmation with the fairly trained projection: bge with learned
projection reaches blocking R@1 **0.743** against **0.773** for raw cosine, so it is
slightly harmful even when trained carefully.

**Settled, and this is the main stage 1 conclusion.**

> The same model, at the same layer, decodes "same entity" at F1 0.853 when both
> records appear in one prompt, and provides essentially no signal when the two
> records are encoded separately and compared by any bilinear form. The predicate is
> not a property of either row that can be read off independently and combined. It
> lives in the interaction between them, which in retrieval terms means it is
> cross-encoder shaped, and cross-encoders do not factor.

**Consequence.** This is a design constraint, not a tuning target. Blocking for a
semantic join must remain a cheap, generic, predicate-agnostic similarity search, and
adjudication cannot be eliminated by a smarter index. This is what redirected
attention to stage 3.

### 4.5 GSI index parameters

From earlier live-cluster sessions, included because they bound what blocking can do
in production.

`nProbes` (IVF cells probed) controls ANN recall against exact:

| rows | nProbes | recall vs exact |
|---|---|---|
| 2k | 1 | 0.685 |
| 2k | >= 2 | 1.000 |
| 50k | 8 | 0.713 |
| 50k | 64 | 1.000 |

`topNScan` is the scan depth and must be at least the LIMIT, or you silently get
fewer rows than asked:

| topNScan | rows returned under LIMIT 200 | recall |
|---|---|---|
| 10 | 20 | 0.10 |
| 50 | 100 | 0.50 |
| 200 | 200 | 1.00 |

`INCLUDE` covered scans: 4.0x faster and 51x less data on a 100-row probe (43.9 ms /
862 KB down to 10.9 ms / 16.9 KB).

`UNION ALL` probe batching: 2.4x to 2.6x for 8 to 32 probes, which is the mechanism
that would let a capped scan be partitioned.

---

## 5. Stage 2: adjudication

### 5.1 Per-pair vs batched (Trummer)

**Motivation.** Per-pair judging costs one call per candidate, so cost is linear in
candidate count and couples directly to blocking recall: widening k to improve recall
makes adjudication proportionally more expensive.
[Trummer, arXiv:2510.08489](https://arxiv.org/abs/2510.08489) observes that putting a
query row and all its candidates in one prompt decouples call count from candidate
count.

**Method.** Both implemented on the 20NG self-join, blocking held fixed, candidate
count swept. 12 left rows, 70 right rows, **one-directional top-k blocking**. Prompt
in section 11.1.

Note for comparing against 5.3: that experiment used *mutual* kNN blocking with
recall pinned at 1.000, so its candidate sets are higher quality and its numbers are
not directly comparable to the table below. Batched-at-34 scores 0.690 there against
batched-at-32 scoring 0.591 here, and the difference is the blocking, not the batch
size.

**Result.**

| adjudication | candidates | calls | precision | recall | F1 |
|---|---|---|---|---|---|
| per pair | 8 | 96 | 0.682 | 0.278 | 0.395 |
| per pair | 16 | 192 | 0.754 | 0.454 | 0.566 |
| per pair | 24 | 288 | 0.740 | 0.528 | 0.616 |
| batched | 8 | 12 | 0.862 | 0.231 | 0.365 |
| batched | 16 | 12 | 0.849 | 0.417 | 0.559 |
| **batched** | **24** | **12** | 0.933 | 0.519 | **0.667** |
| batched | 32 | 12 | 0.922 | 0.435 | 0.591 |
| batched | 48 | 12 | 0.860 | 0.398 | 0.544 |
| batched | 69 | 12 | 0.920 | 0.213 | 0.346 |

**Settled.** Batched at 24 candidates reaches 0.667 in 12 calls against 0.616 in 288
calls for per-pair, so it is both better and roughly **24x cheaper**. Quality traces
an inverted U over batch size, peaking near 24.

**The decline is driven entirely by recall.** Precision stays between 0.86 and 0.93
across the whole sweep while recall rises to 0.519 and falls to 0.213.

### 5.2 Why does batched quality decline past ~24? Three hypotheses, all refuted

This consumed a large share of total effort and produced no explanation.

**Hypothesis A: context window.** Refuted by arithmetic. At 69 candidates the prompt
is roughly 20,000 characters, trivially inside the limit, and the decline is already
visible at 32.

**Hypothesis B: lost in the middle.** [Liu et al.](https://arxiv.org/abs/2307.03172)
predict accuracy highest at the ends of a context and degrading in the middle.
Refuted in section 5.6b: under hard negatives there is **no middle penalty at all**.
Positions 12 through 48 are statistically indistinguishable and the only depressed
slot is position 1, which is the opposite end from where the hypothesis puts the
damage.

**Hypothesis C: a fixed output budget.** If the model rations its answer to a fixed
size, recall must fall as true match count rises. Tested by holding candidates at 24
and varying planted true matches:

| true matches planted | avg candidates named | avg correct | recall | precision |
|---|---|---|---|---|
| 1 | 1.80 | 0.65 | 0.65 | 0.36 |
| 2 | 1.55 | 1.35 | 0.68 | 0.87 |
| 4 | 2.65 | 2.55 | 0.64 | 0.96 |
| 8 | 4.60 | 4.60 | 0.57 | 1.00 |
| 12 | 7.10 | 7.10 | 0.59 | 1.00 |

Refuted: the model scales its output from 1.80 to 7.10 named candidates as the true
count rises, so there is no fixed budget. Recall is roughly flat at 0.57 to 0.68.

**Still open.** The batch-size effect is real, replicated and unexplained. This is
the most interesting loose end in the whole investigation.

**Bonus finding from Hypothesis C.** At m=1 precision is **0.36**: the model names
1.80 candidates when only one is correct. It will not answer "none", so when the
right answer is "nothing here matches" it invents roughly one false positive per
prompt. Precision climbs monotonically to 1.00 as true match count rises. This
single observation explains the density flip (5.3) and the LLM's failure in the fair
fight (6.3).

### 5.3 Does splitting the prompt help? No, monotonically

**Motivation.** If wide candidate lists degrade quality, split them and union the
results.

**Method.** Blocking pinned at 1.000 via mutual kNN k=24, giving 34.4 candidates per
row on average. Only chunk size varied, so match density is constant by construction.

| candidates per prompt | prompts per row | calls | precision | recall | F1 |
|---|---|---|---|---|---|
| **34 (all at once)** | 1.2 | **15** | **0.937** | **0.546** | **0.690** |
| 17 | 2.3 | 28 | 0.862 | 0.519 | 0.647 |
| 12 | 3.5 | 42 | 0.824 | 0.519 | 0.636 |
| 8 | 4.8 | 58 | 0.871 | 0.500 | 0.635 |
| 4 | 9.1 | 109 | 0.797 | 0.472 | 0.593 |

**Settled.** Monotonically worse on precision, recall and F1 while costing up to 7x
more calls. Not a density effect, since a fixed candidate set is being partitioned.

**Interpretation.** The model judges *comparatively*. A wider candidate list gives it
more context for calibrating what counts as a match, and removing that context pushes
it toward guessing. Consistent with the m=1 precision collapse.

### 5.4 The strategy flips with match density

**Motivation.** 5.1 was run on 20NG where each left row has about nine true matches
among 34 candidates. Abt-Buy is the opposite: about 0.6 matches per record among 2.5
candidates.

**Result.**

| dataset | matches per left row | batched SELECT | per-pair MATCH |
|---|---|---|---|
| 20 Newsgroups | ~9 of 34 candidates | wins, ~30x cheaper | loses |
| Abt-Buy | ~0.6 of 2.5 candidates | 0.900 | **0.945** |

Abt-Buy detail: batched P 0.990 / R 0.825, per-pair P 0.965 / R 0.925. Batched keeps
higher precision as it does on 20NG, but when matches are sparse the recall deficit
dominates and the ranking inverts. The batched picks were a strict subset of the
per-pair picks on this data, so union recovered exactly per-pair and intersection
exactly batched.

**Settled, and this is a planner rule.** Match density is estimable from the blocking
score distribution before any LLM call. Dense, batch. Sparse, judge per pair.

**Mechanism.** The m=1 collapse in 5.2. Per-pair judging is structurally the m=1
regime, one candidate per prompt, which is exactly where precision is 0.36.

**Caveat.** The Abt-Buy figures here came from a stratified 200-record subsample with
a 60% positive rate against the natural 28%, so they are not comparable to the
whole-join numbers in section 6. This is why section 6 re-ran the comparison rather
than quoting these.

### 5.5 Predicate wording, and a labelling error of ours

**The largest stage 2 effect came from changing the question, not the technique.**
Same candidates, same batched strategy, same 12 calls.

| predicate | precision | recall | F1 |
|---|---|---|---|
| "the two documents are about the same newsgroup topic" | 0.931 | 0.500 | 0.651 |
| "come from the same newsgroup, even if they discuss different subjects" | 0.967 | 0.537 | 0.690 |
| the above + the list of the seven newsgroup names | 0.957 | **0.620** | **0.753** |

Recall rises 24% relative while precision holds, so this is not the model becoming
permissive. Supplying the label space is legitimate: in a real join you know your
schema.

**Why the effect existed at all.** The ground truth is "these two posts share a
newsgroup" and the prompt asked "are these two posts about the same topic". Different
predicates. Quantified two ways.

Same-newsgroup pairs have mean cosine 0.617 against 0.535 for different-newsgroup
pairs, barely separated, and **10.5% of same-newsgroup pairs are less similar than
the average different-newsgroup pair.**

More persuasively, the least-similar pairs the ground truth calls matches:

```
[comp.graphics] cosine 0.485
  A: "We are interested in purchasing a grayscale printer that offers good
      resolution for grayscale medical images... under $5000?"
  B: ".DXF can not be changed over to .IFF format, what it can be changed to
      is an object format used by one of the 3D programs on the Amiga."

[rec.sport.hockey] cosine 0.540
  A: "And Dino Ciccarelli and Ray Sheppard and so on and so on..... : )"
  B: "Stanley Cup Playoff stats through games played Sunday, April 25 1993"
```

Both members share a newsgroup. Neither pair is about the same topic. When the model
refused these it was correct and our label was wrong.

**Consequence.** The recall ceiling of roughly 0.55 that I spent most of the effort
attacking, and attributed to the model, was substantially a property of the labels.
Relative comparisons between strategies survive since all were scored against the
same labels. Absolute recall on 20NG is not a measurement of model capability.

### 5.6 Candidate position within the prompt

**Motivation.** Batched adjudication puts candidates in an ordered list. Liu et al.
establish lost-in-the-middle on multi-document QA and key-value retrieval.

**Method.** Needle-in-a-haystack adapted to the join task, **paired**. Each trial
constructs a query document from newsgroup C, one true match from the same newsgroup,
and k-1 distractors from other newsgroups. The same candidate set is re-tested with
the match relocated to each position, so document difficulty is constant and any
difference is position alone. Success is the model naming the position where the
match sits.

A pilot at n=20 per cell gave a suggestive pattern, but SE on a proportion near 0.5
at that n is about 0.11 across roughly fifteen comparisons. Since the model is
deterministic (3.5), repeated sampling adds nothing, so the rerun used **150 distinct
paired trials per cell, 1,350 calls**, analysed with McNemar's exact test on
discordant pairs and bootstrap 95% CIs.

**Result.**

| candidates | first | middle | last |
|---|---|---|---|
| 8 | 0.433 [0.353, 0.513] | 0.573 [0.493, 0.653] | 0.587 [0.507, 0.660] |
| 24 | 0.373 [0.293, 0.453] | 0.573 [0.493, 0.653] | 0.367 [0.293, 0.447] |
| 48 | **0.160** [0.107, 0.220] | **0.480** [0.400, 0.560] | 0.300 [0.227, 0.373] |

McNemar, first vs middle discordant counts: 3 vs 24 at k=8, 3 vs 33 at k=24, **3 vs
51** at k=48, all p < 0.0001. At k=8 middle and last are indistinguishable (p=0.83)
and only position 1 is depressed.

**Position 1 is depressed. The rest of the claim did not survive replication.** I
wrote up the table above as an inverted U, the inverse of lost-in-the-middle, with
the middle as the best place for a match. The distractors here are drawn from other
newsgroups, which makes them easy, and the shape is an artifact of that. Section
5.6b re-ran it with hard negatives on two datasets. **Only the position-1 penalty
replicates; the inverted U does not.** The corrected statement is in 5.6b and that
is the one to cite.

Average candidates named is constant across positions within a given k (1.37, 1.37,
1.53 at k=48), so the position-1 effect is a discrimination failure rather than the
model naming fewer things. That part holds up in the rerun.

### 5.6b Position, settled under hard negatives

**Motivation.** 5.6 drew distractors from other newsgroups. Real blocking hands the
model near misses, so the whole result could be an artifact of easy distractors. It
also rested on one dataset.

**Method.** Same paired relocation, but distractors are **blocking neighbours with
true matches removed**, and their order is shuffled so there is no sortedness signal
for the needle's position to violate. Two datasets, three list lengths, 150 paired
trials per cell, McNemar's exact test. Abt-Buy k=48:

| position | rel% | recall | 95% CI | vs middle |
|---|---|---|---|---|
| 1 | 2% | **0.553** | [0.473, 0.627] | 2 vs 59, p < 0.0001 |
| 12 | 25% | 0.960 | [0.927, 0.987] | ns, p=0.34 |
| 24 | 50% | 0.933 | [0.893, 0.973] | reference |
| 36 | 75% | 0.933 | [0.893, 0.967] | ns, p=1.00 |
| 48 | 100% | 0.933 | [0.893, 0.967] | ns, p=1.00 |

Same shape at k=8 (0.620 at position 1, 0.953 to 0.993 everywhere else) and k=24
(0.653 against 0.973 to 0.987). On 20NG k=48 the spread is 0.500 to 0.633 and **no
comparison survives multiplicity**; position 1 is not even significantly worse
there.

**Settled.** **One poisoned slot and a flat tail.** Position 1 is depressed on
Abt-Buy at every length, p < 0.0001. Every other position is indistinguishable from
the middle. There is no inverted U, the middle is not special, and the last position
is not special. 5.6's shape was easy distractors.

**Practical magnitude, revised sharply upward.** 5.6 called this minor, worth under
a point of aggregate recall. That was also wrong, and wrong because a
needle-in-a-haystack recall measurement is the wrong instrument: it scores one slot
in isolation. Measured as join F1 on real candidate sets, moving the best candidate
out of slot 1 is worth **+0.294 F1** (5.6c). The mechanism is precision, not recall:
a poisoned slot 1 makes the model over-accept across the whole list.

### 5.6c What candidate order is worth on the real join

**Method.** Identical candidate sets everywhere (bge top-10, blocking recall 0.985),
all 1081 left rows, pooled ground truth, only the arrangement inside the prompt
varies.

| arrangement | what sits in slot 1 | P | R | F1 | pairs | named |
|---|---|---|---|---|---|---|
| sim_desc | the best candidate | 0.384 | 0.818 | 0.523 | 2191 | 2.03 |
| demote_best | the 2nd-best candidate | 0.390 | 0.928 | 0.549 | 2446 | 2.27 |
| random | a random candidate | 0.560 | 0.946 | 0.703 | 1736 | - |
| best_middle | a mid candidate | 0.655 | 0.909 | 0.762 | 1425 | - |
| swap_1_k | the worst candidate | 0.676 | 0.953 | 0.791 | 1449 | 1.34 |
| **sim_asc** | the worst candidate | 0.711 | 0.960 | **0.817** | 1388 | 1.28 |
| dummy_first | a decoy, cannot match | 0.324 | 0.980 | 0.487 | 3109 | 2.88 |

**Attribution.** `swap_1_k` puts the worst candidate in slot 1 and leaves the rest of
the ordering untouched, reaching 0.791 of the 0.817. So **+0.269 is attributable to
slot-1 contents and only +0.026 to gradient direction.** The fix is a two-element
swap; the list does not have to be reversed.

`dummy_first` bounds what sacrificing the slot can buy and is *worse* than the
baseline at 0.487, because the decoy displaces a real candidate and the model then
names 2.88 pairs per row. Sacrificing the slot is not the move. Putting your worst
real candidate there is.

**Settled.** Candidate order is worth **+0.294 F1 at zero marginal cost**, which is
larger than what the adjudicating LLM itself is worth over a cosine baseline
(+0.243, section 6.3). Presentation is a plan knob, not a prompt detail.

### 5.7 Query placement

**Motivation.** In our prompt the query text sits immediately above `CANDIDATES:`
followed by `1. ...`, so candidate 1 abuts the query and may be partially absorbed
into it. Suggests moving the query below the list.

**Test 1, position 1 only**, k=48, needle always first, 150 paired trials:

| layout | recall at position 1 | 95% CI | avg named |
|---|---|---|---|
| baseline, query first | 0.140 | [0.087, 0.200] | 1.69 |
| explicit delimiter between query and list | 0.240 | [0.173, 0.313] | 1.71 |
| **query last** | **0.453** | [0.373, 0.533] | 0.93 |

Query-last recovers position 1 from 0.140 to 0.453, McNemar 3 vs 50, p < 0.00001. I
reported this as a fix before checking whether it merely moved the problem.

**Test 2, all positions.** It moves the problem.

| position | query first | query last | McNemar |
|---|---|---|---|
| first | 0.153 | **0.465** | 4 vs 49, p<0.00001 |
| middle | **0.583** | 0.326 | 42 vs 5, p<0.00001 |
| last | **0.444** | 0.287 | 32 vs 9, p=0.0004 |
| **overall** | **0.394** | 0.360 | query-first better |

**Settled.** The damage follows whichever end abuts the query text, which supports
adjacency as the mechanism, but relocating it costs more than it recovers. Keep the
conventional layout.

### 5.8 Select-then-match hybrid

**Motivation.** ComEM reports selecting has high precision and low recall while
matching has the reverse, and proposes composing them.

**Method.** Batched SELECT first, then per-pair MATCH on residual candidates ranked
by similarity, with the residual budget swept. Blocking at 1.000.

| configuration | calls | precision | recall | F1 |
|---|---|---|---|---|
| select only | 12 | 0.965 | 0.509 | 0.667 |
| select + match top-4 residual | 60 | 0.813 | 0.565 | 0.667 |
| select + match top-8 residual | 108 | 0.759 | 0.583 | 0.660 |
| select + match top-16 residual | 204 | 0.765 | 0.602 | 0.674 |
| select + match all residual | 368 | 0.773 | 0.630 | 0.694 |

**Settled.** The published trade-off reproduces exactly (recall 0.509 to 0.630,
precision 0.965 to 0.773) but the two effects nearly cancel: **+0.027 F1 for 30x the
calls.** The hybrid does not pay at any budget on this data.

### 5.9 Exit depth

**Motivation.** If the decision is computed mid-network, a truncated forward pass plus
a linear probe could substitute for generation.

**Quality**, single predicate, Qwen3-8B:

| method | F1 |
|---|---|
| full generation | 0.889 |
| probe at layer 8 (22% depth) | 0.867 |
| **probe at layer 12 (33%)** | **0.901** |
| probe at layer 20 (56%) | 0.838 |
| probe at layer 36 (100%) | 0.914 |

**Cost, and a correction.** I first measured at batch size 1 and reported **6.9x**,
which is not how anything is served: at batch 1 generation is memory-bandwidth bound
and the GPU is largely idle, flattering a forward-pass-only method. Re-measured as
throughput, one L4, bfloat16:

| batch | generation | probe L36 | probe L20 | probe L12 | L12 speedup |
|---|---|---|---|---|---|
| 1 | 4.5 rows/s | 12.1 | 20.5 | 30.9 | 6.9x |
| 4 | 12.8 | 26.0 | 43.6 | 66.1 | 5.2x |
| 8 | 18.8 | 29.8 | 51.3 | 78.8 | 4.2x |
| 16 | 22.5 | 32.7 | 56.2 | 85.2 | 3.8x |
| 32 | 22.7 | 28.9 | 49.4 | 75.8 | **3.3x** |

**Settled at 3.3x**, converging on the FLOP ratio of 12 layers out of 36.

**Decomposition at batch 32**, which I had not anticipated: skipping the
autoregressive decode is worth only **1.3x**, truncating the layers a further
**2.6x**. At batch 1 the ordering reversed, because sequential decode on an idle GPU
dominates. Under batching the decode cost amortises away and essentially all durable
benefit is layer truncation, a genuine FLOP reduction batching cannot amortise. The
advice is "exit early", not "avoid generating".

**Saturation.** Throughput saturates near batch 16 on this hardware (generation flat
22.5 to 22.7; probe *declines* 85.2 to 75.8 at batch 32, presumably memory pressure).
Benchmark at 16.

**Caveat.** The probe needs labelled training data, generation does not, so this is
not "probing beats the model". Apples to apples only for a cascade that already
trains a proxy on a labelled sample, which semops does.

**Note.** This contradicts 4.3, where matching peaked at layer 20. Reconciliation:
depth tracks *predicate difficulty*, not operator type. 20NG topic membership is easy
and resolves early; Abt-Buy entity matching is harder and resolves later. Exit layer
is a per-predicate parameter.

**That reconciliation was an inference across three experiments that use different
probe targets, token positions, label definitions and splits, so it could not be
tabulated and should not have been stated as a result.** Section 5.9b replaces all
three with one protocol and measures it directly. The conclusion survives; the
evidence for it is now admissible. **Cite 5.9b, not this section**, for any depth
claim.

### 5.9b Exit depth, unified across four predicates

**Motivation.** 5.9 measured one predicate, and our three layer measurements across
this document (5.9, 4.3, and the anatomy probe in 8.4) use different probe targets,
token positions, label definitions and splits, so **they cannot be put in one
table**. The per-predicate claim at the end of 5.9 was an inference across
incompatible experiments rather than a measurement.

**Method.** One script, one protocol, replacing all three. Every predicate becomes a
binary question in the same prompt shape, ending "Answer only yes or no." The probe
reads the **last token** of that prompt and is a logistic regression fitted on a
fixed **60/40 split, train side only**. The generation baseline answers the **same
prompt on the same test split**, parsed tolerantly, scored with the same metric.
Nine depths, four predicates, n=800 each (480 train, 320 test), bootstrap 95% CIs on
every F1, Qwen3-8B on one L4 in bfloat16 at batch 16, the saturation point from 5.9.

Predicates span operator type crossed with difficulty, since the claim under test is
that depth tracks difficulty rather than operator type: `match_product` (Abt-Buy,
same product), `match_topic` (AG News, same subject area), `filter_sentiment`
(Rotten Tomatoes, is negative), `filter_topic` (AG News, is business).

**Two false starts, both recorded in full below and in 9.6.** The pair predicate was
first 20NG "same newsgroup", where generation scored P 0.000 R 0.000 because the
question asks about a taxonomy the model cannot observe; that is the label defect of
5.5 and the predicate was replaced. Then the generation baseline itself was broken by
output parsing, which took two further runs to find.

**Result, F1 with bootstrap 95% CI.**

| depth | match_product | match_topic | filter_sentiment | filter_topic |
|---|---|---|---|---|
| 4 (11%) | 0.662 [.601,.718] | 0.494 [.427,.556] | 0.627 [.561,.686] | 0.717 [.634,.789] |
| 8 (22%) | 0.691 [.631,.749] | 0.514 [.440,.577] | 0.661 [.596,.717] | 0.712 [.627,.785] |
| 12 (33%) | 0.730 [.673,.783] | 0.579 [.512,.642] | 0.742 [.686,.793] | 0.708 [.623,.780] |
| 16 (44%) | 0.823 [.775,.866] | 0.545 [.476,.610] | 0.791 [.739,.838] | 0.727 [.644,.798] |
| 20 (56%) | 0.944 [.915,.967] | 0.708 [.648,.761] | 0.871 [.830,.909] | 0.735 [.653,.804] |
| 24 (67%) | **0.978** [.960,.993] | 0.718 [.658,.773] | **0.896** [.859,.928] | 0.752 [.676,.817] |
| 28 (78%) | 0.962 [.939,.981] | 0.704 [.642,.758] | 0.882 [.841,.918] | **0.805** [.734,.866] |
| 32 (89%) | 0.968 [.946,.986] | 0.708 [.650,.761] | 0.885 [.845,.920] | 0.793 [.723,.854] |
| 36 (100%) | 0.969 [.948,.987] | **0.721** [.662,.777] | 0.890 [.851,.925] | 0.787 [.713,.851] |
| generation | 0.945 [.916,.967] | 0.435 [.348,.516] | 0.848 [.807,.888] | 0.502 [.429,.570] |

**Finding 1: the top third of the network contributes nothing, on all four
predicates.** Comparing layer 24 with layer 36: 0.978 against 0.969, 0.718 against
0.721, 0.896 against 0.890, 0.752 against 0.787. Every pair is flat or slightly
down, and every CI overlaps. **Twelve of 36 layers are dead weight for all four
predicates**, which is the cleanest result in this section and the one an optimizer
can act on without knowing the predicate.

**Finding 2: required depth is per-predicate, with a 5x spread.** `filter_topic` has
**no significant depth dependence**: layer 4 at 0.717 and the best layer at 0.805
have overlapping CIs, so an eighth of the network decides it. The other three show a
large step at layer 20 with disjoint CIs (`match_product` 0.823 to 0.944,
`match_topic` 0.545 to 0.708, `filter_sentiment` 0.791 to 0.871) and plateau at 24.
So `l*` is **4 for the easy predicate and 20 to 24 for the rest**, and difficulty
rather than operator type is what selects it: the easy predicate here is a *filter*
and the two hardest are a *join* and a *filter*, so the split is not by operator.

**Finding 3: a probe at 56% to 67% depth matches or slightly exceeds generation at
roughly 2x the throughput.**

| predicate | generation | probe at `l*` | throughput | FLOPs |
|---|---|---|---|---|
| match_product | 0.945 [.916,.967] | 0.944 at L20 [.915,.967] | **1.99x** | 1.8x |
| match_product | 0.945 | 0.978 at L24 [.960,.993] | 1.67x | 1.6x |
| filter_sentiment | 0.848 [.807,.888] | 0.871 at L20 [.830,.909] | **2.35x** | 1.8x |
| filter_sentiment | 0.848 | 0.896 at L24 [.859,.928] | 1.97x | 1.6x |

Stated precisely: at layer 20 the probe is **statistically indistinguishable** from
generation (CIs almost coincide) for half the compute; at layer 24 its point estimate
is higher on both predicates but the CIs still overlap, so the honest word is
"matches", not "beats". **The result is equal quality at 1.6x to 2.4x less work, not
superior quality.**

On `match_topic` and `filter_topic` the probe beats generation outright (0.721
against 0.435, 0.805 against 0.502), but that is generation being weak on those two
rather than the probe being strong, and the probe is itself poor there.

**The caveat that keeps this honest.** The probe needs 480 labels; generation needs
zero. This is apples to apples only for a cascade that already labels a calibration
sample, which ours does. It is **not** the latent-knowledge claim refuted in 8.1.

**Finding 4: depth truncation buys compute, not memory.** Peak allocation is flat
across depths (18.00 GB at layer 4 against 18.37 GB at layer 36 on `match_product`)
because the full weights stay resident. Truncating layers does not let you serve a
larger batch, which is the natural but wrong assumption.

**Finding 5: the instruction is worth more than the architecture choice here.**
Adding "Answer only yes or no." raised generation on `match_product` from 0.683 to
0.945 and simultaneously made it **faster** (10.5 to 14.6 rows/s) because the model
emits two tokens instead of twenty-four. A one-line prompt change beat every
inference-level optimisation in this section on both axes at once. That is worth
stating plainly in any paper that also reports a 1.8x from layer truncation.

**The error that produced two wasted runs, recorded because it is the transferable
part.** The first two versions of this experiment did not instruct the model to
answer yes or no, and scored a positive only when the reply began with "y".
Qwen3-8B reasons aloud:

```
'Yes, **A and B are the same product** - same manufacturer (Sharp) and ...'   -> 1
'To determine if the two product listings refer to the **same product** ...'  -> 0  WRONG
```

A diagnostic at n=200 found **19% to 74% of replies** did not begin with y or n,
depending on predicate, and were therefore all counted as negatives:

| predicate | replies not starting y/n | as measured | tolerant parser | + yes/no instruction |
|---|---|---|---|---|
| match_product | 38/200 (19%) | 0.773 | 0.954 | 0.944 |
| match_topic | 94/200 (47%) | 0.077 | 0.077 | 0.391 |
| filter_sentiment | 122/200 (61%) | 0.713 | 0.682 | 0.835 |
| filter_topic | 149/200 (74%) | 0.542 | 0.542 | 0.536 |

This produced a confident, systematic, four-predicate finding that generation is
precision-heavy and recall-poor (P >= 0.700 everywhere, R 0.114 to 0.645) and that
the probe therefore beats it. **Both were artifacts and both are retracted.** With
the instruction, generation's error direction varies by predicate rather than being
uniformly conservative: `match_product` is balanced (P 0.986, R 0.906),
`filter_topic` massively over-accepts (P 0.338, R 0.974).

**This is the third baseline of mine broken by output handling rather than by
method** (9.1 was the chat template plus thinking mode, 9.5 was diagnosing from
theory instead of logs). A weak baseline is never self-evidently wrong, so it has to
be checked deliberately. The check is cheap and should be routine: **dump raw model
output before scoring it, and report the fraction of replies the parser cannot
classify.** That number belongs in the writeup next to the baseline.

---

### 5.10 Distilling the judge into a small model

**Motivation.** The cascade already spends LLM calls labelling a calibration sample.
Those labels are sunk cost. If a small model can learn the operator from them,
subsequent queries skip the large model.

**Method.** Fine-tune Qwen3-0.6B with a sequence classification head on Abt-Buy
pairs, learning curve over label count.

| labelled pairs | positives | F1 | precision | recall |
|---|---|---|---|---|
| 100 | 13 | 0.176 | 0.124 | 0.306 |
| 250 | 33 | 0.244 | 0.221 | 0.272 |
| 500 | 60 | 0.281 | 0.333 | 0.243 |
| **1000** | **121** | **0.767** | 0.713 | 0.830 |
| 2000 | 222 | 0.823 | 0.763 | 0.893 |
| *reference: Qwen3-8B zero-shot* | | *0.855* | | |

**Settled.** A sharp phase transition rather than a slope, jumping 0.281 to 0.767
between 500 and 1000 labels. The threshold is in **positives**, roughly 120, which is
unsurprising at a 10.8% positive rate. It never crosses over: at 2000 labels the
student is still below the teacher.

**Consequence.** Distillation is not free. Our cascade calibrates on 100 to 300
labels, squarely in the dead zone where F1 is 0.18 to 0.28. A viable student needs a
deliberate spend of 1000+ judge calls and still lands below its teacher.

**Caveats.** I discarded the throughput column entirely: trained in float32 with fixed
128-token padding while probe measurements used bfloat16 with dynamic padding, so
52.8 rows/s is an artifact of my setup. Nothing was tuned (fixed LR, 3 epochs, no
early stopping) where published Abt-Buy fine-tunes reach ~0.89. The curve *shape* and
the positive-count threshold are the trustworthy parts.

---

## 6. Stage 3: resolution

### 6.1 The structure that thresholding discards

Section 3.1 measured the Abt-Buy matching: 98.8% of matched left rows have exactly
one match, right-side degree similarly concentrated. The join is very nearly a
bipartite matching.

Independent thresholding cannot use this, and fails two specific ways. A textually
attractive right row can be accepted as the match for many left rows, since each
decision is locally plausible and the threshold is a per-pair rule. And a correct
pair whose absolute score sits below the cut is discarded even when it is
unambiguously the best available match for both rows.

### 6.2 Five resolvers

All operate on the same candidate set and the same scores.

**threshold** keeps every pair scoring at or above a cut, chosen to maximise F1 on
validation positives.

**top-1** keeps each left row's single highest-scoring candidate. A one-sided
constraint, no solver.

**mutual best** keeps `(a,b)` only when `b` is `a`'s best and `a` is `b`'s. A cheap
two-sided approximation to 1:1, the same reciprocal idea that fixed blocking in 4.1.

**assignment** solves for the pair set maximising total score subject to each row
appearing at most once, via the Hungarian algorithm
(`scipy.optimize.linear_sum_assignment` on the negated score matrix, non-candidates
set to a large negative constant).

**assignment + floor** runs assignment then discards assigned pairs below a floor,
since assignment pairs every left row including the 6% that match nothing. The floor
is swept over achieved assignment scores and chosen on validation.

**Floor sweep bug, recorded.** My first sweep ran quantiles 0.5 to 0.999 of the
assigned scores, which never considers keeping everything, so `assignment + floor`
came back *worse* than plain `assignment`, which is impossible if the sweep is
correct. The corrected sweep starts from a floor that keeps all pairs.

### 6.3 The fair fight

**Motivation.** The question is which stage is worth more, so resolution and
adjudication have to be compared under identical conditions. The earlier Gemini
numbers (5.4) came from a stratified subsample and could not support this.

**Method.** Every cell uses identical blocking: bge-small top-10 per left row, 10,810
candidate pairs out of 1,180,452, blocking recall 0.985. Every threshold and floor
tuned on validation positives. Every cell scored over the whole join, all 1081 left
rows against pooled 1028-pair ground truth. Adjudicator is Qwen3-8B with the batched
SELECT prompt over each left row's ten candidates, 1081 calls, batches of 8,
`max_new_tokens=24`, 378 seconds on one L4.

**Result.**

| scorer | resolver | precision | recall | F1 | pairs | LLM calls |
|---|---|---|---|---|---|---|
| cosine | threshold | 0.745 | 0.210 | 0.328 | 290 | 0 |
| cosine | top-1 | 0.734 | 0.771 | 0.752 | 1081 | 0 |
| cosine | mutual best | 0.920 | 0.683 | 0.784 | 763 | 0 |
| **cosine** | **assignment** | 0.845 | 0.879 | **0.862** | 1070 | **0** |
| cosine | assignment + floor | 0.915 | 0.712 | 0.801 | 800 | 0 |
| Qwen3-8B SELECT | its own yes/no | 0.445 | 0.793 | 0.570 | 1830 | 1081 |
| Qwen3-8B SELECT | assignment | 0.885 | 0.763 | 0.819 | 886 | 1081 |
| Qwen3-8B SELECT | assignment + floor | 0.917 | 0.724 | 0.809 | 811 | 1081 |

**Reading the ingredients** against the same 0.328 baseline (cosine + conventional
independent threshold):

| ingredient added | F1 | change | cost |
|---|---|---|---|
| constraint solver, no LLM | 0.862 | **+0.534** | 0 calls |
| LLM, no constraint solver | 0.570 | +0.243 | 1081 calls, 378 s |
| both | 0.819 | +0.491 | 1081 calls, 378 s |

**Correction to the decomposition above: the ratio is not a publishable number.**
Both "+0.534" and "+0.243" are differences from the cosine + threshold cell, and
that cell is sensitive to how hard its threshold is tuned. A rerun under the same
protocol with a finer threshold grid scores it **0.478 rather than 0.328**, the same
class of grid-coverage error already recorded for the floor sweep in 6.2. The
decomposition then reads:

| ingredient | F1 | vs baseline 0.328 | vs baseline 0.478 |
|---|---|---|---|
| constraint solver, no LLM | 0.862 | +0.534 | +0.384 |
| LLM, no constraint solver | 0.570 | +0.242 | +0.092 |
| both | 0.819 | +0.491 | +0.341 |
| **solver value / LLM value** | | **2.2x** | **4.2x** |

The better-tuned baseline makes our own claim *stronger*, so nothing was being
flattered, but a ratio that moves from 2.2x to 4.2x on a baseline-tuning choice
should not be quoted at all. **Use the two comparisons that need no baseline:**

- cosine + assignment reaches **0.862 with 0 LLM calls**, above every cell that
  uses the LLM.
- adding the LLM on top of the solver **costs 0.043 F1 for 1081 calls**.

Both are direct cell-to-cell comparisons under identical blocking and tuning.

**Three findings.**

1. The constraint solver is worth more than the LLM on this join and costs nothing.
   The magnitude depends on the baseline, as above; the ordering does not.
2. Cosine plus assignment at **0.862** exceeds the 8B model's own adjudication at
   **0.570** on the same candidates.
3. The LLM's marginal contribution on top of the solver is **negative**: 0.862 falls
   to 0.819, so 1081 calls purchase a loss of 0.043 F1.

**Where the gain comes from.** The intermediate resolvers decompose it. Plain top-1,
a one-sided constraint with no solver, already reaches 0.752 from 0.328. Mutual best
adds the second side, reaching 0.784 with much higher precision (0.920) and lower
recall (0.683), since it refuses to commit when two rows disagree about their best
partner. Assignment resolves those disagreements globally instead of dropping them,
which is where the remaining recall comes from (0.683 to 0.879). **Most of the effect
is available from very cheap approximations**, and the solver captures the rest.

### 6.4 Why the LLM underperforms here

The LLM cell's precision is **0.445**, from 1,830 accepted pairs against 1,028 true
matches. It nearly doubled the correct answer set.

This is not mysterious and not new within this document. It is exactly section 5.2's
m=1 effect. Blocking hands the model ten candidates of which at most one is usually
correct, close to the m=1 regime where measured precision is 0.36, and the model will
not answer "none".

Two pipeline features then combine badly. The model over-accepts, and its output is a
set of independent binary decisions with nothing downstream to reconcile conflicts.
Assignment is precisely that reconciliation, which is why it lifts the LLM cell from
0.570 to 0.819. It does better still from a graded cosine score than from the model's
binary mask, **because a binary mask has discarded the information the solver uses**
to choose between competing claims.

**The claim, stated narrowly.** Not that LLMs are unsuitable for joins. That on a
join with strong cardinality structure, enforcing that structure is worth more than
judging the pairs, and that an LLM asked to judge candidates independently will
systematically over-accept in the sparse regime blocking itself creates.

### 6.4b Reading the judge as a ranker, which overturns 6.3

**Motivation.** 6.4 explains the headline result with a specific mechanism: the solver
"does better from a graded cosine score than from the model's binary mask, **because a
binary mask has discarded the information the solver uses** to choose between competing
claims." That is a falsifiable claim about our own result and it had never been tested.
If it is right, handing the solver a *graded* LLM score should recover the loss.

**Method.** FIRST-style (EMNLP 2024). Instead of parsing generated text, read the
next-token distribution at the single answer position and restrict it to the candidate
identifier tokens, log-softmaxed over the k identifiers. That is a graded score per
candidate from **one forward pass per left row with no decoding at all**. Identifiers
are single letters A..J, asserted to be one token each; numeric identifiers would break
this because "1" is a prefix of "10" so their first tokens collide. Everything else
holds to the 6.3 protocol: bge-small top-10, all 1081 left rows, pooled ground truth,
thresholds and floors tuned on validation only. Cosine cells reproduce 6.3 to within
0.001 (assignment 0.862, mutual best 0.783, top-1 0.754), so the protocol replicates.

**Result.**

| resolver | cosine | LLM binary mask | **LLM identifier logits** |
|---|---|---|---|
| threshold | 0.478 | 0.455 | **0.909** |
| top-1 | 0.754 | 0.753 | **0.909** |
| mutual best | 0.783 | 0.492 | **0.934** |
| assignment | **0.862** | 0.446 | 0.901 |
| assignment + floor | 0.869 | 0.446 | **0.917** |

**6.4's mechanism claim is confirmed, and it reverses 6.3's conclusion.** The graded
read beats the binary mask on every resolver, by up to **+0.442** (mutual best, 0.492 to
0.934). It also beats cosine on every resolver. Best configuration overall is
**identifier logits + mutual best at 0.934**, against cosine + assignment + floor at
0.869, so the LLM is worth **+0.065** rather than the **-0.043** recorded in 6.3.

**And it is cheaper than the generation it replaces.** One forward pass, no decode:
3.9 rows/s against 1.8 rows/s for generation on the same prompts and hardware,
**2.13x**. So the better number is also the cheaper one. The earlier conclusion that
"the LLM's marginal contribution is negative" was a statement about *how we were
reading the model*, not about the model.

**The binary-mask path is prompt-fragile, which is the deeper reason to abandon it.**
This prompt's mask **under**-accepts: 432 pairs named, P 0.769, R 0.323. The 6.3
prompt's mask **over**-accepts: 1830 pairs named, P 0.445, R 0.793. Same model, same
blocking, same data, opposite error direction from a prompt rewording (lettered plus
"all matching candidates" versus numbered plus "comma-separated"). A pipeline whose
error direction flips on wording is not a pipeline you can tune. The graded read has no
such knob because nothing is being parsed.

**The resolver ranking depends on the scorer, which matters for an optimizer.** With
cosine, assignment dominates mutual best (0.862 against 0.783). With identifier logits
the order **inverts**: mutual best 0.934 against assignment 0.901. Assignment is forced
to pair every left row (1070 pairs) while mutual best may decline (1011), and that
refusal is worth more once the scorer is good enough to be trusted. So "use assignment"
is not a fixed rule; the resolver and the scorer have to be chosen together.

**Why this belongs to the open-weights argument.** Identifier logits need access to the
next-token distribution. Exit depth (5.9b) needs access to hidden states. Neither is
reachable through a text-only completion API, so both are physical plans that open
weights make available and a closed endpoint does not. This is the strongest form of
the paper's claim that open weights expand the plan space, and it is now a constructive
result rather than an observation.

**Scope.** One dataset, one model, k=10. The cross-prompt mask comparison is confounded
(two prompt features changed at once) so it establishes fragility, not its cause. The
cosine cells replicating 6.3 is the evidence that the rest is comparable.

**Which cells here depend on parsing, and which do not.** The binary-mask column is
produced by regex-matching identifier letters in generated text, and 5.9b records a
case where exactly that kind of parsing corrupted a baseline. `\bA\b` can also match
the letter "A" used as an English word, so the mask column may over-name. **The
central claim of this section does not touch a parser on either side**: identifier
logits (0.934) and cosine (0.869) are both numeric scores read without decoding or
parsing anything. So "logits beat cosine by +0.065" is firm, while "logits beat the
mask by +0.442" should be read as the weaker claim that parsed-text readouts are
unreliable, which is the point of the fragility result anyway.

---

### 6.5 Scope conditions

Abt-Buy is 98.8% one-to-one with 94% of left rows matched, which is the structure
assignment exploits and the reason plain assignment beats the floored variant here.
On a many-to-many join, or one where most left rows match nothing, unfloored
assignment is actively wrong since it pairs every left row. The floored variant at
0.801 is the version for lower match rates and still beats thresholding
substantially, but no genuinely sparse or many-to-many join was tested.

Assignment is cubic in table size. Negligible at 1081 x 1092. A million-row join
needs sparse, streaming or auction variants, none measured.

---

## 7. End to end results

### 7.1 Real LLM runs, gemini-flash-lite-latest

20NG, matched vector-only baselines, same data:

| config | vector only | semops + Gemini | calls |
|---|---|---|---|
| join, block_k=3 | 0.319 | 0.305 (P 0.87, R 0.185) | 36 vs 840 naive |
| join, block_k=8 | 0.441 | 0.478 (P 0.745, R 0.352) | 96 vs 840 naive |
| join, best config (batched k=24, mutual kNN, aligned predicate) | 0.540 | **0.667** | 12 |
| dedup, pairwise F1 | 0.500 | 0.517 | 183 |
| dedup, clustering purity | 0.143 | 0.786 | 95 vs 861 all-pairs |

The block_k=3 row is instructive: recall was capped at ~0.33 by blocking before
Gemini saw anything, so the operator could not win. This is what motivated the
blocking work in section 4.

### 7.2 Perfect-oracle runs, earlier sessions

These substitute the dataset's own label for the LLM to isolate the machinery from
judge error. They are **not** LLM quality measurements.

| benchmark | task | vector only | semops with oracle |
|---|---|---|---|
| Emails (Trummer) | contradiction join | 0.18 | 1.00 |
| BioDEX (LOTUS/Abacus corpus) | paper to reaction join | 0.21 | 1.00 |
| 20NG | dedup purity | n/a | 1.000 |

The contradiction join is worth noting: embedding search pulled the contradicting
email into the top 10 of 100 candidates every time, while judging those same
candidates by embedding similarity scored 0.10 precision. Retrieval and judgment are
different problems, which is the same lesson as section 4.4.

**Known bug in this configuration:** with a real LLM, the block-join adjudication path
thrashed and never emitted its completion sentinel. Only per-pair worked. This is why
later real-LLM runs used `block_adjudicate=False`.

### 7.3 Blocking fan-out, and why `block_k` is the dominant knob

`block_k` is the biggest correctness lever in the join and there is no estimator for
it. Set too low, candidate pairs never form, the LLM is never asked, and recall is
capped with nothing in the logs to say so. The right value swings by more than an
order of magnitude across tasks: 2 to 4x fan-out on 20NG, roughly 83x on BioDEX.

---

## 8. Directions that failed

Recorded because the cost of testing them was low and because two initially appeared
to succeed.

### 8.1 Does the model know matches it declines to name?

**Hypothesis.** If the recall ceiling were a decoding limitation rather than a
comprehension one, the information would be recoverable from activations, and the fix
would be cheap.

**Method.** Linear probe on layer 27 hidden states at each candidate's last token,
compared against the model's own generation, Abt-Buy.

**Result.** Generation F1 **0.855**, best probe **0.786**. Generation is better, so
there is no reservoir of unspoken matches.

**Note.** The first version of this showed generation at recall 0.053 against probe
0.774, which was entirely the chat-template and thinking-mode bug (3.4, 9.1).

### 8.2 Can a predicate direction be read without labels?

**Hypothesis.** If the model could hand you a query vector for a predicate from the
predicate text alone, index pushdown would need no training data.

**Method.** Seven 20NG categories as seven binary predicates. Supervised probe versus
two label-free constructions: a direct encoding of the predicate text, and a
contrastive difference between positive and negative framings.

| | mean over 7 predicates |
|---|---|
| supervised probe F1 | 0.772 |
| zero-shot, direct encoding | 0.295 |
| zero-shot, contrastive | 0.288 |
| all-positive baseline | 0.250 |
| **cosine between the two directions** | **+0.010 / +0.006** |

**Settled.** The model's representation of a predicate is **orthogonal** to the
direction that classifies rows by it. Individual cosines scatter around zero, several
negative. Zero-shot performance is barely above the trivial baseline.

**Context.** Consistent with the steering-vector literature, which documents these
constructions as brittle and reports many behaviours unsteerable across all layers
and strengths. There is a paper titled *A Sober Look at Steering Vectors for LLMs*.

**What survives.** Rows encoded alone are linearly separable at 0.772, so hidden
states are usable index keys **under supervision**, which is the filter case already
in production.

### 8.3 Are operators transportable as function vectors?

**Hypothesis.** Following [Todd et al.](https://arxiv.org/abs/2310.15213), a task
vector is the mean activation at the final token of in-context prompts, which when
injected into a zero-shot prompt causes the model to perform the task. Is a semantic
operator such a vector?

**Method.** Three operators with deliberately disjoint answer vocabularies so the
executed operation is identifiable. Inject at layer 14, 10-shot extraction, 120 eval
items per task.

| task | zero-shot | in-context | own vector injected |
|---|---|---|---|
| sentiment | 0.750 | 0.950 | 0.817 |
| topic | 0.950 | 0.983 | 0.950 |
| match | 0.517 | 0.908 | 0.517 |

The sentiment row looks like a success. The cross-injection matrix settles it:

| injected vector | sentiment | topic | match |
|---|---|---|---|
| sentiment | 0.817 | 0.958 | 0.517 |
| topic | 0.783 | 0.950 | 0.517 |
| **match** | **0.883** | 0.950 | 0.517 |

**Settled.** Injecting the *match* vector into a sentiment prompt scores **0.883**,
higher than sentiment's own vector at 0.817. The gain is a generic nudge toward
emitting a label token, not operator identity. By the criterion fixed before looking
at the data, this licenses nothing.

Matching gains nothing at all: 0.517 with and without injection, identical to three
decimals, against an in-context ceiling of 0.908.

**Scope limit.** This is the simpler layer-level construction, not Todd et al.'s
head-level method with causal attribution.

### 8.4 Operator anatomy: where is matching computed?

**Method.** Probe every fourth layer of Qwen3-8B on three tasks varying predicate type
and domain.

| layer | match (Abt-Buy) | match (DBLP-ACM) | filter (Rotten Tomatoes) |
|---|---|---|---|
| 4 | 0.639 | 0.898 | 0.675 |
| 12 | 0.682 | 0.916 | 0.825 |
| **20** | **0.853** | **0.940** | **0.891** |
| 28 | 0.847 | 0.892 | 0.882 |
| 36 | 0.748 | 0.836 | 0.879 |

**Result.** All three peak at layer 20 (56% depth) and decline after. My prediction
that unary predicates resolve early and relational ones late was wrong.

**This is a replication, not a discovery.** [How do LLMs Understand
Relevance?](https://arxiv.org/abs/2504.07898) (TOIS 2025) used activation patching and
found a three-stage process: early layers extract inputs, middle layers compute
relevance, later layers format the answer. Our peak-then-decline is that boundary,
reached by a weaker method.

**Transfer, and a withdrawn claim.**

| train -> test | F1 | in-domain | note |
|---|---|---|---|
| products -> publications | 0.761 | 0.917 | same predicate, different domain |
| publications -> products | 0.743 | 0.848 | same predicate, different domain |
| filter -> match | **0.517** | 0.848 | different predicate type, chance |
| match -> filter | 0.690 | 0.884 | different predicate type |

I claimed a domain-general matching direction from the first two rows. **Withdrawn**
after measuring that DBLP-ACM is 97.4% token overlap (3.3), so the transfer may be
detecting lexical similarity rather than anything about entities.

What survives: crossing predicate *type* costs far more than crossing *domain*.
filter to match at 0.517 is chance on a balanced task, so unary and binary predicates
appear to use different machinery despite being computed at the same depth.

### 8.5 Predicate composition

Not a join result, included because it is the one positive internals finding and it
bears on multi-predicate joins.

**Question.** Given directions for A and B, is the direction for A AND B a linear
combination?

| | cosine with best fit from wA, wB |
|---|---|
| **AND** | **+0.790** (weights +0.64, +0.66) |
| OR | +0.530 (weights +0.25, +0.24) |
| sanity: cos(wA, wB) | -0.101, not collinear |
| sanity: cos(wA, random) | +0.002 |

Classifying the conjunction:

| | F1 |
|---|---|
| all-positive baseline | 0.403 |
| wA alone (control) | 0.680 |
| wB alone (control) | 0.658 |
| **fitted combination** | **0.844** |
| directly trained | 0.808 |

The composed direction beats the directly trained one and clears both single-predicate
controls. Weights come out near equal, which is what conjunction should look like.
AND and OR are recovered *differently*, which is the control that could have shown I
was merely averaging.

**Practical form**, `unit(wA) + unit(wB)`, needing **zero** conjunction labels:

| conjunction | composed, no labels | directly trained |
|---|---|---|
| common (25% base rate) | 0.673 | n=10: 0.619, n=25: 0.692, n=250: 0.779 |
| rare (5.2% base rate) | **0.750** | n=10: 0.501, n=25: 0.577, n=50: 0.678 |

Crossover for the common conjunction is 10 to 25 labels. For the rare one composition
beat every probe that could be fitted, since the dataset ran out of positives at n=96.

**Mechanism, from a hypothesis that died.** I expected the seven predicate directions
to occupy a compressed subspace. They do not: effective rank **6.59 of 7**, variance
spread evenly, pairwise cosines between -0.03 and -0.34. But that negative explains
the positive: near-orthogonal directions add without interfering. Had they been packed
into a low-rank subspace, summing would produce cross-talk.

**Caveats.** One model, one layer, one conjunction pair per condition. In the rare
case `wB` alone reached 0.731 against the composed 0.750, so one predicate nearly
carried it.

---

## 9. Bugs, corrections, retractions

### 9.1 A broken baseline manufactured a headline

The latent-knowledge experiment (8.1) initially showed generation at recall **0.053**
against a probe at 0.774, dramatic support for "the model knows things it will not
say". It was the chat-template and thinking-mode bug: raw string into an
instruction-tuned model, and thinking mode consuming a 12-token budget. Corrected,
generation reached 0.888 and the hypothesis was dead.

A recall of 0.053 from an 8B instruction-tuned model is implausible on its face, and
a comparable 0.825 from a different model was already in my notes.

### 9.2 A control caught overfitting I would have written up as a finding

The learned projection (4.4) dropped bge from 0.772 to 0.184, and the conclusion I was
already inclined toward was that match predicates do not factor. The random-projection
control at 0.728 showed the training was broken rather than the idea. The fairly
trained version produced a negative result anyway, but for an entirely different and
more informative reason. **Including that control was the difference between a wrong
argument and a right one for the same conclusion.**

### 9.3 Evaluating on the easy subset

The first resolution result was **0.906**, scored only over left rows that have a
match, which excludes exactly the cases where a force-pairing resolver does harm. Over
the whole join it is **0.884**. Small deflation, found only by going looking.

Related: I also claimed at one point that most left rows have no match, citing 531 of
737. That figure was for the test *candidate-pair* subset, not the full table. On the
full table only 65 of 1081 (6%) are unmatched.

### 9.4 The floor sweep bug

`assignment + floor` initially came back worse than plain `assignment`, which is
impossible if the sweep is correct. Cause: the sweep ran quantiles 0.5 to 0.999 and
never considered a floor that keeps everything.

### 9.5 Diagnosing from theory rather than logs

A batched run failed twice. I confidently attributed it to CUDA out-of-memory and
redesigned around memory pressure, shrinking batches and adding checkpointing. The log
said `ModuleNotFoundError: No module named 'sklearn'`, omitted from that script's
image. Two relaunches wasted.

### 9.6 Retracted claims, collected

| claim | status |
|---|---|
| "index-served blocking is the lever that moves the join most" | retracted, emphasis backwards (4.1) |
| "the model knows matches it will not say" | retracted, broken baseline (9.1) |
| "there is a domain-general matching circuit" | withdrawn, DBLP-ACM is 97% token overlap (8.4) |
| "matching peaks at layer 27" | corrected to layer 20, coarse grid (8.4) |
| "all operators peak at layer 20" | corrected, depth tracks predicate difficulty (5.9) |
| "early exit is 6.9x" | corrected to 3.3x under batching (5.9) |
| "query-last fixes the position penalty" | retracted, it relocates the damage (5.7) |
| "assignment gets 0.906" | corrected to 0.884 on the whole join (9.3) |
| "the middle is the best place for a match" | **retracted**, artifact of easy distractors; only slot 1 is depressed (5.6b) |
| "the position curve is the inverse of lost-in-the-middle" | **retracted**, under hard negatives there is no middle penalty at all (5.6b) |
| "candidate position is practically minor" | **retracted**, it is worth +0.294 F1 on the real join (5.6c) |
| "the solver is worth twice the LLM" | **withdrawn as a ratio**, moves 2.2x to 4.2x on baseline tuning; use the cell-to-cell comparison (6.3) |
| "cosine + threshold scores 0.328" | corrected to 0.478 under a finer threshold grid (6.3) |
| "depth tracks predicate difficulty" | upheld, and now measured directly rather than inferred across incompatible experiments (5.9b) |
| "greedy generation is systematically conservative" | **retracted**, 19-74% of replies discarded by a startswith("y") parser (5.9b) |
| "the probe beats generation" | **retracted**, generation reaches 0.954 vs probe 0.978 with a working parser and zero labels (5.9b) |
| "early exit is 2.8-5.9x at generation-matching quality" | corrected to **1.6-1.8x** on the predicates where generation works (5.9b) |
| "the LLM's marginal contribution to the join is negative" | **retracted**, that was the readout; identifier logits reach 0.934 vs cosine 0.869 (6.4b) |

### 9.7 Underpowered pilots

Pilots used n=20 per cell, giving SE around 0.11 on a proportion near 0.5, across
roughly fifteen comparisons. The pattern survived at n=150 with paired McNemar tests,
but it could as easily not have, and the pilot was not evidence.

---

## 10. Open questions

Ranked by what I would run next.

**1. Assignment over calibrated LLM scores.** The model currently emits a binary mask,
which discards exactly the graded information the solver consumes, and 6.3 shows the
solver already does better from cosine than from that mask. If per-candidate scores
can be extracted, either from logits in the manner of FIRST or by asking for graded
output, assignment over those scores should dominate every cell in the 2x2. Concrete
prediction, cheap to test.

**2. Per-candidate verdicts instead of free recall.** The over-acceptance in 6.4
arises partly because the prompt invites the model to name whichever candidates it
likes. Requiring an explicit verdict for every numbered candidate forces consideration
of each, and structured output makes it enforceable. Targets the 0.445 precision
directly.

**3. Why does batched quality decline past 24 candidates?** Three mechanisms ruled
out (5.2), none confirmed. The most interesting loose end.

**4. Semantic selectivity estimation.** The oldest problem in query optimisation, and
every semantic-operator system needs it. CADENZA explicitly abandons cross-operator
optimisation for want of it. Our density rule (5.4) needs it as input, and 5.9 makes a
shallow forward pass cost about 32 ms.

**5. Transitivity for self-join and dedup.** Dedup is a self-join and matching is
transitive, so the resolution stage there should enforce transitive closure
consistency rather than 1:1. Untested.

**6. Capacity-constrained assignment for m:n joins.** The 1:1 constraint is the
special case. A join where each left row matches up to c right rows is a
transportation problem, still polynomial. Would extend 6.3 beyond its scope condition.

**7. A second hard benchmark.** Walmart-Amazon and WDC Products. Everything in section
6 is one dataset.

**8. A frontier model over the full join.** The Gemini figures (5.4) are on a
stratified subsample and not comparable. Someone should run a strong model over all
1081 left rows.

---

## 11. Appendix

### 11.1 Prompts

**Batched SELECT** (the workhorse):

```
A QUERY document and numbered CANDIDATES follow. List the numbers of candidates
about the SAME newsgroup topic as the QUERY. Reply only with numbers
comma-separated, or 'none'.

QUERY: {query}

CANDIDATES:
1. {candidate}
2. {candidate}
...
```

**Batched SELECT, Abt-Buy variant:**

```
A QUERY product and numbered CANDIDATE products follow. List the numbers of
candidates that are THE SAME PRODUCT as the query (same manufacturer and model),
or 'none'. Reply only with numbers comma-separated, or 'none'.

QUERY: {query}

CANDIDATES:
1. {candidate}
...
```

**Per-pair MATCH:**

```
Answer only yes or no. Are these two listings the SAME PRODUCT
(same manufacturer and model)?

A: {a}

B: {b}
```

**Predicate-aligned variant** (the +0.102 version, section 5.5):

```
A QUERY post and numbered CANDIDATE posts follow. Every post belongs to exactly
one of these Usenet newsgroups:
comp.graphics, rec.sport.hockey, sci.space, talk.politics.mideast, rec.autos,
sci.med, comp.sys.mac.hardware
List the numbers of candidates belonging to the SAME newsgroup as the QUERY,
even if the specific subject differs. Reply only with numbers comma-separated,
or 'none'.
```

### 11.2 SQL++ captured from the live cluster

Filter pushdown, index-served (`ann_above`). `$qvec` is the **classifier weight
vector**, not a predicate embedding:

```sql
SELECT META(d).id AS _id, d.`text` AS _text,
       APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, 'dot', 8, true, 200) AS _dist,
       d.label AS _label
FROM `default`.`_default`.`polarity_gsi` d
ORDER BY APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, 'dot', 8, true, 200)
LIMIT $k;
```

`EXPLAIN` confirms `IndexScan3` on `idx_polarity_gsi_dot`, no `PrimaryScan3`, no
`Fetch`.

Exact per-key scoring (the default `sem_filter` path, no heap, no cap):

```sql
SELECT META(d).id AS id,
       VECTOR_DISTANCE(d.`embedding`, $q, 'dot') AS dist
FROM `default`.`_default`.`polarity_gsi` d USE KEYS $keys;
```

Index creation:

```sql
CREATE VECTOR INDEX idx_polarity_gsi_dot
ON `default`.`_default`.`polarity_gsi`(embedding VECTOR)
INCLUDE (`text`, `label`)
WITH {'dimension':384,'similarity':'dot','description':'IVF,SQ8',
      'scan_nprobes':8,'train_list':200};
```

Two requirements learned the hard way. The index `similarity` must match the metric
you query with, or the planner silently declines the index and brute-forces the
collection, which is correct but slow and easy to miss. And when a vector index does
serve the query, the returned distance is L2-squared over stored vectors regardless of
the metric string, so the inner product must be recovered: for unit-norm vectors,
`w.x = (|w|^2 + 1 - dist) / 2`, verified to four decimals.

### 11.3 Key configuration values

```
block_k               24 to 32 for batched adjudication (the measured sweet spot)
                      10 for the fair-fight comparison (recall 0.985)
mutual kNN            always on, free, recall 0.889 -> 1.000
batch size (prompt)   ~24 candidates per prompt, inverted U peak
batch size (GPU)      16 on an L4, throughput saturates
exit layer            ~12 of 36 for easy predicates, ~20 for harder ones
nProbes               64 at 50k rows for recall 1.000
topNScan              >= your LIMIT, or you silently get fewer rows
max_heap              8192, caps LIMIT+OFFSET on a vector index scan (error 5433)
```

### 11.4 Experiment scripts

All on Modal, GPU L4, results written to a persistent volume so a client disconnect
cannot lose them.

```
modal_anatomy.py      layer sweep, operator anatomy, transfer
modal_factor.py       predicate/data factorisation, zero-shot directions
modal_compose.py      conjunction composition
modal_rare.py         rare-conjunction learning curve
modal_earlyexit.py    exit depth + predicate subspace geometry
modal_batched.py      throughput vs batch size
modal_finetune.py     distillation learning curve
modal_taskvec.py      function vectors, cross-injection
modal_joinkey.py      blocking representation, learned metric
modal_fairfight.py    the 2x2
```

One operational note: `modal run --detach` does **not** protect a call made with
`.remote()` from a local client disconnect. Use `.spawn()` and poll the volume.
Three runs were killed mid-execution before this was understood.

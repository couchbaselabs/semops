# Optimizing Semantic Joins

Working notes on a systematic pass over the semantic join operator in
[semops](https://github.com/couchbaselabs/semops). A semantic join takes two
relations and a predicate written in English ("these two records describe the same
product"), and returns the pairs that satisfy it, using a vector index to avoid the
cross product and an LLM to do the judging.

These are preliminary experiments rather than a paper, and should be read the way
you would read a colleague walking through results at a lab meeting. Everything is a
run I did myself. Several sections describe things I believed and then disproved,
and I have kept those in with the reasoning that broke them, because in three cases
the way a hypothesis failed was more informative than the hypothesis would have been
had it held.

The document is organised around a three-stage decomposition of the operator, and
within each stage I describe the methods in enough detail to reproduce them, report
the full result rather than the favourable subset, and then say what I think it
licenses.

## Contents

1. The operator, and why three stages
2. Experimental setup
3. Stage 1: candidate generation
4. Stage 2: adjudication
5. Stage 3: resolution
6. A controlled comparison of the three stages against each other
7. Threats to validity
8. What we did not do
9. Process notes and errors

---

## 1. The operator, and why three stages

Written out the way a query optimiser would see it, a semantic join is:

```
for each left row a in R:
    C(a)      = block(a, S)                          # candidate right rows
    verdicts  = { (a,b, judge(a,b)) : b in C(a) }    # per-pair LLM judgments
output        = resolve( all verdicts )              # which verdicts to keep
```

The literature treats this as two stages. Blocking is studied as an information
retrieval problem, with a large body of work on ANN indexes, learned blockers and
recall-oriented candidate generation. Adjudication is studied as a prompting
problem: LOTUS approximates the judge with proxies, ComEM compares matching,
comparing and selecting prompt styles, FIRST reads candidate-identifier logits,
Trummer batches candidates into one prompt to decouple cost from candidate count.

The third stage is almost always a threshold, and it is usually not discussed at
all. That asymmetry is what this document ends up being about, so it is worth
saying why the third stage is not merely a formality.

Stages 1 and 2 both operate on a single pair, or on one left row and its candidate
list. Neither has any view of the join as a whole. If right row `b` is a plausible
match for eleven different left rows, a per-pair judge will happily accept all
eleven, because each decision is locally reasonable. A global threshold cannot fix
this either, since it is also a per-pair rule: it just moves the cut point. Only a
resolution step that reasons over the whole verdict set can use the fact that a join
has cardinality structure, that some joins are one-to-one, that a right row can be
spoken for.

Classical entity resolution knows this and uses bipartite matching. The LLM
entity-matching work I read does not, as far as I can tell, and thresholds
independently. That gap is the thing I eventually measured.

I did not start with this framing. I arrived at it after the stage 1 and stage 2
experiments came back mostly negative, which is why they are presented first and in
full: the negative results are what made the third stage look interesting.

---

## 2. Experimental setup

### 2.1 Datasets

**Abt-Buy** carries the main results. It is a standard entity-matching benchmark
from the Magellan repository, distributed on HuggingFace as `matchbench/Abt-Buy`,
consisting of two product tables plus labelled pairs split into train, validation
and test. The relevant statistics, measured rather than quoted:

```
tableA           1081 rows   (name, description, price)
tableB           1092 rows
cross product    1,180,452 candidate pairs
ground truth     1028 matching pairs, pooled from all three splits
test.csv         1916 labelled pairs, of which 206 positive
```

The cardinality structure matters a great deal later, so I measured it explicitly
rather than assuming the usual "entity matching is one-to-one" folklore:

```
left-row  match counts: {1: 1004, 2: 12}
right-row match counts: {1: 1020, 2: 4}
matched left rows: 1016 of 1081, so 65 left rows (6%) match nothing
fraction of matched left rows with exactly one match: 0.988
```

So this join is very nearly a bipartite matching. I will return to how much that
limits the generality of the main result.

**20 Newsgroups** carries the stage 2 prompt experiments, in a self-join
configuration where the predicate is "these two posts are about the same topic" and
ground truth is "these two posts came from the same newsgroup". Partway through I
established that this ground truth is substantially wrong for the predicate as
written, which is discussed in section 4.4, so the 20NG numbers should be read as
relative comparisons between prompting strategies and not as absolute quality.

**DBLP-ACM** was going to be a second matching dataset for a transfer experiment. I
dropped it after measuring that it is 97.4% solvable by naive token-overlap
similarity (Jaccard over lowercased alphanumeric tokens, best threshold), against
0.736 for Abt-Buy under the same measurement. A dataset that a one-line baseline
nearly solves cannot support conclusions about what a model is doing.

### 2.2 Models

```
embedding    BAAI/bge-small-en-v1.5, 384-dim, via fastembed, run locally
LLM judge    gemini-flash-lite-latest, via the OpenAI-compatible endpoint
open model   Qwen/Qwen3-8B, 36 layers, hidden size 4096, bfloat16
hardware     Modal, single NVIDIA L4 for everything involving Qwen
```

Qwen3-8B was chosen over the newer Qwen3.5-9B deliberately. 3.5 uses a
`Qwen3_5ForConditionalGeneration` class with a nested config, and for work whose
whole point is cleanly extracting per-layer hidden states, the standard
`Qwen3ForCausalLM` path is worth more than the newer weights.

Two implementation details cost me a lot of time and are worth stating because they
silently corrupt results rather than erroring. First, Qwen3 is instruction tuned and
must be prompted through `apply_chat_template`; feeding it a raw string puts it in
raw-completion mode and it stops following instructions. Second, Qwen3 enables a
thinking mode by default, so a small `max_new_tokens` budget gets consumed entirely
by reasoning tokens and the model never emits an answer. Both need explicit handling:

```python
prompt = tok.apply_chat_template(
    [{"role": "user", "content": body}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
...
answer = re.sub(r"<think>.*?</think>", " ", raw, flags=re.S)
```

### 2.3 Determinism, and where statistical power comes from

Before running paired comparisons I checked how much of the variance is the model
rather than the sample. Five distinct prompts, each submitted ten times at
temperature 0, on the batched-adjudication task:

```
5 of 5 prompts returned byte-identical output on all 10 repetitions
5 of 5 parsed match-sets identical
```

At temperature 0 this model is effectively deterministic on this task, so repeating
a prompt buys nothing. This matches the methodology in Liu et al., who use greedy
decoding and no seeds, and draw power from 2,655 queries or 500 examples per
condition instead. It means all statistical power has to come from distinct
examples, which set the sample sizes for the position experiment in section 4.5.

### 2.4 Evaluation protocol

Three conventions apply everywhere, and each of them was introduced after an early
version of an experiment produced a number I could not defend.

**Ground truth is pooled across splits.** For the join experiments the true matching
is every `label=1` pair from train, validation and test. Scoring against test
positives alone would count a correctly recovered train-split match as a false
positive.

**Every threshold and floor is tuned on validation positives only.** Not on test,
and not by maximising the reported metric. An early version tuned the threshold by
sweeping it against the evaluation set, which flatters whichever method has the most
tunable knobs, and in that version it flattered the baseline.

**Scoring covers the whole join.** All 1081 left rows against the pooled truth. The
first version of the resolution experiment scored only left rows that have a match,
which excluded exactly the cases where a resolver that force-pairs every left row
does damage. That version read 0.906; the honest version reads 0.884.

---

## 3. Stage 1: candidate generation

### 3.1 Directionality

The blocking implementation counted a candidate pair `(a,b)` when `b` appeared in
`a`'s top-k under cosine similarity. This is the obvious construction and it is
asymmetric: if `a` is an unusually generic record with many near neighbours, its true
match may sit at rank 30 while `a` sits at rank 2 in the true match's own
neighbourhood.

The change is to admit a pair when either row appears in the other's top-k, which is
the reciprocal or mutual kNN construction from the blocking literature. Measured on
the 20NG self-join, blocking recall against the true pair set:

| k | one-directional | mutual |
|---|---|---|
| 4 | 0.213 | 0.371 |
| 8 | 0.417 | 0.593 |
| 16 | 0.750 | 0.898 |
| 24 | 0.889 | 1.000 |
| 32 | 0.923 | 1.000 |

Candidate recall reaches 1.000 at k=24, against 0.889 for the one-directional
version, at no API cost and with no additional index scans if you already have the
symmetric neighbour lists.

This result reversed a position I had been arguing for several days. I had claimed
that index-served blocking was the lever most likely to move end-to-end join
quality, on the reasoning that the LLM can only judge pairs it is shown. Once
blocking recall was 1.000 and the LLM was still converting only half to two thirds
of the pairs handed to it into correct verdicts, retrieval was demonstrably not the
binding constraint. The claim was not wrong in its mechanism, it was wrong about the
magnitude, and I had not measured the two stages against each other before asserting
which mattered more.

### 3.2 Which representation makes the best join key?

semops already pushes a filter predicate into the vector index, and it works. The
mechanism is that a logistic proxy scores `sigma(w.x + b)`, which is monotonic in
`w.x`, so the proxy's ranking is exactly the maximum-inner-product ranking for query
vector `w`. You hand `w` to the index as the query vector and get the classifier's
own ordering back:

```sql
SELECT META(d).id AS _id, d.`text` AS _text,
       APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, 'dot', 8, true, 200) AS _dist
FROM `default`.`_default`.`polarity_gsi` d
ORDER BY APPROX_VECTOR_DISTANCE(d.`embedding`, $qvec, 'dot', 8, true, 200)
LIMIT $k;
```

On a live single-node cluster this is a genuine index scan: `EXPLAIN` reports
`IndexScan3` on the dot-similarity index with no `PrimaryScan3` and no `Fetch`, the
latter because an `INCLUDE (text, label)` index covers the projection. Two
requirements are easy to get wrong. The index must be created with `similarity`
matching the metric you query with, or the planner silently declines the index and
brute-forces the collection, which is correct but slow and therefore easy to miss.
And when a vector index does serve the query, the distance it returns is L2-squared
over the stored vectors regardless of the metric string, so the inner product has to
be recovered: for unit-norm vectors, `w.x = (|w|^2 + 1 - dist) / 2`.

The natural question is whether the same trick applies to a join. There was direct
encouragement for trying. Probing Qwen3-8B on Abt-Buy pairs, "these two records are
the same entity" is linearly decodable from the layer-20 hidden state at F1 0.853,
which is essentially what full generation achieves on the same data (0.855). If the
representation contains the answer, index the representation.

The first attempt was to index the hidden states directly. Both tables were encoded
row by row with the prompt `Product: {text}`, taking the last-token hidden state at
each of layers 8, 12, 16, 20, 24, 28 and 36, and blocking recall was measured over
the full 1081 x 1092 cross product on the 206 held-out test positives, with no
candidate pre-filtering of any kind.

| representation | dim | R@1 | R@2 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|---|
| bge-small | 384 | 0.772 | 0.883 | 0.951 | 0.990 | 1.000 | 0.853 |
| qwen-L8 | 4096 | 0.073 | 0.092 | 0.121 | 0.184 | 0.286 | 0.112 |
| qwen-L12 | 4096 | 0.058 | 0.083 | 0.102 | 0.165 | 0.233 | 0.095 |
| qwen-L16 | 4096 | 0.044 | 0.063 | 0.087 | 0.117 | 0.184 | 0.075 |
| qwen-L20 | 4096 | 0.024 | 0.034 | 0.073 | 0.102 | 0.175 | 0.052 |
| qwen-L24 | 4096 | 0.019 | 0.024 | 0.068 | 0.078 | 0.112 | 0.045 |
| qwen-L28 | 4096 | 0.019 | 0.044 | 0.068 | 0.078 | 0.126 | 0.048 |
| qwen-L36 | 4096 | 0.073 | 0.083 | 0.141 | 0.180 | 0.218 | 0.108 |

A purpose-trained 384-dimensional embedder outperforms an 8B model's 4096-dimensional
activations by a factor of about thirty at R@1. Every layer is unusable, and there
is an interesting non-monotonicity: layer 20, the best layer for judging whether two
records match, is the worst layer for retrieving the match. The two extremes, layer
8 and the final layer, do marginally better than the middle.

This is less surprising in the context of the embedding literature than it was to
me at the time. A causal decoder's last-token state is optimised for predicting the
next token, and is known to be anisotropic and dominated by a few high-magnitude
directions. That LLM2Vec, E5-Mistral and RepLLaMA all exist, and all involve
substantial training to convert a decoder into an embedder, is evidence that the raw
states are not usable as metric spaces.

### 3.3 Can an indexable metric be learned instead?

The failure of the raw states does not settle the question, because a vector index
does not require the raw representation to be metric. It requires the scoring
function to factor as `f(a) . g(b)`, since the index computes one stored vector
against one query vector. Any bilinear form works: with `score(a,b) = h(a)^T W h(b)`
and `W = L L^T`, the score is `(L^T h(a)) . (L^T h(b))`, so you store `L^T h(b)` and
query with `L^T h(a)`.

So the question becomes whether a low-rank `L` can be learned that makes matching
score higher than non-matching, better than plain cosine does.

**First attempt, and how it went wrong.** I trained `L` at rank 128 with a logistic
objective on the labelled training pairs, using hard negatives mined as the top-20
cosine neighbours of each training positive's left row, with class weighting for the
imbalance, Adam at learning rate 3e-3, for 300 full-batch epochs.

| representation | raw R@1 | learned R@1 |
|---|---|---|
| bge-small | 0.772 | 0.184 |
| qwen-L8 | 0.073 | 0.000 |
| qwen-L20 | 0.024 | 0.000 |
| qwen-L36 | 0.073 | 0.010 |

Learning made every representation dramatically worse. Had I stopped here I might
have written this up as evidence that the match predicate does not factor, which
would have been the conclusion I was already inclined toward and would have been
supported by numbers that in fact showed something else.

The control is what caught it. I had included a random projection to the same rank,
on the reasoning that a claim about a learned 128-dimensional metric needs to be
compared against an arbitrary 128-dimensional metric:

```
bge-small, random projection to rank 128 : R@1 0.728   (raw 0.772)
qwen-L20,  random projection to rank 128 : R@1 0.024   (raw 0.024)
```

A random projection barely dents bge, dropping it from 0.772 to 0.728, while my
learned projection collapsed it to 0.184. Learning was actively destroying
information that a random matrix preserved. That is a signature of overfitting, and
in retrospect the setup guaranteed it: 4096 x 128 is roughly half a million
parameters, the training set contains a few hundred positives, and 300 full-batch
epochs at 3e-3 with no early stopping is enough to memorise them.

**Second attempt, constructed to give the hypothesis a fair chance.** Three changes.
`L` is initialised at the top-128 principal directions of the pooled row
representations, so that training *begins* at approximately the raw cosine geometry
rather than at a random point. The learning rate drops to 1e-4 with weight decay
1e-4. And validation recall@1 is computed every 20 epochs on the validation
positives, with early stopping after three non-improving evaluations, so the
reported model is the best validation checkpoint rather than the final one.

```
bge-small : validation R@1   init 0.762   ep20 0.757   ep40 0.762   ep60 0.757
                             best 0.762, early stopped
qwen-L20  : validation R@1   init 0.015   ep20 0.053   ep40 0.039   ep60 0.044
                             best 0.053, early stopped
```

With the overfitting removed, the learned metric does not improve on cosine at all
for bge, moving from 0.762 to 0.762. On the Qwen representation it triples from a
catastrophic base, 0.015 to 0.053, which is still two orders of magnitude away from
usable. The full-join numbers with the fairly-trained projection confirm it: bge
with the learned projection reaches blocking R@1 0.743 against 0.773 for raw cosine,
so the projection is very slightly harmful even when trained carefully.

Putting the two subsections together produces the one stage 1 conclusion I would
defend. The same model, at the same layer, decodes "same entity" at F1 0.853 when
both records appear in a single prompt, and provides essentially no signal when the
two records are encoded independently and compared by any bilinear form. The
predicate is not a property of either row that can be read off separately and then
combined. It lives in the interaction between them, which in retrieval terms means
it is cross-encoder shaped, and cross-encoders do not factor.

The practical consequence is a design constraint rather than a tuning target.
Blocking for a semantic join has to remain a cheap, generic, predicate-agnostic
similarity search, and adjudication cannot be eliminated by building a smarter
index. This is a more limiting conclusion than I wanted, and it is what redirected
attention to what happens after adjudication.

---

## 4. Stage 2: adjudication

This stage received the most experiments and produced the most results, though none
of them turned out to be the largest effect in the system.

### 4.1 Prompt granularity, and Trummer's batching

The default construction judges one pair per call: "are these two records the same
entity, yes or no." Cost is then linear in the candidate count, which couples
directly to blocking recall, so improving recall by widening k makes adjudication
proportionally more expensive. Trummer's semantic join paper observes that if you
place one query row and all of its candidates in a single prompt and ask which
candidates match, the call count decouples from the candidate count entirely.

I implemented both on the 20NG self-join with blocking held fixed, and swept the
candidate count. Per-pair judging asks a yes/no question about each pair
independently; batched judging uses:

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

| adjudication | candidates per row | calls | precision | recall | F1 |
|---|---|---|---|---|---|
| per pair | 8 | 96 | 0.682 | 0.278 | 0.395 |
| per pair | 16 | 192 | 0.754 | 0.454 | 0.566 |
| per pair | 24 | 288 | 0.740 | 0.528 | 0.616 |
| batched | 8 | 12 | 0.862 | 0.231 | 0.365 |
| batched | 16 | 12 | 0.849 | 0.417 | 0.559 |
| batched | 24 | 12 | 0.933 | 0.519 | 0.667 |
| batched | 32 | 12 | 0.922 | 0.435 | 0.591 |
| batched | 48 | 12 | 0.860 | 0.398 | 0.544 |
| batched | 69 | 12 | 0.920 | 0.213 | 0.346 |

Two observations. Batched adjudication at 24 candidates reaches F1 0.667 using 12
calls, against 0.616 using 288 calls for per-pair at the same candidate count, so it
is both better and roughly 24 times cheaper. And quality traces an inverted U over
batch size, rising to a peak near 24 candidates and then declining, driven almost
entirely by recall: precision stays between 0.86 and 0.93 across the whole sweep
while recall rises to 0.519 and then falls to 0.213.

I spent a considerable amount of the total effort trying to explain that decline and
did not succeed. Three mechanisms were tested and rejected.

It is not the context window. At 69 candidates the prompt is approximately 20,000
characters, well inside the model's limit, and the decline is already visible at 32
candidates.

It is not lost-in-the-middle. Section 4.5 describes the positional experiment in
detail. Under hard negatives there is no middle penalty at all: every position from
a quarter of the way in to the very end is statistically indistinguishable, and the
only depressed slot is the first one, which is the opposite end from where that
hypothesis puts the damage.

It is not an output budget. Holding the candidate count fixed at 24 and varying the
number of planted true matches from 1 to 12, the number of candidates the model
names rises with the number present, from 1.80 to 7.10, so it is not rationing a
fixed-size answer. Recall stays roughly flat at 0.57 to 0.68 across that range.

So the batch-size effect is real, replicated, and unexplained.

### 4.2 Does splitting the prompt help?

Since a wide candidate list degrades quality, the obvious remedy is to split it into
several smaller prompts and union the results. This does not work, and the direction
of the failure is informative. Blocking was pinned at 1.000 using mutual kNN at
k=24, giving an average of 34.4 candidates per row, and only the chunk size varied.

| candidates per prompt | prompts per row | calls | precision | recall | F1 |
|---|---|---|---|---|---|
| 34 (all at once) | 1.2 | 15 | 0.937 | 0.546 | 0.690 |
| 17 | 2.3 | 28 | 0.862 | 0.519 | 0.647 |
| 12 | 3.5 | 42 | 0.824 | 0.519 | 0.636 |
| 8 | 4.8 | 58 | 0.871 | 0.500 | 0.635 |
| 4 | 9.1 | 109 | 0.797 | 0.472 | 0.593 |

Monotonically worse on precision, recall and F1 while costing up to seven times more
calls. Match density is constant across these rows by construction, since a fixed
candidate set is being partitioned, so this is not the density effect from section
4.3. The reading I find most plausible is that the model judges comparatively: a
wider candidate list gives it more context against which to calibrate what counts
as a match, and removing that context pushes it toward guessing. This is consistent
with the m=1 precision collapse described next.

### 4.3 The best strategy depends on match density, and it inverts

The per-pair versus batched comparison above was run on 20NG, where each left row
has roughly nine true matches among 34 candidates. Abt-Buy is the opposite regime,
with roughly 0.6 matches per record among 2.5 candidates in the supplied candidate
set. Running the same two strategies there:

| dataset | matches per left row | batched SELECT | per-pair MATCH |
|---|---|---|---|
| 20 Newsgroups | ~9 of 34 candidates | wins, about 30x cheaper | loses |
| Abt-Buy | ~0.6 of 2.5 candidates | 0.900 | 0.945 |

On Abt-Buy the detailed numbers are batched at precision 0.990 and recall 0.825,
against per-pair at precision 0.965 and recall 0.925. Batched keeps higher
precision, as it does on 20NG, but when matches are sparse the recall deficit
dominates and the ranking inverts. Additionally, the batched picks were a strict
subset of the per-pair picks on this data, so combining them by union recovered
exactly the per-pair result and by intersection exactly the batched result.

There is a mechanism, and it is measurable directly. Holding the candidate list at
24 and planting exactly one true match:

| true matches planted | avg candidates named | avg correct | precision |
|---|---|---|---|
| 1 | 1.80 | 0.65 | 0.36 |
| 2 | 1.55 | 1.35 | 0.87 |
| 4 | 2.65 | 2.55 | 0.96 |
| 8 | 4.60 | 4.60 | 1.00 |
| 12 | 7.10 | 7.10 | 1.00 |

At one true match the model names 1.80 candidates and gets 0.65 of them right, for
precision 0.36. It will not answer "none", so when the correct answer is "almost
nothing here matches" it invents roughly one false positive per prompt. Precision
climbs monotonically to 1.00 as the true match count rises. Per-pair judging is
structurally the m=1 regime, one candidate per prompt, which explains its known
precision problem, and it also explains a much larger effect in section 5.

The operational rule this supports is that match density is estimable from the
blocking score distribution before any LLM call is issued, and the adjudication
strategy should be chosen from it: batch when density is high, judge per pair when
it is low.

### 4.4 Predicate specification, and a labelling error of ours

The single largest effect measured in stage 2 came from changing the wording of the
predicate rather than any technique. Same candidate set, same batched strategy, same
12 calls.

| predicate | precision | recall | F1 |
|---|---|---|---|
| "the two documents are about the same newsgroup topic" | 0.931 | 0.500 | 0.651 |
| "come from the same newsgroup, even if they discuss different subjects" | 0.967 | 0.537 | 0.690 |
| the above, plus the list of the seven newsgroup names | 0.957 | 0.620 | 0.753 |

Recall rises by 24% relative while precision holds, so this is not the model
becoming more permissive. Supplying the label space is legitimate for a real join,
where the schema is known.

The reason this effect existed at all is a mistake in our experimental setup that
took me an embarrassingly long time to find. The ground truth for the 20NG join is
"these two posts share a newsgroup", and the prompt asked "are these two posts about
the same topic". Those are different predicates. I quantified the gap two ways.
Same-newsgroup pairs have mean cosine similarity 0.617 against 0.535 for
different-newsgroup pairs, barely separated, and 10.5% of same-newsgroup pairs are
less similar than the average different-newsgroup pair. More persuasively, inspecting
the least-similar pairs that the ground truth calls matches:

```
[comp.graphics] cosine 0.485
  A: "We are interested in purchasing a grayscale printer that offers good
      resolution for grayscale medical images... under $5000?"
  B: ".DXF can not be changed over to .IFF format, what it can be changed to is
      an object format used by one of the 3D programs on the Amiga."

[rec.sport.hockey] cosine 0.540
  A: "And Dino Ciccarelli and Ray Sheppard and so on and so on..... : )"
  B: "Stanley Cup Playoff stats through games played Sunday, April 25 1993"
```

Both members of each pair share a newsgroup. Neither pair is about the same topic. A
printer purchasing question and a file-format question are both `comp.graphics` and
nothing more. When the model refused these it was correct and our label was wrong.

The consequence is that the recall ceiling of roughly 0.55 that I had spent most of
the effort attacking, and had been attributing to the model, was substantially a
property of the labels. Relative comparisons between prompting strategies survive,
since every strategy was scored against the same labels, but absolute recall on 20NG
is not a measurement of model capability.

### 4.5 Candidate order, which is worth more than the judge

Batched adjudication puts candidates in an ordered list, which raises the question
of whether position matters. Liu et al. establish a "lost in the middle" effect on
multi-document QA and key-value retrieval, with accuracy highest when the relevant
document is at the start or end of the context and degrading in the middle.

The design is a needle-in-a-haystack adapted to the join task, and it is paired.
Each trial constructs one query document from newsgroup C, one true match from the
same newsgroup, and k-1 distractors from other newsgroups. The same candidate set is
then re-tested with the true match relocated to each position under study, so
document difficulty is held constant and any difference is attributable to position
alone. Success is defined as the model naming the position where the match actually
sits.

A pilot at n=20 per cell produced a suggestive pattern, but the standard error on a
recall near 0.5 at that sample size is about 0.11, across roughly fifteen
comparisons, which is not enough. Since section 2.3 established the model is
deterministic, more repetitions of the same prompt would have added nothing, so the
rerun used 150 distinct paired trials per cell, 1,350 calls in total, analysed with
McNemar's exact test on the discordant pairs and bootstrap 95% confidence intervals
on each cell.

| candidates | first | middle | last |
|---|---|---|---|
| 8 | 0.433 [0.353, 0.513] | 0.573 [0.493, 0.653] | 0.587 [0.507, 0.660] |
| 24 | 0.373 [0.293, 0.453] | 0.573 [0.493, 0.653] | 0.367 [0.293, 0.447] |
| 48 | 0.160 [0.107, 0.220] | 0.480 [0.400, 0.560] | 0.300 [0.227, 0.373] |

McNemar on the discordant trials, first versus middle: 3 against 24 at k=8, 3 against
33 at k=24, 3 against 51 at k=48, all with p below 0.0001. At k=8 middle and last are
statistically indistinguishable (p=0.83) and only the first position is depressed.

I wrote this up as the inverse of the lost-in-the-middle prediction, with the middle
of the candidate list as the best place for a match to sit. The first part of that
survived replication and the second part did not, so the paragraph below replaces it.

The distractors above come from other newsgroups, which makes them easy. Real
blocking hands the model near misses. So the rerun drew distractors from the
blocking neighbourhood with true matches removed, shuffled their order so there is
no sortedness signal for the needle's position to violate, and repeated the design
on a second dataset. Abt-Buy at k=48, 150 paired trials per cell:

| position | recall | 95% CI | versus middle |
|---|---|---|---|
| 1 | 0.553 | [0.473, 0.627] | 2 against 59, p below 0.0001 |
| 12 | 0.960 | [0.927, 0.987] | not significant, p=0.34 |
| 24 | 0.933 | [0.893, 0.973] | reference |
| 36 | 0.933 | [0.893, 0.967] | not significant, p=1.00 |
| 48 | 0.933 | [0.893, 0.967] | not significant, p=1.00 |

The same shape holds at k=8 and k=24. On 20 Newsgroups at k=48 nothing survives
multiplicity at all; position 1 is not even reliably worse there. So the settled
result is one poisoned slot and a flat tail. Position 1 is bad, every other position
is interchangeable, the middle is not special and neither is the end. The inverted U
was easy distractors.

The average number of candidates named is roughly constant across positions within a
given k (1.37, 1.37 and 1.53 at k=48), so the position-1 effect is a discrimination
failure rather than the model naming fewer things. That part holds up.

I also called the effect practically small, worth under a point of aggregate recall,
and that was wrong for a methodological reason worth stating. A
needle-in-a-haystack measurement scores one slot in isolation, which is the wrong
instrument for a knob that applies to every row of a join. Measured as join F1 on
real candidate sets, with everything else held fixed and only the arrangement
varying:

| arrangement | what sits in slot 1 | precision | recall | F1 |
|---|---|---|---|---|
| similarity descending | the best candidate | 0.384 | 0.818 | 0.523 |
| random | a random candidate | 0.560 | 0.946 | 0.703 |
| best in the middle | a mid candidate | 0.655 | 0.909 | 0.762 |
| swap first and last | the worst candidate | 0.676 | 0.953 | 0.791 |
| similarity ascending | the worst candidate | 0.711 | 0.960 | 0.817 |
| a decoy first | cannot match anything | 0.324 | 0.980 | 0.487 |

That is 0.294 F1 for free, which is larger than what the adjudicating model itself
is worth over a plain cosine baseline. The mechanism is precision rather than
recall: a poisoned first slot makes the model over-accept across the whole list,
naming 2.03 pairs per row under similarity descending against 1.28 under similarity
ascending.

Swapping only the first and last elements captures 0.791 of the 0.817, so 0.269 of
the gain is attributable to what sits in slot 1 and only 0.026 to the direction of
the gradient. The fix is a two-element swap and the list does not need reversing.
Putting a decoy there is worse than the baseline, because the decoy displaces a real
candidate, so the move is to put your worst real candidate first rather than to
sacrifice the slot.

None of the underlying bias is novel: ComEM reports position bias in select-style
prompts. What is new here is the size of it when it is measured on the join rather
than on one slot.

### 4.6 Query placement

The positional result suggests an adjacency mechanism, since in our prompt the query
text sits immediately above `CANDIDATES:` followed by `1. ...`, so candidate 1 abuts
the query and may be partially absorbed into it. The obvious remedy is to move the
query below the candidate list.

Tested at k=48 with the needle always at position 1, 150 paired trials, three prompt
layouts:

| layout | recall at position 1 | 95% CI | avg named |
|---|---|---|---|
| baseline, query first | 0.140 | [0.087, 0.200] | 1.69 |
| explicit delimiter between query and list | 0.240 | [0.173, 0.313] | 1.71 |
| query last | 0.453 | [0.373, 0.533] | 0.93 |

Moving the query below the list recovers position 1 from 0.140 to 0.453, McNemar 3
against 50, p below 0.00001. That looked like a fix, and I reported it as one before
checking whether it merely moved the problem.

It moves the problem. Testing all three positions under both layouts, again 150
paired trials each:

| position | query first | query last | McNemar |
|---|---|---|---|
| first | 0.153 | 0.465 | 4 vs 49, p<0.00001 |
| middle | 0.583 | 0.326 | 42 vs 5, p<0.00001 |
| last | 0.444 | 0.287 | 32 vs 9, p=0.0004 |
| overall | 0.394 | 0.360 | query-first better |

The damage follows whichever end of the list abuts the query text, which supports the
adjacency mechanism, but relocating it costs more than it recovers: the middle drops
from 0.583 to 0.326 and the overall average is worse. The recommendation is to keep
the conventional layout.

### 4.7 Exit depth

If the matching decision is computed in the middle of the network rather than at the
end, a truncated forward pass plus a linear probe could substitute for generation.
Probing Qwen3-8B on a single predicate, with quality and cost measured on the same
task so the columns are comparable:

| method | F1 |
|---|---|
| full generation | 0.889 |
| probe at layer 8 (22% depth) | 0.867 |
| probe at layer 12 (33%) | 0.901 |
| probe at layer 20 (56%) | 0.838 |
| probe at layer 36 (100%) | 0.914 |

The cost side of this needs care, and my first version got it wrong. I measured
latency at batch size 1 and reported a 6.9x speedup, which is not how anything is
served: at batch 1 generation is memory-bandwidth bound and the GPU is largely idle,
which flatters a forward-pass-only method. Re-measured as throughput across batch
sizes, on one L4 in bfloat16:

| batch | generation | probe L36 | probe L20 | probe L12 | L12 speedup |
|---|---|---|---|---|---|
| 1 | 4.5 rows/s | 12.1 | 20.5 | 30.9 | 6.9x |
| 4 | 12.8 | 26.0 | 43.6 | 66.1 | 5.2x |
| 8 | 18.8 | 29.8 | 51.3 | 78.8 | 4.2x |
| 16 | 22.5 | 32.7 | 56.2 | 85.2 | 3.8x |
| 32 | 22.7 | 28.9 | 49.4 | 75.8 | 3.3x |

The honest figure is 3.3x, converging on the FLOP ratio of 12 layers out of 36 as it
should. The decomposition at batch 32 is the part I had not anticipated: skipping the
autoregressive decode is worth only 1.3x, while truncating the layers is worth a
further 2.6x. At batch 1 the ordering was reversed, because sequential decode on an
idle GPU dominates everything. Under batching the decode cost amortises almost
entirely and essentially all the durable benefit is the layer truncation, which is a
genuine FLOP reduction that batching cannot amortise away. The advice that follows is
"exit early", not "avoid generating".

Throughput saturates near batch 16 on this hardware, with generation flat from 22.5
to 22.7 and the probe actually declining from 85.2 to 75.8 at batch 32, presumably
memory pressure. Benchmarking at 32 would understate both paths.

One caveat bounds this result. The probe requires labelled training data and
generation does not, so this is not a claim that probing beats the model. It is
apples to apples only for a cascade that already trains a proxy on a labelled
sample, which semops does.

Note also that this contradicts something I claimed earlier from the layer sweep in
section 3.2, where matching peaked at layer 20. Topic classification peaks at layer
12 and is nearly flat from layer 8. The reconciliation is that depth tracks predicate
difficulty rather than operator type: 20NG topic membership is easy and resolves
early, Abt-Buy entity matching is harder and resolves later. The exit layer is a
per-predicate parameter, not a constant.

### 4.8 Distilling the judge into a small model

The cascade already spends LLM calls labelling a calibration sample, so those labels
are a sunk cost. If a small model can learn the operator from them, subsequent
queries could skip the large model entirely. Tested as a learning curve, fine-tuning
Qwen3-0.6B with a sequence classification head on Abt-Buy pairs:

| labelled pairs | positives among them | F1 | precision | recall |
|---|---|---|---|---|
| 100 | 13 | 0.176 | 0.124 | 0.306 |
| 250 | 33 | 0.244 | 0.221 | 0.272 |
| 500 | 60 | 0.281 | 0.333 | 0.243 |
| 1000 | 121 | 0.767 | 0.713 | 0.830 |
| 2000 | 222 | 0.823 | 0.763 | 0.893 |

for reference, Qwen3-8B zero-shot generation on the same test set reaches 0.855.

The curve has a sharp transition rather than a gradual slope, jumping from 0.281 to
0.767 between 500 and 1000 labels, and the threshold is in positives rather than
total labels, at roughly 120 of them, which is unsurprising on a task with a 10.8%
positive rate. It also never crosses over: at 2000 labels the student reaches 0.823,
still below the 8B model it would be distilled from.

The practical consequence is that distillation is not free. Our cascade calibrates on
100 to 300 labels, which lands squarely in the dead zone where F1 is 0.18 to 0.28. A
viable student needs a deliberate spend of a thousand or more judge calls and still
lands below its teacher.

Two caveats. I discarded the throughput column entirely: I trained in float32 with
fixed 128-token padding while the probe measurements used bfloat16 with dynamic
padding, so the measured 52.8 rows/s is an artifact of my setup rather than a
property of the model. And nothing was tuned, with a fixed learning rate, three
epochs and no early stopping, where published Abt-Buy fine-tunes reach approximately
0.89. The curve shape and the positive-count threshold are the trustworthy parts.

### 4.9 Adjudication approaches that did not work

Four further hypotheses were tested and rejected. They are recorded because the cost
of testing them was low and because two of them initially appeared to succeed.

**Does the model know matches it declines to name?** If the recall ceiling were a
decoding limitation rather than a comprehension one, the information would be
recoverable from activations. Measured on Abt-Buy with a linear probe on layer 27
against the model's own generation: generation reaches F1 0.855, the probe 0.786.
Generation is better, so there is no reservoir of unspoken matches. The first version
of this experiment showed generation at recall 0.053 against the probe at 0.774,
which was entirely the chat-template and thinking-mode bug from section 2.2.

**Can a predicate direction be read from the model without labels?** If it could, the
filter pushdown of section 3.2 would need no training data. Seven 20NG categories as
seven binary predicates, comparing a supervised probe against two label-free
constructions, a direct encoding of the predicate text and a contrastive difference
between positive and negative framings:

| | mean over 7 predicates |
|---|---|
| supervised probe F1 | 0.772 |
| zero-shot, direct encoding | 0.295 |
| zero-shot, contrastive | 0.288 |
| all-positive baseline | 0.250 |
| cosine between the two directions | +0.010 / +0.006 |

The model's own representation of a predicate is orthogonal to the direction that
classifies rows by that predicate, with individual cosines scattered around zero and
several negative. Zero-shot performance is barely above the trivial baseline. This is
consistent with the steering-vector literature, which documents these constructions
as brittle and reports many behaviours as unsteerable across all layers and
strengths. What survives is that rows encoded alone are linearly separable at 0.772,
so hidden states are usable index keys under supervision, which is the filter case
already in production.

**Are operators transportable as function vectors?** Following Todd et al., a task
vector is the mean activation at the final token of in-context prompts, which when
injected into a zero-shot prompt causes the model to perform the task. Three
operators with deliberately disjoint answer vocabularies so the executed operation
is identifiable, injecting at layer 14 with 10-shot extraction:

| task | zero-shot | in-context | own vector injected |
|---|---|---|---|
| sentiment | 0.750 | 0.950 | 0.817 |
| topic | 0.950 | 0.983 | 0.950 |
| match | 0.517 | 0.908 | 0.517 |

The sentiment row appears to be a success. The cross-injection matrix is what settles
it: injecting the *match* vector into a sentiment prompt scores 0.883, higher than
sentiment's own vector at 0.817. The gain is a generic nudge toward emitting a label
token, not operator identity, and by the criterion I fixed before looking at the data
this licenses nothing. Matching gains nothing at all, 0.517 both with and without
injection, identical to three decimal places against an in-context ceiling of 0.908.
Scope limit: this is the simpler layer-level construction, not Todd et al.'s
head-level method with causal attribution.

**Does a select-then-match hybrid pay?** ComEM reports that selecting has high
precision and low recall while matching has the reverse, and proposes composing them.
Implemented as batched SELECT followed by per-pair MATCH on the residual candidates,
with the residual budget swept:

| configuration | calls | precision | recall | F1 |
|---|---|---|---|---|
| select only | 12 | 0.965 | 0.509 | 0.667 |
| select + match top-4 residual | 60 | 0.813 | 0.565 | 0.667 |
| select + match top-8 residual | 108 | 0.759 | 0.583 | 0.660 |
| select + match top-16 residual | 204 | 0.765 | 0.602 | 0.674 |
| select + match all residual | 368 | 0.773 | 0.630 | 0.694 |

The published precision-recall trade-off reproduces exactly, recall rising from 0.509
to 0.630 while precision falls from 0.965 to 0.773, but the two effects very nearly
cancel: F1 improves by 0.027 for thirty times the calls. The hybrid does not pay at
any budget on this data.

---

## 5. Stage 3: resolution

### 5.1 The structure that thresholding discards

Section 2.1 reported the measured cardinality of the Abt-Buy ground-truth matching:
98.8% of matched left rows have exactly one match, and the right-side degree
distribution is similarly concentrated. The join is very nearly a bipartite matching.

Independent thresholding cannot use this. Two specific failure modes follow. A
right row that is textually attractive can be accepted as the match for many
different left rows, since each of those decisions is locally plausible and the
threshold is a per-pair rule. And a correct pair whose absolute score happens to sit
below the cut is discarded even when it is unambiguously the best available match for
both of its rows.

The alternative is to treat resolution as a constrained optimisation over the whole
verdict set. I compared five resolvers, all operating on the same candidate set and
the same scores.

**threshold** keeps every pair scoring at or above a cut, with the cut chosen to
maximise F1 against validation positives.

**top-1** keeps each left row's single highest-scoring candidate. A one-sided
constraint.

**mutual best** keeps `(a,b)` only when `b` is `a`'s highest-scoring candidate and
`a` is `b`'s. A cheap two-sided approximation to 1:1, and the same reciprocal idea
that fixed blocking in section 3.1.

**assignment** solves for the set of pairs maximising total score subject to each row
appearing at most once, by the Hungarian algorithm via
`scipy.optimize.linear_sum_assignment` on the negated score matrix, with
non-candidate entries set to a large negative constant.

**assignment + floor** runs assignment and then discards assigned pairs scoring below
a floor, since assignment as stated pairs every left row including the 6% that match
nothing. The floor is swept over the achieved assignment scores and chosen on
validation positives.

The floor sweep is worth a note because I got it wrong first. My initial sweep ran
over quantiles 0.5 to 0.999 of the assigned scores, which never considers keeping
everything, and consequently `assignment + floor` came back worse than plain
`assignment`, which is impossible if the sweep is correct. The corrected sweep starts
from a floor that keeps all pairs.

### 5.2 A controlled comparison

The comparison that matters is between resolution and adjudication, since the
question is which stage is worth more. To make it fair, every cell uses identical
blocking: bge-small top-10 per left row, giving 10,810 candidate pairs out of
1,180,452 with blocking recall 0.985. Every threshold and floor is tuned on
validation positives. Every cell is scored over the whole join, all 1081 left rows
against the pooled 1028-pair ground truth.

The adjudicator is Qwen3-8B with the batched SELECT prompt over each left row's ten
candidates, 1081 calls, generated in batches of 8 with `max_new_tokens=24`, taking
378 seconds on one L4.

| scorer | resolver | precision | recall | F1 | pairs returned | LLM calls |
|---|---|---|---|---|---|---|
| cosine | threshold | 0.745 | 0.210 | 0.328 | 290 | 0 |
| cosine | top-1 | 0.734 | 0.771 | 0.752 | 1081 | 0 |
| cosine | mutual best | 0.920 | 0.683 | 0.784 | 763 | 0 |
| cosine | assignment | 0.845 | 0.879 | 0.862 | 1070 | 0 |
| cosine | assignment + floor | 0.915 | 0.712 | 0.801 | 800 | 0 |
| Qwen3-8B SELECT | its own yes/no | 0.445 | 0.793 | 0.570 | 1830 | 1081 |
| Qwen3-8B SELECT | assignment | 0.885 | 0.763 | 0.819 | 886 | 1081 |
| Qwen3-8B SELECT | assignment + floor | 0.917 | 0.724 | 0.809 | 811 | 1081 |

Reading the two ingredients against the same 0.328 baseline, which is cosine scoring
with the conventional independent threshold:

| ingredient added | F1 | change | cost |
|---|---|---|---|
| constraint solver, no LLM | 0.862 | +0.534 | 0 calls |
| LLM, no constraint solver | 0.570 | +0.243 | 1081 calls, 378 s |
| both | 0.819 | +0.491 | 1081 calls, 378 s |

Three things follow. The constraint solver is worth roughly twice what the LLM is
worth on this join, and it costs nothing. Cosine plus assignment at 0.862 exceeds the
8B model's own adjudication at 0.570 on the same candidates. And the LLM's marginal
contribution on top of the solver is negative, 0.862 falling to 0.819, so on this
join the 1081 calls purchase a loss of 0.043 F1.

The intermediate resolvers are informative about where the gain comes from. Plain
top-1, which is a one-sided constraint requiring no solver at all, already reaches
0.752 from the threshold's 0.328. Mutual best adds the second side and reaches 0.784
with much higher precision but lower recall, since it refuses to commit when the two
rows disagree about their best partner. Assignment resolves those disagreements
globally instead of dropping them, which is where the remaining recall comes from,
0.683 rising to 0.879. So most of the effect is available from very cheap
approximations, and the solver captures the rest.

### 5.3 Why the LLM underperforms here

The LLM cell's precision is 0.445, from 1,830 accepted pairs against 1,028 true
matches. It nearly doubled the size of the correct answer set. This is not mysterious
and it is not a new finding within this document: it is exactly the effect measured
in section 4.3. Blocking hands the model ten candidates of which at most one is
usually correct, which is close to the m=1 regime where measured precision is 0.36,
and the model will not answer "none".

Two structural features of the pipeline then combine badly. The model over-accepts,
and its output is a set of independent binary decisions with nothing downstream to
reconcile conflicts between them. Assignment is precisely that reconciliation, which
is why it lifts the LLM cell from 0.570 to 0.819. It performs better still starting
from a graded cosine score than from the model's binary mask, because a binary mask
has discarded the information the solver uses to choose between competing claims.

The claim this supports is narrow, and I want to state it narrowly. It is not that
LLMs are unsuitable for joins. It is that on a join with strong cardinality
structure, enforcing that structure is worth more than judging the pairs, and that an
LLM asked to judge candidates independently will systematically over-accept in the
sparse regime that blocking itself creates.

---

## 6. Threats to validity

The main result depends on join structure that not every join has. Abt-Buy is 98.8%
one-to-one with 94% of left rows matched, which is the structure assignment
exploits, and it is the reason plain assignment beats the floored variant here. On a
many-to-many join, or one where most left rows match nothing, unfloored assignment
would be actively wrong, since it pairs every left row. The floored variant at 0.801
is the version for lower match rates and it still beats thresholding substantially,
but I have not tested a genuinely sparse or many-to-many join, and the honest position
is that what generalises is the question rather than the number.

The headline rests on one benchmark. Walmart-Amazon and WDC Products are the obvious
next datasets. DBLP-ACM is unsuitable for the reason given in section 2.1.

The LLM cell is one model with one prompt. A stronger model or per-pair judging may do
better: Gemini flash-lite per-pair reached 0.945 on a stratified subsample of this
dataset. That subsample had a 60% positive rate against the natural 28% and its
numbers are therefore not comparable to the whole-join figures in section 5.2, which
is exactly why I re-ran the comparison with Qwen locally instead of quoting it. A
frontier model over the full join is the missing measurement.

Every probing result here is correlational. A probe demonstrates that information is
linearly decodable from a representation, not that the model uses it. Activation
patching is the causal instrument and I did not use it, which also means the layer
anatomy in section 3.2 is weaker evidence than the published activation-patching
result it happens to agree with.

Assignment is cubic in the table size. It is negligible at 1081 by 1092. A
million-row join needs the sparse or streaming variants, or auction algorithms, and I
have not measured any of those.

The 20NG absolute numbers are unreliable for the labelling reason in section 4.4.
Comparisons between strategies on that dataset are valid; the levels are not.

---

## 7. What we did not do

The knobs, marked honestly:

| stage | knob | status |
|---|---|---|
| blocking | directionality | done, 0.889 to 1.000 |
| | representation and layer | done, LLM states 30x worse than bge |
| | learned indexable metric | done, does not factor |
| | index parameters (nProbes, topNScan) | done in earlier sessions |
| | selectivity estimation | not done |
| adjudication | per pair versus batched | done, inverts on density |
| | batch size | done, inverted U, mechanism unknown |
| | prompt chunking | done, harmful |
| | predicate wording | done, +0.102 |
| | candidate order | done, +0.294, slot 1 only |
| | query placement | done, no net gain |
| | exit depth | done, 3.3x under batching |
| | distillation | done, needs 10x the labels we have |
| | per-candidate verdicts instead of free recall | not done |
| resolution | independent threshold | done, the baseline |
| | top-1, mutual best, assignment, floored assignment | done, the main result |
| | assignment on calibrated LLM scores | not done |
| | transitivity for self-join and dedup | not done |
| | capacity-constrained assignment for m:n joins | not done |

Three of these are worth more than the rest.

The first is assignment over calibrated LLM scores. The model currently emits a
binary mask, which discards precisely the graded information the solver consumes, and
section 5.3 shows the solver already does better from cosine than from that mask. If
per-candidate scores can be extracted, either from logits in the manner of FIRST or
by asking for graded output, assignment over those scores should dominate every cell
in section 5.2. That is a concrete prediction and it is cheap to test.

The second is per-candidate verdicts rather than free recall. The over-acceptance in
section 5.3 arises partly because the prompt invites the model to name whichever
candidates it likes. Requiring an explicit verdict for every numbered candidate
forces it to consider each one, and structured output makes this enforceable.

The third is semantic selectivity estimation, which is the oldest problem in query
optimisation and which every semantic-operator system needs. CADENZA explicitly
abandons cross-operator optimisation for want of it. Our own density rule from section
4.3 needs it as an input, and section 4.7 makes it cheaper than it used to be, since
a shallow forward pass costs about 32 ms.

---

## 8. Process notes and errors

Four mistakes shaped the results enough that omitting them would misrepresent how the
work went.

**A broken baseline manufactured a headline.** The latent-knowledge experiment in
section 4.9 initially showed generation at recall 0.053 against a probe at 0.774,
which is dramatic support for "the model knows things it will not say". It was the
chat-template and thinking-mode bug from section 2.2. Corrected, generation reached
0.888 and the hypothesis was dead. A recall of 0.053 from an 8B instruction-tuned
model is implausible on its face, and a comparable figure of 0.825 from a different
model was already in my notes, so the number should have been questioned before it
was interpreted.

**A control caught overfitting that I would otherwise have written up as a finding.**
The learned projection in section 3.3 dropped bge from 0.772 to 0.184, and the
conclusion I was already inclined toward was that match predicates do not factor. The
random-projection control, at 0.728, showed the training was broken rather than the
idea. The fairly-trained version then produced a negative result anyway, but for an
entirely different and more informative reason. Including that control was the
difference between a wrong argument and a right one for the same conclusion.

**I evaluated on the easy subset.** The first resolution result was 0.906, scored only
over left rows that have a match, which excludes exactly the cases where a
force-pairing resolver does harm. Over the whole join it is 0.884. The deflation is
small, but I found it myself only because I went looking, and the version that stopped
at 0.906 would have been published.

**I diagnosed from theory rather than logs.** When a batched run failed twice I
confidently attributed it to a CUDA out-of-memory condition and redesigned around
memory pressure, shrinking batch sizes and adding incremental checkpointing. The log
said `ModuleNotFoundError: No module named 'sklearn'`, because I had omitted it from
that script's image. Two relaunches wasted on a theory that fit my expectations.

The pattern common to the first, second and third is that the failure which confirms
your hypothesis is the failure you will not check. In two cases the exciting result
was my own harness, and in both a number that should have looked implausible went
unexamined because it pointed in the direction I wanted.

One methodological note that generalises. Section 2.3 established that the model is
deterministic at temperature 0, which means all statistical power comes from distinct
examples rather than repeated sampling. My pilot experiments used n=20 per cell, which
gives a standard error of roughly 0.11 on a proportion near 0.5, across roughly
fifteen comparisons. The pattern happened to survive at n=150 with paired McNemar
tests, but it could as easily not have, and the pilot was not evidence.

---

## 9. Summary

The three stages of a semantic join were measured against each other on a common
benchmark with identical blocking, and the ordering was the reverse of where the
effort usually goes.

Candidate generation is close to solved by cheap methods, reaching perfect candidate
recall with a reciprocal-neighbour construction that costs nothing, and it cannot be
made predicate-aware, because the join predicate lives in the interaction between two
records and does not factor into per-row index keys. Adjudication is where nearly all
published effort goes and it yields a substantial collection of real but individually
modest effects: batching that is simultaneously cheaper and better, a strategy choice
that inverts with match density, a prompt-wording effect larger than any technique in
the section, a positional effect that is statistically overwhelming and practically
negligible, and an exit-depth saving of 3.3x under realistic batching. Resolution is
almost never studied, is a threshold by default, and on this join was worth more than
twice what the language model was worth while costing nothing.

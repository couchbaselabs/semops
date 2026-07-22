# `sem_filter` architecture

```
rows ──► sample K rows ──► oracle labels them        (K LLM calls)
     │                          │
     │                     calibrate τ⁻ / τ⁺          (Wilson bounds, pure math)
     ▼                          ▼
   proxy scores ──────────► band every row ──► accept | escalate→LLM | reject
```

## Proxy, two modes

1. **Cosine-to-predicate (default).** Embed the predicate, score each row by
   similarity to it. Zero setup, and weak: it only works when the predicate is
   topic-shaped, and not reliably even then. Measured F1 0.914 on a 20 Newsgroups
   topic filter but only 0.512 on an AG News one, same embedding model.

2. **Learned (`proxy_model=`).** Fit a logistic regression on the sampled
   `(embedding → label)` pairs. This is what the literature's big wins use. It
   learns which directions in embedding space predict the label, instead of
   assuming proximity to a phrase.

## Calibration

Thresholds are set by Wilson score lower bounds at confidence `1 − delta/2`, so
they are statistically defensible rather than tuned. If a target cannot be
certified from the sample, that band vanishes and its rows escalate.

The sample is **cross-fitted, not split.** An earlier version held out a third of
the sample for calibration and fit on the rest. That left too few labelled points
for the bounds to certify anything: at 100 samples, 34 calibration points could
not certify precision 0.9 (you need roughly 45 consecutive positives), so `τ⁺`
stayed at `+inf` and every row escalated. `_oof_proba()` now scores each labelled
row with a model that did not train on it, so all labels reach the calibrator
while the thresholds stay honest.

Calibration happens in probability space and the thresholds are then mapped onto
the index's dot-product scale via `logit(p) − bias`. Fold models' raw dot scales
are not comparable to each other; their probabilities are.

## Server-side pushdown

An LR proxy is linear, so its score is a dot product with a weight vector, which
is exactly what a vector index computes. Train on the small sample in-process,
ship the weight vector to the database, score everything with
`VECTOR_DISTANCE(embedding, $w, 'dot')`. Only scalars come back.

Measured, identical F1 and identical oracle calls either way:

| collection | vectors over the wire |
|---|---|
| 2,000 reviews | 2,000 → 100 |
| 50,000 news | 50,000 → 2,500 |

95% at both sizes, and not by coincidence: the only embeddings that leave are the
calibration sample's, so the saving is exactly `1 − sample_frac` at any collection
size.

This only holds because `sess.scan()` leaves vectors in the store when the engine
can score server-side. Scan them out first and the pushdown is decorative: the
vectors have already crossed the wire and the measured saving is 0%. Passing
`with_vectors=True` puts you back in that case.

## Index-served filtering (`Session.sem_filter` over a source)

The cascade never needs a score for a rejected row, only for the accept and
escalate bands. So calibrate on the sample, then ask the index for the rows above
`τ⁻` and reject everything it does not return. See [`gsi_notes.md`](gsi_notes.md)
for the three constraints that govern whether this is sound.

## Knobs

| knob | meaning |
|---|---|
| `recall` / `precision` / `delta` | guarantee targets |
| `min_sample` / `max_sample` | calibration sample size |
| `policy` | `cascade` \| `oracle` (judge everything) \| `proxy` (never escalate) |
| `on_error` | `reject` \| `accept` \| `raise` |
| `workers` | parallel oracle calls |
| `est_k` | starting fetch size for the index-served path |

## Failure modes

- Severe class imbalance starves the sample of positives, so bands cannot certify
  and everything escalates. Safe, but not cheap.
- A learned proxy imitates the oracle, so it inherits the oracle's mistakes.
- A weak proxy costs money, not correctness: `τ⁺` fails to certify, the accept
  band vanishes, and every kept row gets an oracle call.

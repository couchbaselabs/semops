# Resumable cascades

**Status:** first experiment, November 2026. Not a paper.

## What it is

A semantic filter cascade usually has two separate models: a cheap proxy that
decides the easy rows, and an expensive LLM (the oracle) for the rest. semops does
this today with a logistic-regression proxy on `bge-small` embeddings and an LLM
judge.

A **resumable cascade** makes the cheap stage the *first layers of the expensive
model*. A linear probe reads the oracle's residual stream at layer L and sorts
each row into ACCEPT, REJECT or ESCALATE with semops' existing three-band
calibration. Escalated rows do not start over: the forward pass resumes from
layer L.

Sharing the work gives three properties a two-model cascade does not have:

1. **Escalation costs only the remaining layers.** Even if every row escalates,
   the total is about one full pass, so the cascade is never meaningfully worse
   than running the model alone.
2. **Calibration labels are free.** The calibration sample must be labelled by
   the oracle anyway. Those same forward passes produce the hidden states at every
   layer, so the probe's training data costs nothing extra.
3. **The guarantee is about the model itself.** Precision and recall are certified
   against the full model's own answer, not against a different judge.

## Why it should work

From [FINDINGS.md](../../FINDINGS.md), Qwen3-8B on an NVIDIA L4:

- Operator decisions are computed mid-network: probes on three tasks peak at
  layer 20 of 36 and decline after (§4.3), matching activation-patching work on
  relevance judgments.
- A probe at layer 12 matched generation (F1 0.901 against 0.889) at about 3.3x
  the throughput at batch 32 (§4.8).
- The current proxy is weakest where it matters: embedding similarity measures
  what a text is *about*, not what it *concludes* (§1).

## Design

```
Stage 1 (every row)       [instruction + row] through layers 1..L, probe -> ACCEPT / REJECT / ESCALATE
Stage 2 (escalated rows)  resume layers L+1..N from the saved layer-L state, read yes/no from the head
```

- **Calibration** is `semops.cascade.calibrate` unchanged (Wilson bounds), with
  out-of-fold probe scores as in `operators._oof_proba`.
- **The exit layer is a planner choice per predicate**, picked on the calibration
  sample by expected cost: `t_stage1(L) + e(L) * t_resume(L)`, where `e(L)` is the
  share of the sample in the escalate band.
- **What stage 2 needs** for a single-token answer is only each token's hidden
  state at layer L: for Qwen3-8B, 4,096 numbers (8 KB in bf16) per token, against
  48 KB to keep 12 layers of KV and 144 KiB for the full cache.
- **Rows leaving a batch mid-pass** become two physical operators (layers 1..L and
  L+1..N) with the cascade as a filter between them, so escalated rows from many
  stage-1 batches are re-batched for stage 2.

## Experiment (November 2026)

**Setup.**

- **Oracle:** Qwen3-8B in bf16. Its single-token yes/no answer, read from the output head with no decode, is the oracle, so every precision and recall number below is relative to what the model itself answers.
- **Predicates:** 2,000 rows each.

  | Predicate | Data | Model agrees with labels |
  |---|---|---|
  | positive review | Rotten Tomatoes | 0.909 |
  | about business | AG News | 0.773 |
  | about science or technology | AG News | 0.862 |

  On business the model says yes to 43% of articles against a 26% true rate.
- **Cascade:** `semops.cascade.calibrate` with `sem_filter`'s defaults (recall 0.9, precision 0.9, δ 0.1).
  - The proxy is standardized logistic regression (C = 0.1) with 5-fold out-of-fold scores.
  - The calibration sample is n = 100 (the default for a 2,000-row table) or n = 400 (the default 5% of an 8,000-row table).
  - Each setting is repeated over 30 random samples.
- **Arms:**
  - `bge`: the current semops proxy (bge-small embeddings of the row).
  - `qwen3-0.6b`: a separate small model's last hidden state.
  - `Lk`: a probe on Qwen3-8B's residual stream at layer k, then *restart* (a full pass) or *resume* (layers k+1..36) for escalated rows.
  - `chosen`: the exit layer picked per sample by expected cost, using the calibration sample only.
- **Timing:** milliseconds per row on an NVIDIA A10 (Modal; L4s were out of capacity), batch 16, median of 3 runs over 256 rows. Cost per row = stage-1 cost + escalation rate × (full pass or resume).
- **Exactness, checked on the GPU:** resuming from layers 8, 12 and 20 reproduces the full pass exactly (max difference in the yes/no margin: 0.0). The manual layer loop also matches the Hugging Face forward pass exactly.

**Results.** Speedup over one full single-token pass (higher is better), with the escalation rate in brackets:

| Predicate | n | bge cascade | Qwen3-0.6B cascade | early exit, restart | early exit, resume | exit layers chosen |
|---|---|---|---|---|---|---|
| RT positive | 100 | 1.40x (0.69) | 1.04x (0.82) | 1.39x | **1.53x** (0.12) | 20-24 |
| RT positive | 400 | 1.74x (0.56) | 1.64x (0.47) | 1.70x | **1.75x** (0.04) | 18-20 |
| AG business | 100 | 1.39x (0.69) | 1.06x (0.81) | 1.42x | **1.50x** (0.06) | 20-24 |
| AG business | 400 | **2.49x** (0.37) | 1.90x (0.39) | 1.74x | 1.92x (0.33) | mostly 4-6 |
| AG sci/tech | 100 | 1.10x (0.88) | 0.97x (0.90) | 0.82x | **1.24x** (0.47) | 24, 36 |
| AG sci/tech | 400 | 1.44x (0.67) | 1.21x (0.69) | 1.44x | **1.54x** (0.11) | 20-24 |

Both targets (precision and recall ≥ 0.9 against the model) were met in 93-100% of samples by the early-exit cascade. The bge cascade met them in 87-100%, and fell below the 90% design point once (RT, n = 100).

**What the numbers say.**

1. **Resuming makes the cascade safe; restarting doesn't.** Across all 78 layer, predicate and sample-size cells, resume never fell below 0.96x of a full pass; the 4% is repeated input setup. Restart fell as low as 0.73x. With resume, a bad exit-layer choice costs almost nothing.
2. **The gains are modest.** Resumable early exit matched or beat the better separate proxy in 5 of 6 settings, by 0 to 13%. It lost by 23% on AG business at n = 400, a topical predicate where bge escalates only 37% of rows.
3. **The floor is the exit depth.** A *guaranteed* exit needed layers 18 to 24 of 36. At layer 12, the depth §4.8 used, the probe escalated 52-89% of rows once the Wilson bounds apply. This matches the mid-network peak in §4.3. So every row pays about 55% of a pass, and a near-free embedding proxy wins whenever it escalates fewer than about 55% of rows.
4. **Early exit's strength is the escalation rate.** On Rotten Tomatoes at n = 400 it sends 3.6% of rows to the full model, against 56% for bge, at equal or better quality (F1 against labels: 0.900 against 0.892).

**Next.** Points 3 and 4 suggest composing the stages instead of choosing between them:
- bge first, at almost no cost;
- the layer-20 probe only on bge's escalated band;
- resume only where both are unsure.

This is not measured yet.

**Limits.**
- One model, three predicates, 2,000 rows each.
- An A10, not the L4 used in FINDINGS.
- Batch 16, and no prefix caching in any arm.
- The guarantee is relative to the model, which over-includes on AG business.

## Reproduce

```
modal deploy modal_run.py
python -c "import modal; modal.Function.from_name('semops-resumable-cascades', 'pipeline').spawn()"
modal volume get semops-resumable-cascades results/analyze.json results/
```

Raw numbers: [`results/analyze.json`](results/analyze.json) (all arms and layers), [`results/timing.json`](results/timing.json), [`results/extract.json`](results/extract.json).

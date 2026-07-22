"""Re-measure every learned-proxy number under the fixed (cross-fitted) calibration.

The old fit/calibrate split left ~34 labelled points for the Wilson bounds, which
could not certify precision 0.9 — so tau went infinite and the cascade escalated
100% of rows. Every learned-proxy result we published was therefore really
"oracle on all rows", which is why quality looked perfect and savings looked
absent. This re-runs them with _oof_proba in place.

Oracles are label-based (free, perfect) so the numbers isolate the cascade from
LLM noise — the same convention used for the earlier runs being replaced.

  ./.venv/bin/python examples/rerun_benchmarks.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import InMemoryEngine, Row
from semops.vectormath import cosine_similarity
from cb_common import FastEmbedEmbedder
from eval_classification import SklearnLRProxy, load_20ng, DEFAULT_CATEGORIES


class LabelOracle:
    def __init__(self, emb, truth_by_text):
        self._e, self.truth = emb, truth_by_text
        self.embed_model, self.chat_model = emb.embed_model, "label-oracle"
        self.spent_usd, self.calls = 0.0, 0

    def embed(self, t):
        return self._e.embed(t)

    def judge(self, predicate, text):
        self.calls += 1
        return bool(self.truth.get(text, False))


def prf(truth, pred):
    tp = sum(1 for y, p in zip(truth, pred) if y and p)
    fp = sum(1 for y, p in zip(truth, pred) if p and not y)
    fn = sum(1 for y, p in zip(truth, pred) if y and not p)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def best_threshold_f1(scores, labels):
    """Oracle-tuned vector-only baseline: the best F1 any single cosine cutoff
    could reach, chosen WITH the labels. A generous upper bound, not achievable
    in production — stated as such wherever it is quoted."""
    best = (0.0, 0.0, 0.0)
    for t in sorted(set(scores)):
        pred = [s >= t for s in scores]
        p, r, f = prf(labels, pred)
        if f > best[2]:
            best = (p, r, f)
    return best


def run(name, texts, labels, predicate, emb, targets=(0.9, 0.9)):
    vecs = emb.embed(texts)
    pv = emb.embed([predicate])[0]
    truth_by_text = dict(zip(texts, labels))

    # vector-only baseline (unaffected by the calibration fix, re-run for parity)
    sims = [cosine_similarity(pv, v) for v in vecs]
    bp, br, bf = best_threshold_f1(sims, labels)

    eng = InMemoryEngine()
    rows = [Row(id=f"r{i}", text=t, embedding=v, doc={"text": t, "label": l})
            for i, (t, v, l) in enumerate(zip(texts, vecs, labels))]
    eng.add("c", rows)
    model = LabelOracle(emb, truth_by_text)
    rec, prec = targets
    sess = semops.connect(engine=eng, model=model, workers=8)
    res = sess.rows(rows).sem_filter(predicate, proxy_model=SklearnLRProxy(),
                                     recall=rec, precision=prec)
    kept = {r.id for r in res.rows_}
    pred = [f"r{i}" in kept for i in range(len(rows))]
    p, r, f = prf(labels, pred)
    st = res.stats()
    print(f"  {name:28} n={len(rows):5} pos={sum(labels)/len(labels):5.1%}")
    print(f"    vector-only (oracle-tuned)  P={bp:.3f} R={br:.3f} F1={bf:.3f}")
    print(f"    cascade r{rec}/p{prec}          P={p:.3f} R={r:.3f} F1={f:.3f}  "
          f"calls={st['llm_calls']:5}/{len(rows)}  savings={len(rows)/max(st['llm_calls'],1):.2f}x")
    print(f"    tau=({st['tau_minus']:.3f}, {st['tau_plus']:.3f})  "
          f"accept={st['n_accept']} reject={st['n_reject']} escalate={st['n_escalate']}")
    return dict(name=name, n=len(rows), base_f1=bf, f1=f, calls=st["llm_calls"],
                savings=len(rows) / max(st["llm_calls"], 1))


def main():
    emb = FastEmbedEmbedder()
    out = []
    print("=" * 78)
    print("RE-RUN under cross-fitted calibration (label oracle)")
    print("=" * 78)

    # 1. 20 Newsgroups topic predicate — the README's 0.914 / 0.981 pair
    t, l = load_20ng(DEFAULT_CATEGORIES, ["rec.sport.baseball", "rec.sport.hockey"], 2000, 0)
    out.append(run("20NG  is-about-sports", t, l,
                   "this document is about a sport (baseball, hockey, etc.)", emb))

    # 2. Rotten Tomatoes polarity — the README's 0.661 / 0.992 pair
    from cb_ingest import load_rotten
    t, l = load_rotten(2000, 0)
    out.append(run("Rotten polarity", t, l,
                   "this is a negative or critical movie review", emb))

    print()
    print("=" * 78)
    print("SUMMARY (compare against the README's current numbers)")
    print("=" * 78)
    for o in out:
        print(f"  {o['name']:28} vector-only F1={o['base_f1']:.3f}  "
              f"cascade F1={o['f1']:.3f}  savings={o['savings']:.2f}x  calls={o['calls']}")


if __name__ == "__main__":
    main()

"""A SPECIFIC predicate, judged by a real LLM — the harder case.

"is about science or technology" is topic-shaped, which is precisely where an
embedding proxy has an easy time. This runs a narrow predicate that cuts across
the corpus's topic labels, so the proxy has to separate something the embedding
was never organised around.

Evaluation protocol: AG News has no ground truth for a predicate like this, and
inventing one would be worse than useless. The cascade's guarantee is "close to
what the ORACLE would have said", so that is what we measure — a held-out random
sample is judged directly by the oracle and compared against the pipeline's
verdicts on those same rows. Rows the pipeline already escalated are answered from
cache, so the check is nearly free.

  ./.venv/bin/python examples/bench_specific.py --n 10000 --holdout 400
"""
import argparse
import os
import re
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import CouchbaseEngine, HttpQueryCluster, OpenAICompatClient
from semops.vectormath import cosine_similarity
from cb_common import BUCKET, PW, QUERY, SCOPE, USER, FastEmbedEmbedder
from eval_classification import SklearnLRProxy

PREDICATES = {
    "merger": "this story reports a merger, an acquisition, or one company buying another",
    "legal": "this story reports a lawsuit, court ruling, indictment, or other legal proceeding",
    "layoffs": "this story reports job cuts, layoffs, or a company reducing its workforce",
    # Deterministic ground truth, and deliberately orthogonal to topic: a dollar
    # figure shows up in Business, World and Sci/Tech alike, so the embedding has
    # not organised the corpus around it. Free oracle, so it runs at 50k where a
    # rate-limited LLM cannot.
    "dollar": "this story quotes a specific dollar amount of money",
}

DOLLAR_RE = re.compile(r"\$\s?\d|\bUS\$|\b\d[\d,.]*\s*(?:billion|million|trillion)\s*dollars?\b",
                       re.IGNORECASE)


class RegexOracle:
    """Exact, free oracle for the `dollar` predicate."""

    def __init__(self, emb):
        self._e = emb
        self.embed_model, self.chat_model = emb.embed_model, "regex-oracle"
        self.spent_usd, self.calls = 0.0, 0

    def embed(self, t):
        return self._e.embed(t)

    def judge(self, predicate, text):
        self.calls += 1
        return bool(DOLLAR_RE.search(text or ""))


def gemini(emb, model_name):
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("set GEMINI_API_KEY (source semops/.env.local)")
    g = OpenAICompatClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key=key, chat_model=model_name)

    class M:
        embed_model, chat_model = emb.embed_model, g.chat_model
        spent_usd, calls = 0.0, 0
        def embed(self, t): return emb.embed(t)
        def judge(self, p, t): M.calls += 1; return g.judge(p, t)
        def match_block(self, p, q, c): M.calls += 1; return g.match_block(p, q, c)
    return M()


def prf(truth, pred):
    tp = sum(1 for y, p in zip(truth, pred) if y and p)
    fp = sum(1 for y, p in zip(truth, pred) if p and not y)
    fn = sum(1 for y, p in zip(truth, pred) if y and not p)
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 1.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--holdout", type=int, default=400)
    ap.add_argument("--predicate", default="merger", choices=list(PREDICATES))
    ap.add_argument("--recall", type=float, default=0.9)
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--model", default="gemini-flash-latest")
    ap.add_argument("--oracle", default="llm", choices=["llm", "regex"])
    args = ap.parse_args()
    predicate = PREDICATES[args.predicate]

    q = HttpQueryCluster(QUERY, USER, PW, timeout=1800)
    emb = FastEmbedEmbedder()
    ks = f"`{BUCKET}`.`{SCOPE}`.`news`"
    rows = q.query(f"SELECT META(d).id AS id, d.text AS t, d.embedding AS v "
                   f"FROM {ks} d LIMIT {args.n}")
    print(f"{len(rows)} rows | predicate: {predicate!r}")
    print(f"oracle: {args.oracle if args.oracle == 'regex' else args.model}\n")

    model = (RegexOracle(emb) if args.oracle == "regex"
             else gemini(emb, args.model))
    eng = CouchbaseEngine("", USER, PW, BUCKET, SCOPE, cluster=q, nprobes=8,
                          covering_index="idx_news_vec")
    sess = semops.connect(engine=eng, model=model, workers=args.workers)

    t0 = time.time()
    res = sess.scan("news", limit=args.n).sem_filter(
        predicate, proxy_model=SklearnLRProxy(),
        recall=args.recall, precision=args.precision)
    wall = time.time() - t0
    st = res.stats()
    kept = {r.id for r in res.rows_}
    print(f"  oracle calls   {st['llm_calls']:6} / {len(rows)}   savings {len(rows)/max(st['llm_calls'],1):.2f}x")
    print(f"  bands          accept={st['n_accept']} escalate={st['n_escalate']} reject={st['n_reject']}")
    print(f"  tau            ({st['tau_minus']:.3f}, {st['tau_plus']:.3f})   errors={st['errors']}")
    print(f"  selectivity    {len(kept)/len(rows):.2%} kept   wall {wall:.0f}s\n")

    # --- held-out check against the oracle itself (the thing we actually certify)
    rng = random.Random(7)
    sample = rng.sample(rows, min(args.holdout, len(rows)))
    print(f"  judging {len(sample)} held-out rows directly with the oracle...")
    from semops.parallel import pmap
    truth = pmap(lambda r: model.judge(predicate, r["t"]), sample, args.workers)
    pred = [r["id"] in kept for r in sample]
    p, rr, f = prf(truth, pred)
    print(f"  vs oracle on held-out:  P={p:.3f} R={rr:.3f} F1={f:.3f}  "
          f"(oracle says {sum(truth)}/{len(sample)} = {sum(truth)/len(sample):.1%} true)")

    # --- what would vector search alone have done on this predicate?
    pv = emb.embed([predicate])[0]
    sims = {r["id"]: cosine_similarity(pv, r["v"]) for r in rows}
    ss = [sims[r["id"]] for r in sample]
    best = (0.0, 0.0, 0.0)
    for t in sorted(set(ss)):
        pp, rrr, ff = prf(truth, [s >= t for s in ss])
        if ff > best[2]:
            best = (pp, rrr, ff)
    print(f"  vector-only (oracle-tuned threshold):  P={best[0]:.3f} R={best[1]:.3f} F1={best[2]:.3f}")


if __name__ == "__main__":
    main()

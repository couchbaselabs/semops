"""BioDEX semantic-join benchmark (the corpus LOTUS and Abacus both publish on).

Task: join biomedical case reports to the adverse REACTIONS they describe.
  left  = papers (title + abstract)
  right = distinct MedDRA-style reaction terms
  predicate = "the medical report describes this adverse reaction"
  ground truth = the paper's own `reactions:` list from BioDEX-ICSR's target field

This is an ASYMMETRIC join (long document <-> short label), which stresses
embedding blocking differently from the symmetric 20NG case: the fan-out is tiny
(~2-4 true reactions per paper) but the right side is a large vocabulary.

  ./.venv/bin/python examples/bench_biodex.py --oracle truth     # isolates blocking
  ./.venv/bin/python examples/bench_biodex.py --oracle gemini    # real LLM reasoning
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import InMemoryEngine, Row
from semops.vectormath import cosine_similarity
from cb_common import FastEmbedEmbedder

PREDICATE = "the medical report describes this adverse reaction"


def parse_reactions(target: str) -> list[str]:
    m = re.search(r"reactions:\s*(.+)", target)
    if not m:
        return []
    return [r.strip() for r in m.group(1).split(",") if r.strip()]


def load(n_papers, seed=0, max_chars=700):
    from datasets import load_dataset
    d = load_dataset("BioDEX/BioDEX-ICSR", split="train")
    papers, truth_by_paper = [], {}
    for i in range(len(d)):
        if len(papers) >= n_papers:
            break
        rx = parse_reactions(d[i]["target"] or "")
        title = (d[i]["title"] or "").strip()
        abstract = (d[i]["abstract"] or "").strip()
        if not rx or not abstract:
            continue
        pid = f"p{len(papers)}"
        papers.append({"id": pid, "text": (title + ". " + abstract)[:max_chars]})
        truth_by_paper[pid] = set(rx)
    vocab = sorted({r for s in truth_by_paper.values() for r in s})
    return papers, vocab, truth_by_paper


def prf(truth, got):
    tp = len(truth & got)
    p = tp / len(got) if got else 1.0
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


class TruthOracle:
    def __init__(self, emb, truth_pairs, by_text):
        self._e, self.truth, self.by_text = emb, truth_pairs, by_text
        self.embed_model, self.chat_model = emb.embed_model, "truth-oracle"
        self.spent_usd, self.calls = 0.0, 0

    def embed(self, t):
        return self._e.embed(t)

    def _pair(self, a, b):
        ia, ib = self.by_text.get(a), self.by_text.get(b)
        return (ia, ib) in self.truth or (ib, ia) in self.truth

    def judge(self, predicate, text):
        self.calls += 1
        try:
            lt, rt = text.split("ITEM A:\n", 1)[1].split("\n\nITEM B:\n", 1)
        except Exception:
            return False
        return self._pair(lt, rt)

    def match_block(self, predicate, query, candidates):
        self.calls += 1
        return [i for i, c in enumerate(candidates) if self._pair(query, c)], True


def gemini_model(emb):
    from semops import OpenAICompatClient
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("set GEMINI_API_KEY (source semops/.env.local)")
    g = OpenAICompatClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key=key, chat_model=os.environ.get("SEMOPS_CHAT_MODEL", "gemini-flash-latest"))

    class M:
        embed_model, chat_model = emb.embed_model, g.chat_model
        spent_usd, calls = 0.0, 0
        def embed(self, t): return emb.embed(t)
        def judge(self, p, t): M.calls += 1; return g.judge(p, t)
        def match_block(self, p, q, c): M.calls += 1; return g.match_block(p, q, c)
    return M()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-papers", dest="n_papers", type=int, default=150)
    ap.add_argument("--block-k", dest="block_k", type=int, default=30)
    ap.add_argument("--oracle", default="truth", choices=["truth", "gemini"])
    args = ap.parse_args()

    emb = FastEmbedEmbedder()
    print(f"loading BioDEX-ICSR ({args.n_papers} papers)...")
    papers, vocab, truth_by_paper = load(args.n_papers)
    truth = {(p["id"], f"r{vocab.index(rx)}")
             for p in papers for rx in truth_by_paper[p["id"]]}
    fan = len(truth) / len(papers)
    print(f"  {len(papers)} papers x {len(vocab)} distinct reactions = {len(papers)*len(vocab)} pairs")
    print(f"  true pairs: {len(truth)}  fan-out ~{fan:.1f} reactions/paper  "
          f"selectivity {len(truth)/(len(papers)*len(vocab)):.4f}")
    print(f"  e.g. reactions: {vocab[:6]}\n")

    p_vecs = emb.embed([p["text"] for p in papers])
    r_rows = [{"id": f"r{i}", "text": t} for i, t in enumerate(vocab)]
    r_vecs = emb.embed([r["text"] for r in r_rows])
    by_text = {p["text"]: p["id"] for p in papers}
    by_text.update({r["text"]: r["id"] for r in r_rows})

    # baseline: pure embedding top-k (no LLM) at the true fan-out
    k = max(1, round(fan))
    got = set()
    for p, pv in zip(papers, p_vecs):
        order = sorted(range(len(r_rows)), key=lambda i: cosine_similarity(pv, r_vecs[i]), reverse=True)
        for i in order[:k]:
            got.add((p["id"], r_rows[i]["id"]))
    pp, rr, ff = prf(truth, got)
    print(f"{'embedding top-k (k=fan-out, no LLM)':42} P={pp:.2f} R={rr:.2f} F1={ff:.2f}  calls=0")

    eng = InMemoryEngine()
    eng.add("reactions", [Row(id=r["id"], text=r["text"], embedding=v, doc=r)
                          for r, v in zip(r_rows, r_vecs)])
    left = [Row(id=p["id"], text=p["text"], embedding=v, doc=p)
            for p, v in zip(papers, p_vecs)]
    model = TruthOracle(emb, truth, by_text) if args.oracle == "truth" else gemini_model(emb)

    for name, kw in [
        ("sem_join cascade + block-join", dict(block_adjudicate=True, recall=0.9,
                                               precision=0.9, min_sample=40)),
    ]:
        sess = semops.connect(engine=eng, model=model, workers=4)
        j = sess.rows(left).sem_join("reactions", PREDICATE, block_k=args.block_k, **kw)
        pp, rr, ff = prf(truth, set(j.id_pairs()))
        st = j.stats.as_dict()
        print(f"{name:42} P={pp:.2f} R={rr:.2f} F1={ff:.2f}  "
              f"calls={st['oracle_calls']:4} (block={st['block_calls']}) "
              f"cand={st['candidate_pairs']} vs nested_loop={st['nested_loop_calls']}")

    # how much recall did BLOCKING make reachable at this block_k?
    reach = set()
    for p, pv in zip(papers, p_vecs):
        order = sorted(range(len(r_rows)), key=lambda i: cosine_similarity(pv, r_vecs[i]), reverse=True)
        for i in order[:args.block_k]:
            reach.add((p["id"], r_rows[i]["id"]))
    print(f"\n  blocking recall @ block_k={args.block_k}: {len(truth & reach)/len(truth):.2f} "
          f"(ceiling on join recall)")


if __name__ == "__main__":
    main()

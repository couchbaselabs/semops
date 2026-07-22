"""Trummer's "Emails" semantic-join benchmark (arXiv 2510.08489, Table 2).

Enron-flavoured CONTRADICTION join — the case where embedding-based matching is
documented to collapse (his embedding join scored F1 = 0; LOTUS inherited it).

Spec from the paper:
  statements: "[Name]: I first heard about the losses in [Month Year]"   (10 rows)
  emails:     "I first told [Name] about the losses [TimeFrame]"        (100 rows)
  predicate:  "the two texts contradict each other"
  selectivity 0.01  -> 10 true contradictions out of 1000 pairs

Ground truth: an email contradicts a statement iff SAME name AND the email's
date is EARLIER than the date the person claims they first heard.

  ./.venv/bin/python examples/bench_emails.py --oracle truth
  ./.venv/bin/python examples/bench_emails.py --oracle gemini
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import InMemoryEngine, Row
from semops.vectormath import cosine_similarity
from cb_common import FastEmbedEmbedder

NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry", "Irene", "Jack"]
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
PREDICATE = "the two texts contradict each other"


def build(seed=0, emails_per_name=10):
    """10 statements, 100 emails, exactly 1 contradiction per name (selectivity 0.01)."""
    rng = random.Random(seed)
    claimed = {n: rng.randint(4, 9) for n in NAMES}  # month index the person claims

    statements, emails, truth = [], [], set()
    for i, n in enumerate(NAMES):
        statements.append({
            "id": f"s{i}",
            "text": f"{n}: I first heard about the losses in {MONTHS[claimed[n]]} 2022",
            "name": n})

    e = 0
    for n in NAMES:
        # exactly one email EARLIER than the claim (a contradiction), rest later
        offsets = [-rng.randint(1, 3)] + [rng.randint(1, 3) for _ in range(emails_per_name - 1)]
        rng.shuffle(offsets)
        for off in offsets:
            m = max(0, min(11, claimed[n] + off))
            emails.append({
                "id": f"e{e}",
                "text": f"I first told {n} about the losses in {MONTHS[m]} 2022",
                "name": n, "month": m})
            if m < claimed[n]:
                sid = f"s{NAMES.index(n)}"
                truth.add((sid, f"e{e}"))
            e += 1
    return statements, emails, truth, claimed


def prf(truth, got):
    tp = len(truth & got)
    p = tp / len(got) if got else 1.0
    r = tp / len(truth) if truth else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


class TruthOracle:
    """Perfect contradiction oracle (isolates BLOCKING from oracle quality)."""

    def __init__(self, emb, truth_pairs, by_text):
        self._e = emb
        self.truth = truth_pairs
        self.by_text = by_text
        self.embed_model = emb.embed_model
        self.chat_model = "truth-oracle"
        self.spent_usd = 0.0
        self.calls = 0

    def embed(self, texts):
        return self._e.embed(texts)

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

    class M:  # local embeddings + Gemini reasoning
        embed_model = emb.embed_model
        chat_model = g.chat_model
        spent_usd = 0.0
        calls = 0
        def embed(self, t): return emb.embed(t)
        def judge(self, p, t): M.calls += 1; return g.judge(p, t)
        def match_block(self, p, q, c): M.calls += 1; return g.match_block(p, q, c)
    return M()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="truth", choices=["truth", "gemini"])
    ap.add_argument("--block-k", dest="block_k", type=int, default=100)
    args = ap.parse_args()

    emb = FastEmbedEmbedder()
    statements, emails, truth, claimed = build()
    print(f"Emails benchmark (Trummer 2510.08489): {len(statements)} statements x {len(emails)} emails "
          f"= {len(statements)*len(emails)} pairs")
    print(f"  true contradictions: {len(truth)}  (selectivity {len(truth)/(len(statements)*len(emails)):.3f})")
    print(f"  predicate: {PREDICATE!r}\n")
    print(f"  e.g. statement: {statements[0]['text']}")
    ex = next(e for e in emails if (statements[0]['id'], e['id']) in truth)
    print(f"       contradicting email: {ex['text']}\n")

    s_vecs = emb.embed([s["text"] for s in statements])
    e_vecs = emb.embed([e["text"] for e in emails])
    by_text = {s["text"]: s["id"] for s in statements}
    by_text.update({e["text"]: e["id"] for e in emails})

    # --- baseline 1: Trummer's "embedding join" = match each email to its most similar statement
    got = set()
    for e, ev in zip(emails, e_vecs):
        best = max(range(len(statements)), key=lambda i: cosine_similarity(ev, s_vecs[i]))
        got.add((statements[best]["id"], e["id"]))
    p, r, f = prf(truth, got)
    print(f"{'embedding join (1-NN, his baseline)':42} P={p:.2f} R={r:.2f} F1={f:.2f}  llm_calls=0")

    # --- our operator
    eng = InMemoryEngine()
    eng.add("emails", [Row(id=e["id"], text=e["text"], embedding=v, doc=e)
                       for e, v in zip(emails, e_vecs)])
    left = [Row(id=s["id"], text=s["text"], embedding=v, doc=s)
            for s, v in zip(statements, s_vecs)]

    model = TruthOracle(emb, truth, by_text) if args.oracle == "truth" else gemini_model(emb)

    for name, kw in [
        ("sem_join blocking-only (no LLM)", dict(policy="blocking")),
        ("sem_join cascade + block-join",   dict(block_adjudicate=True, recall=0.9, precision=0.9,
                                                 min_sample=30)),
        ("sem_join policy=oracle",          dict(policy="oracle", block_adjudicate=True)),
    ][: 2 if args.oracle == "gemini" else 3]:   # full-oracle = 1000 calls, skip on rate-limited key
        sess = semops.connect(engine=eng, model=model, workers=4)
        j = sess.rows(left).sem_join("emails", PREDICATE, block_k=args.block_k, **kw)
        p, r, f = prf(truth, set(j.id_pairs()))
        st = j.stats.as_dict()
        print(f"{name:42} P={p:.2f} R={r:.2f} F1={f:.2f}  "
              f"calls={st['oracle_calls']:4} (block={st['block_calls']}) cand={st['candidate_pairs']}")

    print(f"\n  oracle={args.oracle}; paper reports embedding-join F1=0.0 on Emails, "
          f"and block/adaptive joins ~2x the tuple join's F1")


if __name__ == "__main__":
    main()

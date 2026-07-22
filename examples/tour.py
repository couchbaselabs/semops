"""Every operator, end to end, on the support-ticket data from the README.

Runs with no API key and no cluster. Embeddings are real (fastembed, local), and
the oracle is a small scripted stand-in rather than an LLM, so the output is
deterministic and you can see exactly what each operator decides and why. Swap
ScriptedOracle for OpenAICompatClient / AnthropicClient and the same code runs
against a real model.

    ./.venv/bin/python examples/tour.py

For the real thing: examples/demo_couchbase.py (live cluster),
examples/eval_couchbase.py (measured against labels).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import semops
from semops import InMemoryEngine, Row

TICKETS = [
    ("t1", "Payment failed three times but my card was charged anyway. "
           "Refund me or I'm cancelling my account."),
    ("t2", "Charged twice for the same order, please refund the duplicate."),
    ("t3", "The export button does nothing on Safari."),
    ("t4", "Can't export my report in Safari, the button is dead."),
    ("t5", "Love the new dashboard, just wanted to say thanks!"),
]

KNOWN_ISSUES = [
    ("k1", "Duplicate charge when a payment retries after a gateway timeout."),
    ("k2", "Export fails silently on Safari 17 due to a popup-blocker change."),
]

# What a competent human (or a good LLM) would say about this toy data. Scripted
# so the demo is reproducible; the operators do not know these answers, they only
# get to ask about one row or one pair at a time.
CANCELLING = {"t1"}
CAUSED_BY = {("t1", "k1"), ("t2", "k1"), ("t3", "k2"), ("t4", "k2")}
SAME_PROBLEM = {("t3", "t4")}


class ScriptedOracle:
    """Stands in for the LLM. Same interface: judge(predicate, text) -> bool."""

    def __init__(self, embedder):
        self._e = embedder
        self.embed_model, self.chat_model = embedder.embed_model, "scripted"
        self.spent_usd, self.calls = 0.0, 0
        self._by_text = {t: i for i, t in TICKETS + KNOWN_ISSUES}

    def embed(self, texts):
        return self._e.embed(texts)

    def judge(self, predicate, text):
        self.calls += 1
        if "ITEM A:" in text:                      # a pair, from sem_join/sem_dedup
            a, b = text.split("ITEM A:\n", 1)[1].split("\n\nITEM B:\n", 1)
            ia, ib = self._by_text.get(a.strip()), self._by_text.get(b.strip())
            pair = {(ia, ib), (ib, ia)}
            return bool(pair & (CAUSED_BY | SAME_PROBLEM))
        return self._by_text.get(text.strip()) in CANCELLING

    def match_block(self, predicate, query, candidates):
        self.calls += 1
        qi = self._by_text.get(query.strip())
        out = []
        for i, c in enumerate(candidates):
            ci = self._by_text.get(c.strip())
            if {(qi, ci), (ci, qi)} & (CAUSED_BY | SAME_PROBLEM):
                out.append(i)
        return out, True


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    from cb_common import FastEmbedEmbedder
    model = ScriptedOracle(FastEmbedEmbedder())

    engine = InMemoryEngine()
    for name, rows in (("tickets", TICKETS), ("known_issues", KNOWN_ISSUES)):
        vecs = model.embed([t for _, t in rows])
        engine.add(name, [Row(id=i, text=t, embedding=v, doc={"id": i, "text": t})
                          for (i, t), v in zip(rows, vecs)])
    sess = semops.connect(engine=engine, model=model, workers=4)

    # ---------------------------------------------------------------- search
    rule("search: vector top-k. No LLM. This is the ANN lookup that feeds the rest.")
    print('  query: "problems with refunds"\n')
    for r in sess.search("tickets", "problems with refunds", k=3):
        print(f"  {r.id}  {r.text[:62]}")
    print("\n  Similarity only. Note it happily returns near-misses: that is the")
    print("  behaviour the semantic operators exist to correct.")

    # ------------------------------------------------------------ sem_filter
    rule("sem_filter: a WHERE clause written in English")
    print('  predicate: "the customer is threatening to cancel"\n')
    p = sess.scan("tickets").sem_filter(
        "the customer is threatening to cancel", recall=0.9, precision=0.9)
    for r in p:
        print(f"  kept  {r.id}  {r.text[:62]}")
    st = p.stats()
    print(f"\n  oracle calls {st['llm_calls']} of {st['n_rows']} rows")
    print("  t2 is also a refund complaint, but it is not threatening to cancel.")
    print("  No similarity threshold draws that line. The oracle does.")

    # -------------------------------------------------------------- sem_join
    rule("sem_join: a JOIN ... ON written in English")
    print('  predicate: "the ticket is caused by this known issue"\n')
    j = sess.scan("tickets").sem_join(
        "known_issues", "the ticket is caused by this known issue",
        block_k=2, block_adjudicate=True)
    for left, right in sorted(j.id_pairs()):
        print(f"  {left} caused by {right}")
    js = j.stats.as_dict()
    print(f"\n  {js['candidate_pairs']} candidate pairs after blocking, "
          f"{js['nested_loop_calls']} for a nested loop")
    print(f"  oracle calls {js['oracle_calls']}, of which {js['block_calls']} were "
          f"batched block-join prompts")
    print("  t5 matches nothing, correctly.")

    # ------------------------------------------------------------- sem_dedup
    rule("sem_dedup: collapse rows describing the same thing")
    d = sess.sem_dedup("tickets", "the two tickets report the same underlying problem",
                       block_k=4, block_adjudicate=True)
    groups = d.duplicate_groups()
    for grp in groups:
        print(f"  duplicates: {sorted(r.id for r in grp)}")
    if not groups:
        print("  (none found)")
    print(f"  canonical set: {sorted(r.id for r in d.canonical())}")
    print("\n  t3 and t4 are the same Safari bug in different words.")
    print("  t1 and t2 are both billing, but different problems, so they stay apart.")

    # ---------------------------------------------------------- sem_group_by
    rule("sem_group_by: cluster into themes nobody named in advance. Zero LLM calls.")
    for grp in sess.sem_group_by("tickets", k=3).groups:
        print(f"  group: {sorted(r.id for r in grp.rows)}")
    print("\n  Pure k-means over the embeddings. Pass name_clusters=True to spend")
    print("  one LLM call per cluster naming it.")

    # ------------------------------------------------------------- composing
    rule("Composing: narrow with cheap operators before paying for expensive ones")
    print("""  (sess.search("tickets", "billing problems", k=4)     # vector: 4 candidates
       .sem_filter("the customer is threatening to cancel")  # LLM: uncertain only
       .sem_join("known_issues", "...", block_adjudicate=True))\n""")
    out = (sess.search("tickets", "billing problems", k=4)
               .sem_filter("the customer is threatening to cancel")
               .sem_join("known_issues", "the ticket is caused by this known issue",
                         block_k=2, block_adjudicate=True))
    print(f"  final pairs: {sorted(out.id_pairs())}")

    print(f"\n{'=' * 72}")
    print("Operator internals: docs/sem_filter.md, docs/sem_join.md,")
    print("docs/sem_dedup_and_group_by.md, docs/gsi_notes.md")


if __name__ == "__main__":
    main()

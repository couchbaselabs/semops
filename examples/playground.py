"""semops playground — poke at sem_filter interactively or from the CLI.

Runs OFFLINE by default (deterministic FakeModelClient, no keys, no network).
Set --real (plus env vars) to use a real OpenAI-compatible LLM/embeddings.

Examples
--------
# offline, synthetic data, one predicate:
python3 examples/playground.py --predicate "battery life" --n 2000

# offline, YOUR docs (one per line), show each row's decision:
python3 examples/playground.py --file mydocs.txt --predicate "is a complaint" --explain

# interactive: type predicates, see results live (Ctrl-D / 'quit' to exit):
python3 examples/playground.py --n 1500

# real LLM (in-memory engine, still no cluster):
export SEMOPS_LLM_KEY=sk-...            # OpenAI-compatible key
python3 examples/playground.py --real --file mydocs.txt --predicate "mentions refund"
"""
import argparse
import os
import random
import sys


import semops
from semops import (
    AnthropicClient,
    FakeModelClient,
    InMemoryEngine,
    OpenAICompatClient,
    Row,
)
from semops import cascade as _cascade

FILLER = ["device", "product", "review", "quality", "design", "value", "everyday"]


def synth(n, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        roll, f = rng.random(), " ".join(rng.sample(FILLER, 3))
        if roll < 0.35:
            out.append(f"the battery life is great and lasts all day {f}")
        elif roll < 0.80:
            out.append(f"the screen is bright and the camera is sharp {f}")
        elif roll < 0.90:
            out.append(f"the battery drains quickly while gaming {f}")
        else:
            out.append(f"long product life and a sturdy build {f}")
    return out


def load_texts(args):
    if args.file:
        with open(args.file) as fh:
            texts = [ln.strip() for ln in fh if ln.strip()]
    else:
        texts = synth(args.n)
    if args.limit:
        texts = texts[:args.limit]
    return texts


def _openai_embedder():
    """Optional neural proxy for Claude: OpenAI embeddings if SEMOPS_EMBED_KEY set."""
    key = os.environ.get("SEMOPS_EMBED_KEY", "")
    if not key:
        return None
    return OpenAICompatClient(
        base_url=os.environ.get("SEMOPS_EMBED_URL", "https://api.openai.com/v1"),
        api_key=key,
        embed_model=os.environ.get("SEMOPS_EMBED_MODEL", "text-embedding-3-small"),
    )


def make_model(provider):
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            sys.exit("provider 'anthropic' needs ANTHROPIC_API_KEY in the environment.")
        embedder = _openai_embedder()  # None -> local lexical proxy
        note = "OpenAI embeddings" if embedder else "local lexical proxy"
        print(f"  oracle=Claude ({os.environ.get('SEMOPS_CHAT_MODEL','claude-haiku-4-5-20251001')}), "
              f"proxy={note}")
        return AnthropicClient(
            api_key=key,
            chat_model=os.environ.get("SEMOPS_CHAT_MODEL", "claude-haiku-4-5-20251001"),
            embedder=embedder,
        )
    if provider == "gemini":
        # Gemini has BOTH embeddings and an LLM, and a Google OpenAI-compatible
        # endpoint -> one key powers the whole cascade (proxy + oracle).
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            sys.exit("provider 'gemini' needs GEMINI_API_KEY (or GOOGLE_API_KEY).")
        chat = os.environ.get("SEMOPS_CHAT_MODEL", "gemini-flash-latest")
        emb = os.environ.get("SEMOPS_EMBED_MODEL", "gemini-embedding-001")
        print(f"  oracle={chat}, proxy=Gemini {emb}")
        return OpenAICompatClient(
            base_url=os.environ.get(
                "SEMOPS_LLM_URL", "https://generativelanguage.googleapis.com/v1beta/openai"),
            api_key=key,
            chat_model=chat,
            embed_model=emb,
        )
    if provider == "openai":
        key = os.environ.get("SEMOPS_LLM_KEY", "")
        if not key:
            sys.exit("provider 'openai' needs SEMOPS_LLM_KEY (and optionally SEMOPS_LLM_URL).")
        return OpenAICompatClient(
            base_url=os.environ.get("SEMOPS_LLM_URL", "https://api.openai.com/v1"),
            api_key=key,
            chat_model=os.environ.get("SEMOPS_CHAT_MODEL", "gpt-4o-mini"),
            embed_model=os.environ.get("SEMOPS_EMBED_MODEL", "text-embedding-3-small"),
        )
    return FakeModelClient(dims=64)


def run(sess, source, rows, predicate, args):
    pipe = sess.scan(source).sem_filter(
        predicate, recall=args.recall, precision=args.precision, policy=args.policy,
        min_sample=args.min_sample)
    st = pipe.stats()
    kept_ids = {r["id"] for r in pipe.collect()}

    print(f"\npredicate: {predicate!r}   policy={args.policy}")
    print(f"  bands   : accept={st['n_accept']}  escalate={st['n_escalate']}  reject={st['n_reject']}")
    print(f"  thresh  : reject<= {st['tau_minus']:.3f} | escalate | {st['tau_plus']:.3f} <=accept"
          f"   collapsed={st['proxy_collapsed']}")
    print(f"  LLM calls: {st['llm_calls']} (naive={st['n_rows']})  savings={st['savings_ratio']}x"
          f"  cache_hits={st['cache_hits']}  errors={st['errors']}")
    if st["errors"]:
        print(f"  ⚠ {st['errors']} oracle call(s) failed and fell back to '{args.policy}' default "
              f"(rate limit? try smaller --limit).")
    print(f"  kept    : {len(kept_ids)} / {st['n_rows']} rows")

    if args.explain:
        # recompute proxy + thresholds via the public API to show each decision.
        # Reuse the session cache so we don't re-hit the API for rows the main run
        # already judged.
        from semops.cache import make_key
        model = sess.model

        def cached_judge(row):
            key = make_key("judge", getattr(model, "chat_model", model.__class__.__name__),
                           predicate, row.id)
            hit = sess.cache.get(key)
            if hit is not None:
                return hit
            v = model.judge(predicate, row.text)
            sess.cache.put(key, v)
            return v

        pv = sess.cache.get(make_key("embed", getattr(model, "embed_model",
             model.__class__.__name__), predicate)) or model.embed([predicate])[0]
        scores = sess.engine.proxy_scores(rows, pv)
        idx = random.Random(0).sample(range(len(rows)), min(len(rows), max(100, len(rows) // 20)))
        th = _cascade.calibrate([(scores[i], cached_judge(rows[i])) for i in idx],
                                args.recall, args.precision)
        print("  --- sample of decisions ---")
        for r, s in list(zip(rows, scores))[:args.show]:
            band = th.band(s)
            mark = "KEEP" if r.id in kept_ids else "drop"
            print(f"    [{mark}] {band.value:8} s={s:.3f}  {r.text[:70]}")
    else:
        print("  kept examples:")
        for d in pipe.collect()[:args.show]:
            print(f"    - {d.get('text', d)[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predicate", help="NL predicate; omit for interactive mode")
    ap.add_argument("--file", help="text file, one document per line (else synthetic data)")
    ap.add_argument("--n", type=int, default=2000, help="synthetic dataset size")
    ap.add_argument("--limit", type=int, default=0, help="cap docs loaded (stay under rate limits)")
    ap.add_argument("--provider", default="fake",
                    choices=["fake", "openai", "anthropic", "gemini"],
                    help="model provider (fake=offline; gemini/openai do embeddings+oracle; "
                         "anthropic=Claude oracle + separate/ local proxy)")
    ap.add_argument("--real", action="store_true", help="alias for --provider openai")
    ap.add_argument("--policy", default="cascade", choices=["cascade", "oracle", "proxy"])
    ap.add_argument("--recall", type=float, default=0.9)
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--explain", action="store_true", help="show per-row band decisions")
    ap.add_argument("--show", type=int, default=8, help="how many rows to print")
    ap.add_argument("--min-sample", dest="min_sample", type=int, default=100,
                    help="oracle sample size for calibration (the cost/quality knob)")
    ap.add_argument("--budget", type=float, default=5.0, help="USD budget guard")
    args = ap.parse_args()

    provider = "openai" if args.real else args.provider
    model = make_model(provider)
    texts = load_texts(args)
    vecs = model.embed(texts)
    rows = [Row(id=str(i), text=t, embedding=v, doc={"id": str(i), "text": t})
            for i, (t, v) in enumerate(zip(texts, vecs))]
    eng = InMemoryEngine()
    eng.add("docs", rows)
    sess = semops.connect(engine=eng, model=model, budget_usd=args.budget)

    print(f"semops playground — provider={provider}, {len(rows)} docs in-memory")

    if args.predicate:
        run(sess, "docs", rows, args.predicate, args)
        return
    print("Enter a predicate (blank line or 'quit' to exit):")
    while True:
        try:
            p = input("\npredicate> ").strip()
        except EOFError:
            break
        if not p or p.lower() in ("quit", "exit"):
            break
        run(sess, "docs", rows, p, args)


if __name__ == "__main__":
    main()

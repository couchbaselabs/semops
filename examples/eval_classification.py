"""Real-world eval of sem_filter on a labeled text-classification dataset.

Turns a classification task into a boolean predicate and measures the cascade
against GROUND TRUTH — plus a "vector-search-alone" baseline, so we can quantify
what the semantic operator adds over pure similarity.

Default: 20 Newsgroups, target sci.space, predicate "is about space...",
embedder = local lexical (free), oracle = ground-truth labels (a perfect, free
oracle -> isolates cascade mechanics + proxy quality at full scale).

    ./.venv/bin/python examples/eval_classification.py --n 3000

Real semantics/cost with an LLM oracle (smaller n; uses your key + quota):
    set -a; . ./.env.local; set +a
    ./.venv/bin/python examples/eval_classification.py --n 300 --oracle gemini --embedder gemini
"""
import argparse
import os
import random
import sys


import semops
from semops import InMemoryEngine, LocalHashingEmbedder, OpenAICompatClient, Row
from semops import cascade as _cascade

DEFAULT_CATEGORIES = [
    "sci.space", "rec.sport.baseball", "rec.sport.hockey",
    "comp.graphics", "sci.med", "talk.politics.mideast", "rec.autos",
]


def load_20ng(categories, targets, n, seed):
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", categories=categories,
                           remove=("headers", "footers", "quotes"))
    tset = set(targets)
    idx = [i for i in range(len(d.data)) if d.data[i].strip()]
    random.Random(seed).shuffle(idx)
    if n:
        idx = idx[:n]
    texts = [d.data[i] for i in idx]
    labels = [d.target_names[d.target[i]] in tset for i in idx]
    return texts, labels


class SklearnLRProxy:
    """Learned proxy: logistic regression on embeddings -> P(positive)."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression(max_iter=1000)
        self._model = None
        self._const = 0.0

    def fit(self, X, y):
        yy = [1 if v else 0 for v in y]
        if len(set(yy)) < 2:            # need both classes
            self._model, self._const = None, float(yy[0]) if yy else 0.0
            return
        self._model = self._lr.fit(X, yy)

    def predict_proba(self, X):
        if self._model is None:
            return [self._const] * len(X)
        j = list(self._model.classes_).index(1)
        return [float(row[j]) for row in self._model.predict_proba(X)]

    def linear_params(self):
        """(weights, bias) so the proxy can be pushed down as a dot product.
        Returns None when there's no usable model (degenerate sample)."""
        if self._model is None:
            return None
        return (self._model.coef_[0].tolist(), float(self._model.intercept_[0]))


def prf(truth, pred):
    tp = sum(1 for y, p in zip(truth, pred) if y and p)
    fp = sum(1 for y, p in zip(truth, pred) if p and not y)
    fn = sum(1 for y, p in zip(truth, pred) if y and not p)
    P = tp / (tp + fp) if (tp + fp) else 1.0
    R = tp / (tp + fn) if (tp + fn) else 1.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def best_threshold(scores, labels):
    """Oracle-tuned single threshold on the proxy = best-case vector-search-alone."""
    best = (0.0, 0.0, 0.0, None)  # F, P, R, thr
    for t in sorted(set(scores)):
        p, r, f = prf(labels, [s >= t for s in scores])
        if f > best[0]:
            best = (f, p, r, t)
    return best


class LsaEmbedder:
    """TF-IDF + LSA (TruncatedSVD) — a real classical semantic proxy, fit on the
    corpus. Free, instant, no heavy deps. Much stronger than raw bag-of-words."""

    def __init__(self, corpus, dims=256):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.vec = TfidfVectorizer(stop_words="english", max_features=20000, min_df=2)
        X = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(n_components=min(dims, X.shape[1] - 1), random_state=0)
        self.svd.fit(X)
        self.embed_model = f"tfidf-lsa-{dims}"
        self.spent_usd = 0.0
        self.calls = 0

    def embed(self, texts):
        return self.svd.transform(self.vec.transform(list(texts))).tolist()


class SplitModel:
    """Compose an embedder (proxy) with a judge fn (oracle)."""

    def __init__(self, embedder, judge_fn, chat_model="split-oracle"):
        self._e = embedder
        self._judge = judge_fn
        self.embed_model = getattr(embedder, "embed_model", "local")
        self.chat_model = chat_model
        self.spent_usd = 0.0
        self.calls = 0

    def embed(self, texts):
        return self._e.embed(texts)

    def judge(self, predicate, text):
        self.calls += 1
        return self._judge(predicate, text)


def _real_client():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        sys.exit("real oracle/embedder needs GEMINI_API_KEY in env.")
    return OpenAICompatClient(
        base_url=os.environ.get("SEMOPS_LLM_URL",
                                "https://generativelanguage.googleapis.com/v1beta/openai"),
        api_key=key,
        chat_model=os.environ.get("SEMOPS_CHAT_MODEL", "gemini-flash-latest"),
        embed_model=os.environ.get("SEMOPS_EMBED_MODEL", "gemini-embedding-001"),
    )


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs="*", default=["sci.space"])
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--predicate", default="is about space, astronomy, or spacecraft")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--embedder", default="lsa", choices=["lsa", "local", "gemini", "openai"])
    ap.add_argument("--oracle", default="label", choices=["label", "gemini", "openai"])
    ap.add_argument("--proxy", default="learned", choices=["learned", "predicate"],
                    help="learned=LR on sample embeddings (strong); predicate=cosine-to-phrase")
    ap.add_argument("--recall", type=float, default=0.9)
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--min-sample", dest="min_sample", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"loading 20 Newsgroups: target={args.target}, {len(args.categories)} categories, n<={args.n}")
    texts, labels = load_20ng(args.categories, args.target, args.n, args.seed)
    truth = {t: y for t, y in zip(texts, labels)}
    pos = sum(labels)
    print(f"  {len(texts)} docs, {pos} positive ({pct(pos/len(texts))})   predicate={args.predicate!r}")

    # embedder (proxy) + oracle (judge)
    real = None
    if args.embedder in ("gemini", "openai") or args.oracle in ("gemini", "openai"):
        real = _real_client()
    if args.embedder == "local":
        embedder = LocalHashingEmbedder(dims=512)
    elif args.embedder == "lsa":
        embedder = LsaEmbedder(texts, dims=256)
    else:
        embedder = real
    if args.oracle == "label":
        judge_fn = lambda pred, text: truth.get(text, False)
        oracle_name = "ground-truth labels (perfect, free)"
    else:
        judge_fn = real.judge
        oracle_name = f"{args.oracle} LLM"
    model = SplitModel(embedder, judge_fn)
    print(f"  embedder={args.embedder} ({model.embed_model}), oracle={oracle_name}")

    # embed + build rows
    print("  embedding...")
    vecs = model.embed(texts)
    rows = [Row(id=str(i), text=t, embedding=v, doc={"id": str(i), "text": t})
            for i, (t, v) in enumerate(zip(texts, vecs))]

    # proxy scores (for the baseline + shared by the cascade)
    pred_vec = model.embed([args.predicate])[0]
    eng = InMemoryEngine(); eng.add("docs", rows)
    scores = eng.proxy_scores(rows, pred_vec)

    # (1) vector-search-alone, oracle-tuned best threshold
    vf, vp, vr, vt = best_threshold(scores, labels)

    # (2) sem_filter cascade
    proxy_model = SklearnLRProxy() if args.proxy == "learned" else None
    print(f"  proxy={args.proxy}")
    sess = semops.connect(engine=eng, model=model, budget_usd=50.0)
    pipe = sess.scan("docs").sem_filter(
        args.predicate, recall=args.recall, precision=args.precision,
        min_sample=args.min_sample, seed=args.seed, proxy_model=proxy_model)
    st = pipe.stats()
    kept = {r["id"] for r in pipe.collect()}
    cpred = [r.id in kept for r in rows]
    cp, cr, cf = prf(labels, cpred)

    # (3) oracle ceiling — the F1 the oracle itself achieves vs ground truth.
    # label oracle = 1.0 by construction; a real LLM has its own error, which we
    # measure for free on the rows it actually judged (from the cache).
    n_judged = st["n_rows"]
    if args.oracle == "label":
        of = 1.0
    else:
        from semops.cache import make_key
        jt, jp = [], []
        for i, r in enumerate(rows):
            v = sess.cache.get(make_key("judge", getattr(model, "chat_model", "x"),
                                        args.predicate, r.id))
            if v is not None:
                jp.append(v)
                jt.append(labels[i])
        of = prf(jt, jp)[2] if jp else None
        n_judged = len(jp)

    print("\n=== RESULTS ===")
    print(f"{'method':<34}{'P':>7}{'R':>7}{'F1':>7}{'LLM calls':>12}{'savings':>9}")
    print("-" * 76)
    print(f"{'vector search alone (best thr)':<34}{vp:>7.3f}{vr:>7.3f}{vf:>7.3f}{0:>12}{'--':>9}")
    print(f"{'sem_filter cascade':<34}{cp:>7.3f}{cr:>7.3f}{cf:>7.3f}"
          f"{st['llm_calls']:>12}{str(st['savings_ratio'])+'x':>9}")
    ceil = f"{of:.3f}" if of is not None else "n/a"
    olabel = "oracle ceiling (every row)" if args.oracle == "label" \
        else f"oracle ceiling ({n_judged} judged)"
    print(f"{olabel:<34}{'':>7}{'':>7}{ceil:>7}{n_judged:>12}{'1.0x':>9}")
    print("-" * 76)
    print(f"bands: accept={st['n_accept']} escalate={st['n_escalate']} reject={st['n_reject']}"
          f"  errors={st['errors']}  collapsed={st['proxy_collapsed']}")
    print(f"thresholds: reject<= {st['tau_minus']:.3f} | escalate | {st['tau_plus']:.3f} <=accept")
    print(f"\ntakeaway: cascade F1 {cf:.3f} vs vector-alone {vf:.3f} "
          f"({'+' if cf>=vf else ''}{cf-vf:+.3f}), using {st['llm_calls']}/{st['n_rows']} oracle calls "
          f"({st['savings_ratio']}x fewer than full-oracle).")


if __name__ == "__main__":
    main()

"""Semantic operators. First (and reference) operator: sem_filter.

The operator is engine-agnostic: it asks the engine for proxy scores (native
pushdown if offered, portable in-service cosine otherwise), runs the SHARED
cascade, and spends the oracle only on the uncertain middle band. If the engine
advertises a native in-engine implementation (`caps.native_ops`), it short-
circuits to that instead — the graduation path.
"""
from __future__ import annotations

import math
import random
import threading
from typing import Optional, Sequence

from . import cascade as _cascade
from .backends import ModelClient
from .budget import Budget
from .cache import Cache, make_key
from .engines.base import BaseEngine
from .parallel import pmap
from .telemetry import Telemetry
from .types import Band, CascadeStats, FilterResult, JoinResult, JoinStats, Row


def _embed_one(model: ModelClient, cache: Cache, text: str) -> list[float]:
    key = make_key("embed", getattr(model, "embed_model", model.__class__.__name__), text)
    hit = cache.get(key)
    if hit is not None:
        return hit
    vec = model.embed([text])[0]
    cache.put(key, vec)
    return vec


def _ensure_embeddings(model: ModelClient, cache: Cache, rows: Sequence[Row]) -> None:
    missing = [r for r in rows if not r.embedding]
    if not missing:
        return
    # try cache first, then batch-embed the rest
    to_embed, keys = [], []
    for r in missing:
        key = make_key("embed", getattr(model, "embed_model", model.__class__.__name__), r.text)
        hit = cache.get(key)
        if hit is not None:
            r.embedding = hit
        else:
            to_embed.append(r)
            keys.append(key)
    if to_embed:
        vecs = model.embed([r.text for r in to_embed])
        for r, key, vec in zip(to_embed, keys, vecs):
            r.embedding = vec
            cache.put(key, vec)


def sem_filter(
    engine: BaseEngine,
    model: ModelClient,
    rows: Sequence[Row],
    predicate: str,
    *,
    recall: float = 0.9,
    precision: float = 0.9,
    delta: float = 0.1,
    sample_frac: float = 0.05,
    min_sample: int = 100,
    max_sample: Optional[int] = None,
    policy: str = "cascade",           # "cascade" | "oracle" | "proxy"
    on_error: str = "reject",          # "reject" | "accept" | "raise"
    proxy_model=None,                  # optional learned proxy: .fit(X,y)/.predict_proba(X)
    workers: int = 1,                  # parallel LLM calls (I/O-bound); 1 = sequential
    seed: int = 0,
    cache: Optional[Cache] = None,
    budget: Optional[Budget] = None,
    telemetry: Optional[Telemetry] = None,
) -> FilterResult:
    import time
    t0 = time.time()
    cache = cache or Cache()
    budget = budget or Budget()
    rows = list(rows)
    stats = CascadeStats(n_rows=len(rows))

    # 0. native graduation short-circuit
    if "sem_filter" in engine.caps.native_ops:
        native = engine.native_operator("sem_filter")
        if native is not None:
            return native(rows, predicate, recall=recall, precision=precision, delta=delta)

    if not rows:
        return FilterResult([], stats)

    # oracle wrapper: cached + budget-charged; updates stats.
    # Safe to call from worker threads: cache/budget lock internally, stats under slock.
    chat_cost = getattr(model, "chat_cost_per_call", 0.0005)
    slock = threading.Lock()

    def oracle(row: Row) -> bool:
        key = make_key("judge", getattr(model, "chat_model", model.__class__.__name__),
                       predicate, row.id)
        hit = cache.get(key)
        if hit is not None:
            with slock:
                stats.cache_hits += 1
            return hit
        budget.charge(chat_cost)  # raises BudgetExceeded if over limit -> query stops
        try:
            val = model.judge(predicate, row.text)  # network call: NOT holding a lock
        except Exception:
            if on_error == "raise":
                raise
            with slock:
                stats.errors += 1
            val = (on_error == "accept")
        cache.put(key, val)
        with slock:
            stats.llm_calls += 1
        return val

    # 1. predicate embedding
    pred_vec = _embed_one(model, cache, predicate)
    server_side = getattr(engine.caps, "server_side_scoring", False)

    keep = [False] * len(rows)
    stats.n_sample = 0

    if policy == "oracle":
        keep = pmap(oracle, rows, workers)
        stats.n_escalate += len(rows)
    else:
        # 2. sample + oracle-label FIRST (a learned proxy needs labels before scoring)
        idx = _sample_indices(len(rows), sample_frac, min_sample, max_sample, seed)
        sample_set = set(idx)
        sample_labels = dict(zip(idx, pmap(lambda i: oracle(rows[i]), idx, workers)))
        stats.n_escalate += len(idx)
        stats.n_sample = len(idx)

        # 3. proxy scores — pushed into the engine when it can score server-side
        #    (embeddings never leave the store); computed in-service otherwise.
        th_pre = None  # set when the proxy path calibrates in probability space
        if proxy_model is not None:
            # Cross-fit rather than hold out a calibration third: the holdout left
            # too few labelled points for the Wilson bounds to certify anything
            # (34 of 100 could not certify precision 0.9, so tau_plus stayed +inf
            # and every row escalated). See _oof_proba.
            _ensure_row_vectors(engine, model, cache, rows, idx)  # only the sample's vectors
            X = [rows[i].embedding for i in idx]
            y = [sample_labels[i] for i in idx]
            oof = _oof_proba(type(proxy_model), X, y, seed=seed)
            th_pre = _cascade.calibrate(list(zip(oof, y)), recall, precision, delta)
            proxy_model.fit(X, y)  # final model, fit on the whole sample
            lin = getattr(proxy_model, "linear_params", lambda: None)()
            if server_side and lin is not None:
                # a logistic-regression proxy is linear: its score is a dot product
                # with the learned weight vector -> push it down as VECTOR_DISTANCE(.,w,'dot')
                weights, bias = lin
                scores = engine.proxy_scores(rows, weights, metric="dot")
                th_pre = _cascade.Thresholds(
                    tau_minus=_proba_tau_to_dot(th_pre.tau_minus, bias),
                    tau_plus=_proba_tau_to_dot(th_pre.tau_plus, bias),
                    collapsed=th_pre.collapsed)
            else:
                _ensure_embeddings(model, cache, rows)
                scores = proxy_model.predict_proba([r.embedding for r in rows])
            calib = []
        else:
            if not server_side:
                _ensure_embeddings(model, cache, rows)  # in-service cosine needs local vectors
            scores = engine.proxy_scores(rows, pred_vec)  # server-side if the engine supports it
            calib = [(scores[i], sample_labels[i]) for i in idx]

        # 5. calibrate thresholds, then band every non-sampled row
        th = th_pre if th_pre is not None else _cascade.calibrate(calib, recall, precision, delta)
        stats.tau_minus, stats.tau_plus, stats.proxy_collapsed = (
            th.tau_minus, th.tau_plus, th.collapsed)

        if policy == "proxy":  # A/B: decide everything by the proxy, never escalate
            boundary = (th.tau_plus if math.isfinite(th.tau_plus)
                        else th.tau_minus if math.isfinite(th.tau_minus)
                        else sorted(scores)[len(scores) // 2])
            for i, r in enumerate(rows):
                if i in sample_set:
                    keep[i] = sample_labels[i]
                    continue
                keep[i] = scores[i] >= boundary
                stats.n_accept += 1 if keep[i] else 0
                stats.n_reject += 0 if keep[i] else 1
        else:  # cascade (default)
            escalate_idx = []
            for i, r in enumerate(rows):
                if i in sample_set:
                    keep[i] = sample_labels[i]
                    continue
                band = th.band(scores[i])
                if band is Band.ACCEPT:
                    keep[i] = True
                    stats.n_accept += 1
                elif band is Band.REJECT:
                    keep[i] = False
                    stats.n_reject += 1
                else:
                    escalate_idx.append(i)  # defer so the pool can run them together
            stats.n_escalate += len(escalate_idx)
            for i, v in zip(escalate_idx, pmap(lambda j: oracle(rows[j]), escalate_idx, workers)):
                keep[i] = v

    stats.llm_calls_saved = max(0, stats.n_rows - stats.llm_calls)
    stats.est_cost_usd = stats.llm_calls * chat_cost
    result = FilterResult([r for r, k in zip(rows, keep) if k], stats)

    if telemetry is not None:
        telemetry.log(
            operator="sem_filter",
            engine=engine.name,
            params={"predicate": predicate, "recall": recall, "precision": precision,
                    "delta": delta, "policy": policy, "sample_frac": sample_frac},
            stats=stats.as_dict(),
            latency_ms=(time.time() - t0) * 1000.0,
        )
    return result


def sem_filter_source(
    engine,
    model,
    source: str,
    predicate: str,
    *,
    proxy_model=None,
    recall: float = 0.9,
    precision: float = 0.9,
    delta: float = 0.05,
    sample_frac: float = 0.05,
    min_sample: int = 100,
    # Uncapped, matching sem_filter. (sem_join caps this at 500 because its sample
    # is drawn from candidate PAIRS, which grow as n_left * block_k. Rows do not
    # explode that way, and capping here starves the calibrator: at 50k rows a 500
    # cap vs the natural 2500 was the difference between certifying tau_plus and
    # leaving it at +inf, which doubled the oracle calls.)
    max_sample: Optional[int] = None,
    filters: Optional[dict] = None,
    est_k: int = 256,
    workers: int = 1,
    on_error: str = "reject",
    seed: int = 0,
    cache: Optional[Cache] = None,
    budget: Optional[Budget] = None,
    telemetry: Optional[Telemetry] = None,
    chat_cost: float = 0.0,
) -> FilterResult:
    """sem_filter over a whole collection WITHOUT scoring every row.

    The row-list `sem_filter` scores all n rows to band them. But the cascade only
    ever needs two things: the escalate band (to ask the oracle) and the accept
    band (to return). Everything below tau_minus is rejected — and a rejected row
    does not need a score, it just needs to not come back.

    So: calibrate tau on the labelled sample only, then ask the index for the rows
    above tau_minus (see CouchbaseEngine.ann_above — a linear proxy is a dot
    product, and on normalised vectors that is cosine to the weight vector, so an
    ordinary vector index serves it). Work becomes proportional to the number of
    rows that survive, not to collection size.

    Requires a linear proxy (`linear_params()`) and an ANN engine; callers without
    both should use the row-list sem_filter.
    """
    import time
    t0 = time.time()
    cache = cache or Cache(enabled=False)
    budget = budget or Budget(None)
    stats = CascadeStats()
    slock = threading.Lock()

    lin = None
    if proxy_model is None:
        raise ValueError("sem_filter_source needs a linear proxy_model (got None)")

    # 1. sample the collection and oracle-label it — the only full-corpus touch,
    #    and it reads ids/text only (no vectors).
    sample_rows = engine.scan(source, filters, limit=None, with_vectors=False)
    stats.n_rows = len(sample_rows)
    idx = _sample_indices(len(sample_rows), sample_frac, min_sample, max_sample, seed)
    picked = [sample_rows[i] for i in idx]

    def oracle(row: Row) -> bool:
        key = make_key("judge", getattr(model, "chat_model", model.__class__.__name__),
                       predicate, row.id)
        hit = cache.get(key)
        if hit is not None:
            with slock:
                stats.cache_hits += 1
            return hit
        budget.charge(chat_cost)
        try:
            val = model.judge(predicate, row.text)
        except Exception:
            if on_error == "raise":
                raise
            with slock:
                stats.errors += 1
            val = (on_error == "accept")
        cache.put(key, val)
        with slock:
            stats.llm_calls += 1
        return val

    labels = pmap(oracle, picked, workers)
    stats.n_sample = len(picked)
    stats.n_escalate += len(picked)

    # 2. cross-fit the proxy: every labelled row gets a score from a model that did
    #    not see it, so all labels reach the calibrator (see _oof_proba). Only the
    #    sample's vectors are pulled — not the collection's.
    vecs = engine.fetch_vectors(source, [r.id for r in picked])
    keep_i = [i for i, r in enumerate(picked) if r.id in vecs]
    X = [vecs[picked[i].id] for i in keep_i]
    y = [labels[i] for i in keep_i]
    if len(set(y)) < 2:
        raise ValueError("sample has a single class — cannot fit a proxy")

    oof = _oof_proba(type(proxy_model), X, y, seed=seed)
    th_p = _cascade.calibrate(list(zip(oof, y)), recall, precision, delta)

    proxy_model.fit(X, y)                      # final model, fit on the whole sample
    lin = getattr(proxy_model, "linear_params", lambda: None)()
    if lin is None:
        raise ValueError("proxy_model produced no linear params (degenerate sample)")
    weights, bias = lin

    # thresholds were certified on probabilities; move them onto the index's scale
    th = _cascade.Thresholds(
        tau_minus=_proba_tau_to_dot(th_p.tau_minus, bias),
        tau_plus=_proba_tau_to_dot(th_p.tau_plus, bias),
        collapsed=th_p.collapsed)
    stats.tau_minus, stats.tau_plus, stats.proxy_collapsed = (
        th.tau_minus, th.tau_plus, th.collapsed)

    # 3. pull only what clears tau_minus. Below it is rejected without being read.
    labelled = {picked[i].id: labels[i] for i in range(len(picked))}
    if math.isfinite(th.tau_minus):
        survivors, exhaustive = engine.ann_above(
            source, weights, th.tau_minus, filters, est_k=est_k)
        stats.proxy_exhaustive = exhaustive
    else:
        # no usable lower threshold -> the cascade cannot prune; fall back to all rows
        survivors = [(r, 0.0) for r in sample_rows]
        stats.proxy_exhaustive = True
    stats.n_scored = len(survivors)
    stats.n_reject += max(0, stats.n_rows - len(survivors))

    # 4. band the survivors; only the escalate band costs an oracle call
    keep_rows, escalate = [], []
    for row, score in survivors:
        if row.id in labelled:
            if labelled[row.id]:
                keep_rows.append(row)
            continue
        band = th.band(score)
        if band is Band.ACCEPT:
            keep_rows.append(row)
            stats.n_accept += 1
        elif band is Band.REJECT:
            stats.n_reject += 1
        else:
            escalate.append(row)
    stats.n_escalate += len(escalate)
    for row, verdict in zip(escalate, pmap(oracle, escalate, workers)):
        if verdict:
            keep_rows.append(row)
    # sampled rows that cleared the oracle but sat below tau_minus still belong
    for i, row in enumerate(picked):
        if labels[i] and all(k.id != row.id for k in keep_rows):
            keep_rows.append(row)

    stats.llm_calls_saved = max(0, stats.n_rows - stats.llm_calls)
    stats.est_cost_usd = stats.llm_calls * chat_cost
    result = FilterResult(keep_rows, stats)
    if telemetry is not None:
        telemetry.log(
            operator="sem_filter_source", engine=engine.name,
            params={"predicate": predicate, "recall": recall, "precision": precision,
                    "delta": delta, "est_k": est_k},
            stats=stats.as_dict(), latency_ms=(time.time() - t0) * 1000.0)
    return result


def sem_join(
    engine: BaseEngine,
    model: ModelClient,
    left_rows: Sequence[Row],
    right_source: str,
    predicate: str,
    *,
    block_k: int = 10,                 # ANN candidates retrieved per left row
    recall: float = 0.9,
    precision: float = 0.9,
    delta: float = 0.1,
    sample_frac: float = 0.1,
    min_sample: int = 50,
    # candidate_pairs can be huge (n_left * block_k), and sample_frac would scale
    # the calibration sample with it. Cap it: a few hundred labelled pairs is
    # plenty to calibrate, and with block-join adjudication escalation is O(n_left)
    # so being conservative costs almost nothing. (BioDEX: capping 4860->200
    # samples took F1 0.98->1.00 and calls 5010->350.)
    max_sample: Optional[int] = 500,
    policy: str = "cascade",           # "cascade" | "oracle" | "blocking"
    block_adjudicate: bool = False,    # batch escalate-band pairs into one prompt per left (Trummer block join)
    block_alpha: int = 4,              # shrink factor on overflow (adaptive selectivity)
    self_join: bool = False,           # dedupe symmetric candidate pairs (for sem_dedup)
    workers: int = 1,                  # parallel ANN queries + LLM calls
    probe_batch: int = 32,             # ANN probes per round trip (engine.ann_batch)
    on_error: str = "reject",
    seed: int = 0,
    cache: Optional[Cache] = None,
    budget: Optional[Budget] = None,
    telemetry: Optional[Telemetry] = None,
) -> JoinResult:
    """Join left_rows to the right_source on a natural-language predicate.

    Strategy (LOTUS/Trummer): the vector index BLOCKS — for each left row we
    ANN-retrieve its top-block_k nearest right rows (embedding similarity between
    join keys), pruning the quadratic space. Then the cascade adjudicates the
    surviving candidate pairs: very-similar pairs auto-accept, low-similarity
    auto-reject, and only the ambiguous middle goes to the LLM.

    Assumes a similarity-shaped predicate (same-entity / same-topic). For
    complementary predicates (contradiction/negation), embedding blocking misses
    matches (F1 collapses) — use policy='oracle' on a pre-blocked set instead.

    CHOOSING block_k (measured on 20NG, fan-out ~39 true matches/row):
        block_k = 1x fan-out  -> blocking recall 0.56
        block_k = 2x fan-out  -> blocking recall 0.84
        block_k = 4x fan-out  -> blocking recall 0.94
    ANN similarity is only a proxy for the predicate, so a row's true matches are
    NOT all among its nearest neighbours — set block_k to 2-4x the expected number
    of matches per left row, not 1x. Recall lost here can never be recovered later.

    This is cheap: with block_adjudicate=True the LLM cost is O(n_left) (one
    batched prompt per left row), NOT O(candidate_pairs) — widening block_k costs
    more tokens per prompt but not more calls, until the prompt hits the context
    limit and the adaptive-alpha shrink starts splitting batches.
    """
    import time
    t0 = time.time()
    cache = cache or Cache()
    budget = budget or Budget()
    left_rows = list(left_rows)
    stats = JoinStats(n_left=len(left_rows))
    if not left_rows:
        return JoinResult([], stats)
    stats.n_right = engine.count(right_source)

    _ensure_row_vectors(engine, model, cache, left_rows, range(len(left_rows)))

    chat_cost = getattr(model, "chat_cost_per_call", 0.0005)
    slock = threading.Lock()

    def adjudicate(pair) -> bool:
        lrow, rrow = pair
        key = make_key("join", getattr(model, "chat_model", model.__class__.__name__),
                       predicate, lrow.id, rrow.id)
        hit = cache.get(key)
        if hit is not None:
            with slock:
                stats.cache_hits += 1
            return hit
        budget.charge(chat_cost)
        text = f"ITEM A:\n{lrow.text}\n\nITEM B:\n{rrow.text}"
        try:
            val = model.judge(predicate, text)  # network call, unlocked
        except Exception:
            if on_error == "raise":
                raise
            with slock:
                stats.errors += 1
            val = (on_error == "accept")
        cache.put(key, val)
        with slock:
            stats.oracle_calls += 1
        return val

    # 1. BLOCKING via the vector index: candidate pairs = each left row's ANN neighbors.
    #    Probes are batched into one statement per chunk (Couchbase: UNION ALL, one
    #    round trip instead of one per left row — measured 2.4-2.5x), and the chunks
    #    run through the pool on top of that. Right-side embeddings are never read
    #    after blocking, so with_vectors=False leaves them in the store (51x less
    #    data per probe, and an INCLUDE index can then cover the scan entirely).
    embedded = [l for l in left_rows if l.embedding]
    chunks = [embedded[i:i + probe_batch] for i in range(0, len(embedded), probe_batch)]
    neighbor_lists = [
        nl for chunk_result in pmap(
            lambda ch: engine.ann_batch(right_source, [l.embedding for l in ch], block_k,
                                        with_vectors=False),
            chunks, workers)
        for nl in chunk_result
    ]
    cand: list[tuple[Row, Row, float]] = []
    for lrow, neighbors in zip(embedded, neighbor_lists):
        for rrow, sim in neighbors:
            if rrow.id == lrow.id:
                continue  # skip trivial self-match
            cand.append((lrow, rrow, sim))
    if self_join:
        # collapse symmetric candidate pairs so each unordered pair is judged once
        seen, dedup = set(), []
        for l, r, s in cand:
            key = (l.id, r.id) if l.id < r.id else (r.id, l.id)
            if key not in seen:
                seen.add(key)
                dedup.append((l, r, s))
        cand = dedup
    stats.candidate_pairs = len(cand)
    if not cand:
        return JoinResult([], stats)

    scores = [c[2] for c in cand]
    matched = [False] * len(cand)

    if policy == "oracle":
        matched = pmap(adjudicate, [(l, r) for l, r, _s in cand], workers)
        stats.n_escalate += len(cand)
    elif policy == "blocking":
        # pure embedding join: accept every candidate pair, no LLM
        for i in range(len(cand)):
            matched[i] = True
            stats.n_accept += 1
    else:  # cascade
        idx = _sample_indices(len(cand), sample_frac, min_sample, max_sample, seed)
        sample_set = set(idx)
        sample_labels = dict(zip(idx, pmap(
            lambda i: adjudicate((cand[i][0], cand[i][1])), idx, workers)))
        stats.n_escalate += len(idx)
        th = _cascade.calibrate([(scores[i], sample_labels[i]) for i in idx],
                                recall, precision, delta)
        stats.tau_minus, stats.tau_plus = th.tau_minus, th.tau_plus

        use_block = block_adjudicate and hasattr(model, "match_block")
        escalate_by_left: dict[str, list[int]] = {}  # left.id -> candidate indices
        for i, (l, r, s) in enumerate(cand):
            if i in sample_set:
                matched[i] = sample_labels[i]
                continue
            band = th.band(s)
            if band is Band.ACCEPT:
                matched[i] = True
                stats.n_accept += 1
            elif band is Band.REJECT:
                matched[i] = False
                stats.n_reject += 1
            else:  # escalate to the LLM
                stats.n_escalate += 1
                escalate_by_left.setdefault(l.id, []).append(i)  # defer (batch and/or parallelize)

        if use_block:
            # block-join adjudication: one batched prompt per left row (Trummer Alg. 2/3),
            # and the per-left blocks themselves run in parallel.
            groups = list(escalate_by_left.values())

            def _run_block(idxs):
                lrow = cand[idxs[0]][0]
                rights = [cand[i][1] for i in idxs]
                return _block_adjudicate(model, predicate, lrow, rights, block_alpha,
                                         cache, budget, stats, chat_cost, on_error, slock)

            for idxs, match_ids in zip(groups, pmap(_run_block, groups, workers)):
                for i in idxs:
                    matched[i] = cand[i][1].id in match_ids
        else:
            flat = [i for idxs in escalate_by_left.values() for i in idxs]
            for i, v in zip(flat, pmap(
                    lambda j: adjudicate((cand[j][0], cand[j][1])), flat, workers)):
                matched[i] = v

    pairs = [(cand[i][0], cand[i][1]) for i in range(len(cand)) if matched[i]]
    stats.matches = len(pairs)

    if telemetry is not None:
        telemetry.log(
            operator="sem_join", engine=engine.name,
            params={"predicate": predicate, "block_k": block_k, "policy": policy,
                    "recall": recall, "precision": precision},
            stats=stats.as_dict(), latency_ms=(time.time() - t0) * 1000.0)
    return JoinResult(pairs, stats)


def sem_group_by(
    engine: BaseEngine,
    model: ModelClient,
    source: str,
    *,
    k: int,
    method: str = "embedding",          # "embedding" (cluster row vectors) | "llm_label" (LOTUS)
    name_clusters: bool = False,        # ask the LLM for a human-readable label per group
    group_prompt: str = "the topic",    # what to group/label by
    rows: Optional[Sequence[Row]] = None,
    workers: int = 1,
    seed: int = 0,
    cache: Optional[Cache] = None,
    budget: Optional[Budget] = None,
    telemetry: Optional[Telemetry] = None,
) -> "GroupByResult":
    """Cluster rows into k semantic groups whose labels aren't known in advance.

    method="embedding" (default): k-means over the rows' embeddings — cheap, zero
    LLM calls, leans entirely on the vector space.
    method="llm_label" (LOTUS): the LLM projects a candidate label per row, those
    labels are embedded and clustered — more controllable, one LLM call per row.
    name_clusters=True adds one LLM call per group to name it.
    """
    import time
    from .clustering import kmeans
    from .types import GroupByResult, GroupByStats, SemGroup
    t0 = time.time()
    cache = cache or Cache()
    budget = budget or Budget()
    rows = list(rows) if rows is not None else engine.scan(source)
    stats = GroupByStats(n_rows=len(rows), method=method)
    if not rows:
        return GroupByResult([], stats)

    chat_cost = getattr(model, "chat_cost_per_call", 0.0005)

    if method == "llm_label" and hasattr(model, "generate"):
        labels = []
        for r in rows:
            budget.charge(chat_cost)
            labels.append(model.generate(
                f"In a few words, name {group_prompt} of this text.\n\n{r.text}\n\nLabel:"))
            stats.llm_calls += 1
        vectors = model.embed(labels)
    else:  # embedding
        _ensure_row_vectors(engine, model, cache, rows, range(len(rows)))
        vectors = [r.embedding for r in rows]

    assign = kmeans(vectors, k, seed=seed)
    buckets: dict[int, list[Row]] = {}
    for r, a in zip(rows, assign):
        buckets.setdefault(a, []).append(r)
    groups = [SemGroup(id=i, rows=members)
              for i, (_gid, members) in enumerate(sorted(buckets.items()))]
    stats.k = len(groups)

    if name_clusters and hasattr(model, "generate"):
        for g in groups:
            sample = "\n".join(f"- {r.text[:120]}" for r in g.rows[:6])
            budget.charge(chat_cost)
            g.label = model.generate(
                f"Give a short (2-4 word) label for {group_prompt} shared by these:\n{sample}\nLabel:")
            stats.llm_calls += 1

    if telemetry is not None:
        telemetry.log(operator="sem_group_by", engine=engine.name,
                      params={"k": k, "method": method, "name_clusters": name_clusters},
                      stats=stats.as_dict(), latency_ms=(time.time() - t0) * 1000.0)
    return GroupByResult(groups, stats)


def sem_dedup(
    engine: BaseEngine,
    model: ModelClient,
    source: str,
    predicate: str = "ITEM A and ITEM B refer to the same real-world entity",
    *,
    rows: Optional[Sequence[Row]] = None,
    block_k: int = 10,
    recall: float = 0.9,
    precision: float = 0.9,
    delta: float = 0.1,
    sample_frac: float = 0.1,
    min_sample: int = 50,
    policy: str = "cascade",
    block_adjudicate: bool = False,
    block_alpha: int = 4,
    workers: int = 1,
    on_error: str = "reject",
    seed: int = 0,
    cache: Optional[Cache] = None,
    budget: Optional[Budget] = None,
    telemetry: Optional[Telemetry] = None,
) -> "DedupResult":
    """Deduplicate a collection: cluster rows that refer to the same entity.

    A self-`sem_join` under a "same entity" predicate (vector-index blocking +
    cascade adjudication, optionally block-batched), followed by connected-
    components clustering (union-find) over the matched pairs.

    Note: connected components take the transitive closure, so A~B and B~C put
    A, B, C in one entity even if A~C was never directly confirmed — the standard
    ER behavior, which can over-merge on noisy matches.
    """
    from .types import DedupResult, DedupStats
    rows = list(rows) if rows is not None else engine.scan(source)
    jr = sem_join(
        engine, model, rows, source, predicate,
        block_k=block_k, recall=recall, precision=precision, delta=delta,
        sample_frac=sample_frac, min_sample=min_sample, policy=policy,
        block_adjudicate=block_adjudicate, block_alpha=block_alpha, self_join=True,
        workers=workers, on_error=on_error, seed=seed, cache=cache, budget=budget,
        telemetry=telemetry)

    # union-find over matched pairs -> entity clusters
    parent = {r.id: r.id for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for l, r in jr.pairs:
        rl, rr = find(l.id), find(r.id)
        if rl != rr:
            parent[rl] = rr

    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(find(r.id), []).append(r)
    clusters = list(groups.values())

    js = jr.stats
    stats = DedupStats(
        n_rows=len(rows), candidate_pairs=js.candidate_pairs, matched_pairs=js.matches,
        n_clusters=len(clusters), n_duplicate_rows=len(rows) - len(clusters),
        oracle_calls=js.oracle_calls, block_calls=js.block_calls, overflows=js.overflows)
    return DedupResult(clusters, stats)


def _block_adjudicate(model, predicate, left, rights, alpha, cache, budget, stats,
                      chat_cost, on_error, slock=None):
    """Adjudicate a left row against many right candidates in batched prompts.
    Start optimistic (all rights in one prompt); on overflow (output truncated,
    no 'Finished'), shrink the batch by alpha and retry — Trummer's adaptive join.
    Returns the set of matching right ids."""
    import contextlib
    lock = slock or contextlib.nullcontext()
    matched: set = set()
    key_model = getattr(model, "chat_model", model.__class__.__name__)
    batch = max(1, len(rights))
    i = 0
    while i < len(rights):
        chunk = rights[i:i + batch]
        budget.charge(chat_cost)
        try:
            idxs, complete = model.match_block(predicate, left.text, [r.text for r in chunk])
        except Exception:
            if on_error == "raise":
                raise
            idxs, complete = [], True
            with lock:
                stats.errors += 1
        with lock:
            stats.block_calls += 1
            stats.oracle_calls += 1
        if not complete and batch > 1:
            batch = max(1, batch // alpha)  # adaptive: reserve more output room, retry same chunk
            with lock:
                stats.overflows += 1
            continue
        hit = {chunk[j].id for j in idxs if 0 <= j < len(chunk)}
        matched |= hit
        for r in chunk:  # populate per-pair cache so re-runs are cheap/consistent
            cache.put(make_key("join", key_model, predicate, left.id, r.id), r.id in hit)
        i += len(chunk)
    return matched


def _sample_indices(n: int, frac: float, min_sample: int, max_sample: Optional[int],
                    seed: int) -> list[int]:
    target = max(min_sample, math.ceil(frac * n))
    if max_sample is not None:
        target = min(target, max_sample)
    target = min(target, n)
    rng = random.Random(seed)
    return rng.sample(range(n), target)


def _ensure_row_vectors(engine, model, cache, rows, indices):
    """Ensure the given rows carry embeddings, preferring the engine's stored
    vectors (fetch_vectors) over re-embedding text — so we pull only what we need
    (e.g. just the fit sample) rather than the whole collection's vectors."""
    missing = [rows[i] for i in indices if not rows[i].embedding]
    if not missing:
        return
    fetch = getattr(engine, "fetch_vectors", None)
    src = getattr(engine, "_last_source", None)
    if fetch and src:
        got = fetch(src, [r.id for r in missing])
        for r in missing:
            if got.get(r.id):
                r.embedding = got[r.id]
        missing = [r for r in missing if not r.embedding]
    if missing:  # fallback: recompute from text via the model (cached)
        _ensure_embeddings(model, cache, missing)


def _split(idx: list[int], seed: int, cal_frac: float = 0.34) -> tuple[list[int], list[int]]:
    """Split the labeled sample into (fit, calibrate) so thresholds are set on
    data the learned proxy did NOT train on — keeps the guarantees honest."""
    order = list(idx)
    random.Random(seed + 1).shuffle(order)
    k = max(1, int(len(order) * cal_frac))
    return order[k:], order[:k]  # fit, calibrate


def _oof_proba(make_proxy, X, y, folds: int = 5, seed: int = 0) -> list[float]:
    """Out-of-fold proxy probabilities: every labelled row is scored by a model
    that did not train on it.

    Why not a plain fit/calibrate split: the split spends most labels on fitting,
    and the Wilson bounds then have too few points left to certify anything. On
    2k reviews a 100-row sample left 34 calibration points, which cannot certify
    precision 0.9 at delta=0.05 (that needs ~45 consecutive positives) — so
    tau_plus stayed +inf and the cascade escalated every row. Cross-fitting keeps
    the thresholds honest (no row is scored by a model that saw it) while letting
    ALL labels reach the calibrator.
    """
    n = len(X)
    order = list(range(n))
    random.Random(seed + 1).shuffle(order)
    out = [0.5] * n
    for f in range(folds):
        test = [order[i] for i in range(f, n, folds)]
        if not test:
            continue
        tset = set(test)
        train = [i for i in range(n) if i not in tset]
        ytr = [y[i] for i in train]
        if len(set(ytr)) < 2:      # degenerate fold — leave the neutral prior
            continue
        m = make_proxy()
        m.fit([X[i] for i in train], ytr)
        for i, p in zip(test, m.predict_proba([X[i] for i in test])):
            out[i] = p
    return out


def _proba_tau_to_dot(tau_p: float, bias: float) -> float:
    """Map a threshold on P(positive) to the dot-product scale the index scores in.

    The proxy is p = sigmoid(w.x + b), so w.x = logit(p) - b. Calibrating in
    probability space keeps fold models comparable (their raw dot scales are not);
    this converts the result onto the final model's scale exactly.
    """
    if tau_p <= 0.0:
        return float("-inf")
    if tau_p >= 1.0:
        return float("inf")
    return math.log(tau_p / (1.0 - tau_p)) - bias

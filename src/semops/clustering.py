"""Lightweight k-means for semantic grouping.

Uses numpy when available (fast, vectorized) and falls back to a pure-Python
implementation so the operator still runs with zero dependencies. Returns a flat
list of cluster ids, one per input vector.
"""
from __future__ import annotations

import math
import random
from typing import Sequence


def kmeans(vectors: Sequence[Sequence[float]], k: int, iters: int = 50, seed: int = 0,
           n_init: int = 8, normalize: bool = True) -> list[int]:
    """k-means returning a cluster id per vector. Runs n_init restarts and keeps
    the lowest-inertia solution (k-means is sensitive to initialization).
    normalize=True L2-normalizes first, so clustering uses cosine geometry —
    the right choice for embeddings."""
    n = len(vectors)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))  # degenerate: each point its own cluster
    try:
        import numpy as np
        return _kmeans_numpy(vectors, k, iters, seed, n_init, normalize, np)
    except Exception:
        return _kmeans_pure(vectors, k, iters, seed, n_init, normalize)


def _kmeans_numpy(vectors, k, iters, seed, n_init, normalize, np):
    X = np.asarray(vectors, dtype=float)
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = X.shape[0]
    best_labels, best_inertia = None, math.inf
    for r in range(n_init):
        rng = np.random.default_rng(seed + r)
        centers = X[rng.choice(n, k, replace=False)].copy()
        labels = np.full(n, -1)
        for it in range(iters):
            # argmin_c ||x-c||^2 == argmin_c (||c||^2 - 2 x·c)
            d = (centers ** 2).sum(1)[None, :] - 2.0 * X.dot(centers.T)
            new = d.argmin(1)
            if it > 0 and np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                m = X[labels == c]
                if len(m):
                    centers[c] = m.mean(0)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.tolist()
    return best_labels


def _sqdist(a, b):
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def _l2(v):
    nrm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / nrm for x in v]


def _kmeans_pure(vectors, k, iters, seed, n_init, normalize):
    V = [_l2(v) for v in vectors] if normalize else [list(v) for v in vectors]
    n = len(V)
    best_labels, best_inertia = None, math.inf
    for r in range(n_init):
        rng = random.Random(seed + r)
        centers = [list(V[i]) for i in rng.sample(range(n), k)]
        labels = [-1] * n
        for it in range(iters):
            changed = False
            for i, x in enumerate(V):
                best = min(range(k), key=lambda c: _sqdist(x, centers[c]))
                if best != labels[i]:
                    labels[i] = best
                    changed = True
            if it > 0 and not changed:
                break
            for c in range(k):
                members = [V[i] for i in range(n) if labels[i] == c]
                if members:
                    dim = len(members[0])
                    centers[c] = [sum(m[d] for m in members) / len(members) for d in range(dim)]
        inertia = sum(_sqdist(V[i], centers[labels[i]]) for i in range(n))
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels

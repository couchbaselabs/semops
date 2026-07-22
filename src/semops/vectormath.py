""" 
If numpy is installed these could be swapped for vectorized versions, but
correctness does not depend on it.
"""
from __future__ import annotations

import math
from typing import Sequence


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1]. Higher = more similar.

    We use *similarity* (not distance) as the proxy score throughout, so that
    "higher score -> more likely the predicate is TRUE" holds and the cascade
    thresholds read naturally (accept above tau_plus, reject below tau_minus).
    """
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)

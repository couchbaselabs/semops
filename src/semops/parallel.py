"""Bounded thread-pool map for I/O-bound work (LLM calls, N1QL round-trips).

Everything expensive in semops is network-bound, so threads (not processes) are
the right tool — the GIL is released during socket I/O.

`pmap` is ORDER-PRESERVING: results come back in input order regardless of
completion order, so parallel runs produce byte-identical results to sequential
ones. workers<=1 runs inline with no threads at all.
"""
from __future__ import annotations

from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def pmap(fn: Callable[[T], R], items: Iterable[T], workers: int = 1) -> list[R]:
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))  # ex.map preserves input order

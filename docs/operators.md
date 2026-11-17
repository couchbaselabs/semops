---
title: Operators
nav_order: 2
has_children: true
permalink: /operators/
---

# Operators

Each operator takes a condition written in natural language and judges it with an
LLM, but spends the LLM only where a cheaper signal cannot decide. A vector index
narrows the candidates, and a calibrated cascade settles the easy rows with
precision and recall guarantees.

| Operator | SQL analogue | How it stays affordable |
|---|---|---|
| [`sem_filter`](sem_filter.md) | `WHERE` | A learned proxy bands every row into accept, reject or escalate; only the escalate band reaches the LLM. With the `DOT` metric the index itself serves the proxy. |
| [`sem_join`](sem_join.md) | `JOIN` | `APPROX_VECTOR_DISTANCE` blocking prunes the cross product, a per-pair cascade settles easy pairs, and the uncertain pairs are judged in batched prompts. |
| [`sem_dedup` and `sem_group_by`](sem_dedup_and_group_by.md) | `DISTINCT` and `GROUP BY` | Dedup is a self-join plus union-find. Group-by clusters the stored vectors, with no LLM calls at all. |

The pages below show each operator's architecture. The
[home page](../README.md) has the quickstart and the benchmarks.

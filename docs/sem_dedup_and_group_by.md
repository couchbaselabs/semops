# `sem_dedup` and `sem_group_by` architecture

## `sem_dedup`

A self-`sem_join` plus clustering:

```
rows ──► sem_join(rows, rows, "same entity", self_join=True)
           └── symmetric pairs collapsed so each unordered pair is judged once
         matched pairs ──► union-find ──► entity clusters
```

`.duplicate_groups()` gives the groups with more than one member.
`.canonical()` gives one representative each.

Everything in [`sem_join.md`](sem_join.md) applies, since this *is* a join.
`block_k` is still the ceiling on what can be found.

**Transitive closure, the caveat that actually bites.** Connected components merge
anything transitively linked. If A~B and B~C then A, B and C land in one entity
even if A~C was never confirmed, and even if A~C is false. That is standard
entity-resolution behaviour and it over-merges on noisy matches. A high-precision
oracle is what keeps it honest, so prefer precision over recall when setting
targets for a dedup.

Under-merging is the opposite failure: if blocking never connects two members of a
group, the group fragments. We measured 9 clusters for 7 true categories on 20
Newsgroups, which is fragmentation, not over-merging.

## `sem_group_by`

The cheapest operator. No LLM at all in the default path:

```
rows ──► embeddings ──► k-means ──► k groups
              (numpy fast path; pure-Python fallback;
               n_init restarts; L2-normalised, so this is cosine geometry)
    │
    ├── optional: name_clusters=True  ──► 1 LLM call per cluster to name it
    └── optional: method="llm_label"  ──► LOTUS style: the LLM proposes a label
                  per row, then the LABELS are embedded and clustered. One call
                  per row, so expensive, but it gives you control over what
                  "similar" means. Useful when you want grouping by intent
                  rather than by topic.
```

`k` is yours to choose. This is the classic k-means problem and we do not estimate
it. If you want the LLM to discover the number of groups, that is `sem_agg`
territory and `sem_agg` is not built.

Because the default path never calls the LLM, this operator has no cascade, no
thresholds, and no guarantee attached. Cluster purity is whatever the embedding
geometry gives you: measured 0.90 on 20 Newsgroups with `k` set to the true number
of categories.

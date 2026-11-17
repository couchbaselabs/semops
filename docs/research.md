---
title: Research
nav_order: 4
has_children: true
permalink: /research/
---

# Research

The experiments behind semops, kept as working records rather than polished
papers. Results that overturned earlier conclusions are left in, next to the
correction, and each claim carries a confidence marker: **[solid]**,
**[tentative]** or **[retracted]**.

| Page | What it covers |
|---|---|
| [Findings](../FINDINGS.md) | The consolidated record: filter, join and dedup quality with a real LLM judge; the cascade on a live GSI cluster; what Qwen3-8B's hidden states reveal about operator decisions; and the measurement mistakes worth not repeating. |
| [Optimizing semantic joins](optimizing-semantic-joins.md) | A systematic pass over `sem_join`, written like a lab-meeting walkthrough. |
| [Semantic join: complete record](semantic-join-everything.md) | Every join experiment in one place: motivation, setup, full results and what each one settled. |
| [Resumable cascades](../ideas/resumable-cascades/README.md) | November 2026. Make the cascade's cheap stage the first layers of the LLM itself, and resume the forward pass for uncertain rows. First results on Qwen3-8B. |

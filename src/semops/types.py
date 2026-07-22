"""Core data types shared across semops. Backend-agnostic."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Row:
    """A unit of data flowing through semantic operators.

    `id`        stable identifier (used as the cache key for oracle calls).
    `text`      the text handed to the LLM oracle / embedded for the proxy.
    `embedding` the proxy vector. May be None; operators will embed `text` lazily.
    `doc`       the full source document (returned to the user on collect()).
    """

    id: str
    text: str
    embedding: Optional[list[float]] = None
    doc: dict[str, Any] = field(default_factory=dict)


class Band(enum.Enum):
    """Which of the three cascade bands a row's proxy score falls into."""

    ACCEPT = "accept"      # proxy is confident TRUE  -> keep, no LLM
    ESCALATE = "escalate"  # proxy is uncertain       -> ask the oracle LLM
    REJECT = "reject"      # proxy is confident FALSE -> drop, no LLM


@dataclass
class CascadeStats:
    """Diagnostics from one cascade run. This is the ROI evidence."""

    n_rows: int = 0
    n_sample: int = 0
    tau_minus: float = float("-inf")
    tau_plus: float = float("inf")
    n_accept: int = 0
    n_reject: int = 0
    n_escalate: int = 0
    llm_calls: int = 0            # oracle calls actually made (sample + escalate, deduped)
    llm_calls_saved: int = 0      # calls a naive per-row oracle would have made, minus ours
    est_cost_usd: float = 0.0
    cache_hits: int = 0
    errors: int = 0               # oracle calls that failed (and fell back per on_error)
    proxy_collapsed: bool = False  # thresholds crossed -> single-threshold decision, no middle band
    n_scored: int = 0             # rows the proxy actually scored (< n_rows on the index-served path)
    proxy_exhaustive: bool = True  # False => the tau_minus sweep hit its k cap, so the
                                   # accept/escalate set may be incomplete

    def savings_ratio(self) -> float:
        """Naive oracle cost / cascade oracle cost. Higher is better."""
        denom = self.llm_calls if self.llm_calls else 1
        return self.n_rows / denom

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_sample": self.n_sample,
            "tau_minus": self.tau_minus,
            "tau_plus": self.tau_plus,
            "n_accept": self.n_accept,
            "n_reject": self.n_reject,
            "n_escalate": self.n_escalate,
            "llm_calls": self.llm_calls,
            "llm_calls_saved": self.llm_calls_saved,
            "est_cost_usd": round(self.est_cost_usd, 6),
            "cache_hits": self.cache_hits,
            "errors": self.errors,
            "proxy_collapsed": self.proxy_collapsed,
            "n_scored": self.n_scored,
            "proxy_exhaustive": self.proxy_exhaustive,
            "savings_ratio": round(self.savings_ratio(), 2),
        }


@dataclass
class FilterResult:
    """Return value of sem_filter: kept rows + how we got there."""

    rows: list[Row]
    stats: CascadeStats

    def docs(self) -> list[dict[str, Any]]:
        return [r.doc if r.doc else {"id": r.id, "text": r.text} for r in self.rows]


@dataclass
class JoinStats:
    """Diagnostics from a sem_join. Savings are vs a full nested-loop LLM join."""

    n_left: int = 0
    n_right: int = 0
    candidate_pairs: int = 0     # pairs surviving ANN blocking
    oracle_calls: int = 0        # LLM adjudications actually made
    matches: int = 0
    n_accept: int = 0            # pairs auto-accepted by the proxy (no LLM)
    n_reject: int = 0            # pairs auto-rejected by the proxy (no LLM)
    n_escalate: int = 0          # pairs sent to the LLM
    block_calls: int = 0         # batched (block-join) LLM calls
    overflows: int = 0           # block calls that overflowed -> adaptively shrank
    cache_hits: int = 0
    errors: int = 0
    tau_minus: float = float("-inf")
    tau_plus: float = float("inf")

    def savings_ratio(self) -> float:
        """(full nested-loop calls) / (calls we made). Higher is better."""
        denom = self.oracle_calls if self.oracle_calls else 1
        return (self.n_left * self.n_right) / denom

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_left": self.n_left, "n_right": self.n_right,
            "nested_loop_calls": self.n_left * self.n_right,
            "candidate_pairs": self.candidate_pairs,
            "oracle_calls": self.oracle_calls, "matches": self.matches,
            "n_accept": self.n_accept, "n_reject": self.n_reject,
            "n_escalate": self.n_escalate, "block_calls": self.block_calls,
            "overflows": self.overflows, "cache_hits": self.cache_hits,
            "errors": self.errors,
            "tau_minus": self.tau_minus, "tau_plus": self.tau_plus,
            "savings_ratio": round(self.savings_ratio(), 2),
        }


@dataclass
class JoinResult:
    """Return value of sem_join: matched (left, right) row pairs + diagnostics."""

    pairs: list[tuple[Row, Row]]
    stats: JoinStats

    def id_pairs(self) -> list[tuple[str, str]]:
        return [(l.id, r.id) for l, r in self.pairs]


@dataclass
class DedupStats:
    """Diagnostics from sem_dedup. Savings are vs all-pairs (n choose 2) comparison."""

    n_rows: int = 0
    candidate_pairs: int = 0
    matched_pairs: int = 0
    n_clusters: int = 0            # distinct entities found
    n_duplicate_rows: int = 0     # n_rows - n_clusters (rows that duplicate another)
    oracle_calls: int = 0
    block_calls: int = 0
    overflows: int = 0

    def savings_ratio(self) -> float:
        naive = self.n_rows * (self.n_rows - 1) / 2
        return naive / (self.oracle_calls if self.oracle_calls else 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows, "all_pairs": int(self.n_rows * (self.n_rows - 1) / 2),
            "candidate_pairs": self.candidate_pairs, "matched_pairs": self.matched_pairs,
            "n_clusters": self.n_clusters, "n_duplicate_rows": self.n_duplicate_rows,
            "oracle_calls": self.oracle_calls, "block_calls": self.block_calls,
            "overflows": self.overflows, "savings_ratio": round(self.savings_ratio(), 2),
        }


@dataclass
class DedupResult:
    """Return value of sem_dedup: entity clusters (each a list of duplicate rows)."""

    clusters: list[list[Row]]
    stats: DedupStats

    def duplicate_groups(self) -> list[list[Row]]:
        return [c for c in self.clusters if len(c) > 1]

    def canonical(self) -> list[Row]:
        """One representative row per distinct entity."""
        return [c[0] for c in self.clusters]


@dataclass
class SemGroup:
    """One semantic group produced by sem_group_by."""

    id: int
    rows: list[Row]
    label: Optional[str] = None


@dataclass
class GroupByStats:
    n_rows: int = 0
    k: int = 0
    method: str = "embedding"
    llm_calls: int = 0            # 0 for pure embedding clustering; +k if naming; +n if llm_label

    def as_dict(self) -> dict[str, Any]:
        return {"n_rows": self.n_rows, "k": self.k, "method": self.method,
                "llm_calls": self.llm_calls}


@dataclass
class GroupByResult:
    """Return value of sem_group_by: semantic groups over the rows."""

    groups: list[SemGroup]
    stats: GroupByStats

    def assignments(self) -> dict[str, int]:
        return {r.id: g.id for g in self.groups for r in g.rows}

    def sizes(self) -> dict[int, int]:
        return {g.id: len(g.rows) for g in self.groups}

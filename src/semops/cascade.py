"""The three-band cascade — the correctness core (KB Part II).

Given a cheap proxy score s(row) (higher => more likely the predicate is TRUE)
and an expensive oracle, learn two thresholds on a labeled sample:

    s >= tau_plus     -> ACCEPT   (auto-TRUE, no oracle)   guaranteed precision
    tau_minus < s < tau_plus -> ESCALATE (ask oracle)
    s <= tau_minus    -> REJECT   (auto-FALSE, no oracle)  guaranteed recall

Guarantees (following LOTUS): with a labeled sample we set thresholds so that,
with confidence >= 1 - delta/2 each:
  - precision of the ACCEPT region >= precision_target
  - recall of the KEPT set {s > tau_minus} >= recall_target
We use the Wilson score interval lower bound as the (conservative, distribution-
free-ish) confidence bound on each proportion, so the guarantees are honest for
finite samples rather than point estimates.

This module knows nothing about Couchbase, LLMs, or embeddings. It is pure math
over (score, label) pairs, which is exactly why it can be unit-tested to death.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from .types import Band

NEG_INF = float("-inf")
POS_INF = float("inf")


def _z_for_two_sided_split(delta: float) -> float:
    """z for a one-sided confidence of (1 - delta/2) — each guarantee gets delta/2."""
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return statistics.NormalDist().inv_cdf(1.0 - delta / 2.0)


def wilson_lower_bound(k: int, n: int, z: float) -> float:
    """Lower bound of the Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    return max(0.0, (center - margin) / denom)


@dataclass
class Thresholds:
    tau_minus: float
    tau_plus: float
    collapsed: bool  # thresholds crossed -> single decision boundary, no ESCALATE band

    def band(self, score: float) -> Band:
        if score >= self.tau_plus:
            return Band.ACCEPT
        if score <= self.tau_minus:
            return Band.REJECT
        return Band.ESCALATE


def calibrate(
    sample: Sequence[tuple[float, bool]],
    recall_target: float,
    precision_target: float,
    delta: float = 0.1,
) -> Thresholds:
    """Learn (tau_minus, tau_plus) from labeled (score, label) sample points.

    Degrades safely: if the proxy is too weak to meet a target, the corresponding
    threshold goes to infinity (accept nothing / reject nothing) so those rows
    ESCALATE to the oracle rather than being wrongly auto-labeled. No false
    guarantees are ever emitted.
    """
    n = len(sample)
    if n == 0:
        return Thresholds(NEG_INF, POS_INF, False)  # everything escalates
    if not (0.0 <= recall_target <= 1.0) or not (0.0 <= precision_target <= 1.0):
        raise ValueError("targets must be in [0, 1]")

    z = _z_for_two_sided_split(delta)
    scores = sorted({s for s, _ in sample})
    total_pos = sum(1 for _, y in sample if y)

    # tau_plus: the SMALLEST t whose ACCEPT region {s >= t} has guaranteed
    # precision >= precision_target. Smallest passing t => largest guaranteed region.
    tau_plus = POS_INF
    for t in scores:  # ascending
        k = sum(1 for s, y in sample if s >= t and y)
        m = sum(1 for s, _ in sample if s >= t)
        if m > 0 and wilson_lower_bound(k, m, z) >= precision_target:
            tau_plus = t
            break

    # tau_minus: the LARGEST t whose KEPT set {s > t} has guaranteed
    # recall >= recall_target. Largest passing t => largest guaranteed reject region.
    # If there are no positives in the sample we cannot certify recall by sampling,
    # so we refuse to auto-reject (tau_minus = -inf).
    tau_minus = NEG_INF
    if total_pos > 0:
        for t in sorted(scores, reverse=True):  # descending
            pos_kept = sum(1 for s, y in sample if y and s > t)
            if wilson_lower_bound(pos_kept, total_pos, z) >= recall_target:
                tau_minus = t
                break

    # If the bands cross (proxy strong enough to decide everything), collapse to a
    # single decision boundary with no ESCALATE band.
    collapsed = False
    if tau_minus != NEG_INF and tau_plus != POS_INF and tau_minus >= tau_plus:
        theta = (tau_plus + tau_minus) / 2.0
        tau_plus = tau_minus = theta
        collapsed = True

    return Thresholds(tau_minus, tau_plus, collapsed)


# --- helpers for evaluation / tests ---------------------------------------

def empirical_precision_recall(
    scored: Sequence[tuple[float, bool]],
    predicted_keep: Sequence[bool],
) -> tuple[float, float]:
    """Precision & recall of a boolean prediction vs the true labels."""
    tp = sum(1 for (_, y), p in zip(scored, predicted_keep) if p and y)
    fp = sum(1 for (_, y), p in zip(scored, predicted_keep) if p and not y)
    fn = sum(1 for (_, y), p in zip(scored, predicted_keep) if (not p) and y)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall

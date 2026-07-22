"""Correctness tests for the cascade math — the load-bearing piece.

These are pure (no engine, no model, no network): (score, label) in -> thresholds
out, and we assert the *guarantees actually hold* on held-out data and that the
calibrator degrades safely when the proxy is uninformative.
"""
import math
import random
import unittest

from semops.cascade import (
    calibrate,
    empirical_precision_recall,
    wilson_lower_bound,
)
from semops.types import Band


class TestWilson(unittest.TestCase):
    def test_bounds_are_conservative(self):
        # LB <= point estimate, and within [0,1]
        for k, n in [(5, 10), (9, 10), (50, 100), (0, 10), (10, 10)]:
            lb = wilson_lower_bound(k, n, z=1.64)
            self.assertGreaterEqual(lb, 0.0)
            self.assertLessEqual(lb, 1.0)
            self.assertLessEqual(lb, k / n + 1e-9)

    def test_tightens_with_n(self):
        # same proportion, more samples -> tighter (higher) lower bound
        lb_small = wilson_lower_bound(8, 10, z=1.64)
        lb_big = wilson_lower_bound(800, 1000, z=1.64)
        self.assertLess(lb_small, lb_big)

    def test_zero_n(self):
        self.assertEqual(wilson_lower_bound(0, 0, z=1.64), 0.0)


def _band_labels(scored, th):
    """Apply thresholds; ESCALATE rows are decided by their true label (the
    oracle would return truth), so this measures the guaranteed accept/reject
    bands against ground truth."""
    keep = []
    for s, y in scored:
        b = th.band(s)
        if b is Band.ACCEPT:
            keep.append(True)
        elif b is Band.REJECT:
            keep.append(False)
        else:
            keep.append(y)  # oracle == truth in this synthetic setting
    return keep


class TestCalibrate(unittest.TestCase):
    def _make(self, n, sep, rng):
        """Positives centered at +sep, negatives at -sep, gaussian noise=1."""
        data = []
        for _ in range(n):
            y = rng.random() < 0.4
            mu = sep if y else -sep
            data.append((rng.gauss(mu, 1.0), y))
        return data

    def test_strong_proxy_meets_targets(self):
        rng = random.Random(0)
        train = self._make(2000, sep=1.5, rng=rng)
        test = self._make(4000, sep=1.5, rng=rng)
        th = calibrate(train, recall_target=0.9, precision_target=0.9, delta=0.1)
        self.assertLessEqual(th.tau_minus, th.tau_plus)
        keep = _band_labels(test, th)
        precision, recall = empirical_precision_recall(test, keep)
        # honest statistical guarantees: allow modest slack below the 0.9 targets
        self.assertGreaterEqual(recall, 0.85, f"recall {recall}")
        self.assertGreaterEqual(precision, 0.85, f"precision {precision}")

    def test_accept_region_precision_holds(self):
        # the ACCEPT band alone (no oracle) must satisfy the precision target
        rng = random.Random(1)
        train = self._make(3000, sep=1.2, rng=rng)
        test = self._make(6000, sep=1.2, rng=rng)
        th = calibrate(train, recall_target=0.9, precision_target=0.9, delta=0.1)
        accept = [(s, y) for s, y in test if s >= th.tau_plus]
        if accept:  # only meaningful if we auto-accept anything
            pos = sum(1 for _, y in accept if y)
            self.assertGreaterEqual(pos / len(accept), 0.85)

    def test_weak_proxy_refuses_to_autoaccept(self):
        # scores independent of labels -> cannot certify precision -> tau_plus = +inf
        rng = random.Random(2)
        data = [(rng.gauss(0, 1), rng.random() < 0.4) for _ in range(1500)]
        th = calibrate(data, recall_target=0.9, precision_target=0.9, delta=0.1)
        self.assertEqual(th.tau_plus, float("inf"),
                         "weak proxy must not auto-accept anything")

    def test_no_positives_refuses_to_autoreject(self):
        data = [(random.Random(3).gauss(0, 1), False) for _ in range(200)]
        th = calibrate(data, recall_target=0.9, precision_target=0.9, delta=0.1)
        self.assertEqual(th.tau_minus, float("-inf"))
        self.assertEqual(th.tau_plus, float("inf"))

    def test_perfectly_separable_collapses(self):
        # positives all >0, negatives all <0 -> bands cross -> single boundary
        data = [(1.0 + i * 0.01, True) for i in range(100)] + \
               [(-1.0 - i * 0.01, False) for i in range(100)]
        th = calibrate(data, recall_target=0.9, precision_target=0.9, delta=0.1)
        self.assertTrue(th.collapsed)
        self.assertEqual(th.tau_minus, th.tau_plus)
        # the single boundary separates the classes
        keep = _band_labels(data, th)
        precision, recall = empirical_precision_recall(data, keep)
        self.assertGreaterEqual(precision, 0.95)
        self.assertGreaterEqual(recall, 0.95)

    def test_empty_sample(self):
        th = calibrate([], recall_target=0.9, precision_target=0.9, delta=0.1)
        self.assertEqual(th.tau_minus, float("-inf"))
        self.assertEqual(th.tau_plus, float("inf"))
        # -> band() sends everything to ESCALATE
        self.assertIs(th.band(0.0), Band.ESCALATE)


if __name__ == "__main__":
    unittest.main()

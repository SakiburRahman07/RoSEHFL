import unittest

import numpy as np

from rosehfl.utils.robust_agg import aggregate_with_rule


class RobustAggregationTests(unittest.TestCase):
    def test_trust_aggregation_downweights_outlier(self):
        weights = [
            [np.array([0.0, 0.0], dtype=np.float32)],
            [np.array([0.1, -0.1], dtype=np.float32)],
            [np.array([10.0, 10.0], dtype=np.float32)],
        ]
        aggregate, info = aggregate_with_rule(
            rule="trust",
            node_ids=[0, 1, 2],
            weights_list=weights,
            sizes=[10, 10, 10],
            phi={0: 1.0, 1: 1.0, 2: 1.0},
        )
        self.assertLess(float(info["alpha"][2]), 0.2)
        self.assertLess(np.linalg.norm(aggregate[0]), 1.0)

    def test_shrinkage_downweights_outlier_on_small_edge(self):
        # Three-node edge: the RoSE bottleneck case where the legacy
        # exp(-beta * dev / median_dev) collapses to extreme concentration.
        weights = [
            [np.array([0.0, 0.0], dtype=np.float32)],
            [np.array([0.1, -0.1], dtype=np.float32)],
            [np.array([10.0, 10.0], dtype=np.float32)],
        ]
        aggregate, info = aggregate_with_rule(
            rule="trust",
            node_ids=[0, 1, 2],
            weights_list=weights,
            sizes=[10, 10, 10],
            phi={0: 1.0, 1: 1.0, 2: 1.0},
        )
        self.assertTrue(bool(info["use_shrinkage"]))
        self.assertLess(float(info["alpha"][2]), 0.1)
        # alpha cap = 2/n = 0.667; no non-outlier should exceed it.
        self.assertLessEqual(float(np.max(info["alpha"])), 2.0 / 3.0 + 1e-6)
        self.assertLess(np.linalg.norm(aggregate[0]), 1.0)

    def test_shrinkage_handles_two_nodes(self):
        # n=2 previously destabilised median-based trust; shrinkage must cope.
        weights = [
            [np.array([0.0, 0.0], dtype=np.float32)],
            [np.array([5.0, -5.0], dtype=np.float32)],
        ]
        aggregate, info = aggregate_with_rule(
            rule="trust",
            node_ids=[0, 1],
            weights_list=weights,
            sizes=[20, 20],
        )
        self.assertEqual(info["alpha"].shape, (2,))
        self.assertTrue(np.all(np.isfinite(info["alpha"])))
        self.assertAlmostEqual(float(info["alpha"].sum()), 1.0, places=5)

    def test_dev_clipping_reduces_tail_influence(self):
        # An extreme outlier should not distort the prior scale auto-fit.
        weights = [
            [np.array([0.0, 0.0], dtype=np.float32)],
            [np.array([0.1, -0.1], dtype=np.float32)],
            [np.array([0.2, -0.2], dtype=np.float32)],
            [np.array([0.15, -0.15], dtype=np.float32)],
            [np.array([500.0, 500.0], dtype=np.float32)],
        ]
        _, info_clipped = aggregate_with_rule(
            rule="trust",
            node_ids=[0, 1, 2, 3, 4],
            weights_list=weights,
            sizes=[10] * 5,
            dev_clip_q=0.8,
        )
        _, info_unclipped = aggregate_with_rule(
            rule="trust",
            node_ids=[0, 1, 2, 3, 4],
            weights_list=weights,
            sizes=[10] * 5,
            dev_clip_q=1.0,
        )
        # Clipping keeps the outlier's alpha at least as small.
        self.assertLessEqual(
            float(info_clipped["alpha"][4]),
            float(info_unclipped["alpha"][4]) + 1e-9,
        )
        self.assertLess(float(info_clipped["alpha"][4]), 0.05)

    def test_legacy_rule_still_available(self):
        weights = [
            [np.array([0.0, 0.0], dtype=np.float32)],
            [np.array([0.1, -0.1], dtype=np.float32)],
            [np.array([10.0, 10.0], dtype=np.float32)],
        ]
        _, info = aggregate_with_rule(
            rule="trust_legacy",
            node_ids=[0, 1, 2],
            weights_list=weights,
            sizes=[10, 10, 10],
        )
        self.assertFalse(bool(info["use_shrinkage"]))

    def test_krum_returns_non_outlier(self):
        weights = [
            [np.array([0.0, 0.0], dtype=np.float32)],
            [np.array([0.05, -0.05], dtype=np.float32)],
            [np.array([9.0, 9.0], dtype=np.float32)],
        ]
        aggregate, info = aggregate_with_rule(
            rule="krum",
            node_ids=[0, 1, 2],
            weights_list=weights,
            sizes=[1, 1, 1],
            phi=None,
            krum_f=1,
        )
        self.assertEqual(info["rule"], "krum")
        self.assertLess(np.linalg.norm(aggregate[0]), 1.0)


if __name__ == "__main__":
    unittest.main()


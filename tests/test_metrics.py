"""Tests for the paper-facing UAV localization metrics."""

from __future__ import annotations

import math
import unittest

import torch

from uavgeo.metrics import compute_bearing_metrics, heading_errors_degrees


def heading(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


class BearingMetricsTest(unittest.TestCase):
    def test_primary_metrics(self) -> None:
        metrics = compute_bearing_metrics(
            predicted_tile_ids=["a", "b", "x", "d"],
            target_tile_ids=["a", "b", "c", "d"],
            predicted_xy=[[0, 0], [3, 4], [60, 0], [64, 0]],
            target_xy=torch.zeros(4, 2),
            predicted_heading=[heading(0), heading(10), heading(15), heading(20)],
            target_heading=[heading(0)] * 4,
            cities=["citya", "citya", "cityb", "cityb"],
            meters_per_pixel_xy=[0.25, 0.25],
        )
        self.assertAlmostEqual(metrics.recall_at_1, 75.0)
        self.assertAlmostEqual(metrics.lsr_at_15, 75.0)
        self.assertAlmostEqual(metrics.hsr_at_15, 75.0)
        self.assertAlmostEqual(metrics.mle, 8.0625)
        self.assertAlmostEqual(metrics.mhe, 11.25)

    def test_heading_wraparound(self) -> None:
        error = heading_errors_degrees([heading(359)], [heading(1)])
        self.assertAlmostEqual(float(error[0]), 2.0)

    def test_zero_heading_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            heading_errors_degrees([[0.0, 0.0]], [[1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()

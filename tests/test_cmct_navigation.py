"""Unit tests for the navigation-only CMCT temporal filter."""

from __future__ import annotations

import unittest

from uavgeo.cmct_navigation import (
    CMCTConfig,
    CMCTFilter,
    CMCTObservation,
    circular_distance_deg,
    circular_fusion_deg,
)
from uavgeo.data.catalog import MapRecord
from uavgeo.navigation import NavigationPose


def map_record() -> MapRecord:
    return MapRecord(
        map_id="map",
        city="citya",
        width=4096,
        height=4096,
        center_latitude=35.0,
        center_longitude=139.0,
        lat_per_pixel=2.25e-6,
        lng_per_pixel=2.75e-6,
        meters_per_pixel_x=0.25,
        meters_per_pixel_y=0.25,
    )


def observation(record: MapRecord, x: float, y: float, heading: float, confidence: float):
    latitude, longitude = record.pixel_to_geo(x, y)
    return CMCTObservation(
        NavigationPose("citya", x, y, latitude, longitude, heading), confidence
    )


class CMCTNavigationTest(unittest.TestCase):
    def test_circular_heading_wraparound(self) -> None:
        self.assertAlmostEqual(circular_distance_deg(359.0, 1.0), 2.0)
        fused = circular_fusion_deg(359.0, 1.0, 0.5)
        self.assertLess(circular_distance_deg(fused, 0.0), 1e-6)

    def test_consistent_observation_updates_state(self) -> None:
        record = map_record()
        temporal = CMCTFilter(record, CMCTConfig())
        temporal.initialize(1000.0, 1000.0, 0.0)
        result = temporal.update(
            observation(record, 1014.0, 1000.0, 0.0, 0.9),
            executed_heading_deg=0.0,
            executed_distance_m=2.5,
        )
        self.assertFalse(result.observation_rejected)
        self.assertGreater(result.update_gain, 0.15)
        self.assertGreater(result.state.x, result.prior_x)

    def test_large_jump_is_rejected(self) -> None:
        record = map_record()
        temporal = CMCTFilter(record, CMCTConfig())
        temporal.initialize(1000.0, 1000.0, 90.0)
        result = temporal.update(
            observation(record, 3500.0, 3500.0, 270.0, 0.99),
            executed_heading_deg=90.0,
            executed_distance_m=25.0,
        )
        self.assertTrue(result.observation_rejected)
        self.assertEqual(result.update_gain, 0.0)

    def test_cross_city_observation_is_rejected(self) -> None:
        record = map_record()
        temporal = CMCTFilter(record, CMCTConfig())
        temporal.initialize(1000.0, 1000.0, 90.0)
        latitude, longitude = record.pixel_to_geo(1000.0, 900.0)
        result = temporal.update(
            CMCTObservation(
                NavigationPose("cityb", 1000.0, 900.0, latitude, longitude, 90.0),
                0.99,
            ),
            executed_heading_deg=90.0,
            executed_distance_m=25.0,
        )
        self.assertTrue(result.observation_rejected)


if __name__ == "__main__":
    unittest.main()

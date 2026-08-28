"""Optional integration test against a locally generated UAV90K dataset."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from uavgeo.data.catalog import UAV90KCatalog
from uavgeo.data.datasets import LocalRegistrationDataset


class UAV90KDataTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("UAV90K_ROOT"), "UAV90K_ROOT is not set")
    def test_local_registration_sample(self) -> None:
        root = Path(os.environ["UAV90K_ROOT"])
        catalog = UAV90KCatalog(root)
        dataset = LocalRegistrationDataset(catalog, "train")
        sample = dataset[0]
        self.assertEqual(tuple(sample["query_image"].shape), (3, 252, 252))
        self.assertEqual(tuple(sample["positive_satellite_tile"].shape), (3, 252, 252))
        self.assertEqual(tuple(sample["satellite_tiles"].shape), (9, 3, 252, 252))
        self.assertEqual(tuple(sample["target_xy"].shape), (2,))
        self.assertTrue(((sample["target_xy"] >= 0) & (sample["target_xy"] <= 1)).all())
        self.assertEqual(tuple(sample["tile_validity"].shape), (3, 3))


if __name__ == "__main__":
    unittest.main()

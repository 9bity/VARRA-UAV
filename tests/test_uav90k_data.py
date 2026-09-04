"""Optional integration test against a locally generated UAV90K dataset."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
from torch import nn

from uavgeo.data.catalog import UAV90KCatalog
from uavgeo.data.datasets import (
    DINOImageTransform,
    LocalRegistrationDataset,
    SatelliteCandidateLoader,
    SingleStagePairDataset,
    UAVQueryDataset,
)
from uavgeo.inference import GlobalToLocalInference
from uavgeo.models.backbone import DINOv2Backbone
from uavgeo.models.retrieval import SatelliteFeatureIndex
from uavgeo.models.system import GlobalToLocalModel


class FakeDINO(nn.Module):
    def __init__(self, output_dim: int = 768) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, output_dim, kernel_size=14, stride=14)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        return {
            "x_norm_clstoken": tokens.mean(dim=1),
            "x_norm_patchtokens": tokens,
        }


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

    @unittest.skipUnless(os.environ.get("UAV90K_ROOT"), "UAV90K_ROOT is not set")
    def test_single_query_inference_contract(self) -> None:
        catalog = UAV90KCatalog(Path(os.environ["UAV90K_ROOT"]))
        backbone = DINOv2Backbone(
            model_name="dinov2_vitb14", freeze=True, model=FakeDINO()
        )
        model = GlobalToLocalModel(
            backbone=backbone,
            retrieval_dim=32,
            local_model_dim=32,
            adapter_dim=16,
            num_heads=4,
        )
        center_id = sorted(catalog.tiles)[0]
        index = SatelliteFeatureIndex([center_id], torch.randn(1, 32))
        engine = GlobalToLocalInference(
            model, catalog, index, torch.device("cpu"), amp_enabled=False
        )
        engine.candidate_loader = SatelliteCandidateLoader(
            catalog, DINOImageTransform(28)
        )
        query = UAVQueryDataset(catalog, "test", DINOImageTransform(28))[0]
        prediction = engine.predict(query["image"], top_k=1)
        self.assertEqual(prediction.retrieved_top1_tile_id, center_id)
        self.assertTrue(0.0 <= prediction.global_x < 4096.0)
        self.assertTrue(0.0 <= prediction.global_y < 4096.0)
        self.assertAlmostEqual(
            prediction.heading_cos**2 + prediction.heading_sin**2, 1.0, places=4
        )

    @unittest.skipUnless(os.environ.get("UAV90K_ROOT"), "UAV90K_ROOT is not set")
    def test_negative_candidate_has_masked_pose_target(self) -> None:
        catalog = UAV90KCatalog(Path(os.environ["UAV90K_ROOT"]))
        record = catalog.queries_for_split("train")[0]
        other_city = next(city for city in catalog.maps_by_city if city != record.city)
        negative_center = catalog.tile_id(other_city, 8, 8)
        self.assertIsNotNone(negative_center)
        dataset = LocalRegistrationDataset(
            catalog,
            "train",
            query_transform=DINOImageTransform(28),
            tile_transform=DINOImageTransform(28),
            negative_candidates={record.sample_id: [negative_center]},
            negative_probability=1.0,
        )
        sample = dataset[0]
        self.assertEqual(float(sample["candidate_label"]), 0.0)
        self.assertTrue(torch.equal(sample["target_xy"], torch.tensor([0.5, 0.5])))

    @unittest.skipUnless(os.environ.get("UAV90K_ROOT"), "UAV90K_ROOT is not set")
    def test_single_stage_pair_has_positive_and_negative_candidates(self) -> None:
        catalog = UAV90KCatalog(Path(os.environ["UAV90K_ROOT"]))
        records = catalog.queries_for_split("val")
        candidates: dict[str, list[str]] = {}
        tiles = tuple(catalog.tiles.values())
        for record in records:
            negative = next(
                tile.tile_id for tile in tiles if tile.city != record.city
            )
            candidates[record.sample_id] = [negative]
        dataset = SingleStagePairDataset(
            catalog,
            "val",
            candidates,
            query_transform=DINOImageTransform(28),
            tile_transform=DINOImageTransform(28),
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["positive_satellite_tiles"].shape), (9, 3, 28, 28))
        self.assertEqual(tuple(sample["negative_satellite_tiles"].shape), (9, 3, 28, 28))
        self.assertEqual(tuple(sample["positive_tile_validity"].shape), (3, 3))
        self.assertEqual(tuple(sample["negative_tile_validity"].shape), (3, 3))
        self.assertTrue(((sample["target_xy"] >= 0) & (sample["target_xy"] <= 1)).all())


if __name__ == "__main__":
    unittest.main()

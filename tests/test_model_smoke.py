"""Small tensor-contract test that does not download DINOv2 weights."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from uavgeo.losses import GlobalToLocalLoss
from uavgeo.models.backbone import DINOv2Backbone
from uavgeo.models.system import GlobalToLocalModel
from uavgeo.models.varra import weighted_similarity_transform
from uavgeo.training import build_multi_positive_mask


class FakeDINO(nn.Module):
    def __init__(self, output_dim: int = 768) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, output_dim, kernel_size=14, stride=14)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.patch_embed(images)
        tokens = feature_map.flatten(2).transpose(1, 2)
        return {
            "x_norm_clstoken": tokens.mean(dim=1),
            "x_norm_patchtokens": tokens,
        }


class GlobalToLocalSmokeTest(unittest.TestCase):
    def test_multi_positive_mask(self) -> None:
        mask = build_multi_positive_mask(
            ["tile_a", "tile_a", "tile_b"],
            ["tile_a", "tile_b", "tile_a"],
        )
        expected = torch.tensor(
            [[True, False, True], [True, False, True], [False, True, False]]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_weighted_similarity_transform(self) -> None:
        source = torch.tensor(
            [[[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]]
        )
        angle = torch.tensor(0.4)
        rotation = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle)],
                [torch.sin(angle), torch.cos(angle)],
            ]
        ).unsqueeze(0)
        expected_scale = torch.tensor([0.35])
        expected_translation = torch.tensor([[0.2, -0.1]])
        target = expected_scale[:, None, None] * torch.einsum(
            "bij,bnj->bni", rotation, source
        ) + expected_translation[:, None, :]
        estimated_rotation, estimated_translation, estimated_scale = (
            weighted_similarity_transform(source, target, torch.ones(1, 4))
        )
        self.assertTrue(torch.allclose(estimated_rotation, rotation, atol=1e-5))
        self.assertTrue(
            torch.allclose(estimated_translation, expected_translation, atol=1e-5)
        )
        self.assertTrue(torch.allclose(estimated_scale, expected_scale, atol=1e-5))

        half_rotation, half_translation, half_scale = weighted_similarity_transform(
            source.half(), target.half(), torch.ones(1, 4, dtype=torch.float16)
        )
        self.assertEqual(half_rotation.dtype, torch.float16)
        self.assertTrue(torch.isfinite(half_rotation).all())
        self.assertTrue(torch.isfinite(half_translation).all())
        self.assertTrue(torch.isfinite(half_scale).all())

    def test_forward_and_backward(self) -> None:
        backbone = DINOv2Backbone(
            model_name="dinov2_vitb14",
            freeze=True,
            model=FakeDINO(),
        )
        model = GlobalToLocalModel(
            backbone=backbone,
            retrieval_dim=32,
            local_model_dim=32,
            adapter_dim=16,
            num_heads=4,
        )
        query = torch.randn(2, 3, 28, 28)
        positive_tile = torch.randn(2, 3, 28, 28)
        satellite_tiles = torch.randn(2, 9, 3, 28, 28)
        tile_validity = torch.ones(2, 3, 3, dtype=torch.bool)
        output = model(query, positive_tile, satellite_tiles, tile_validity)

        self.assertEqual(tuple(output.query_descriptor.shape), (2, 32))
        self.assertEqual(tuple(output.satellite_descriptor.shape), (2, 32))
        self.assertEqual(tuple(output.localization.position_xy.shape), (2, 2))
        self.assertEqual(tuple(output.localization.heading.shape), (2, 2))
        self.assertEqual(tuple(output.localization.confidence_logit.shape), (2,))
        self.assertEqual(tuple(output.localization.attention.heatmap.shape), (2, 6, 6))

        positive_mask = torch.tensor([[True, False], [False, True]])
        losses = GlobalToLocalLoss()(
            output,
            target_xy=torch.tensor([[0.4, 0.6], [0.5, 0.5]]),
            target_heading=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            positive_mask=positive_mask,
            candidate_label=torch.ones(2),
        )
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        trainable_gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(trainable_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in trainable_gradients))


if __name__ == "__main__":
    unittest.main()

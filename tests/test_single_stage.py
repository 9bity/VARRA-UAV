"""Tensor-contract tests for the one-run model path."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from uavgeo.models.backbone import DINOv2Backbone
from uavgeo.models.system import GlobalToLocalModel
from uavgeo.single_stage_losses import SingleStageGlobalToLocalLoss


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


class SingleStageSmokeTest(unittest.TestCase):
    def test_positive_negative_joint_forward_and_backward(self) -> None:
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
        batch = 2
        output = model.forward_single_stage(
            query_images=torch.randn(batch, 3, 28, 28),
            positive_satellite_tiles=torch.randn(batch, 3, 28, 28),
            positive_candidate_tiles=torch.randn(batch, 9, 3, 28, 28),
            negative_candidate_tiles=torch.randn(batch, 9, 3, 28, 28),
            positive_tile_validity=torch.ones(batch, 3, 3, dtype=torch.bool),
            negative_tile_validity=torch.ones(batch, 3, 3, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.query_descriptor.shape), (batch, 32))
        self.assertEqual(
            tuple(output.positive_localization.position_xy.shape), (batch, 2)
        )
        self.assertEqual(
            tuple(output.negative_localization.confidence_logit.shape), (batch,)
        )
        self.assertEqual(
            tuple(output.positive_localization.attention.reciprocal_score.shape),
            (batch,),
        )
        self.assertEqual(
            tuple(output.positive_localization.attention.geometric_residual.shape),
            (batch,),
        )

        losses = SingleStageGlobalToLocalLoss()(
            output,
            target_xy=torch.tensor([[0.4, 0.6], [0.5, 0.5]]),
            target_heading=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            positive_mask=torch.eye(batch, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(losses.total))
        self.assertTrue(torch.isfinite(losses.quality))
        losses.total.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_quality_loss_prefers_positive_candidate(self) -> None:
        loss = SingleStageGlobalToLocalLoss()
        better = torch.tensor([2.0])
        worse = torch.tensor([-2.0])
        calibrated = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat((better, worse)), torch.tensor([1.0, 0.0])
        ) + torch.nn.functional.softplus(worse - better + loss.quality_margin).mean()
        reversed_quality = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat((worse, better)), torch.tensor([1.0, 0.0])
        ) + torch.nn.functional.softplus(better - worse + loss.quality_margin).mean()
        self.assertLess(float(calibrated), float(reversed_quality))


if __name__ == "__main__":
    unittest.main()

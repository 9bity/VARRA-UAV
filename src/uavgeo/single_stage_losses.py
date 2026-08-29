"""One-run joint objectives for retrieval, pose, and candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .losses import heatmap_point_nll, multi_positive_contrastive_loss
from .models.system import SingleStageOutput


@dataclass
class SingleStageLossOutput:
    total: Tensor
    retrieval: Tensor
    position: Tensor
    heatmap: Tensor
    heading: Tensor
    quality: Tensor


class SingleStageGlobalToLocalLoss(nn.Module):
    """Train every model component from the first epoch.

    The positive candidate receives continuous position and heading
    supervision.  Positive/negative candidates jointly train a calibrated
    quality score and a margin-ranking objective, removing the need for a
    separately trained confidence stage.
    """

    def __init__(
        self,
        retrieval_weight: float = 1.0,
        position_weight: float = 5.0,
        heatmap_weight: float = 1.0,
        heading_weight: float = 2.0,
        quality_weight: float = 0.5,
        retrieval_temperature: float = 0.07,
        quality_margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.retrieval_weight = retrieval_weight
        self.position_weight = position_weight
        self.heatmap_weight = heatmap_weight
        self.heading_weight = heading_weight
        self.quality_weight = quality_weight
        self.retrieval_temperature = retrieval_temperature
        self.quality_margin = quality_margin

    def forward(
        self,
        output: SingleStageOutput,
        target_xy: Tensor,
        target_heading: Tensor,
        positive_mask: Tensor,
    ) -> SingleStageLossOutput:
        logits = output.query_descriptor @ output.satellite_descriptor.transpose(0, 1)
        logits = logits / self.retrieval_temperature
        retrieval_query = multi_positive_contrastive_loss(logits, positive_mask)
        retrieval_reference = multi_positive_contrastive_loss(
            logits.transpose(0, 1), positive_mask.transpose(0, 1)
        )
        retrieval = 0.5 * (retrieval_query + retrieval_reference)

        positive = output.positive_localization
        negative = output.negative_localization
        position = F.smooth_l1_loss(positive.position_xy, target_xy)
        heatmap = heatmap_point_nll(positive.attention.heatmap, target_xy)
        normalized_target_heading = F.normalize(target_heading, dim=-1)
        heading = (
            1.0 - (positive.heading * normalized_target_heading).sum(dim=-1)
        ).mean()

        positive_logits = positive.confidence_logit
        negative_logits = negative.confidence_logit
        binary_logits = torch.cat((positive_logits, negative_logits), dim=0)
        binary_targets = torch.cat(
            (torch.ones_like(positive_logits), torch.zeros_like(negative_logits)),
            dim=0,
        )
        calibration = F.binary_cross_entropy_with_logits(binary_logits, binary_targets)
        ranking = F.softplus(
            negative_logits - positive_logits + self.quality_margin
        ).mean()
        quality = calibration + ranking

        total = (
            self.retrieval_weight * retrieval
            + self.position_weight * position
            + self.heatmap_weight * heatmap
            + self.heading_weight * heading
            + self.quality_weight * quality
        )
        return SingleStageLossOutput(
            total=total,
            retrieval=retrieval,
            position=position,
            heatmap=heatmap,
            heading=heading,
            quality=quality,
        )

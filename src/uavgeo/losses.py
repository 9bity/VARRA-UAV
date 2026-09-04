"""Multi-task objectives for global retrieval and local registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .models.system import GlobalToLocalOutput


@dataclass
class LossOutput:
    total: Tensor
    retrieval: Tensor
    position: Tensor
    heatmap: Tensor
    heading: Tensor
    confidence: Tensor


def multi_positive_contrastive_loss(logits: Tensor, positive_mask: Tensor) -> Tensor:
    """InfoNCE where one query may have several valid reference positives."""

    if logits.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("logits and positive_mask must have the same [Q,R] shape")
    if positive_mask.dtype is not torch.bool:
        positive_mask = positive_mask.bool()
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every query must have at least one positive reference")
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positive_mask, negative_infinity)
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()


def heatmap_point_nll(
    heatmap: Tensor, target_xy: Tensor, reduction: str = "mean"
) -> Tensor:
    """Bilinearly sample probability at a continuous [0,1] target position."""

    if heatmap.ndim != 3 or target_xy.shape != (heatmap.shape[0], 2):
        raise ValueError("Expected heatmap [B,H,W] and target_xy [B,2]")
    sampling_grid = (2.0 * target_xy - 1.0).view(-1, 1, 1, 2)
    probability = F.grid_sample(
        heatmap.unsqueeze(1),
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).flatten()
    losses = -probability.clamp_min(1e-8).log()
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class GlobalToLocalLoss(nn.Module):
    def __init__(
        self,
        retrieval_weight: float = 1.0,
        position_weight: float = 5.0,
        heatmap_weight: float = 1.0,
        heading_weight: float = 2.0,
        confidence_weight: float = 0.5,
        retrieval_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.retrieval_weight = retrieval_weight
        self.position_weight = position_weight
        self.heatmap_weight = heatmap_weight
        self.heading_weight = heading_weight
        self.confidence_weight = confidence_weight
        self.retrieval_temperature = retrieval_temperature

    def forward(
        self,
        output: GlobalToLocalOutput,
        target_xy: Tensor,
        target_heading: Tensor,
        positive_mask: Optional[Tensor] = None,
        candidate_label: Optional[Tensor] = None,
    ) -> LossOutput:
        batch = output.query_descriptor.shape[0]
        logits = output.query_descriptor @ output.satellite_descriptor.transpose(0, 1)
        logits = logits / self.retrieval_temperature
        if positive_mask is None:
            positive_mask = torch.eye(batch, device=logits.device, dtype=torch.bool)
        retrieval_query = multi_positive_contrastive_loss(logits, positive_mask)
        retrieval_reference = multi_positive_contrastive_loss(
            logits.transpose(0, 1), positive_mask.transpose(0, 1)
        )
        retrieval = 0.5 * (retrieval_query + retrieval_reference)

        if candidate_label is None:
            local_mask = torch.ones(batch, device=logits.device, dtype=torch.bool)
        else:
            local_mask = candidate_label.to(logits.device).bool()
        position_per_sample = F.smooth_l1_loss(
            output.localization.position_xy, target_xy, reduction="none"
        ).mean(dim=-1)
        position = masked_mean(position_per_sample, local_mask)
        heatmap_per_sample = heatmap_point_nll(
            output.localization.attention.heatmap, target_xy, reduction="none"
        )
        heatmap = masked_mean(heatmap_per_sample, local_mask)
        normalized_target_heading = F.normalize(target_heading, dim=-1)
        heading_per_sample = 1.0 - (
            output.localization.heading * normalized_target_heading
        ).sum(dim=-1)
        heading = masked_mean(heading_per_sample, local_mask)
        if candidate_label is None:
            confidence = output.localization.confidence_logit.new_zeros(())
        else:
            confidence = F.binary_cross_entropy_with_logits(
                output.localization.confidence_logit, candidate_label.float()
            )

        total = (
            self.retrieval_weight * retrieval
            + self.position_weight * position
            + self.heatmap_weight * heatmap
            + self.heading_weight * heading
            + self.confidence_weight * confidence
        )
        return LossOutput(total, retrieval, position, heatmap, heading, confidence)

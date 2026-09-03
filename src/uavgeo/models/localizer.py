"""Continuous position, heading, and candidate-confidence prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .varra import VARRA, VARRAOutput


@dataclass
class LocalizerOutput:
    position_xy: Tensor
    heading: Tensor
    confidence_logit: Tensor
    attention: VARRAOutput


def spatial_soft_argmax(heatmap: Tensor) -> Tensor:
    batch, height, width = heatmap.shape
    y = torch.linspace(0.0, 1.0, height, device=heatmap.device, dtype=heatmap.dtype)
    x = torch.linspace(0.0, 1.0, width, device=heatmap.device, dtype=heatmap.dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack(
        (
            (heatmap * grid_x).sum(dim=(1, 2)),
            (heatmap * grid_y).sum(dim=(1, 2)),
        ),
        dim=-1,
    ).reshape(batch, 2)


class LocalRegistrationHead(nn.Module):
    def __init__(
        self,
        backbone_dim: int,
        model_dim: int = 256,
        adapter_dim: int = 128,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.varra = VARRA(backbone_dim, model_dim, adapter_dim, num_heads)
        pose_dim = 2 * model_dim + 3
        self.position_residual = nn.Sequential(
            nn.LayerNorm(pose_dim),
            nn.Linear(pose_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 2),
        )
        self.heading_head = nn.Sequential(
            nn.LayerNorm(pose_dim),
            nn.Linear(pose_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 2),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(pose_dim + 2),
            nn.Linear(pose_dim + 2, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
        )

    def forward(
        self,
        uvp_tokens: Tensor,
        satellite_tokens: Tensor,
        uvp_grid: tuple[int, int],
        satellite_grid: tuple[int, int],
        satellite_valid_mask: Optional[Tensor] = None,
    ) -> LocalizerOutput:
        attention = self.varra(
            uvp_tokens,
            satellite_tokens,
            uvp_grid,
            satellite_grid,
            satellite_valid_mask,
        )
        query_pool = attention.query_features.mean(dim=1)
        satellite_weights = attention.heatmap.flatten(1).unsqueeze(-1)
        satellite_pool = (attention.satellite_features * satellite_weights).sum(dim=1)
        pose_features = torch.cat(
            (
                query_pool,
                satellite_pool,
                attention.geometric_heading,
                attention.scale.unsqueeze(-1),
            ),
            dim=-1,
        )

        coarse_position = spatial_soft_argmax(attention.heatmap)
        residual = 0.05 * torch.tanh(self.position_residual(pose_features))
        position = (coarse_position + residual).clamp(0.0, 1.0)
        learned_heading = self.heading_head(pose_features)
        heading = F.normalize(learned_heading + attention.geometric_heading, dim=-1)

        flat_heatmap = attention.heatmap.flatten(1)
        peak = flat_heatmap.amax(dim=-1, keepdim=True)
        entropy = -(flat_heatmap * flat_heatmap.clamp_min(1e-8).log()).sum(
            dim=-1, keepdim=True
        )
        confidence = self.confidence_head(torch.cat((pose_features, peak, entropy), dim=-1))
        return LocalizerOutput(position, heading, confidence.squeeze(-1), attention)

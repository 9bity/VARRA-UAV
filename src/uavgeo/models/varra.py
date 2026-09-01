"""View-Adaptive Rotation-aware Reciprocal Attention (VARRA)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import ViewSpecificAligner


@dataclass
class VARRAOutput:
    query_features: Tensor
    satellite_features: Tensor
    correspondence: Tensor
    heatmap: Tensor
    rotation_matrix: Tensor
    translation: Tensor
    scale: Tensor
    geometric_heading: Tensor


def normalized_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)


def weighted_similarity_transform(source: Tensor, target: Tensor, weights: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate a differentiable 2D scale-rotation-translation transform."""

    eps = torch.finfo(source.dtype).eps
    weights = weights.clamp_min(0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    source_mean = (weights.unsqueeze(-1) * source).sum(dim=1)
    target_mean = (weights.unsqueeze(-1) * target).sum(dim=1)
    source_centered = source - source_mean.unsqueeze(1)
    target_centered = target - target_mean.unsqueeze(1)
    covariance = torch.einsum(
        "bn,bni,bnj->bij", weights, source_centered, target_centered
    )
    u, singular_values, vh = torch.linalg.svd(covariance, full_matrices=False)
    correction = torch.ones((source.shape[0], 2), device=source.device, dtype=source.dtype)
    correction[:, -1] = torch.where(
        torch.det(vh.transpose(-1, -2) @ u.transpose(-1, -2)) < 0,
        -1.0,
        1.0,
    )
    diagonal = torch.diag_embed(correction)
    rotation = vh.transpose(-1, -2) @ diagonal @ u.transpose(-1, -2)
    variance = (weights.unsqueeze(-1) * source_centered.square()).sum(dim=(1, 2))
    scale = (singular_values * correction).sum(dim=-1) / variance.clamp_min(eps)
    transformed_mean = torch.einsum("bij,bj->bi", rotation, source_mean)
    translation = target_mean - scale.unsqueeze(-1) * transformed_mean
    return rotation, translation, scale


class VARRA(nn.Module):
    """Reciprocal semantic attention refined by a soft similarity transform.

    Initial UVP-to-satellite matches are accepted only when the reverse
    satellite-to-UVP attention agrees. Their soft correspondences estimate a 2D
    scale/rotation/translation transform, which gates geometrically inconsistent
    pairs before the final feature exchange and heatmap construction.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 256,
        adapter_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.aligner = ViewSpecificAligner(input_dim, model_dim, adapter_dim)
        self.query_uvp = nn.Linear(model_dim, model_dim)
        self.key_uvp = nn.Linear(model_dim, model_dim)
        self.value_uvp = nn.Linear(model_dim, model_dim)
        self.query_sat = nn.Linear(model_dim, model_dim)
        self.key_sat = nn.Linear(model_dim, model_dim)
        self.value_sat = nn.Linear(model_dim, model_dim)
        self.query_output = nn.Linear(model_dim, model_dim)
        self.satellite_output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))
        self.geometry_strength_logit = nn.Parameter(torch.tensor(-2.0))
        self.geometry_sigma_raw = nn.Parameter(torch.tensor(0.0))
        self.query_norm = nn.LayerNorm(model_dim)
        self.satellite_norm = nn.LayerNorm(model_dim)

    def _heads(self, tensor: Tensor) -> Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        uvp_tokens: Tensor,
        satellite_tokens: Tensor,
        uvp_grid: tuple[int, int],
        satellite_grid: tuple[int, int],
        satellite_valid_mask: Optional[Tensor] = None,
    ) -> VARRAOutput:
        batch, query_count, _ = uvp_tokens.shape
        satellite_count = satellite_tokens.shape[1]
        if query_count != uvp_grid[0] * uvp_grid[1]:
            raise ValueError("UVP token count does not match uvp_grid")
        if satellite_count != satellite_grid[0] * satellite_grid[1]:
            raise ValueError("Satellite token count does not match satellite_grid")

        uvp = self.aligner.forward_uvp(uvp_tokens)
        satellite = self.aligner.forward_satellite(satellite_tokens)
        q_uvp = self._heads(self.query_uvp(uvp))
        k_uvp = self._heads(self.key_uvp(uvp))
        v_uvp = self._heads(self.value_uvp(uvp))
        q_sat = self._heads(self.query_sat(satellite))
        k_sat = self._heads(self.key_sat(satellite))
        v_sat = self._heads(self.value_sat(satellite))

        temperature = self.log_temperature.exp().clamp(0.01, 1.0)
        forward_scores = torch.matmul(q_uvp, k_sat.transpose(-1, -2)) / (
            math.sqrt(self.head_dim) * temperature
        )
        reverse_scores = torch.matmul(q_sat, k_uvp.transpose(-1, -2)) / (
            math.sqrt(self.head_dim) * temperature
        )
        if satellite_valid_mask is not None:
            if satellite_valid_mask.shape != (batch, satellite_count):
                raise ValueError("satellite_valid_mask must have shape [B,Ns]")
            forward_scores = forward_scores.masked_fill(
                ~satellite_valid_mask[:, None, None, :], torch.finfo(forward_scores.dtype).min
            )

        forward_attention = F.softmax(forward_scores, dim=-1)
        reverse_attention = F.softmax(reverse_scores, dim=-1).transpose(-1, -2)
        reciprocal = torch.sqrt((forward_attention * reverse_attention).clamp_min(1e-12))

        semantic_correspondence = reciprocal.mean(dim=1)
        semantic_normalized = semantic_correspondence / semantic_correspondence.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        uvp_coordinates = normalized_grid(
            *uvp_grid, device=uvp_tokens.device, dtype=uvp_tokens.dtype
        ).unsqueeze(0).expand(batch, -1, -1)
        satellite_coordinates = normalized_grid(
            *satellite_grid, device=satellite_tokens.device, dtype=satellite_tokens.dtype
        ).unsqueeze(0).expand(batch, -1, -1)
        expected_satellite = semantic_normalized @ satellite_coordinates
        query_reliability = semantic_correspondence.amax(dim=-1)
        rotation, translation, scale = weighted_similarity_transform(
            uvp_coordinates, expected_satellite, query_reliability
        )

        projected_uvp = scale[:, None, None] * torch.einsum(
            "bij,bnj->bni", rotation, uvp_coordinates
        ) + translation[:, None, :]
        squared_distance = (
            projected_uvp[:, :, None, :] - satellite_coordinates[:, None, :, :]
        ).square().sum(dim=-1)
        sigma = F.softplus(self.geometry_sigma_raw) + 0.05
        geometric_gate = torch.exp(-squared_distance / (2.0 * sigma.square()))
        strength = torch.sigmoid(self.geometry_strength_logit)
        reciprocal = reciprocal * (
            (1.0 - strength) + strength * geometric_gate[:, None, :, :]
        )
        reciprocal = reciprocal / reciprocal.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        reciprocal = self.dropout(reciprocal)

        attended_satellite = torch.matmul(reciprocal, v_sat)
        attended_uvp = torch.matmul(reciprocal.transpose(-1, -2), v_uvp)
        attended_satellite = attended_satellite.transpose(1, 2).reshape(
            batch, query_count, self.model_dim
        )
        attended_uvp = attended_uvp.transpose(1, 2).reshape(
            batch, satellite_count, self.model_dim
        )
        query_features = self.query_norm(uvp + self.query_output(attended_satellite))
        satellite_features = self.satellite_norm(
            satellite + self.satellite_output(attended_uvp)
        )

        correspondence = reciprocal.mean(dim=1)
        heatmap = correspondence.sum(dim=1).reshape(batch, *satellite_grid)
        heatmap = heatmap / heatmap.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        heading = torch.stack((rotation[:, 0, 0], rotation[:, 1, 0]), dim=-1)
        heading = F.normalize(heading, dim=-1)
        return VARRAOutput(
            query_features=query_features,
            satellite_features=satellite_features,
            correspondence=correspondence,
            heatmap=heatmap,
            rotation_matrix=rotation,
            translation=translation,
            scale=scale,
            geometric_heading=heading,
        )

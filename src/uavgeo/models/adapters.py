"""Lightweight residual adapters for view-specific domain alignment."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class ResidualAdapter(nn.Module):
    def __init__(
        self, input_dim: int, bottleneck_dim: int, output_dim: Optional[int] = None
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, output_dim)
        self.skip = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        self.gate = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: Tensor) -> Tensor:
        adapted = self.up(self.activation(self.down(self.norm(features))))
        return self.skip(features) + torch.sigmoid(self.gate) * adapted


class ViewSpecificAligner(nn.Module):
    """Separate UVP/satellite adapters followed by a shared semantic projection."""

    def __init__(self, input_dim: int, model_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.uvp_adapter = ResidualAdapter(input_dim, bottleneck_dim, model_dim)
        self.satellite_adapter = ResidualAdapter(input_dim, bottleneck_dim, model_dim)
        self.shared = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim))

    def forward_uvp(self, features: Tensor) -> Tensor:
        return self.shared(self.uvp_adapter(features))

    def forward_satellite(self, features: Tensor) -> Tensor:
        return self.shared(self.satellite_adapter(features))

"""Global cross-view descriptor learning and exact-search utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import ViewSpecificAligner


class RetrievalHead(nn.Module):
    def __init__(self, input_dim: int, descriptor_dim: int = 256, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.aligner = ViewSpecificAligner(input_dim, descriptor_dim, bottleneck_dim)
        self.projector = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
        )

    def encode_uvp(self, descriptor: Tensor) -> Tensor:
        return F.normalize(self.projector(self.aligner.forward_uvp(descriptor)), dim=-1)

    def encode_satellite(self, descriptor: Tensor) -> Tensor:
        return F.normalize(self.projector(self.aligner.forward_satellite(descriptor)), dim=-1)


@dataclass
class SearchResult:
    scores: Tensor
    indices: Tensor


class ExactSatelliteIndex:
    """Torch cosine index used as a dependency-free reference implementation."""

    def __init__(self, descriptors: Tensor) -> None:
        if descriptors.ndim != 2:
            raise ValueError("Satellite descriptors must have shape [N,D]")
        self.descriptors = F.normalize(descriptors, dim=-1)

    def search(self, queries: Tensor, top_k: int) -> SearchResult:
        if top_k <= 0 or top_k > self.descriptors.shape[0]:
            raise ValueError("Invalid top_k")
        scores = F.normalize(queries, dim=-1) @ self.descriptors.transpose(0, 1)
        values, indices = scores.topk(top_k, dim=-1, largest=True, sorted=True)
        return SearchResult(scores=values, indices=indices)


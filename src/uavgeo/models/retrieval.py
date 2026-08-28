"""Global cross-view descriptor learning and exact-search utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import ViewSpecificAligner


class RetrievalHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        descriptor_dim: int = 256,
        bottleneck_dim: int = 256,
    ) -> None:
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


class SatelliteFeatureIndex:
    """Serializable descriptor database with stable tile-ID ordering."""

    FORMAT_VERSION = 1

    def __init__(self, tile_ids: Sequence[str], descriptors: Tensor) -> None:
        if not tile_ids or len(tile_ids) != descriptors.shape[0]:
            raise ValueError("tile_ids must match the descriptor row count")
        if len(set(tile_ids)) != len(tile_ids):
            raise ValueError("Satellite index contains duplicate tile IDs")
        self.tile_ids = tuple(tile_ids)
        self.descriptors = F.normalize(descriptors.detach().float().cpu(), dim=-1)
        self.index = ExactSatelliteIndex(self.descriptors)

    def search(self, queries: Tensor, top_k: int) -> SearchResult:
        descriptors = self.index.descriptors.to(queries.device)
        scores = F.normalize(queries.float(), dim=-1) @ descriptors.transpose(0, 1)
        values, indices = scores.topk(top_k, dim=-1, largest=True, sorted=True)
        return SearchResult(scores=values, indices=indices)

    def tile_ids_for(self, indices: Tensor) -> list[list[str]]:
        if indices.ndim != 2:
            raise ValueError("indices must have shape [B,K]")
        return [
            [self.tile_ids[int(index)] for index in row]
            for row in indices.detach().cpu()
        ]

    def save(
        self,
        path: Union[str, Path],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(
            {
                "format_version": self.FORMAT_VERSION,
                "tile_ids": self.tile_ids,
                "descriptors": self.descriptors,
                "metadata": metadata or {},
            },
            temporary,
        )
        temporary.replace(destination)

    @classmethod
    def load(
        cls, path: Union[str, Path]
    ) -> tuple["SatelliteFeatureIndex", dict[str, Any]]:
        payload = torch.load(Path(path), map_location="cpu")
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported satellite-index format version")
        return cls(payload["tile_ids"], payload["descriptors"]), payload.get("metadata", {})

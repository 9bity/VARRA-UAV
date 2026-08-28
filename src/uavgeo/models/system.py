"""Complete shared-DINO global retrieval and local registration model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor, nn

from .backbone import DINOFeatures, DINOv2Backbone
from .localizer import LocalRegistrationHead, LocalizerOutput
from .retrieval import RetrievalHead


@dataclass
class GlobalToLocalOutput:
    query_descriptor: Tensor
    satellite_descriptor: Tensor
    localization: LocalizerOutput


class GlobalToLocalModel(nn.Module):
    """One shared DINOv2 backbone with trainable retrieval and VARRA heads."""

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        backbone_pretrained: bool = True,
        freeze_backbone: bool = True,
        retrieval_dim: int = 256,
        local_model_dim: int = 256,
        adapter_dim: int = 128,
        num_heads: int = 8,
        backbone: Optional[DINOv2Backbone] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone or DINOv2Backbone(
            model_name=backbone_name,
            pretrained=backbone_pretrained,
            freeze=freeze_backbone,
        )
        self.retrieval = RetrievalHead(
            self.backbone.output_dim,
            descriptor_dim=retrieval_dim,
            bottleneck_dim=adapter_dim,
        )
        self.localizer = LocalRegistrationHead(
            self.backbone.output_dim,
            model_dim=local_model_dim,
            adapter_dim=adapter_dim,
            num_heads=num_heads,
        )

    def extract(self, images: Tensor) -> DINOFeatures:
        return self.backbone(images)

    def encode_query(self, images: Tensor) -> tuple[Tensor, DINOFeatures]:
        features = self.extract(images)
        descriptor = self.retrieval.encode_uvp(features.global_descriptor)
        return descriptor, features

    def encode_satellite(self, images: Tensor) -> tuple[Tensor, DINOFeatures]:
        features = self.extract(images)
        descriptor = self.retrieval.encode_satellite(features.global_descriptor)
        return descriptor, features

    def extract_satellite_grid(
        self, satellite_tiles: Tensor
    ) -> tuple[Tensor, tuple[int, int]]:
        """Encode nine tiles independently and stitch their patch-token grids."""

        if satellite_tiles.ndim != 5 or satellite_tiles.shape[1] != 9:
            raise ValueError("satellite_tiles must have shape [B,9,3,H,W]")
        batch, tile_count, channels, height, width = satellite_tiles.shape
        flat_tiles = satellite_tiles.reshape(batch * tile_count, channels, height, width)
        features = self.extract(flat_tiles)
        tile_grid_h, tile_grid_w = features.grid_size
        tokens = features.patch_tokens.reshape(
            batch, 3, 3, tile_grid_h, tile_grid_w, self.backbone.output_dim
        )
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(
            batch,
            3 * tile_grid_h * 3 * tile_grid_w,
            self.backbone.output_dim,
        )
        return tokens, (3 * tile_grid_h, 3 * tile_grid_w)

    def localize_candidates(
        self,
        query: DINOFeatures,
        satellite_tiles: Tensor,
        tile_validity: Optional[Tensor] = None,
    ) -> LocalizerOutput:
        """Localize one encoded query against one or more 3x3 candidates."""

        satellite_tokens, satellite_grid = self.extract_satellite_grid(satellite_tiles)
        candidate_count = satellite_tiles.shape[0]
        query_tokens = query.patch_tokens
        if query_tokens.shape[0] == 1 and candidate_count > 1:
            query_tokens = query_tokens.expand(candidate_count, -1, -1)
        elif query_tokens.shape[0] != candidate_count:
            raise ValueError("Query and candidate batch sizes are incompatible")
        satellite_valid_mask = (
            self.expand_tile_validity(tile_validity, satellite_grid)
            if tile_validity is not None
            else None
        )
        return self.localizer(
            query_tokens,
            satellite_tokens,
            query.grid_size,
            satellite_grid,
            satellite_valid_mask,
        )

    @staticmethod
    def expand_tile_validity(
        tile_validity: Tensor, satellite_grid: tuple[int, int]
    ) -> Tensor:
        if tile_validity.ndim != 3 or tile_validity.shape[1:] != (3, 3):
            raise ValueError("tile_validity must have shape [B,3,3]")
        grid_h, grid_w = satellite_grid
        if grid_h % 3 or grid_w % 3:
            raise ValueError("Satellite token grid must be divisible into 3x3 tiles")
        patch_mask = tile_validity.repeat_interleave(grid_h // 3, dim=1)
        patch_mask = patch_mask.repeat_interleave(grid_w // 3, dim=2)
        return patch_mask.flatten(1)

    def forward(
        self,
        query_images: Tensor,
        positive_satellite_tiles: Tensor,
        satellite_tiles: Tensor,
        tile_validity: Optional[Tensor] = None,
    ) -> GlobalToLocalOutput:
        query_descriptor, query = self.encode_query(query_images)
        satellite_descriptor, _ = self.encode_satellite(positive_satellite_tiles)
        localization = self.localize_candidates(query, satellite_tiles, tile_validity)
        return GlobalToLocalOutput(query_descriptor, satellite_descriptor, localization)

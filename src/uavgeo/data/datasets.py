"""PyTorch datasets for retrieval and local 3x3 registration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .catalog import QueryRecord, SatelliteTile, UAV90KCatalog


ImageTransform = Callable[[Image.Image], Tensor]
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


class DINOImageTransform:
    """Resize and normalize a PIL image for a DINOv2 ViT/14 backbone."""

    def __init__(self, size: Union[int, tuple[int, int]]) -> None:
        self.size = (size, size) if isinstance(size, int) else size
        if self.size[0] % 14 or self.size[1] % 14:
            raise ValueError("DINOv2 input dimensions must be divisible by patch size 14")

    def __call__(self, image: Image.Image) -> Tensor:
        image = image.convert("RGB").resize((self.size[1], self.size[0]), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class UAVQueryDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        catalog: UAV90KCatalog,
        split: str,
        transform: Optional[ImageTransform] = None,
    ) -> None:
        self.catalog = catalog
        self.records = catalog.queries_for_split(split)
        self.transform = transform or DINOImageTransform(252)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self.transform(load_rgb(record.uav_path))
        return {
            "sample_id": record.sample_id,
            "image": image,
            "global_xy": torch.tensor((record.global_x, record.global_y), dtype=torch.float32),
            "latitude_longitude": torch.tensor(
                (record.latitude, record.longitude), dtype=torch.float64
            ),
            "heading": torch.tensor((record.heading_cos, record.heading_sin), dtype=torch.float32),
            "gt_tile_id": record.gt_tile_id,
            "city": record.city,
        }


class SatelliteTileDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        catalog: UAV90KCatalog,
        transform: Optional[ImageTransform] = None,
    ) -> None:
        self.catalog = catalog
        self.tiles = sorted(catalog.tiles.values(), key=lambda tile: tile.tile_id)
        self.transform = transform or DINOImageTransform(252)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> dict[str, Any]:
        tile = self.tiles[index]
        return {
            "tile_id": tile.tile_id,
            "image": self.transform(load_rgb(tile.image_path)),
            "center_xy": torch.tensor((tile.center_x, tile.center_y), dtype=torch.float32),
            "city": tile.city,
        }


class SatelliteCandidateLoader:
    """Load a center tile's padded 3x3 neighborhood for local registration."""

    def __init__(
        self,
        catalog: UAV90KCatalog,
        transform: Optional[ImageTransform] = None,
        tile_size: int = 256,
    ) -> None:
        self.catalog = catalog
        self.transform = transform or DINOImageTransform(252)
        self.tile_size = tile_size

    def load(self, center_tile_id: str) -> tuple[Tensor, Tensor, Tensor]:
        center = self.catalog.tiles[center_tile_id]
        black_tile = Image.new("RGB", (self.tile_size, self.tile_size))
        tensors: list[Tensor] = []
        validity = torch.zeros((3, 3), dtype=torch.bool)
        for index, tile in enumerate(self.catalog.neighborhood(center_tile_id)):
            grid_row, grid_col = divmod(index, 3)
            if tile is None:
                image = black_tile
            else:
                image = load_rgb(tile.image_path)
                validity[grid_row, grid_col] = True
            tensors.append(self.transform(image))
        origin = torch.tensor(
            (
                (center.col - 1) * self.tile_size,
                (center.row - 1) * self.tile_size,
            ),
            dtype=torch.float32,
        )
        return torch.stack(tensors, dim=0), validity, origin


class LocalRegistrationDataset(Dataset[dict[str, Any]]):
    """Return a UVP and a positive 3x3 candidate with continuous local targets.

    The candidate center changes deterministically with the epoch. Consequently,
    the true location can occur in any of the nine cells rather than always in
    the central tile.
    """

    def __init__(
        self,
        catalog: UAV90KCatalog,
        split: str,
        query_transform: Optional[ImageTransform] = None,
        tile_transform: Optional[ImageTransform] = None,
        tile_size: int = 256,
        negative_candidates: Optional[Mapping[str, Sequence[str]]] = None,
        negative_probability: float = 0.0,
    ) -> None:
        if not 0.0 <= negative_probability <= 1.0:
            raise ValueError("negative_probability must be in [0,1]")
        self.catalog = catalog
        self.records = catalog.queries_for_split(split)
        self.query_transform = query_transform or DINOImageTransform(252)
        self.tile_transform = tile_transform or DINOImageTransform(252)
        self.tile_size = tile_size
        self.negative_candidates = negative_candidates or {}
        self.negative_probability = negative_probability
        self.candidate_loader = SatelliteCandidateLoader(
            catalog, self.tile_transform, tile_size
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _candidate_center(self, record: QueryRecord) -> tuple[str, bool]:
        key = f"{record.sample_id}:{self.epoch}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        negatives = self.negative_candidates.get(record.sample_id, ())
        use_negative = (
            bool(negatives)
            and int.from_bytes(digest[:8], "little") / 2**64
            < self.negative_probability
        )
        if use_negative:
            choice = int.from_bytes(digest[8:16], "little") % len(negatives)
            center = negatives[choice]
            if center not in self.catalog.tiles:
                raise ValueError(f"Unknown negative candidate center: {center}")
            neighborhood_ids = {
                tile.tile_id
                for tile in self.catalog.neighborhood(center)
                if tile is not None
            }
            if record.gt_tile_id in neighborhood_ids:
                raise ValueError(
                    f"Mislabeled negative contains GT tile: {record.sample_id}, {center}"
                )
            return center, False
        positives = self.catalog.positive_candidate_centers(record)
        choice = int.from_bytes(digest[8:16], "little") % len(positives)
        return positives[choice], True

    def _load_satellite_grid(self, center_tile_id: str) -> tuple[Tensor, Tensor]:
        tiles, validity, _ = self.candidate_loader.load(center_tile_id)
        return tiles, validity

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        positive_tile = self.catalog.tiles[record.gt_tile_id]
        center_tile_id, is_positive = self._candidate_center(record)
        satellite_grid, tile_validity, mosaic_origin = self.candidate_loader.load(
            center_tile_id
        )
        mosaic_origin_x = float(mosaic_origin[0])
        mosaic_origin_y = float(mosaic_origin[1])
        if is_positive:
            target_x = (record.global_x - mosaic_origin_x) / (3 * self.tile_size)
            target_y = (record.global_y - mosaic_origin_y) / (3 * self.tile_size)
        else:
            # Local pose labels are masked for negatives by GlobalToLocalLoss.
            target_x = target_y = 0.5
        if is_positive and not (0.0 <= target_x <= 1.0 and 0.0 <= target_y <= 1.0):
            raise RuntimeError(f"Target outside positive mosaic: {record.sample_id}")

        return {
            "sample_id": record.sample_id,
            "gt_tile_id": record.gt_tile_id,
            "city": record.city,
            "query_image": self.query_transform(load_rgb(record.uav_path)),
            "positive_satellite_tile": self.tile_transform(
                load_rgb(positive_tile.image_path)
            ),
            "satellite_tiles": satellite_grid,
            "candidate_center_tile_id": center_tile_id,
            "mosaic_origin_xy": torch.tensor(
                (mosaic_origin_x, mosaic_origin_y), dtype=torch.float32
            ),
            "target_xy": torch.tensor((target_x, target_y), dtype=torch.float32),
            "global_xy": torch.tensor((record.global_x, record.global_y), dtype=torch.float32),
            "heading": torch.tensor((record.heading_cos, record.heading_sin), dtype=torch.float32),
            "candidate_label": torch.tensor(float(is_positive), dtype=torch.float32),
            "tile_validity": tile_validity,
        }

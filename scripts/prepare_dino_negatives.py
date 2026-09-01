"""Prepare fixed UAV90K hard negatives with frozen public DINOv2 features.

This is deterministic data preprocessing, not a first model-training stage.
The resulting manifest can be versioned and reused by every reproduction run.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from uavgeo.data.catalog import UAV90KCatalog
from uavgeo.data.datasets import SatelliteTileDataset, UAVQueryDataset
from uavgeo.checkpoints import file_sha256
from uavgeo.mining import write_negative_manifest
from uavgeo.models.backbone import DINOv2Backbone
from uavgeo.training import dataset_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--search-k", type=int, default=200)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def encode_satellites(
    model: DINOv2Backbone,
    catalog: UAV90KCatalog,
    device: torch.device,
    amp_enabled: bool,
    batch_size: int,
    num_workers: int,
) -> tuple[list[str], torch.Tensor]:
    loader = DataLoader(
        SatelliteTileDataset(catalog),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    tile_ids: list[str] = []
    descriptors: list[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        with autocast_context(device, amp_enabled):
            features = model(images)
        tile_ids.extend(batch["tile_id"])
        descriptors.append(F.normalize(features.global_descriptor.float(), dim=-1).cpu())
    return tile_ids, torch.cat(descriptors, dim=0)


@torch.no_grad()
def mine_split(
    split: str,
    model: DINOv2Backbone,
    catalog: UAV90KCatalog,
    tile_ids: list[str],
    satellite_descriptors: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    batch_size: int,
    num_workers: int,
    search_k: int,
    negatives_per_query: int,
) -> dict[str, tuple[str, ...]]:
    dataset = UAVQueryDataset(catalog, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    database = satellite_descriptors.to(device)
    mined: dict[str, tuple[str, ...]] = {}
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        with autocast_context(device, amp_enabled):
            query = model(images).global_descriptor
        similarities = F.normalize(query.float(), dim=-1) @ database.transpose(0, 1)
        indices = similarities.topk(search_k, dim=-1).indices.cpu()
        for sample_id, gt_tile_id, row in zip(
            batch["sample_id"], batch["gt_tile_id"], indices
        ):
            candidates: list[str] = []
            for index in row.tolist():
                center_id = tile_ids[index]
                neighborhood = {
                    tile.tile_id
                    for tile in catalog.neighborhood(center_id)
                    if tile is not None
                }
                if gt_tile_id not in neighborhood:
                    candidates.append(center_id)
                if len(candidates) == negatives_per_query:
                    break
            if len(candidates) != negatives_per_query:
                raise RuntimeError(
                    f"Only {len(candidates)} valid negatives for {sample_id}; "
                    "increase --search-k"
                )
            mined[sample_id] = tuple(candidates)
    return mined


def main() -> None:
    args = parse_args()
    catalog = UAV90KCatalog(args.dataset)
    if not 0 < args.search_k <= len(catalog.tiles):
        raise ValueError("search-k must be within the satellite database size")
    if not 0 < args.negatives_per_query <= args.search_k:
        raise ValueError("negatives-per-query must be in [1, search-k]")
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    model = DINOv2Backbone(args.backbone, freeze=True).to(device).eval()
    tile_ids, satellite_descriptors = encode_satellites(
        model,
        catalog,
        device,
        amp_enabled,
        args.batch_size,
        args.num_workers,
    )
    all_candidates: dict[str, tuple[str, ...]] = {}
    for split in args.splits:
        split_candidates = mine_split(
            split,
            model,
            catalog,
            tile_ids,
            satellite_descriptors,
            device,
            amp_enabled,
            args.batch_size,
            args.num_workers,
            args.search_k,
            args.negatives_per_query,
        )
        overlap = set(all_candidates).intersection(split_candidates)
        if overlap:
            raise ValueError(f"Repeated samples across splits: {len(overlap)}")
        all_candidates.update(split_candidates)
    write_negative_manifest(args.output, all_candidates)
    metadata = {
        "dataset": dataset_fingerprint(args.dataset),
        "backbone": args.backbone,
        "splits": list(args.splits),
        "search_k": args.search_k,
        "negatives_per_query": args.negatives_per_query,
        "samples": len(all_candidates),
    }
    dino_weights = os.environ.get("DINOV2_WEIGHTS")
    if dino_weights:
        metadata["dinov2_weights"] = {
            "path": dino_weights,
            "sha256": file_sha256(dino_weights),
        }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved fixed DINOv2 negatives for {len(all_candidates)} samples to {args.output}")


if __name__ == "__main__":
    main()

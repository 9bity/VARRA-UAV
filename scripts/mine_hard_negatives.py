"""Mine Top-K satellite centers whose 3x3 areas exclude the query ground truth."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from uavgeo.checkpoints import load_trained_model
from uavgeo.data.catalog import UAV90KCatalog
from uavgeo.data.datasets import UAVQueryDataset
from uavgeo.mining import write_negative_manifest
from uavgeo.models.retrieval import SatelliteFeatureIndex
from uavgeo.models.system import GlobalToLocalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--search-k", type=int, default=50)
    parser.add_argument("--negatives-per-query", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


@torch.no_grad()
def mine_split(
    split: str,
    model: GlobalToLocalModel,
    catalog: UAV90KCatalog,
    feature_index: SatelliteFeatureIndex,
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
        persistent_workers=False,
    )
    mined: dict[str, tuple[str, ...]] = {}
    processed = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with context:
            descriptors, _ = model.encode_query(images)
        result = feature_index.search(descriptors, search_k)
        retrieved = feature_index.tile_ids_for(result.indices)
        for sample_id, gt_tile_id, centers in zip(
            batch["sample_id"], batch["gt_tile_id"], retrieved
        ):
            negatives: list[str] = []
            for center_id in centers:
                neighborhood = {
                    tile.tile_id
                    for tile in catalog.neighborhood(center_id)
                    if tile is not None
                }
                if gt_tile_id not in neighborhood:
                    negatives.append(center_id)
                if len(negatives) == negatives_per_query:
                    break
            if negatives:
                mined[sample_id] = tuple(negatives)
        processed += len(batch["sample_id"])
        if processed % 1000 == 0:
            print(f"{split}: processed {processed}/{len(dataset)}")
    return mined


def main() -> None:
    args = parse_args()
    if args.search_k <= 0 or args.negatives_per_query <= 0:
        raise ValueError("search-k and negatives-per-query must be positive")
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    model, checkpoint = load_trained_model(args.checkpoint, device)
    feature_index, metadata = SatelliteFeatureIndex.load(args.index)
    if metadata.get("checkpoint_epoch") != int(checkpoint["epoch"]):
        raise ValueError("Satellite index and model checkpoint epochs do not match")
    if args.search_k > len(feature_index.tile_ids):
        raise ValueError("search-k exceeds the satellite database size")
    catalog = UAV90KCatalog(args.dataset)

    all_candidates: dict[str, tuple[str, ...]] = {}
    for split in args.splits:
        overlap = set(all_candidates) & {
            item.sample_id for item in catalog.queries_for_split(split)
        }
        if overlap:
            raise ValueError(f"Repeated samples across requested splits: {len(overlap)}")
        all_candidates.update(
            mine_split(
                split,
                model,
                catalog,
                feature_index,
                device,
                amp_enabled,
                args.batch_size,
                args.num_workers,
                args.search_k,
                args.negatives_per_query,
            )
        )
    write_negative_manifest(args.output, all_candidates)
    print(f"saved hard negatives for {len(all_candidates)} samples to {args.output}")


if __name__ == "__main__":
    main()

"""Encode the UAV90K satellite database and save a cosine-search index."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from uavgeo.checkpoints import load_trained_model
from uavgeo.data.catalog import UAV90KCatalog
from uavgeo.data.datasets import SatelliteTileDataset
from uavgeo.models.retrieval import SatelliteFeatureIndex
from uavgeo.models.system import GlobalToLocalModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
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
def encode_database(
    model: GlobalToLocalModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    amp_enabled: bool,
) -> tuple[list[str], torch.Tensor]:
    tile_ids: list[str] = []
    batches: list[torch.Tensor] = []
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with context:
            descriptors, _ = model.encode_satellite(images)
        tile_ids.extend(batch["tile_id"])
        batches.append(descriptors.float().cpu())
        if step % 10 == 0:
            print(f"encoded {len(tile_ids)} satellite tiles")
    if not batches:
        raise RuntimeError("Satellite dataset is empty")
    return tile_ids, torch.cat(batches, dim=0)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    model, checkpoint = load_trained_model(args.checkpoint, device)
    catalog = UAV90KCatalog(args.dataset)
    dataset = SatelliteTileDataset(catalog)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    tile_ids, descriptors = encode_database(model, loader, device, amp_enabled)
    index = SatelliteFeatureIndex(tile_ids, descriptors)
    index.save(
        args.output,
        metadata={
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "dataset": str(args.dataset.resolve()),
            "descriptor_dim": int(descriptors.shape[1]),
        },
    )
    print(f"saved {len(tile_ids)} descriptors to {args.output.resolve()}")


if __name__ == "__main__":
    main()

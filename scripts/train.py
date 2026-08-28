"""Train the UAV90K global-retrieval and local-registration network."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from uavgeo.data.catalog import UAV90KCatalog
from uavgeo.data.datasets import LocalRegistrationDataset
from uavgeo.losses import GlobalToLocalLoss, LossOutput
from uavgeo.mining import read_negative_manifest
from uavgeo.models.system import GlobalToLocalModel
from uavgeo.training import (
    append_jsonl,
    atomic_torch_save,
    build_multi_positive_mask,
    capture_rng_state,
    restore_rng_state,
    seed_everything,
    trainable_parameters,
)


LOSS_NAMES = ("total", "retrieval", "position", "heatmap", "heading", "confidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backbone", default="dinov2_vitb14")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--retrieval-dim", type=int, default=256)
    parser.add_argument("--local-model-dim", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--negative-manifest", type=Path)
    parser.add_argument("--negative-probability", type=float, default=0.0)
    parser.add_argument("--confidence-weight", type=float, default=0.0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def make_loader(
    dataset: LocalRegistrationDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    generator: torch.Generator,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=False,
        generator=generator,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    keys = (
        "query_image",
        "positive_satellite_tile",
        "satellite_tiles",
        "tile_validity",
        "target_xy",
        "heading",
        "candidate_label",
    )
    return {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in keys
    }


def autocast_context(device: torch.device, enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    # The reciprocal-attention path contains very small probabilities.  BF16
    # keeps FP32's exponent range, avoiding FP16 underflow and non-finite
    # gradients while retaining Tensor Core acceleration on supported GPUs.
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def batch_losses(
    model: GlobalToLocalModel,
    criterion: GlobalToLocalLoss,
    batch: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
) -> LossOutput:
    tensors = move_batch(batch, device)
    positive_mask = build_multi_positive_mask(
        batch["gt_tile_id"], batch["gt_tile_id"]
    ).to(device)
    with autocast_context(device, amp_enabled):
        output = model(
            tensors["query_image"],
            tensors["positive_satellite_tile"],
            tensors["satellite_tiles"],
            tensors["tile_validity"],
        )
        return criterion(
            output,
            tensors["target_xy"],
            tensors["heading"],
            positive_mask=positive_mask,
            candidate_label=tensors["candidate_label"],
        )


def update_totals(totals: dict[str, float], losses: LossOutput, batch_size: int) -> None:
    for name in LOSS_NAMES:
        totals[name] += float(getattr(losses, name).detach()) * batch_size


def averaged(totals: dict[str, float], samples: int) -> dict[str, float]:
    if samples == 0:
        raise RuntimeError("No samples were processed")
    return {name: value / samples for name, value in totals.items()}


def limited_batches(
    loader: DataLoader[dict[str, Any]], limit: Optional[int]
) -> Iterator[dict[str, Any]]:
    for index, batch in enumerate(loader):
        if limit is not None and index >= limit:
            break
        yield batch


def train_epoch(
    model: GlobalToLocalModel,
    criterion: GlobalToLocalLoss,
    loader: DataLoader[dict[str, Any]],
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_clip: float,
    log_interval: int,
    limit: Optional[int],
) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in LOSS_NAMES}
    samples = 0
    for step, batch in enumerate(limited_batches(loader, limit), start=1):
        optimizer.zero_grad(set_to_none=True)
        losses = batch_losses(model, criterion, batch, device, amp_enabled)
        if not torch.isfinite(losses.total):
            raise FloatingPointError(
                f"Non-finite training loss at step {step}: {float(losses.total)}"
            )
        scaler.scale(losses.total).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0:
            clip_grad_norm_(
                trainable_parameters(model), grad_clip, error_if_nonfinite=True
            )
        scaler.step(optimizer)
        scaler.update()

        batch_size = len(batch["sample_id"])
        samples += batch_size
        update_totals(totals, losses, batch_size)
        if step % log_interval == 0:
            print(f"train step={step} samples={samples} loss={totals['total'] / samples:.6f}")
    return averaged(totals, samples)


@torch.no_grad()
def validate_epoch(
    model: GlobalToLocalModel,
    criterion: GlobalToLocalLoss,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    amp_enabled: bool,
    limit: Optional[int],
) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in LOSS_NAMES}
    samples = 0
    for batch in limited_batches(loader, limit):
        losses = batch_losses(model, criterion, batch, device, amp_enabled)
        batch_size = len(batch["sample_id"])
        samples += batch_size
        update_totals(totals, losses, batch_size)
    return averaged(totals, samples)


def save_configuration(args: argparse.Namespace, path: Path) -> None:
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 because retrieval needs in-batch negatives")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 0.0 <= args.negative_probability <= 1.0:
        raise ValueError("negative-probability must be in [0,1]")
    if args.negative_probability > 0 and args.negative_manifest is None:
        raise ValueError("negative-probability requires --negative-manifest")
    if args.confidence_weight > 0 and args.negative_probability <= 0:
        raise ValueError("confidence loss requires negative candidate sampling")
    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError("Use either --init-checkpoint or --resume, not both")

    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_configuration(args, args.output_dir / "config.json")

    catalog = UAV90KCatalog(args.dataset)
    negative_candidates = (
        read_negative_manifest(args.negative_manifest)
        if args.negative_manifest is not None
        else None
    )
    train_dataset = LocalRegistrationDataset(
        catalog,
        "train",
        negative_candidates=negative_candidates,
        negative_probability=args.negative_probability,
    )
    val_dataset = LocalRegistrationDataset(
        catalog,
        "val",
        negative_candidates=negative_candidates,
        negative_probability=args.negative_probability,
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers, True, loader_generator
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, args.num_workers, False, loader_generator
    )

    model = GlobalToLocalModel(
        backbone_name=args.backbone,
        freeze_backbone=not args.unfreeze_backbone,
        retrieval_dim=args.retrieval_dim,
        local_model_dim=args.local_model_dim,
        adapter_dim=args.adapter_dim,
        num_heads=args.num_heads,
    ).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(initial["model"])
        print(f"initialized model weights from {args.init_checkpoint}")
    criterion = GlobalToLocalLoss(confidence_weight=args.confidence_weight)
    optimizer = AdamW(
        trainable_parameters(model),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_val = math.inf

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val = float(checkpoint["best_val"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    print(
        f"device={device} amp={amp_enabled} train={len(train_dataset)} "
        f"val={len(val_dataset)} trainable={sum(p.numel() for p in trainable_parameters(model)):,}"
    )
    history_path = args.output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_dataset.set_epoch(epoch)
        # Keep validation candidates fixed so losses remain comparable by epoch.
        val_dataset.set_epoch(0)
        train_metrics = train_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            args.grad_clip,
            args.log_interval,
            args.limit_train_batches,
        )
        val_metrics = validate_epoch(
            model,
            criterion,
            val_loader,
            device,
            amp_enabled,
            args.limit_val_batches,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(history_path, record)
        print(json.dumps(record, ensure_ascii=False))

        improved = val_metrics["total"] < best_val
        if improved:
            best_val = val_metrics["total"]
        checkpoint = {
            "epoch": epoch,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "loader_generator_state": loader_generator.get_state(),
            "rng_state": capture_rng_state(),
            "args": vars(args),
        }
        atomic_torch_save(checkpoint, args.output_dir / "latest.pt")
        if improved:
            atomic_torch_save(checkpoint, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()

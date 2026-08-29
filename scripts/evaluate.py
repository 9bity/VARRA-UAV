"""Run global-to-local inference on a UAV90K split and report paper metrics."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import torch

from uavgeo.checkpoints import file_sha256, load_trained_model
from uavgeo.data.catalog import UAV90KCatalog
from uavgeo.data.datasets import UAVQueryDataset
from uavgeo.inference import GlobalToLocalInference
from uavgeo.metrics import compute_bearing_metrics
from uavgeo.models.retrieval import SatelliteFeatureIndex
from uavgeo.training import dataset_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--confidence-weight", type=float, default=0.0)
    parser.add_argument(
        "--mle-protocol",
        choices=("bearing-compatible", "global-geodesic"),
        default="bearing-compatible",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    device = resolve_device(args.device)
    model, checkpoint = load_trained_model(args.checkpoint, device)
    expected_dataset = checkpoint.get("reproducibility", {}).get("dataset", {})
    actual_dataset = dataset_fingerprint(args.dataset)
    if expected_dataset:
        if expected_dataset.get("sha256") != actual_dataset["sha256"]:
            raise ValueError("Dataset metadata/split fingerprint differs from training")
    else:
        warnings.warn(
            "Legacy checkpoint has no dataset fingerprint; exact split verification is unavailable"
        )
    index, index_metadata = SatelliteFeatureIndex.load(args.index)
    if index_metadata.get("checkpoint_epoch") != int(checkpoint["epoch"]):
        raise ValueError("Satellite index and model checkpoint epochs do not match")
    if index_metadata.get("checkpoint_sha256") != file_sha256(args.checkpoint):
        raise ValueError("Satellite index was built from a different checkpoint")

    catalog = UAV90KCatalog(args.dataset)
    dataset = UAVQueryDataset(catalog, args.split)
    engine = GlobalToLocalInference(
        model,
        catalog,
        index,
        device,
        confidence_weight=args.confidence_weight,
        candidate_batch_size=args.candidate_batch_size,
        amp_enabled=args.amp,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / f"{args.split}_predictions.csv"

    predicted_tiles: list[str] = []
    target_tiles: list[str] = []
    predicted_xy: list[list[float]] = []
    target_xy: list[list[float]] = []
    predicted_heading: list[list[float]] = []
    target_heading: list[list[float]] = []
    target_cities: list[str] = []
    predicted_cities: list[str] = []
    predicted_latlon: list[list[float]] = []
    target_latlon: list[list[float]] = []
    meters_per_pixel: list[list[float]] = []

    fields = (
        "sample_id",
        "retrieved_top1_tile_id",
        "predicted_tile_id",
        "target_tile_id",
        "predicted_city",
        "target_city",
        "global_x",
        "global_y",
        "target_x",
        "target_y",
        "latitude",
        "longitude",
        "heading_cos",
        "heading_sin",
        "selected_rank",
    )
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        sample_count = len(dataset) if args.limit is None else min(len(dataset), args.limit)
        for index_in_split in range(sample_count):
            sample = dataset[index_in_split]
            prediction = engine.predict(sample["image"], top_k=args.top_k)
            target = dataset.records[index_in_split]
            writer.writerow(
                {
                    "sample_id": target.sample_id,
                    "retrieved_top1_tile_id": prediction.retrieved_top1_tile_id,
                    "predicted_tile_id": prediction.predicted_tile_id,
                    "target_tile_id": target.gt_tile_id,
                    "predicted_city": prediction.city,
                    "target_city": target.city,
                    "global_x": prediction.global_x,
                    "global_y": prediction.global_y,
                    "target_x": target.global_x,
                    "target_y": target.global_y,
                    "latitude": prediction.latitude,
                    "longitude": prediction.longitude,
                    "heading_cos": prediction.heading_cos,
                    "heading_sin": prediction.heading_sin,
                    "selected_rank": prediction.selected_rank,
                }
            )
            predicted_tiles.append(prediction.retrieved_top1_tile_id)
            target_tiles.append(target.gt_tile_id)
            predicted_xy.append([prediction.global_x, prediction.global_y])
            target_xy.append([target.global_x, target.global_y])
            predicted_heading.append([prediction.heading_cos, prediction.heading_sin])
            target_heading.append([target.heading_cos, target.heading_sin])
            target_cities.append(target.city)
            predicted_cities.append(prediction.city)
            predicted_latlon.append([prediction.latitude, prediction.longitude])
            target_latlon.append([target.latitude, target.longitude])
            map_record = catalog.maps[target.map_id]
            meters_per_pixel.append(
                [map_record.meters_per_pixel_x, map_record.meters_per_pixel_y]
            )
            if (index_in_split + 1) % 100 == 0:
                print(f"evaluated {index_in_split + 1}/{sample_count}")

    metrics = compute_bearing_metrics(
        predicted_tiles,
        target_tiles,
        predicted_xy,
        target_xy,
        predicted_heading,
        target_heading,
        target_cities,
        meters_per_pixel,
        predicted_cities=predicted_cities,
        predicted_latlon=predicted_latlon,
        target_latlon=target_latlon,
        mle_protocol=args.mle_protocol,
    )
    cross_city_failures = sum(
        predicted != target
        for predicted, target in zip(predicted_cities, target_cities)
    )
    summary = {
        "split": args.split,
        "samples": sample_count,
        "top_k": args.top_k,
        "confidence_weight": args.confidence_weight,
        "mle_protocol": args.mle_protocol,
        "cross_city_failures": cross_city_failures,
        "cross_city_failure_rate": 100.0 * cross_city_failures / sample_count,
        "dataset_fingerprint": actual_dataset["sha256"],
        **metrics.as_dict(),
    }
    summary_path = args.output_dir / f"{args.split}_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

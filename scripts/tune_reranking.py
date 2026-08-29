"""Cache validation candidates once and search deterministic reranking settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

from uavgeo.checkpoints import file_sha256, load_trained_model
from uavgeo.data.catalog import QueryRecord, UAV90KCatalog
from uavgeo.data.datasets import UAVQueryDataset
from uavgeo.inference import GlobalToLocalInference
from uavgeo.metrics import compute_bearing_metrics
from uavgeo.models.retrieval import SatelliteFeatureIndex
from uavgeo.training import dataset_fingerprint


CACHE_FIELDS = (
    "sample_id",
    "rank",
    "center_tile_id",
    "retrieval_score",
    "confidence_logit",
    "global_x",
    "global_y",
    "heading_cos",
    "heading_sin",
)


def parse_numbers(values: Iterable[str], cast: type) -> list[Any]:
    return [cast(value) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--max-top-k", type=int, default=15)
    parser.add_argument("--candidate-batch-size", type=int, default=15)
    parser.add_argument("--top-k", nargs="+", default=("1", "3", "5", "10", "15"))
    parser.add_argument(
        "--sigmoid-weights",
        nargs="+",
        default=("0", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.75", "1"),
    )
    parser.add_argument(
        "--logit-weights",
        nargs="+",
        default=("0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.5"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def expected_cache_metadata(
    args: argparse.Namespace, dataset_sha256: str
) -> dict[str, Any]:
    return {
        "split": args.split,
        "max_top_k": args.max_top_k,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "index_sha256": file_sha256(args.index),
        "dataset_fingerprint": dataset_sha256,
    }


def build_cache(
    args: argparse.Namespace,
    catalog: UAV90KCatalog,
    model: torch.nn.Module,
    feature_index: SatelliteFeatureIndex,
    device: torch.device,
    cache_path: Path,
) -> None:
    dataset = UAVQueryDataset(catalog, args.split)
    engine = GlobalToLocalInference(
        model,
        catalog,
        feature_index,
        device,
        confidence_weight=0.0,
        candidate_batch_size=args.candidate_batch_size,
        amp_enabled=args.amp,
    )
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        for index_in_split in range(len(dataset)):
            sample = dataset[index_in_split]
            prediction = engine.predict(sample["image"], top_k=args.max_top_k)
            for candidate in prediction.candidates:
                writer.writerow(
                    {
                        "sample_id": dataset.records[index_in_split].sample_id,
                        "rank": candidate.retrieval_rank,
                        "center_tile_id": candidate.center_tile_id,
                        "retrieval_score": candidate.retrieval_score,
                        "confidence_logit": candidate.confidence_logit,
                        "global_x": candidate.global_x,
                        "global_y": candidate.global_y,
                        "heading_cos": candidate.heading_cos,
                        "heading_sin": candidate.heading_sin,
                    }
                )
            if (index_in_split + 1) % 100 == 0:
                print(f"cached {index_in_split + 1}/{len(dataset)} queries")
    temporary.replace(cache_path)


def load_cache(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["sample_id"]].append(row)
    for candidates in grouped.values():
        candidates.sort(key=lambda item: int(item["rank"]))
    return dict(grouped)


def confidence_value(logit: float, transform: str) -> float:
    if transform == "logit":
        return logit
    return 1.0 / (1.0 + math.exp(-logit))


def evaluate_setting(
    catalog: UAV90KCatalog,
    records: list[QueryRecord],
    cache: dict[str, list[dict[str, str]]],
    top_k: int,
    weight: float,
    transform: str,
) -> dict[str, Any]:
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
    selected_ranks: list[int] = []

    for target in records:
        candidates = cache[target.sample_id]
        if len(candidates) < top_k:
            raise ValueError(f"Cache has fewer than {top_k} candidates: {target.sample_id}")
        considered = candidates[:top_k]
        chosen = max(
            considered,
            key=lambda item: float(item["retrieval_score"])
            + weight
            * confidence_value(float(item["confidence_logit"]), transform),
        )
        predicted_tiles.append(candidates[0]["center_tile_id"])
        target_tiles.append(target.gt_tile_id)
        city = catalog.tiles[chosen["center_tile_id"]].city
        map_record = catalog.maps_by_city[city]
        global_x = min(max(float(chosen["global_x"]), 0.0), map_record.width - 1.0)
        global_y = min(max(float(chosen["global_y"]), 0.0), map_record.height - 1.0)
        latitude, longitude = map_record.pixel_to_geo(global_x, global_y)
        predicted_xy.append([global_x, global_y])
        target_xy.append([target.global_x, target.global_y])
        predicted_heading.append(
            [float(chosen["heading_cos"]), float(chosen["heading_sin"])]
        )
        target_heading.append([target.heading_cos, target.heading_sin])
        target_cities.append(target.city)
        predicted_cities.append(city)
        predicted_latlon.append([latitude, longitude])
        target_latlon.append([target.latitude, target.longitude])
        target_map = catalog.maps[target.map_id]
        meters_per_pixel.append(
            [target_map.meters_per_pixel_x, target_map.meters_per_pixel_y]
        )
        selected_ranks.append(int(chosen["rank"]))

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
        mle_protocol="bearing-compatible",
    )
    cross_city = sum(
        predicted != target
        for predicted, target in zip(predicted_cities, target_cities)
    )
    return {
        "top_k": top_k,
        "confidence_transform": transform,
        "confidence_weight": weight,
        "cross_city_failures": cross_city,
        "mean_selected_rank": sum(selected_ranks) / len(selected_ranks),
        **metrics.as_dict(),
    }


def main() -> None:
    args = parse_args()
    top_k_values = parse_numbers(args.top_k, int)
    if min(top_k_values) <= 0 or max(top_k_values) > args.max_top_k:
        raise ValueError("top-k values must be within [1,max-top-k]")
    device = resolve_device(args.device)
    dataset_info = dataset_fingerprint(args.dataset)
    model, checkpoint = load_trained_model(args.checkpoint, device)
    trained_dataset = checkpoint.get("reproducibility", {}).get("dataset", {})
    if trained_dataset and trained_dataset.get("sha256") != dataset_info["sha256"]:
        raise ValueError("Checkpoint and tuning dataset fingerprints differ")
    if not trained_dataset:
        warnings.warn("Legacy checkpoint has no embedded dataset fingerprint")
    feature_index, index_metadata = SatelliteFeatureIndex.load(args.index)
    if index_metadata.get("checkpoint_sha256") != file_sha256(args.checkpoint):
        raise ValueError("Satellite index was built from a different checkpoint")

    catalog = UAV90KCatalog(args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / f"{args.split}_top{args.max_top_k}_candidates.csv"
    metadata_path = cache_path.with_suffix(".meta.json")
    expected_metadata = expected_cache_metadata(args, dataset_info["sha256"])
    if cache_path.exists() and not args.rebuild_cache:
        if not metadata_path.is_file():
            raise ValueError("Candidate cache metadata is missing")
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual_metadata != expected_metadata:
            raise ValueError("Candidate cache metadata does not match this run")
        print(f"reusing {cache_path}")
    else:
        build_cache(args, catalog, model, feature_index, device, cache_path)
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2) + "\n", encoding="utf-8"
        )

    cache = load_cache(cache_path)
    records = catalog.queries_for_split(args.split)
    if len(cache) != len(records):
        raise ValueError("Candidate cache does not cover the complete split")
    settings: list[tuple[str, float]] = [
        ("sigmoid", value) for value in parse_numbers(args.sigmoid_weights, float)
    ] + [("logit", value) for value in parse_numbers(args.logit_weights, float)]
    results = [
        evaluate_setting(catalog, records, cache, top_k, weight, transform)
        for top_k in top_k_values
        for transform, weight in settings
    ]
    results.sort(
        key=lambda item: (
            -item["lsr_at_15"],
            -item["hsr_at_15"],
            item["mle"],
            item["mhe"],
            item["cross_city_failures"],
        )
    )
    output = {
        "split": args.split,
        "samples": len(records),
        "dataset_fingerprint": dataset_info["sha256"],
        "selection_rule": "LSR desc, HSR desc, MLE asc, MHE asc, cross-city asc",
        "best": results[0],
        "results": results,
    }
    output_path = args.output_dir / f"{args.split}_reranking_sweep.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["best"], indent=2))
    print(f"saved {len(results)} settings to {output_path}")


if __name__ == "__main__":
    main()

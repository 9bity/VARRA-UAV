"""Evaluate CMCT-Naver with the unchanged trained single-stage localizer."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import fields
from pathlib import Path

import torch
from PIL import Image

from uavgeo.checkpoints import file_sha256, load_trained_model
from uavgeo.cmct_navigation import (
    CMCTConfig,
    CMCTObservation,
    run_cmct_navigation_episode,
)
from uavgeo.data.catalog import UAV90KCatalog
from uavgeo.data.datasets import DINOImageTransform
from uavgeo.inference import GlobalToLocalInference
from uavgeo.models.retrieval import SatelliteFeatureIndex
from uavgeo.navigation import NavigationPose, compute_navigation_metrics, load_navigation_routes
from uavgeo.training import dataset_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--evaluation-lock", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--cmct-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--step-m", type=float, default=25.0)
    parser.add_argument("--arrival-radius-m", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-id", action="append")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def verify_single_stage(checkpoint: dict[str, object]) -> None:
    arguments = checkpoint.get("args")
    if not isinstance(arguments, dict):
        raise ValueError("Checkpoint has no recorded training arguments")
    if "quality_weight" not in arguments or "quality_margin" not in arguments:
        raise ValueError("Checkpoint is not identifiable as the single-stage model")
    if arguments.get("init_checkpoint"):
        raise ValueError("Checkpoint unexpectedly records a stage-1 initializer")
    if not bool(arguments.get("deterministic", False)):
        raise ValueError("Single-stage checkpoint was not trained deterministically")


def load_cmct_config(path: Path) -> tuple[CMCTConfig, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {field.name for field in fields(CMCTConfig)}
    config = CMCTConfig(**{name: payload[name] for name in names})
    config.validate()
    return config, payload


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def main() -> None:
    args = parse_args()
    if args.step_m <= 0 or args.arrival_radius_m <= 0 or args.max_steps <= 0:
        raise ValueError("Navigation parameters must be positive")
    config, config_payload = load_cmct_config(args.cmct_config)
    lock = json.loads(args.evaluation_lock.read_text(encoding="utf-8"))
    selected = lock["selected"]
    top_k = int(selected["top_k"])
    candidate_batch_size = int(selected["candidate_batch_size"])
    confidence_weight = float(selected["confidence_weight"])
    confidence_transform = str(selected["confidence_transform"])

    device = resolve_device(args.device)
    model, checkpoint = load_trained_model(args.checkpoint, device)
    verify_single_stage(checkpoint)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    index_sha256 = file_sha256(args.index)
    if checkpoint_sha256 != lock["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the locked evaluation config")
    if index_sha256 != lock["index_sha256"]:
        raise ValueError("Index does not match the locked evaluation config")
    actual_dataset = dataset_fingerprint(args.dataset)
    if actual_dataset["sha256"] != lock["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint does not match the locked config")
    trained_dataset = checkpoint.get("reproducibility", {}).get("dataset", {})
    if trained_dataset.get("sha256") != actual_dataset["sha256"]:
        raise ValueError("Checkpoint and navigation dataset fingerprints differ")

    index, index_metadata = SatelliteFeatureIndex.load(args.index)
    if index_metadata.get("checkpoint_epoch") != int(checkpoint["epoch"]):
        raise ValueError("Satellite index and checkpoint epochs do not match")
    if index_metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("Satellite index was built from another checkpoint")

    catalog = UAV90KCatalog(args.dataset)
    engine = GlobalToLocalInference(
        model,
        catalog,
        index,
        device,
        confidence_weight=confidence_weight,
        confidence_transform=confidence_transform,
        candidate_batch_size=candidate_batch_size,
        amp_enabled=args.amp,
    )
    transform = DINOImageTransform(252)
    routes = load_navigation_routes(args.routes)
    if args.route_id:
        requested = set(args.route_id)
        routes = tuple(route for route in routes if route.route_id in requested)
        missing = requested - {route.route_id for route in routes}
        if missing:
            raise ValueError(f"Unknown route IDs: {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for route_number, route in enumerate(routes, start=1):
        map_record = catalog.maps_by_city[route.city]
        with Image.open(catalog.root / "maps" / route.city / "map.jpg") as handle:
            map_image = handle.convert("RGB")

        def predict(observation: Image.Image) -> CMCTObservation:
            result = engine.predict(transform(observation), top_k=top_k)
            chosen = result.candidates[result.selected_rank - 1]
            return CMCTObservation(
                pose=NavigationPose(
                    city=result.city,
                    x=result.global_x,
                    y=result.global_y,
                    latitude=result.latitude,
                    longitude=result.longitude,
                    heading_deg=math.degrees(
                        math.atan2(result.heading_sin, result.heading_cos)
                    )
                    % 360.0,
                ),
                confidence=sigmoid(chosen.confidence_logit),
            )

        episode = run_cmct_navigation_episode(
            route,
            map_record,
            map_image,
            predict,
            config,
            step_m=args.step_m,
            arrival_radius_m=args.arrival_radius_m,
            max_steps=args.max_steps,
        )
        episodes.append(episode)
        (args.output_dir / f"{route.route_id}.json").write_text(
            json.dumps(episode.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"route {route_number}/{len(routes)} {route.route_id}: "
            f"success={episode.success} NE={episode.navigation_error_m:.2f}m "
            f"SPL={episode.spl:.2f} steps={episode.steps} "
            f"rejected={episode.rejected_observations}",
            flush=True,
        )

    metrics = compute_navigation_metrics(episodes, args.arrival_radius_m)
    summary: dict[str, object] = {
        "protocol": "Bearing-Naver-2D-closed-loop",
        "navigation_algorithm": "CMCT-Naver",
        "localizer": "unchanged-trained-single-stage-global-to-local",
        "observation": "rotated satellite-view crop at real UAV pose",
        "step_m": args.step_m,
        "arrival_radius_m": args.arrival_radius_m,
        "max_steps": args.max_steps,
        "top_k": top_k,
        "candidate_batch_size": candidate_batch_size,
        "confidence_weight": confidence_weight,
        "confidence_transform": confidence_transform,
        "cmct_config": config_payload,
        "cmct_config_sha256": file_sha256(args.cmct_config),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "index": str(args.index.resolve()),
        "index_sha256": index_sha256,
        "evaluation_lock": str(args.evaluation_lock.resolve()),
        "evaluation_lock_sha256": file_sha256(args.evaluation_lock),
        "routes": str(args.routes.resolve()),
        "routes_sha256": file_sha256(args.routes),
        "dataset_fingerprint": actual_dataset["sha256"],
        "total_rejected_observations": sum(
            episode.rejected_observations for episode in episodes
        ),
        "total_relocalizations": sum(episode.relocalizations for episode in episodes),
        **metrics.as_dict(),
    }
    if args.baseline_metrics:
        baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
        summary["baseline_metrics"] = {
            key: baseline[key] for key in ("SR@20", "SPL", "NE")
        }
        summary["delta_vs_baseline"] = {
            "SR@20": metrics.sr_at_20 - float(baseline["SR@20"]),
            "SPL": metrics.spl - float(baseline["SPL"]),
            "NE": metrics.ne_m - float(baseline["NE"]),
        }
        summary["baseline_metrics_sha256"] = file_sha256(args.baseline_metrics)
    (args.output_dir / "navigation_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "navigation_episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields_csv = (
            "route_id",
            "success",
            "spl",
            "navigation_error_m",
            "shortest_path_m",
            "actual_path_m",
            "steps",
            "reached_waypoints",
            "termination",
            "rejected_observations",
            "relocalizations",
            "mean_effective_confidence",
            "mean_uncertainty_m",
        )
        writer = csv.DictWriter(handle, fieldnames=fields_csv)
        writer.writeheader()
        for episode in episodes:
            row = episode.as_dict()
            row.pop("trajectory")
            writer.writerow(row)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

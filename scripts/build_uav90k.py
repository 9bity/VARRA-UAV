"""Build the UAV90K global-retrieval dataset from Bearing-UAV-90K.

The conversion is non-destructive. By default, image files are exposed through
NTFS hard links, while all task-specific labels and indices are newly generated.
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


CITY_NAMES = ("citya", "cityb", "cityc", "cityd")
TILE_SIZE = 256
BLOCK_GRID_SIZE = 15
TILE_GRID_SIZE = BLOCK_GRID_SIZE + 1
UNIT_PIXEL = TILE_SIZE // 2
SAMPLES_PER_CITY = 22_500
SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.85, "val": 0.05, "test": 0.10}


@dataclass(frozen=True)
class CitySource:
    name: str
    root: Path
    metadata: Path
    satellite_dir: Path
    uav_dir: Path
    map_image: Path
    map_json: Path
    map_data: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "manifest"),
        default="hardlink",
        help="How image data is exposed in UAV90K.",
    )
    parser.add_argument(
        "--skip-duplicate-hash-check",
        action="store_true",
        help="Skip SHA256 validation of RST duplicates (not recommended).",
    )
    return parser.parse_args()


def resolve_source_path(source_root: Path, raw_path: str) -> Path:
    normalized = raw_path.replace("\\", "/").lstrip("./")
    marker = source_root.name + "/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return source_root / Path(normalized)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if mode == "manifest":
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size:
            raise RuntimeError(f"Existing destination has a different size: {destination}")
        return

    if mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:  # pragma: no cover - argparse constrains this value
        raise ValueError(mode)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    os.replace(temporary, path)
    return count


def load_map_catalog(source_root: Path) -> list[tuple[Path, Path, dict[str, Any]]]:
    catalog: list[tuple[Path, Path, dict[str, Any]]] = []
    for json_path in sorted((source_root / "city_rsi").glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = json_path.with_name(data["image"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        catalog.append((image_path, json_path, data))
    if not catalog:
        raise RuntimeError("No city maps found under city_rsi")
    return catalog


def first_uav_sidecar(uav_dir: Path) -> dict[str, Any]:
    path = next(iter(sorted(uav_dir.glob("*_v3d.json"))), None)
    if path is None:
        raise RuntimeError(f"No UAV sidecar JSON found in {uav_dir}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_city_map(
    uav_dir: Path,
    catalog: list[tuple[Path, Path, dict[str, Any]]],
) -> tuple[Path, Path, dict[str, Any]]:
    sample = first_uav_sidecar(uav_dir)
    candidates = [
        entry
        for entry in catalog
        if entry[2].get("width_pixel") == 4096
        and entry[2].get("height_pixel") == 4096
        and math.isclose(float(entry[2].get("lngm_per_pixel", 0.0)), 0.25)
        and math.isclose(float(entry[2].get("latm_per_pixel", 0.0)), 0.25)
    ]
    if not candidates:
        raise RuntimeError("No compatible 4096x4096, 0.25 m/px city map found")

    def distance(entry: tuple[Path, Path, dict[str, Any]]) -> float:
        data = entry[2]
        return (float(data["lat"]) - float(sample["lat"])) ** 2 + (
            float(data["lng"]) - float(sample["lng"])
        ) ** 2

    return min(candidates, key=distance)


def discover_cities(source_root: Path) -> list[CitySource]:
    catalog = load_map_catalog(source_root)
    cities: list[CitySource] = []
    for name in CITY_NAMES:
        root = source_root / name
        metadata = root / "rawmetadata.csv"
        satellite_dirs = sorted(root.glob("sat_*"))
        uav_dirs = sorted(root.glob("uav_*"))
        if not metadata.is_file() or len(satellite_dirs) != 1 or len(uav_dirs) != 1:
            raise RuntimeError(f"Unexpected source layout for {name}: {root}")
        map_image, map_json, map_data = select_city_map(uav_dirs[0], catalog)
        cities.append(
            CitySource(
                name=name,
                root=root,
                metadata=metadata,
                satellite_dir=satellite_dirs[0],
                uav_dir=uav_dirs[0],
                map_image=map_image,
                map_json=map_json,
                map_data=map_data,
            )
        )
    return cities


def map_pixel_to_geo(map_data: dict[str, Any], x: float, y: float) -> tuple[float, float]:
    longitude = (
        (x - float(map_data["width_pixel"]) / 2.0) * float(map_data["lng_per_pixel"])
        + float(map_data["lng"])
    )
    latitude = (
        float(map_data["lat"])
        - (y - float(map_data["height_pixel"]) / 2.0) * float(map_data["lat_per_pixel"])
    )
    return latitude, longitude


def build_split_assignments(sample_ids: list[str]) -> dict[str, str]:
    shuffled = list(sample_ids)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    train_end = int(len(shuffled) * SPLIT_RATIOS["train"])
    val_end = train_end + int(len(shuffled) * SPLIT_RATIOS["val"])
    assignments: dict[str, str] = {}
    for sample_id in shuffled[:train_end]:
        assignments[sample_id] = "train"
    for sample_id in shuffled[train_end:val_end]:
        assignments[sample_id] = "val"
    for sample_id in shuffled[val_end:]:
        assignments[sample_id] = "test"
    return assignments


def read_city_rows(city: CitySource) -> list[dict[str, str]]:
    with city.metadata.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != SAMPLES_PER_CITY:
        raise RuntimeError(
            f"{city.name} has {len(rows)} rows; expected {SAMPLES_PER_CITY}"
        )
    return rows


def source_tile_candidates(city: CitySource) -> dict[tuple[int, int], list[Path]]:
    candidates: dict[tuple[int, int], list[Path]] = {
        (row, col): []
        for row in range(TILE_GRID_SIZE)
        for col in range(TILE_GRID_SIZE)
    }
    for block_x in range(BLOCK_GRID_SIZE):
        for block_y in range(BLOCK_GRID_SIZE):
            for dx in (0, 1):
                for dy in (0, 1):
                    source = city.satellite_dir / (
                        f"block_{block_x}_{block_y}_base_{dx}{dy}.jpg"
                    )
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    candidates[(block_y + dy, block_x + dx)].append(source)
    return candidates


def validate_tile_duplicates(
    candidates: dict[tuple[int, int], list[Path]],
    skip_hash_check: bool,
) -> None:
    if skip_hash_check:
        return
    for coordinate, paths in candidates.items():
        expected = sha256(paths[0])
        for path in paths[1:]:
            if sha256(path) != expected:
                raise RuntimeError(
                    f"RST duplicates disagree at row/col {coordinate}: {paths[0]} vs {path}"
                )


def build_maps(
    cities: list[CitySource], output_root: Path, link_mode: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for city in cities:
        map_id = f"{city.name}_map"
        image_rel = Path("maps") / city.name / "map.jpg"
        json_rel = Path("maps") / city.name / "map.json"
        materialize(city.map_image, output_root / image_rel, link_mode)
        materialize(city.map_json, output_root / json_rel, link_mode)
        data = city.map_data
        rows.append(
            {
                "map_id": map_id,
                "city": city.name,
                "image_path": image_rel.as_posix(),
                "json_path": json_rel.as_posix(),
                "width": data["width_pixel"],
                "height": data["height_pixel"],
                "center_latitude": data["lat"],
                "center_longitude": data["lng"],
                "lat_per_pixel": data["lat_per_pixel"],
                "lng_per_pixel": data["lng_per_pixel"],
                "meters_per_pixel_x": data["lngm_per_pixel"],
                "meters_per_pixel_y": data["latm_per_pixel"],
            }
        )
    return rows


def build_satellite_tiles(
    cities: list[CitySource],
    output_root: Path,
    link_mode: str,
    skip_hash_check: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tile_rows: list[dict[str, Any]] = []
    neighborhood_rows: list[dict[str, Any]] = []

    for city in cities:
        candidates = source_tile_candidates(city)
        validate_tile_duplicates(candidates, skip_hash_check)
        for row in range(TILE_GRID_SIZE):
            for col in range(TILE_GRID_SIZE):
                tile_id = f"{city.name}_r{row:02d}_c{col:02d}"
                source = candidates[(row, col)][0]
                image_rel = (
                    Path("images")
                    / "satellite"
                    / city.name
                    / f"tile_r{row:02d}_c{col:02d}.jpg"
                )
                materialize(source, output_root / image_rel, link_mode)
                x0, y0 = col * TILE_SIZE, row * TILE_SIZE
                x1, y1 = x0 + TILE_SIZE, y0 + TILE_SIZE
                latitude, longitude = map_pixel_to_geo(
                    city.map_data, x0 + TILE_SIZE / 2.0, y0 + TILE_SIZE / 2.0
                )
                tile_rows.append(
                    {
                        "tile_id": tile_id,
                        "map_id": f"{city.name}_map",
                        "city": city.name,
                        "row": row,
                        "col": col,
                        "image_path": image_rel.as_posix(),
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "center_x": x0 + TILE_SIZE / 2.0,
                        "center_y": y0 + TILE_SIZE / 2.0,
                        "center_latitude": latitude,
                        "center_longitude": longitude,
                    }
                )

                neighbors: dict[str, Any] = {
                    "center_tile_id": tile_id,
                    "city": city.name,
                    "row": row,
                    "col": col,
                }
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        neighbor_row, neighbor_col = row + dy, col + dx
                        key = f"n{dy + 1}{dx + 1}"
                        if 0 <= neighbor_row < TILE_GRID_SIZE and 0 <= neighbor_col < TILE_GRID_SIZE:
                            neighbors[key] = (
                                f"{city.name}_r{neighbor_row:02d}_c{neighbor_col:02d}"
                            )
                        else:
                            neighbors[key] = ""
                neighborhood_rows.append(neighbors)
    return tile_rows, neighborhood_rows


def sidecar_for_uav(uav_path: Path) -> dict[str, Any]:
    sidecar = uav_path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    return json.loads(sidecar.read_text(encoding="utf-8"))


def prepare_sample_ids(
    city_rows: dict[str, list[dict[str, str]]], source_root: Path
) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {}
    seen: set[str] = set()
    for city_name, rows in city_rows.items():
        city_ids: list[str] = []
        for row in rows:
            source = resolve_source_path(source_root, row["target_patch_3d"])
            sample_id = f"{city_name}_{source.stem.removesuffix('_v3d')}"
            if sample_id in seen:
                raise RuntimeError(f"Duplicate sample ID: {sample_id}")
            seen.add(sample_id)
            city_ids.append(sample_id)
        ids[city_name] = city_ids
    return ids


def generate_samples(
    cities: list[CitySource],
    city_rows: dict[str, list[dict[str, str]]],
    city_sample_ids: dict[str, list[str]],
    split_assignments: dict[str, str],
    source_root: Path,
    output_root: Path,
    link_mode: str,
) -> Iterator[dict[str, Any]]:
    for city in cities:
        for raw, sample_id in zip(city_rows[city.name], city_sample_ids[city.name]):
            source_uav = resolve_source_path(source_root, raw["target_patch_3d"])
            uav_rel = Path("images") / "uav" / city.name / source_uav.name
            materialize(source_uav, output_root / uav_rel, link_mode)

            block_x = int(raw["block_x"])
            block_y = int(raw["block_y"])
            x_norm = float(raw["x_norm"])
            y_norm = float(raw["y_norm"])
            global_x = (block_x + 1) * TILE_SIZE + x_norm * UNIT_PIXEL
            global_y = (block_y + 1) * TILE_SIZE + y_norm * UNIT_PIXEL
            if not (0.0 <= global_x < 4096.0 and 0.0 <= global_y < 4096.0):
                raise RuntimeError(
                    f"Global coordinate outside map for {sample_id}: {global_x}, {global_y}"
                )

            tile_col = min(int(global_x // TILE_SIZE), TILE_GRID_SIZE - 1)
            tile_row = min(int(global_y // TILE_SIZE), TILE_GRID_SIZE - 1)
            tile_offset_x = (global_x - tile_col * TILE_SIZE) / TILE_SIZE
            tile_offset_y = (global_y - tile_row * TILE_SIZE) / TILE_SIZE
            theta_raw = float(raw["theta"])
            heading_deg = theta_raw % 360.0
            latitude, longitude = map_pixel_to_geo(city.map_data, global_x, global_y)
            sidecar = sidecar_for_uav(source_uav)

            if abs(latitude - float(sidecar["lat"])) > 1e-10 or abs(
                longitude - float(sidecar["lng"])
            ) > 1e-10:
                raise RuntimeError(f"Geographic coordinate mismatch for {sample_id}")

            yield {
                "sample_id": sample_id,
                "split": split_assignments[sample_id],
                "city": city.name,
                "map_id": f"{city.name}_map",
                "uav_path": uav_rel.as_posix(),
                "source_uav_path": raw["target_patch_3d"].replace("\\", "/"),
                "block_x": block_x,
                "block_y": block_y,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "x_uccs": float(raw["x_uccs"]),
                "y_uccs": float(raw["y_uccs"]),
                "global_x": global_x,
                "global_y": global_y,
                "latitude": latitude,
                "longitude": longitude,
                "gt_tile_id": f"{city.name}_r{tile_row:02d}_c{tile_col:02d}",
                "gt_tile_row": tile_row,
                "gt_tile_col": tile_col,
                "tile_offset_x": tile_offset_x,
                "tile_offset_y": tile_offset_y,
                "theta_raw": theta_raw,
                "heading_deg": heading_deg,
                "heading_cos": float(raw["x_cosa"]),
                "heading_sin": float(raw["y_sina"]),
                "altitude_m": sidecar.get("alt", ""),
                "ground_width_m": sidecar.get("width_meter", ""),
                "ground_height_m": sidecar.get("height_meter", ""),
                "capture_time": sidecar.get("time", ""),
            }


def write_splits(
    split_dir: Path,
    city_sample_ids: dict[str, list[str]],
    assignments: dict[str, str],
) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    ordered_ids = [sample_id for city in CITY_NAMES for sample_id in city_sample_ids[city]]
    for split in ("train", "val", "test"):
        selected = [sample_id for sample_id in ordered_ids if assignments[sample_id] == split]
        temporary = split_dir / f"{split}.txt.tmp"
        temporary.write_text("".join(f"{sample_id}\n" for sample_id in selected), encoding="utf-8")
        os.replace(temporary, split_dir / f"{split}.txt")
        counts[split] = len(selected)
    return counts


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if source_root == output_root:
        raise RuntimeError("Source and output must be different directories")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root = output_root / "metadata"
    (output_root / "features" / "satellite").mkdir(parents=True, exist_ok=True)
    (output_root / "features" / "index").mkdir(parents=True, exist_ok=True)

    cities = discover_cities(source_root)
    city_rows = {city.name: read_city_rows(city) for city in cities}
    city_sample_ids = prepare_sample_ids(city_rows, source_root)
    assignments: dict[str, str] = {}
    for city_name in CITY_NAMES:
        assignments.update(build_split_assignments(city_sample_ids[city_name]))

    print("[1/5] Materializing city maps")
    map_rows = build_maps(cities, output_root, args.link_mode)
    map_fields = list(map_rows[0])
    write_csv(metadata_root / "maps.csv", map_fields, map_rows)

    print("[2/5] Deduplicating and validating satellite tiles")
    tile_rows, neighborhood_rows = build_satellite_tiles(
        cities,
        output_root,
        args.link_mode,
        args.skip_duplicate_hash_check,
    )
    write_csv(metadata_root / "satellite_tiles.csv", list(tile_rows[0]), tile_rows)
    write_csv(
        metadata_root / "neighborhoods.csv",
        list(neighborhood_rows[0]),
        neighborhood_rows,
    )

    print("[3/5] Materializing 90K UAV queries and generating labels")
    sample_fields = [
        "sample_id",
        "split",
        "city",
        "map_id",
        "uav_path",
        "source_uav_path",
        "block_x",
        "block_y",
        "x_norm",
        "y_norm",
        "x_uccs",
        "y_uccs",
        "global_x",
        "global_y",
        "latitude",
        "longitude",
        "gt_tile_id",
        "gt_tile_row",
        "gt_tile_col",
        "tile_offset_x",
        "tile_offset_y",
        "theta_raw",
        "heading_deg",
        "heading_cos",
        "heading_sin",
        "altitude_m",
        "ground_width_m",
        "ground_height_m",
        "capture_time",
    ]
    sample_count = write_csv(
        metadata_root / "samples.csv",
        sample_fields,
        generate_samples(
            cities,
            city_rows,
            city_sample_ids,
            assignments,
            source_root,
            output_root,
            args.link_mode,
        ),
    )

    print("[4/5] Writing deterministic splits")
    split_counts = write_splits(metadata_root / "splits", city_sample_ids, assignments)

    print("[5/5] Writing dataset manifest")
    info = {
        "name": "UAV90K",
        "source_dataset": "Bearing-UAV-90K",
        "source_root": str(source_root),
        "link_mode": args.link_mode,
        "query_modality": "single real UAV-view patch",
        "reference_modality": "offline satellite tile database",
        "tile_size": TILE_SIZE,
        "map_size": [4096, 4096],
        "cities": list(CITY_NAMES),
        "counts": {
            "maps": len(map_rows),
            "satellite_tiles": len(tile_rows),
            "neighborhoods": len(neighborhood_rows),
            "samples": sample_count,
            **split_counts,
        },
        "split": {
            "seed": SPLIT_SEED,
            "method": "per-city Python random.Random shuffle",
            "ratios": SPLIT_RATIOS,
        },
        "coordinate_system": {
            "global_origin": "top-left of 4096x4096 city map",
            "global_x": "right-positive pixels",
            "global_y": "down-positive pixels",
            "heading": "counter-clockwise from image x-axis, normalized to [0,360)",
        },
    }
    temporary_info = output_root / "dataset_info.json.tmp"
    temporary_info.write_text(json.dumps(info, indent=2), encoding="utf-8")
    os.replace(temporary_info, output_root / "dataset_info.json")

    print(json.dumps(info["counts"], indent=2))
    print(f"UAV90K is ready at: {output_root}")


if __name__ == "__main__":
    main()


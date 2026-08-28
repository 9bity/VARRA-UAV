"""Validate a generated UAV90K dataset without loading image pixels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_CITIES = {"citya", "cityb", "cityc", "cityd"}
TILE_SIZE = 256
GRID_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_maps(root: Path) -> set[str]:
    rows = read_csv(root / "metadata" / "maps.csv")
    require(len(rows) == 4, f"Expected 4 maps, found {len(rows)}")
    require({row["city"] for row in rows} == EXPECTED_CITIES, "Map cities mismatch")
    map_ids: set[str] = set()
    for row in rows:
        require(row["map_id"] not in map_ids, f"Duplicate map ID: {row['map_id']}")
        map_ids.add(row["map_id"])
        require(int(row["width"]) == 4096 and int(row["height"]) == 4096, "Map size mismatch")
        require((root / row["image_path"]).is_file(), f"Missing map: {row['image_path']}")
        require((root / row["json_path"]).is_file(), f"Missing map JSON: {row['json_path']}")
    return map_ids


def validate_tiles(root: Path, map_ids: set[str]) -> dict[str, dict[str, str]]:
    rows = read_csv(root / "metadata" / "satellite_tiles.csv")
    require(len(rows) == 4 * GRID_SIZE * GRID_SIZE, f"Expected 1024 tiles, found {len(rows)}")
    tile_by_id: dict[str, dict[str, str]] = {}
    coordinates: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in rows:
        tile_id = row["tile_id"]
        require(tile_id not in tile_by_id, f"Duplicate tile ID: {tile_id}")
        require(row["map_id"] in map_ids, f"Unknown map for {tile_id}")
        tile_by_id[tile_id] = row
        tile_row, tile_col = int(row["row"]), int(row["col"])
        require(0 <= tile_row < GRID_SIZE and 0 <= tile_col < GRID_SIZE, f"Bad tile coordinate: {tile_id}")
        coordinates[row["city"]].add((tile_row, tile_col))
        require(int(row["x0"]) == tile_col * TILE_SIZE, f"Bad x0 for {tile_id}")
        require(int(row["y0"]) == tile_row * TILE_SIZE, f"Bad y0 for {tile_id}")
        require((root / row["image_path"]).is_file(), f"Missing tile image: {tile_id}")
    expected_grid = {(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE)}
    for city in EXPECTED_CITIES:
        require(coordinates[city] == expected_grid, f"Incomplete tile grid for {city}")
    return tile_by_id


def validate_neighborhoods(root: Path, tile_by_id: dict[str, dict[str, str]]) -> None:
    rows = read_csv(root / "metadata" / "neighborhoods.csv")
    require(len(rows) == len(tile_by_id), "Neighborhood count mismatch")
    centers: set[str] = set()
    for item in rows:
        center = item["center_tile_id"]
        require(center not in centers, f"Duplicate neighborhood: {center}")
        centers.add(center)
        city, row, col = item["city"], int(item["row"]), int(item["col"])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                actual = item[f"n{dy + 1}{dx + 1}"]
                nr, nc = row + dy, col + dx
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    expected = f"{city}_r{nr:02d}_c{nc:02d}"
                    require(actual == expected, f"Bad neighbor {center}, offset {dy, dx}")
                    require(actual in tile_by_id, f"Unknown neighbor tile: {actual}")
                else:
                    require(actual == "", f"Out-of-map neighbor is not blank for {center}")
    require(centers == set(tile_by_id), "Neighborhood centers do not cover all tiles")


def validate_samples(root: Path, map_ids: set[str], tile_by_id: dict[str, dict[str, str]]) -> tuple[set[str], Counter[str]]:
    path = root / "metadata" / "samples.csv"
    sample_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            require(sample_id not in sample_ids, f"Duplicate sample ID: {sample_id}")
            sample_ids.add(sample_id)
            split_counts[row["split"]] += 1
            city_counts[row["city"]] += 1
            require(row["map_id"] in map_ids, f"Unknown map for sample {sample_id}")
            require((root / row["uav_path"]).is_file(), f"Missing UAV image: {sample_id}")

            global_x, global_y = float(row["global_x"]), float(row["global_y"])
            require(0.0 <= global_x < 4096.0 and 0.0 <= global_y < 4096.0, f"Bad global coordinate: {sample_id}")
            expected_col = min(int(global_x // TILE_SIZE), GRID_SIZE - 1)
            expected_row = min(int(global_y // TILE_SIZE), GRID_SIZE - 1)
            expected_tile = f"{row['city']}_r{expected_row:02d}_c{expected_col:02d}"
            require(row["gt_tile_id"] == expected_tile, f"GT tile mismatch: {sample_id}")
            require(expected_tile in tile_by_id, f"Missing GT tile: {sample_id}")
            offset_x, offset_y = float(row["tile_offset_x"]), float(row["tile_offset_y"])
            require(0.0 <= offset_x <= 1.0 and 0.0 <= offset_y <= 1.0, f"Bad tile offset: {sample_id}")

            heading_cos, heading_sin = float(row["heading_cos"]), float(row["heading_sin"])
            require(abs(math.hypot(heading_cos, heading_sin) - 1.0) < 1e-6, f"Non-unit heading: {sample_id}")
            require(0.0 <= float(row["heading_deg"]) < 360.0, f"Bad heading range: {sample_id}")

    require(len(sample_ids) == 90_000, f"Expected 90000 samples, found {len(sample_ids)}")
    require(city_counts == Counter({city: 22_500 for city in EXPECTED_CITIES}), f"City counts mismatch: {city_counts}")
    require(split_counts == Counter({"train": 76_500, "val": 4_500, "test": 9_000}), f"Split counts mismatch: {split_counts}")
    return sample_ids, split_counts


def validate_split_files(root: Path, sample_ids: set[str], split_counts: Counter[str]) -> None:
    combined: set[str] = set()
    for split, expected_count in split_counts.items():
        path = root / "metadata" / "splits" / f"{split}.txt"
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(len(ids) == expected_count, f"Split file count mismatch: {split}")
        require(len(set(ids)) == len(ids), f"Duplicate ID in split: {split}")
        require(not (combined & set(ids)), f"Overlapping split: {split}")
        combined.update(ids)
    require(combined == sample_ids, "Split files do not partition all samples")


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    require(info["name"] == "UAV90K", "Unexpected dataset name")
    map_ids = validate_maps(root)
    tile_by_id = validate_tiles(root, map_ids)
    validate_neighborhoods(root, tile_by_id)
    sample_ids, split_counts = validate_samples(root, map_ids, tile_by_id)
    validate_split_files(root, sample_ids, split_counts)
    print(
        json.dumps(
            {
                "status": "valid",
                "maps": len(map_ids),
                "satellite_tiles": len(tile_by_id),
                "samples": len(sample_ids),
                "splits": dict(split_counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


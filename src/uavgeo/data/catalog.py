"""Typed access to UAV90K metadata without importing PyTorch."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class SatelliteTile:
    tile_id: str
    map_id: str
    city: str
    row: int
    col: int
    image_path: Path
    x0: int
    y0: int
    x1: int
    y1: int
    center_x: float
    center_y: float


@dataclass(frozen=True)
class QueryRecord:
    sample_id: str
    split: str
    city: str
    map_id: str
    uav_path: Path
    global_x: float
    global_y: float
    latitude: float
    longitude: float
    gt_tile_id: str
    gt_tile_row: int
    gt_tile_col: int
    heading_deg: float
    heading_cos: float
    heading_sin: float


@dataclass(frozen=True)
class MapRecord:
    map_id: str
    city: str
    width: int
    height: int
    center_latitude: float
    center_longitude: float
    lat_per_pixel: float
    lng_per_pixel: float
    meters_per_pixel_x: float
    meters_per_pixel_y: float

    def pixel_to_geo(self, x: float, y: float) -> tuple[float, float]:
        longitude = (x - self.width / 2.0) * self.lng_per_pixel + self.center_longitude
        latitude = self.center_latitude - (y - self.height / 2.0) * self.lat_per_pixel
        return latitude, longitude


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class UAV90KCatalog:
    """Load query, satellite-tile, and 3x3-neighborhood metadata."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root).resolve()
        metadata = self.root / "metadata"
        if not metadata.is_dir():
            raise FileNotFoundError(metadata)

        self.maps = self._load_maps(metadata / "maps.csv")
        self.maps_by_city = {item.city: item for item in self.maps.values()}
        self.tiles = self._load_tiles(metadata / "satellite_tiles.csv")
        self.neighborhoods = self._load_neighborhoods(metadata / "neighborhoods.csv")
        self.queries = self._load_queries(metadata / "samples.csv")
        self._queries_by_split: dict[str, list[QueryRecord]] = {}
        for query in self.queries:
            self._queries_by_split.setdefault(query.split, []).append(query)

    @staticmethod
    def _load_maps(path: Path) -> dict[str, MapRecord]:
        maps: dict[str, MapRecord] = {}
        for row in _read_csv(path):
            item = MapRecord(
                map_id=row["map_id"],
                city=row["city"],
                width=int(row["width"]),
                height=int(row["height"]),
                center_latitude=float(row["center_latitude"]),
                center_longitude=float(row["center_longitude"]),
                lat_per_pixel=float(row["lat_per_pixel"]),
                lng_per_pixel=float(row["lng_per_pixel"]),
                meters_per_pixel_x=float(row["meters_per_pixel_x"]),
                meters_per_pixel_y=float(row["meters_per_pixel_y"]),
            )
            if item.map_id in maps:
                raise ValueError(f"Duplicate map: {item.map_id}")
            maps[item.map_id] = item
        return maps

    def _load_tiles(self, path: Path) -> dict[str, SatelliteTile]:
        tiles: dict[str, SatelliteTile] = {}
        for row in _read_csv(path):
            tile = SatelliteTile(
                tile_id=row["tile_id"],
                map_id=row["map_id"],
                city=row["city"],
                row=int(row["row"]),
                col=int(row["col"]),
                image_path=self.root / row["image_path"],
                x0=int(row["x0"]),
                y0=int(row["y0"]),
                x1=int(row["x1"]),
                y1=int(row["y1"]),
                center_x=float(row["center_x"]),
                center_y=float(row["center_y"]),
            )
            if tile.tile_id in tiles:
                raise ValueError(f"Duplicate satellite tile: {tile.tile_id}")
            tiles[tile.tile_id] = tile
        return tiles

    @staticmethod
    def _load_neighborhoods(path: Path) -> dict[str, tuple[Optional[str], ...]]:
        neighborhoods: dict[str, tuple[Optional[str], ...]] = {}
        for row in _read_csv(path):
            center = row["center_tile_id"]
            neighborhoods[center] = tuple(
                row[f"n{grid_row}{grid_col}"] or None
                for grid_row in range(3)
                for grid_col in range(3)
            )
        return neighborhoods

    def _load_queries(self, path: Path) -> list[QueryRecord]:
        records: list[QueryRecord] = []
        for row in _read_csv(path):
            records.append(
                QueryRecord(
                    sample_id=row["sample_id"],
                    split=row["split"],
                    city=row["city"],
                    map_id=row["map_id"],
                    uav_path=self.root / row["uav_path"],
                    global_x=float(row["global_x"]),
                    global_y=float(row["global_y"]),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    gt_tile_id=row["gt_tile_id"],
                    gt_tile_row=int(row["gt_tile_row"]),
                    gt_tile_col=int(row["gt_tile_col"]),
                    heading_deg=float(row["heading_deg"]),
                    heading_cos=float(row["heading_cos"]),
                    heading_sin=float(row["heading_sin"]),
                )
            )
        return records

    def queries_for_split(self, split: str) -> list[QueryRecord]:
        try:
            return self._queries_by_split[split]
        except KeyError as error:
            raise ValueError(f"Unknown split: {split}") from error

    def tile_id(self, city: str, row: int, col: int) -> Optional[str]:
        tile_id = f"{city}_r{row:02d}_c{col:02d}"
        return tile_id if tile_id in self.tiles else None

    def neighborhood(self, center_tile_id: str) -> tuple[Optional[SatelliteTile], ...]:
        try:
            ids = self.neighborhoods[center_tile_id]
        except KeyError as error:
            raise ValueError(f"Unknown center tile: {center_tile_id}") from error
        return tuple(self.tiles[tile_id] if tile_id is not None else None for tile_id in ids)

    def positive_candidate_centers(self, query: QueryRecord) -> list[str]:
        """Return centers whose 3x3 area contains the query's GT tile."""

        centers: list[str] = []
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                center = self.tile_id(
                    query.city,
                    query.gt_tile_row + delta_row,
                    query.gt_tile_col + delta_col,
                )
                if center is not None:
                    centers.append(center)
        return centers

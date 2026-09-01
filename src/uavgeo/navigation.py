"""Bearing-Naver-compatible closed-loop navigation and metrics.

The localizer sees only the observation rendered at the real UAV state.  Its
predicted absolute position is used to point the vehicle at the next waypoint;
the simulator then advances the real state by one fixed-length step.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence, Union

from PIL import Image

from .data.catalog import MapRecord


@dataclass(frozen=True)
class NavigationRoute:
    route_id: str
    bearing_map_id: str
    city: str
    shortest_path_m: float
    waypoints_xy: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class NavigationPose:
    city: str
    x: float
    y: float
    latitude: float
    longitude: float
    heading_deg: float


@dataclass(frozen=True)
class NavigationStep:
    step: int
    target_waypoint: int
    real_x: float
    real_y: float
    predicted_city: str
    predicted_x: float
    predicted_y: float
    predicted_latitude: float
    predicted_longitude: float
    predicted_heading_deg: float
    commanded_heading_deg: float
    waypoint_error_real_m: float
    waypoint_error_predicted_m: float


@dataclass(frozen=True)
class NavigationEpisode:
    route_id: str
    success: bool
    spl: float
    navigation_error_m: float
    shortest_path_m: float
    actual_path_m: float
    steps: int
    reached_waypoints: int
    termination: str
    trajectory: tuple[NavigationStep, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationMetrics:
    sr_at_20: float
    spl: float
    ne_m: float
    episodes: int
    successes: int

    def as_dict(self) -> dict[str, Union[int, float]]:
        return {
            "SR@20": self.sr_at_20,
            "SPL": self.spl,
            "NE": self.ne_m,
            "episodes": self.episodes,
            "successes": self.successes,
        }


def load_navigation_routes(path: Union[str, Path]) -> tuple[NavigationRoute, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    routes: list[NavigationRoute] = []
    for item in payload:
        flat = [float(value) for value in item["waypoints_xy_flat"]]
        if len(flat) < 4 or len(flat) % 2:
            raise ValueError(f"Invalid waypoint list for {item['route_id']}")
        waypoints = tuple((flat[index], flat[index + 1]) for index in range(0, len(flat), 2))
        routes.append(
            NavigationRoute(
                route_id=str(item["route_id"]),
                bearing_map_id=str(item["bearing_map_id"]),
                city=str(item["city"]),
                shortest_path_m=float(item["shortest_path_m"]),
                waypoints_xy=waypoints,
            )
        )
    if len({route.route_id for route in routes}) != len(routes):
        raise ValueError("Navigation route IDs must be unique")
    return tuple(routes)


def pixel_distance_m(
    first: tuple[float, float], second: tuple[float, float], map_record: MapRecord
) -> float:
    dx = (second[0] - first[0]) * map_record.meters_per_pixel_x
    dy = (second[1] - first[1]) * map_record.meters_per_pixel_y
    return math.hypot(dx, dy)


def haversine_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Distance between ``(latitude, longitude)`` pairs in meters."""

    radius = 6_371_000.0
    lat1, lon1 = map(math.radians, first)
    lat2, lon2 = map(math.radians, second)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        dlon / 2.0
    ) ** 2
    return radius * 2.0 * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))


def bearing_deg(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Local azimuth: east is 0 degrees and north is 90 degrees."""

    mean_lat = math.radians((first[0] + second[0]) / 2.0)
    east = (second[1] - first[1]) * math.cos(mean_lat)
    north = second[0] - first[0]
    return math.degrees(math.atan2(north, east)) % 360.0


def move_pixel(
    point: tuple[float, float], heading_deg: float, distance_m: float, map_record: MapRecord
) -> tuple[float, float]:
    angle = math.radians(heading_deg)
    east_m = math.cos(angle) * distance_m
    north_m = math.sin(angle) * distance_m
    return (
        point[0] + east_m / map_record.meters_per_pixel_x,
        point[1] - north_m / map_record.meters_per_pixel_y,
    )


def crop_rotated_observation(
    map_image: Image.Image,
    center_xy: tuple[float, float],
    heading_deg: float,
    size: int = 256,
) -> Image.Image:
    """Reproduce Bearing-Naver's rotated 2D satellite-view observation crop."""

    if size <= 0:
        raise ValueError("Observation size must be positive")
    # A square of ceil(size*sqrt(2)) retains a full center crop after rotation.
    outer = int(math.ceil(size * math.sqrt(2.0))) + 4
    x, y = center_xy
    left = int(round(x - outer / 2.0))
    top = int(round(y - outer / 2.0))
    patch = map_image.crop((left, top, left + outer, top + outer))
    rotated = patch.rotate(-heading_deg, resample=Image.Resampling.BILINEAR)
    offset = (outer - size) // 2
    return rotated.crop((offset, offset, offset + size, offset + size)).convert("RGB")


def compute_navigation_metrics(
    episodes: Sequence[NavigationEpisode], success_radius_m: float = 20.0
) -> NavigationMetrics:
    if not episodes:
        raise ValueError("At least one navigation episode is required")
    # Recompute success from final navigation error so an episode cannot carry
    # a contradictory success flag into the paper metrics.
    successes = sum(episode.navigation_error_m < success_radius_m for episode in episodes)
    spl_values = [
        (episode.shortest_path_m / max(episode.shortest_path_m, episode.actual_path_m))
        if episode.navigation_error_m < success_radius_m
        else 0.0
        for episode in episodes
    ]
    count = len(episodes)
    return NavigationMetrics(
        sr_at_20=100.0 * successes / count,
        spl=100.0 * sum(spl_values) / count,
        ne_m=sum(episode.navigation_error_m for episode in episodes) / count,
        episodes=count,
        successes=successes,
    )


PredictionFunction = Callable[[Image.Image], NavigationPose]


def run_navigation_episode(
    route: NavigationRoute,
    map_record: MapRecord,
    map_image: Image.Image,
    predict: PredictionFunction,
    *,
    step_m: float = 25.0,
    arrival_radius_m: float = 20.0,
    max_steps: int = 384,
    observation_size: int = 256,
) -> NavigationEpisode:
    """Run one deterministic Bearing-Naver closed-loop route."""

    if step_m <= 0 or arrival_radius_m <= 0 or max_steps <= 0:
        raise ValueError("step_m, arrival_radius_m, and max_steps must be positive")
    current = route.waypoints_xy[0]
    heading = 90.0
    waypoint_index = 1
    actual_path_m = 0.0
    records: list[NavigationStep] = []
    termination = "max_steps"

    for step in range(max_steps):
        target = route.waypoints_xy[waypoint_index]
        target_lat, target_lon = map_record.pixel_to_geo(*target)
        observation = crop_rotated_observation(
            map_image, current, heading, size=observation_size
        )
        prediction = predict(observation)
        real_error = pixel_distance_m(current, target, map_record)
        predicted_error = haversine_m(
            (prediction.latitude, prediction.longitude), (target_lat, target_lon)
        )
        heading = bearing_deg(
            (prediction.latitude, prediction.longitude), (target_lat, target_lon)
        )
        records.append(
            NavigationStep(
                step=step,
                target_waypoint=waypoint_index,
                real_x=current[0],
                real_y=current[1],
                predicted_city=prediction.city,
                predicted_x=prediction.x,
                predicted_y=prediction.y,
                predicted_latitude=prediction.latitude,
                predicted_longitude=prediction.longitude,
                predicted_heading_deg=prediction.heading_deg,
                commanded_heading_deg=heading,
                waypoint_error_real_m=real_error,
                waypoint_error_predicted_m=predicted_error,
            )
        )

        # Match the released Bearing-Naver code: either the real or predicted
        # pose may advance the local waypoint, but final success uses real pose.
        if min(real_error, predicted_error) < arrival_radius_m:
            if waypoint_index == len(route.waypoints_xy) - 1:
                termination = "endpoint_reached"
                break
            waypoint_index += 1
            continue

        current = move_pixel(current, heading, step_m, map_record)
        actual_path_m += step_m
        if not (0.0 <= current[0] < map_record.width and 0.0 <= current[1] < map_record.height):
            termination = "out_of_map"
            break

    final_error = pixel_distance_m(current, route.waypoints_xy[-1], map_record)
    success = final_error < arrival_radius_m
    spl = (
        100.0 * route.shortest_path_m / max(route.shortest_path_m, actual_path_m)
        if success
        else 0.0
    )
    return NavigationEpisode(
        route_id=route.route_id,
        success=success,
        spl=spl,
        navigation_error_m=final_error,
        shortest_path_m=route.shortest_path_m,
        actual_path_m=actual_path_m,
        steps=len(records),
        reached_waypoints=waypoint_index,
        termination=termination,
        trajectory=tuple(records),
    )

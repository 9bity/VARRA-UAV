"""Confidence- and motion-consistent temporal navigation.

This module is deliberately separate from :mod:`uavgeo.navigation`.  It treats
the trained single-stage localizer as a frozen black box and changes only the
closed-loop navigation state estimator and controller input.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

from PIL import Image

from .data.catalog import MapRecord
from .navigation import (
    NavigationPose,
    NavigationRoute,
    bearing_deg,
    crop_rotated_observation,
    haversine_m,
    move_pixel,
    pixel_distance_m,
)


@dataclass(frozen=True)
class CMCTConfig:
    position_sigma_m: float = 45.0
    heading_sigma_deg: float = 45.0
    min_gain: float = 0.15
    max_gain: float = 0.85
    reject_threshold: float = 0.03
    process_noise_m: float = 3.0
    motion_noise_ratio: float = 0.05
    initial_uncertainty_m: float = 5.0
    max_uncertainty_m: float = 150.0
    reset_after_rejections: int = 3
    reset_consistency_m: float = 30.0
    reset_min_confidence: float = 0.65

    def validate(self) -> None:
        if self.position_sigma_m <= 0 or self.heading_sigma_deg <= 0:
            raise ValueError("CMCT consistency sigmas must be positive")
        if not 0.0 <= self.min_gain <= self.max_gain <= 1.0:
            raise ValueError("CMCT gains must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.reject_threshold <= 1.0:
            raise ValueError("reject_threshold must be in [0,1]")
        if self.process_noise_m < 0 or self.motion_noise_ratio < 0:
            raise ValueError("CMCT motion noise values must be non-negative")
        if self.initial_uncertainty_m < 0 or self.max_uncertainty_m <= 0:
            raise ValueError("CMCT uncertainty values are invalid")
        if self.reset_after_rejections <= 0 or self.reset_consistency_m <= 0:
            raise ValueError("CMCT relocalization values must be positive")
        if not 0.0 <= self.reset_min_confidence <= 1.0:
            raise ValueError("reset_min_confidence must be in [0,1]")


@dataclass(frozen=True)
class CMCTObservation:
    pose: NavigationPose
    confidence: float


@dataclass(frozen=True)
class CMCTState:
    city: str
    x: float
    y: float
    latitude: float
    longitude: float
    heading_deg: float
    uncertainty_m: float


@dataclass(frozen=True)
class CMCTUpdate:
    state: CMCTState
    prior_x: float
    prior_y: float
    position_innovation_m: float
    heading_innovation_deg: float
    motion_consistency: float
    effective_confidence: float
    update_gain: float
    observation_rejected: bool
    relocalized: bool
    rejection_streak: int


@dataclass(frozen=True)
class CMCTNavigationStep:
    step: int
    target_waypoint: int
    real_x: float
    real_y: float
    raw_predicted_city: str
    raw_predicted_x: float
    raw_predicted_y: float
    raw_predicted_heading_deg: float
    visual_confidence: float
    filtered_x: float
    filtered_y: float
    filtered_latitude: float
    filtered_longitude: float
    filtered_heading_deg: float
    uncertainty_m: float
    position_innovation_m: float
    heading_innovation_deg: float
    motion_consistency: float
    effective_confidence: float
    update_gain: float
    observation_rejected: bool
    relocalized: bool
    commanded_heading_deg: float
    waypoint_error_real_m: float
    waypoint_error_raw_m: float
    waypoint_error_filtered_m: float


@dataclass(frozen=True)
class CMCTNavigationEpisode:
    route_id: str
    success: bool
    spl: float
    navigation_error_m: float
    shortest_path_m: float
    actual_path_m: float
    steps: int
    reached_waypoints: int
    termination: str
    rejected_observations: int
    relocalizations: int
    mean_effective_confidence: float
    mean_uncertainty_m: float
    trajectory: tuple[CMCTNavigationStep, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def circular_distance_deg(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def circular_fusion_deg(prior: float, observation: float, gain: float) -> float:
    prior_rad = math.radians(prior)
    observation_rad = math.radians(observation)
    x = (1.0 - gain) * math.cos(prior_rad) + gain * math.cos(observation_rad)
    y = (1.0 - gain) * math.sin(prior_rad) + gain * math.sin(observation_rad)
    if abs(x) + abs(y) < 1e-12:
        return prior % 360.0
    return math.degrees(math.atan2(y, x)) % 360.0


class CMCTFilter:
    """Action-aware robust fusion of consecutive frozen-localizer outputs."""

    def __init__(self, map_record: MapRecord, config: CMCTConfig) -> None:
        config.validate()
        self.map_record = map_record
        self.config = config
        self.state: CMCTState | None = None
        self.rejection_streak = 0
        self._pending_pose: NavigationPose | None = None
        self._pending_count = 0

    def initialize(self, x: float, y: float, heading_deg: float = 90.0) -> CMCTState:
        latitude, longitude = self.map_record.pixel_to_geo(x, y)
        self.state = CMCTState(
            city=self.map_record.city,
            x=float(x),
            y=float(y),
            latitude=latitude,
            longitude=longitude,
            heading_deg=heading_deg % 360.0,
            uncertainty_m=self.config.initial_uncertainty_m,
        )
        self.rejection_streak = 0
        self._pending_pose = None
        self._pending_count = 0
        return self.state

    def _update_pending(self, observation: CMCTObservation) -> bool:
        pose = observation.pose
        if observation.confidence < self.config.reset_min_confidence:
            self._pending_pose = None
            self._pending_count = 0
            return False
        if self._pending_pose is None or self._pending_pose.city != pose.city:
            self._pending_pose = pose
            self._pending_count = 1
        else:
            separation = haversine_m(
                (self._pending_pose.latitude, self._pending_pose.longitude),
                (pose.latitude, pose.longitude),
            )
            if separation <= self.config.reset_consistency_m:
                self._pending_count += 1
                self._pending_pose = pose
            else:
                self._pending_pose = pose
                self._pending_count = 1
        return self._pending_count >= self.config.reset_after_rejections

    def update(
        self,
        observation: CMCTObservation,
        *,
        executed_heading_deg: float,
        executed_distance_m: float,
    ) -> CMCTUpdate:
        if self.state is None:
            raise RuntimeError("CMCTFilter must be initialized before update")
        confidence = min(max(float(observation.confidence), 0.0), 1.0)
        prior_x, prior_y = move_pixel(
            (self.state.x, self.state.y),
            executed_heading_deg,
            executed_distance_m,
            self.map_record,
        )
        prior_heading = executed_heading_deg % 360.0
        prior_uncertainty = math.sqrt(
            self.state.uncertainty_m**2
            + self.config.process_noise_m**2
            + (self.config.motion_noise_ratio * executed_distance_m) ** 2
        )
        pose = observation.pose
        same_city = pose.city == self.map_record.city
        if same_city:
            position_innovation = pixel_distance_m(
                (prior_x, prior_y), (pose.x, pose.y), self.map_record
            )
            heading_innovation = circular_distance_deg(pose.heading_deg, prior_heading)
            exponent = -0.5 * (
                (position_innovation / self.config.position_sigma_m) ** 2
                + (heading_innovation / self.config.heading_sigma_deg) ** 2
            )
            motion_consistency = math.exp(max(exponent, -80.0))
        else:
            position_innovation = math.inf
            heading_innovation = 180.0
            motion_consistency = 0.0
        effective_confidence = confidence * motion_consistency
        rejected = effective_confidence < self.config.reject_threshold
        relocalized = False

        if rejected:
            self.rejection_streak += 1
            can_reset = self._update_pending(observation)
            if can_reset and same_city:
                gain = 1.0
                relocalized = True
                rejected = False
                self.rejection_streak = 0
                self._pending_pose = None
                self._pending_count = 0
            else:
                gain = 0.0
        else:
            self.rejection_streak = 0
            self._pending_pose = None
            self._pending_count = 0
            gain = self.config.min_gain + (
                self.config.max_gain - self.config.min_gain
            ) * effective_confidence

        if gain > 0.0 and same_city:
            filtered_x = (1.0 - gain) * prior_x + gain * pose.x
            filtered_y = (1.0 - gain) * prior_y + gain * pose.y
            filtered_heading = circular_fusion_deg(prior_heading, pose.heading_deg, gain)
            bounded_innovation = min(position_innovation, self.config.max_uncertainty_m)
            uncertainty = math.sqrt(
                ((1.0 - gain) * prior_uncertainty) ** 2
                + (gain * bounded_innovation) ** 2
            )
        else:
            filtered_x, filtered_y = prior_x, prior_y
            filtered_heading = prior_heading
            uncertainty = prior_uncertainty
        filtered_x = min(max(filtered_x, 0.0), self.map_record.width - 1.0)
        filtered_y = min(max(filtered_y, 0.0), self.map_record.height - 1.0)
        uncertainty = min(uncertainty, self.config.max_uncertainty_m)
        latitude, longitude = self.map_record.pixel_to_geo(filtered_x, filtered_y)
        self.state = CMCTState(
            city=self.map_record.city,
            x=filtered_x,
            y=filtered_y,
            latitude=latitude,
            longitude=longitude,
            heading_deg=filtered_heading,
            uncertainty_m=uncertainty,
        )
        return CMCTUpdate(
            state=self.state,
            prior_x=prior_x,
            prior_y=prior_y,
            position_innovation_m=position_innovation,
            heading_innovation_deg=heading_innovation,
            motion_consistency=motion_consistency,
            effective_confidence=effective_confidence,
            update_gain=gain,
            observation_rejected=rejected,
            relocalized=relocalized,
            rejection_streak=self.rejection_streak,
        )


CMCTPredictionFunction = Callable[[Image.Image], CMCTObservation]


def run_cmct_navigation_episode(
    route: NavigationRoute,
    map_record: MapRecord,
    map_image: Image.Image,
    predict: CMCTPredictionFunction,
    config: CMCTConfig,
    *,
    step_m: float = 25.0,
    arrival_radius_m: float = 20.0,
    max_steps: int = 384,
    observation_size: int = 256,
) -> CMCTNavigationEpisode:
    """Run CMCT without changing the frozen localizer or baseline runner."""

    if step_m <= 0 or arrival_radius_m <= 0 or max_steps <= 0:
        raise ValueError("step_m, arrival_radius_m, and max_steps must be positive")
    current = route.waypoints_xy[0]
    real_heading = 90.0
    waypoint_index = 1
    actual_path_m = 0.0
    records: list[CMCTNavigationStep] = []
    termination = "max_steps"
    temporal = CMCTFilter(map_record, config)
    temporal.initialize(*current, heading_deg=real_heading)
    executed_heading = real_heading
    executed_distance = 0.0

    for step in range(max_steps):
        target = route.waypoints_xy[waypoint_index]
        target_lat, target_lon = map_record.pixel_to_geo(*target)
        observation_image = crop_rotated_observation(
            map_image, current, real_heading, size=observation_size
        )
        observation = predict(observation_image)
        update = temporal.update(
            observation,
            executed_heading_deg=executed_heading,
            executed_distance_m=executed_distance,
        )
        state = update.state
        real_error = pixel_distance_m(current, target, map_record)
        raw_error = haversine_m(
            (observation.pose.latitude, observation.pose.longitude),
            (target_lat, target_lon),
        )
        filtered_error = haversine_m(
            (state.latitude, state.longitude), (target_lat, target_lon)
        )
        command_heading = bearing_deg(
            (state.latitude, state.longitude), (target_lat, target_lon)
        )
        records.append(
            CMCTNavigationStep(
                step=step,
                target_waypoint=waypoint_index,
                real_x=current[0],
                real_y=current[1],
                raw_predicted_city=observation.pose.city,
                raw_predicted_x=observation.pose.x,
                raw_predicted_y=observation.pose.y,
                raw_predicted_heading_deg=observation.pose.heading_deg,
                visual_confidence=observation.confidence,
                filtered_x=state.x,
                filtered_y=state.y,
                filtered_latitude=state.latitude,
                filtered_longitude=state.longitude,
                filtered_heading_deg=state.heading_deg,
                uncertainty_m=state.uncertainty_m,
                position_innovation_m=update.position_innovation_m,
                heading_innovation_deg=update.heading_innovation_deg,
                motion_consistency=update.motion_consistency,
                effective_confidence=update.effective_confidence,
                update_gain=update.update_gain,
                observation_rejected=update.observation_rejected,
                relocalized=update.relocalized,
                commanded_heading_deg=command_heading,
                waypoint_error_real_m=real_error,
                waypoint_error_raw_m=raw_error,
                waypoint_error_filtered_m=filtered_error,
            )
        )

        # Keep the released baseline's waypoint policy for comparability: the
        # simulator truth or the navigation estimate may advance a waypoint.
        if min(real_error, filtered_error) < arrival_radius_m:
            executed_distance = 0.0
            if waypoint_index == len(route.waypoints_xy) - 1:
                termination = "endpoint_reached"
                break
            waypoint_index += 1
            continue

        current = move_pixel(current, command_heading, step_m, map_record)
        actual_path_m += step_m
        real_heading = command_heading
        executed_heading = command_heading
        executed_distance = step_m
        if not (
            0.0 <= current[0] < map_record.width
            and 0.0 <= current[1] < map_record.height
        ):
            termination = "out_of_map"
            break

    final_error = pixel_distance_m(current, route.waypoints_xy[-1], map_record)
    success = final_error < arrival_radius_m
    spl = (
        100.0 * route.shortest_path_m / max(route.shortest_path_m, actual_path_m)
        if success
        else 0.0
    )
    effective = [record.effective_confidence for record in records]
    uncertainties = [record.uncertainty_m for record in records]
    return CMCTNavigationEpisode(
        route_id=route.route_id,
        success=success,
        spl=spl,
        navigation_error_m=final_error,
        shortest_path_m=route.shortest_path_m,
        actual_path_m=actual_path_m,
        steps=len(records),
        reached_waypoints=waypoint_index,
        termination=termination,
        rejected_observations=sum(record.observation_rejected for record in records),
        relocalizations=sum(record.relocalized for record in records),
        mean_effective_confidence=sum(effective) / len(effective),
        mean_uncertainty_m=sum(uncertainties) / len(uncertainties),
        trajectory=tuple(records),
    )

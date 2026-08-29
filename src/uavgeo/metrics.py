"""Paper-facing metrics for global UAV localization and heading estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence, Union

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class BearingMetrics:
    recall_at_1: float
    lsr_at_15: float
    hsr_at_15: float
    mle: float
    mhe: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _as_float_tensor(value: Union[Tensor, Sequence[Sequence[float]]]) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def position_errors_meters(
    predicted_xy: Union[Tensor, Sequence[Sequence[float]]],
    target_xy: Union[Tensor, Sequence[Sequence[float]]],
    meters_per_pixel_xy: Union[Tensor, Sequence[float], Sequence[Sequence[float]]],
) -> Tensor:
    """Return per-sample Euclidean location errors in meters."""

    predicted = _as_float_tensor(predicted_xy)
    target = _as_float_tensor(target_xy)
    scale = torch.as_tensor(meters_per_pixel_xy, dtype=torch.float64)
    if predicted.ndim != 2 or predicted.shape[1] != 2 or target.shape != predicted.shape:
        raise ValueError("predicted_xy and target_xy must have matching shape [N,2]")
    if scale.shape == (2,):
        scale = scale.unsqueeze(0).expand_as(predicted)
    if scale.shape != predicted.shape:
        raise ValueError("meters_per_pixel_xy must have shape [2] or [N,2]")
    if (scale <= 0).any():
        raise ValueError("meters-per-pixel values must be positive")
    return torch.linalg.vector_norm((predicted - target) * scale, dim=-1)


def heading_errors_degrees(
    predicted_heading: Union[Tensor, Sequence[Sequence[float]]],
    target_heading: Union[Tensor, Sequence[Sequence[float]]],
) -> Tensor:
    """Return the shortest circular error between heading vectors in degrees."""

    predicted = _as_float_tensor(predicted_heading)
    target = _as_float_tensor(target_heading)
    if predicted.ndim != 2 or predicted.shape[1] != 2 or target.shape != predicted.shape:
        raise ValueError("Heading tensors must have matching shape [N,2]")
    if (torch.linalg.vector_norm(predicted, dim=-1) <= 1e-12).any():
        raise ValueError("Predicted heading contains a zero vector")
    if (torch.linalg.vector_norm(target, dim=-1) <= 1e-12).any():
        raise ValueError("Target heading contains a zero vector")
    cosine = (F.normalize(predicted, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1)
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def haversine_errors_meters(
    predicted_latlon: Union[Tensor, Sequence[Sequence[float]]],
    target_latlon: Union[Tensor, Sequence[Sequence[float]]],
) -> Tensor:
    """Return great-circle distances for `[latitude, longitude]` pairs."""

    predicted = torch.deg2rad(_as_float_tensor(predicted_latlon))
    target = torch.deg2rad(_as_float_tensor(target_latlon))
    if predicted.ndim != 2 or predicted.shape[1] != 2 or target.shape != predicted.shape:
        raise ValueError("Latitude/longitude tensors must have matching shape [N,2]")
    delta = predicted - target
    a = (
        torch.sin(delta[:, 0] / 2).square()
        + torch.cos(predicted[:, 0])
        * torch.cos(target[:, 0])
        * torch.sin(delta[:, 1] / 2).square()
    )
    return 2 * 6_371_008.8 * torch.asin(torch.sqrt(a.clamp(0.0, 1.0)))


def compute_bearing_metrics(
    predicted_tile_ids: Sequence[str],
    target_tile_ids: Sequence[str],
    predicted_xy: Union[Tensor, Sequence[Sequence[float]]],
    target_xy: Union[Tensor, Sequence[Sequence[float]]],
    predicted_heading: Union[Tensor, Sequence[Sequence[float]]],
    target_heading: Union[Tensor, Sequence[Sequence[float]]],
    cities: Sequence[str],
    meters_per_pixel_xy: Union[Tensor, Sequence[float], Sequence[Sequence[float]]] = (
        0.25,
        0.25,
    ),
    predicted_cities: Optional[Sequence[str]] = None,
    predicted_latlon: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
    target_latlon: Optional[Union[Tensor, Sequence[Sequence[float]]]] = None,
    mle_protocol: Literal["bearing-compatible", "global-geodesic"] = (
        "bearing-compatible"
    ),
) -> BearingMetrics:
    """Compute the five primary metrics using Bearing-UAV aggregation behavior.

    Recall@1 is exact global tile retrieval accuracy. LSR and HSR are micro
    success rates over samples. MLE follows the original code's macro average:
    location errors are averaged within each city/map and then across maps. MHE
    is the mean shortest circular heading error over all samples.
    """

    sample_count = len(target_tile_ids)
    if sample_count == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set")
    if len(predicted_tile_ids) != sample_count or len(cities) != sample_count:
        raise ValueError("Tile ID and city collections must have the same length")

    if mle_protocol not in ("bearing-compatible", "global-geodesic"):
        raise ValueError(f"Unsupported MLE protocol: {mle_protocol}")
    local_position_errors = position_errors_meters(
        predicted_xy, target_xy, meters_per_pixel_xy
    )
    position_errors = local_position_errors.clone()
    cross_city = torch.zeros(sample_count, dtype=torch.bool)
    if predicted_cities is not None:
        if len(predicted_cities) != sample_count:
            raise ValueError("predicted_cities must match the sample count")
        cross_city = torch.tensor(
            [predicted != target for predicted, target in zip(predicted_cities, cities)],
            dtype=torch.bool,
        )
        if cross_city.any():
            if predicted_latlon is None or target_latlon is None:
                raise ValueError("Cross-city evaluation requires latitude/longitude")
            geographic_errors = haversine_errors_meters(
                predicted_latlon, target_latlon
            )
            position_errors[cross_city] = geographic_errors[cross_city]
    heading_errors = heading_errors_degrees(predicted_heading, target_heading)
    if len(position_errors) != sample_count or len(heading_errors) != sample_count:
        raise ValueError("Prediction tensors do not match tile ID collection length")

    recall = 100.0 * sum(
        predicted == target
        for predicted, target in zip(predicted_tile_ids, target_tile_ids)
    ) / sample_count
    lsr = 100.0 * float((position_errors <= 15.0).double().mean())
    hsr = 100.0 * float((heading_errors <= 15.0).double().mean())

    city_names = sorted(set(cities))
    city_mle = []
    for city in city_names:
        mask = torch.tensor([item == city for item in cities], dtype=torch.bool)
        if mle_protocol == "bearing-compatible":
            # Bearing-UAV receives the correct local map/RSB, so its MLE has no
            # cross-map distance. Match that local-map precision protocol while
            # continuing to count cross-city predictions as LSR failures.
            mask = mask & ~cross_city
            if not mask.any():
                raise ValueError(f"No same-city predictions available for {city}")
            city_mle.append(local_position_errors[mask].mean())
        elif mask.any():
            city_mle.append(position_errors[mask].mean())
    mle = float(torch.stack(city_mle).mean())
    mhe = float(heading_errors.mean())
    return BearingMetrics(recall, lsr, hsr, mle, mhe)

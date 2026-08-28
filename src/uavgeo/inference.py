"""Single-UVP global retrieval followed by 3x3 local registration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor

from .data.catalog import UAV90KCatalog
from .data.datasets import SatelliteCandidateLoader
from .models.retrieval import SatelliteFeatureIndex
from .models.system import GlobalToLocalModel


@dataclass(frozen=True)
class CandidatePrediction:
    center_tile_id: str
    retrieval_rank: int
    retrieval_score: float
    confidence_logit: float
    global_x: float
    global_y: float
    heading_cos: float
    heading_sin: float


@dataclass(frozen=True)
class QueryPrediction:
    retrieved_top1_tile_id: str
    predicted_tile_id: str
    city: str
    global_x: float
    global_y: float
    latitude: float
    longitude: float
    heading_cos: float
    heading_sin: float
    selected_rank: int
    candidates: tuple[CandidatePrediction, ...]


class GlobalToLocalInference:
    def __init__(
        self,
        model: GlobalToLocalModel,
        catalog: UAV90KCatalog,
        index: SatelliteFeatureIndex,
        device: torch.device,
        confidence_weight: float = 0.0,
        candidate_batch_size: int = 1,
        amp_enabled: bool = True,
    ) -> None:
        if candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        self.model = model.eval()
        self.catalog = catalog
        self.index = index
        self.device = device
        self.confidence_weight = float(confidence_weight)
        self.candidate_batch_size = candidate_batch_size
        self.amp_enabled = bool(amp_enabled and device.type == "cuda")
        self.candidate_loader = SatelliteCandidateLoader(catalog)

    def _autocast(self):
        if self.amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def predict(self, query_image: Tensor, top_k: int = 5) -> QueryPrediction:
        if query_image.ndim == 3:
            query_image = query_image.unsqueeze(0)
        if query_image.ndim != 4 or query_image.shape[0] != 1:
            raise ValueError("Inference currently expects one query image")
        query_image = query_image.to(self.device)
        with self._autocast():
            query_descriptor, query_features = self.model.encode_query(query_image)
        retrieval = self.index.search(query_descriptor, top_k)
        center_ids = self.index.tile_ids_for(retrieval.indices)[0]

        loaded = [self.candidate_loader.load(tile_id) for tile_id in center_ids]
        origins = torch.stack([item[2] for item in loaded]).to(self.device)
        positions: list[Tensor] = []
        headings: list[Tensor] = []
        confidences: list[Tensor] = []
        for start in range(0, len(loaded), self.candidate_batch_size):
            chunk = loaded[start : start + self.candidate_batch_size]
            satellite_tiles = torch.stack([item[0] for item in chunk]).to(self.device)
            validity = torch.stack([item[1] for item in chunk]).to(self.device)
            with self._autocast():
                localization = self.model.localize_candidates(
                    query_features, satellite_tiles, validity
                )
            positions.append(localization.position_xy.float())
            headings.append(localization.heading.float())
            confidences.append(localization.confidence_logit.float())
        position_xy = torch.cat(positions)
        heading = torch.cat(headings)
        confidence_logit = torch.cat(confidences)
        global_xy = (
            origins
            + 3 * self.candidate_loader.tile_size * position_xy
        )

        retrieval_scores = retrieval.scores[0]
        selection_scores = retrieval_scores + self.confidence_weight * torch.sigmoid(
            confidence_logit
        )
        selected = int(selection_scores.argmax())
        candidates: list[CandidatePrediction] = []
        for index, center_id in enumerate(center_ids):
            candidates.append(
                CandidatePrediction(
                    center_tile_id=center_id,
                    retrieval_rank=index + 1,
                    retrieval_score=float(retrieval_scores[index]),
                    confidence_logit=float(confidence_logit[index]),
                    global_x=float(global_xy[index, 0]),
                    global_y=float(global_xy[index, 1]),
                    heading_cos=float(heading[index, 0]),
                    heading_sin=float(heading[index, 1]),
                )
            )

        chosen = candidates[selected]
        city = self.catalog.tiles[chosen.center_tile_id].city
        map_record = self.catalog.maps_by_city[city]
        global_x = min(max(chosen.global_x, 0.0), map_record.width - 1.0)
        global_y = min(max(chosen.global_y, 0.0), map_record.height - 1.0)
        tile_col = min(int(global_x // self.candidate_loader.tile_size), 15)
        tile_row = min(int(global_y // self.candidate_loader.tile_size), 15)
        predicted_tile_id = self.catalog.tile_id(city, tile_row, tile_col)
        if predicted_tile_id is None:
            raise RuntimeError("Predicted position does not map to a satellite tile")
        latitude, longitude = map_record.pixel_to_geo(global_x, global_y)
        return QueryPrediction(
            retrieved_top1_tile_id=center_ids[0],
            predicted_tile_id=predicted_tile_id,
            city=city,
            global_x=global_x,
            global_y=global_y,
            latitude=latitude,
            longitude=longitude,
            heading_cos=chosen.heading_cos,
            heading_sin=chosen.heading_sin,
            selected_rank=selected + 1,
            candidates=tuple(candidates),
        )

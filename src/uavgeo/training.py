"""Training utilities shared by command-line entry points and tests."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_multi_positive_mask(
    query_tile_ids: Sequence[str], reference_tile_ids: Sequence[str]
) -> Tensor:
    """Mark every query/reference pair that belongs to the same satellite tile."""

    if not query_tile_ids or not reference_tile_ids:
        raise ValueError("Tile ID collections must not be empty")
    return torch.tensor(
        [
            [query_id == reference_id for reference_id in reference_tile_ids]
            for query_id in query_tile_ids
        ],
        dtype=torch.bool,
    )


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return parameters


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

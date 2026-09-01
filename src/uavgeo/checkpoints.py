"""Model checkpoint loading helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Union

import torch

from .models.system import GlobalToLocalModel


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trained_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    backbone_pretrained: bool = False,
) -> tuple[GlobalToLocalModel, dict[str, Any]]:
    """Recreate the architecture recorded by the trainer and load its weights."""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location=device, weights_only=False
    )
    arguments = checkpoint.get("args")
    if not isinstance(arguments, dict):
        raise ValueError("Checkpoint does not contain trainer arguments")
    model = GlobalToLocalModel(
        backbone_name=arguments.get("backbone", "dinov2_vitb14"),
        backbone_pretrained=backbone_pretrained,
        freeze_backbone=not arguments.get("unfreeze_backbone", False),
        retrieval_dim=int(arguments.get("retrieval_dim", 256)),
        local_model_dim=int(arguments.get("local_model_dim", 256)),
        adapter_dim=int(arguments.get("adapter_dim", 128)),
        num_heads=int(arguments.get("num_heads", 8)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint

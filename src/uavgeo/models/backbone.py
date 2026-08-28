"""DINOv2 feature extractor wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class DINOFeatures:
    global_descriptor: Tensor
    patch_tokens: Tensor
    grid_size: tuple[int, int]


class DINOv2Backbone(nn.Module):
    """Expose normalized CLS and dense patch tokens from official DINOv2."""

    MODEL_DIMS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        freeze: bool = True,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if model_name not in self.MODEL_DIMS:
            raise ValueError(f"Unsupported DINOv2 model: {model_name}")
        self.model_name = model_name
        self.output_dim = self.MODEL_DIMS[model_name]
        self.patch_size = 14
        self.freeze = freeze
        self.model = model or torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
        )
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "DINOv2Backbone":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, images: Tensor) -> DINOFeatures:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(images.shape)}")
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("Image dimensions must be divisible by DINOv2 patch size 14")

        context = torch.no_grad() if self.freeze else torch.enable_grad()
        with context:
            output = self.model.forward_features(images)
        global_descriptor = output["x_norm_clstoken"]
        patch_tokens = output["x_norm_patchtokens"]
        grid_size = (height // self.patch_size, width // self.patch_size)
        expected_tokens = grid_size[0] * grid_size[1]
        if patch_tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"DINO token count {patch_tokens.shape[1]} does not match grid {grid_size}"
            )
        return DINOFeatures(global_descriptor, patch_tokens, grid_size)

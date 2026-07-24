# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Optional frozen DINOv2 feature encoder for DSRL (posttrain-style obs).

posttrain's DSRL conditions the SAC actor + critic on an *independent* frozen
DINOv2 embedding of the scene camera rather than the base policy's own features.
This module is the opt-in equivalent here (``adapter.dsrl.feature_source=dino``):
a frozen ``facebook/dinov2-*`` encoder that turns the raw camera frame into a
fixed-width embedding fed to the noise actor and the Transformer critic.

Default OFF. GR00T's own VL features (``feature_source=gr00t``) are recommended
because they are aligned with the frozen flow head the noise steers; DINOv2 is
provided for exact posttrain parity / ablation.

NOTE: the image preprocessing here assumes ``model_input.images`` holds RGB
frames (a per-camera dict, or a ``[B, (V,) C, H, W]`` / ``[B, H, W, C]`` tensor).
Values >1.5 are treated as ``[0, 255]`` and rescaled. Validate this against the
env's real image format on the first run.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# ImageNet normalisation (DINOv2's training statistics).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoFeatureEncoder(nn.Module):
    """Frozen DINOv2 image encoder -> ``[B, hidden_size]`` embedding."""

    def __init__(self, model_name: str = "facebook/dinov2-base", image_size: int = 224) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError("adapter.dsrl.feature_source=dino requires `transformers` with DINOv2 support.") from exc

        self.model = AutoModel.from_pretrained(model_name)
        self.model.requires_grad_(False)
        self.model.eval()
        self.output_dim = int(self.model.config.hidden_size)
        self.image_size = int(image_size)
        self.register_buffer("pixel_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        logger.info("[dsrl] DINOv2 feature encoder '%s' loaded (frozen, output_dim=%d)", model_name, self.output_dim)

    def train(self, mode: bool = True):  # keep DINOv2 in eval regardless of parent mode
        super().train(mode)
        self.model.eval()
        return self

    @staticmethod
    def _to_bchw(images: Any) -> torch.Tensor:
        """Coerce the env image container to a single-view ``[B, 3, H, W]`` batch."""
        if isinstance(images, dict):
            if not images:
                raise ValueError("DINOv2 feature_source: empty image dict.")
            images = next(iter(images.values()))
        x = images
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if x.dim() == 5:  # [B, V, C, H, W] -> first view
            x = x[:, 0]
        if x.dim() != 4:
            raise ValueError(f"DINOv2 feature_source expects a 4-D image batch, got shape {tuple(x.shape)}")
        # Channels-last [B, H, W, C] -> channels-first.
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        if x.shape[1] == 1:  # grayscale -> RGB
            x = x.expand(-1, 3, -1, -1)
        return x

    @torch.no_grad()
    def forward(self, images: Any) -> torch.Tensor:
        param = next(self.model.parameters())
        x = self._to_bchw(images).to(device=param.device, dtype=torch.float32)
        if x.amax() > 1.5:  # [0, 255] -> [0, 1]
            x = x / 255.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.pixel_mean) / self.pixel_std
        out = self.model(pixel_values=x.to(param.dtype))
        emb = getattr(out, "pooler_output", None)
        if emb is None:
            emb = out.last_hidden_state.mean(dim=1)
        return emb.float()


__all__ = ["DinoFeatureEncoder"]

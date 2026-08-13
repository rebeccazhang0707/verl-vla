# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""CNN observation encoder shared by pixel actors and critics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DSRLCNNEncoder(nn.Module):
    """Four-layer VALID CNN followed by a normalized tanh bottleneck."""

    def __init__(
        self,
        *,
        image_size: int,
        features: list[int],
        strides: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        if len(features) != len(strides):
            raise ValueError("DSRL CNN features and strides must have equal lengths.")

        layers: list[nn.Module] = []
        in_channels = 3
        spatial_size = int(image_size)
        for out_channels, stride in zip(features, strides, strict=True):
            convolution = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
            nn.init.orthogonal_(convolution.weight, gain=2.0**0.5)
            nn.init.zeros_(convolution.bias)
            layers.extend([convolution, nn.ReLU()])
            in_channels = out_channels
            spatial_size = (spatial_size - 3) // stride + 1
        self.image_size = int(image_size)
        self.convolutions = nn.Sequential(*layers)

        bottleneck = nn.Linear(in_channels * spatial_size * spatial_size, int(latent_dim))
        nn.init.xavier_normal_(bottleneck.weight)
        nn.init.zeros_(bottleneck.bias)
        self.bottleneck = nn.Sequential(bottleneck, nn.LayerNorm(int(latent_dim)), nn.Tanh())

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4:
            raise ValueError(f"DSRL CNN expects a 4D image batch, got {tuple(pixels.shape)}")
        if pixels.shape[-1] == 3:
            pixels = pixels.permute(0, 3, 1, 2)
        elif pixels.shape[1] != 3:
            raise ValueError(f"DSRL CNN expects BCHW or BHWC RGB images, got {tuple(pixels.shape)}")

        normalize_uint8 = pixels.dtype == torch.uint8
        pixels = pixels.float()
        if pixels.shape[-2:] != (self.image_size, self.image_size):
            pixels = F.interpolate(
                pixels,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if normalize_uint8:
            pixels = pixels / 255.0
        return self.bottleneck(self.convolutions(pixels).flatten(start_dim=1))


__all__ = ["DSRLCNNEncoder"]

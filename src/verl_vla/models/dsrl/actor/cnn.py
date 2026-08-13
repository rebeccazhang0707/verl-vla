# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Official-style DSRL actor over RGB observations and robot state."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import DSRLSteeringConfig
from .cnn_encoder import DSRLCNNEncoder

_TANH_EPS = 1e-6


class DSRLCNNActor(nn.Module):
    """Independent CNN policy producing tanh-Gaussian steering noise."""

    def __init__(
        self,
        *,
        feature_dim: int,
        state_dim: int,
        noise_dim: int,
        noise_horizon: int,
        config: DSRLSteeringConfig,
    ) -> None:
        super().__init__()
        del feature_dim
        actor_config = config.cnn
        self.state_dim = int(state_dim)
        self.noise_dim = int(noise_dim)
        self.noise_horizon = int(noise_horizon)
        self.noise_per_step = bool(config.noise_per_step)
        self.noise_bound = float(config.noise_bound)
        self.log_std_min = float(config.log_std_min)
        self.log_std_max = float(config.log_std_max)
        if self.noise_bound <= 0:
            raise ValueError(f"dsrl noise_bound must be positive, got {self.noise_bound}")

        latent_dim = int(actor_config.latent_dim)
        self.encoder = DSRLCNNEncoder(
            image_size=int(actor_config.image_size),
            features=[int(value) for value in actor_config.features],
            strides=[int(value) for value in actor_config.strides],
            latent_dim=latent_dim,
        )
        trunk: list[nn.Module] = []
        in_dim = latent_dim + self.state_dim
        for hidden_dim in (int(value) for value in actor_config.hidden_dims):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight, gain=2.0**0.5)
            nn.init.zeros_(linear.bias)
            trunk.extend([linear, nn.ReLU()])
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk)

        self.out_dim = self.noise_dim * (self.noise_horizon if self.noise_per_step else 1)
        self.mean_head = nn.Linear(in_dim, self.out_dim)
        self.log_std_head = nn.Linear(in_dim, self.out_dim)
        for head in (self.mean_head, self.log_std_head):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, pixels: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.reshape(state.shape[0], -1)[..., : self.state_dim].float()
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"DSRL CNN actor expected state_dim={self.state_dim}, got {state.shape[-1]}")
        hidden = self.trunk(torch.cat([self.encoder(pixels), state], dim=-1))
        mean = self.mean_head(hidden).float()
        log_std = self.log_std_head(hidden).float().clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        pixels: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False,
        noise_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(pixels, state)
        if deterministic:
            pre_tanh = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device, dtype=mean.dtype)
        else:
            std = log_std.exp()
            if noise_scale is not None:
                std = torch.sqrt(std.square() + float(noise_scale) ** 2)
            normal = torch.distributions.Normal(mean, std)
            pre_tanh = normal.rsample()
            squashed = torch.tanh(pre_tanh)
            log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - squashed.pow(2) + _TANH_EPS)
            log_prob = log_prob.sum(dim=-1)
            if self.noise_bound != 1.0:
                log_prob = log_prob - self.out_dim * math.log(self.noise_bound)
        noise_flat = torch.tanh(pre_tanh) * self.noise_bound
        batch_size = noise_flat.shape[0]
        if self.noise_per_step:
            noise = noise_flat.view(batch_size, self.noise_horizon, self.noise_dim)
        else:
            noise = noise_flat.unsqueeze(1).expand(batch_size, self.noise_horizon, self.noise_dim).contiguous()
        return noise, log_prob


__all__ = ["DSRLCNNActor"]

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DSRL noise actor variant: a Transformer chunking SAC policy over flow noise.

Same contract as :class:`~verl_vla.models.dsrl.actor.mlp.DSRLNoiseActor` (map
a frozen-backbone feature vector plus the raw robot state to the flow sampler's
initial noise ``x0``), but the trunk mirrors the posttrain reference DSRL actor
(``isaac_rl_posttraining/modules/transformer/actor.py``): the observation is
embedded once, repeated over the noise horizon, differentiated only by a
sinusoidal positional encoding, and mixed by a Transformer encoder so every flow
step gets its own tanh-squashed Gaussian latent while sharing parameters.

The tanh change-of-variables correction uses the numerically exact softplus form
the reference DSRL stack relies on, rather than the ``log(1 - tanh^2 + eps)``
form of the MLP actor.

Selected with ``adapter.dsrl.actor_type=transformer``; the MLP actor stays the
default. ``noise_per_step`` picks an independent latent per horizon step
(GR00T/posttrain parity) or one shared latent broadcast over the chunk (pi0
RLinf parity, i.e. a single-token transformer).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import DSRLSteeringConfig

_LOG_2 = math.log(2.0)


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (posttrain parity)."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class DSRLTransformerNoiseActor(nn.Module):
    """Transformer chunking tanh-Gaussian policy over the initial flow noise.

    Args:
        feature_dim: Width of the frozen backbone feature vector (e.g. 2048).
        state_dim: Flattened raw/normalized state width fed to the actor.
        noise_dim: Per-step noise width (the steered action DOF).
        noise_horizon: Flow action horizon the noise seeds.
        config: Shared :class:`DSRLSteeringConfig`.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        state_dim: int,
        noise_dim: int,
        noise_horizon: int,
        config: DSRLSteeringConfig | None = None,
    ) -> None:
        super().__init__()
        config = config or DSRLSteeringConfig()
        actor_config = config.transformer
        self.feature_dim = int(feature_dim)
        self.state_dim = int(state_dim)
        self.noise_dim = int(noise_dim)
        self.noise_horizon = int(noise_horizon)
        self.noise_per_step = bool(config.noise_per_step)
        self.noise_bound = float(config.noise_bound)
        self.log_std_min = float(config.log_std_min)
        self.log_std_max = float(config.log_std_max)
        if self.noise_bound <= 0:
            raise ValueError(f"dsrl noise_bound must be positive, got {self.noise_bound}")

        # One latent per horizon step (posttrain), or one shared latent broadcast
        # over the chunk (a single-token transformer, RLinf parity).
        self.chunk_size = self.noise_horizon if self.noise_per_step else 1

        d_model = int(actor_config.d_model)
        nhead = int(actor_config.nhead)
        if d_model % nhead != 0:
            raise ValueError(f"dsrl d_model ({d_model}) must be divisible by nhead ({nhead})")
        if actor_config.activation not in ("relu", "gelu"):
            raise ValueError(f"dsrl transformer activation must be 'relu' or 'gelu', got {actor_config.activation}")
        self.d_model = d_model

        # Observation featurizer: concat the frozen features and raw state, then
        # project to the token width (posttrain concatenates per-key features and
        # projects to d_model with a single linear).
        self.obs_projection = nn.Linear(self.feature_dim + self.state_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.position_embedding = _PositionalEncoding(
            d_model, max_len=max(self.chunk_size, 1), dropout=float(actor_config.positional_dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=float(actor_config.dropout),
            activation=str(actor_config.activation),
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(actor_config.num_encoder_layers),
            norm=nn.LayerNorm(d_model),
        )
        self.pre_output_norm = nn.LayerNorm(d_model)

        # Per-token mean / log-std heads (shared across steps), posttrain layout.
        self.mean_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, self.noise_dim)
        )
        self.log_std_processor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, self.noise_dim)
        )
        # posttrain init: a zero log-std head starts the policy at std=exp(log_std_init).
        with torch.no_grad():
            self.log_std_processor[-1].weight.zero_()
            self.log_std_processor[-1].bias.fill_(float(actor_config.log_std_init))

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-step pre-tanh Gaussian ``(mean, log_std)`` in float32.

        Shapes: ``(B, chunk_size, noise_dim)`` each.
        """
        features = features.reshape(features.shape[0], -1).float()
        state = state.reshape(state.shape[0], -1).float()
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"DSRL noise actor expected feature_dim={self.feature_dim}, got {features.shape[-1]}")
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"DSRL noise actor expected state_dim={self.state_dim}, got {state.shape[-1]}")

        obs = self.obs_projection(torch.cat([features, state], dim=-1))  # [B, d_model]
        # One token per chunk step: same observation, distinguished by the
        # positional encoding; the encoder mixes information across steps.
        tokens = obs.unsqueeze(1).expand(obs.shape[0], self.chunk_size, self.d_model)
        tokens = self.input_norm(tokens)
        tokens = self.position_embedding(tokens)
        hidden = self.transformer_encoder(tokens)
        hidden = self.pre_output_norm(hidden)

        mean = self.mean_processor(hidden).float()
        log_std = self.log_std_processor(hidden).float().clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
        deterministic: bool = False,
        noise_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample steering noise for the flow head.

        Returns:
            noise: (B, noise_horizon, noise_dim) float32 in
                ``[-noise_bound, noise_bound]``, ready to seed the flow ODE.
            log_prob: (B,) float32, summed over all steered noise entries
                (0 when ``deterministic``), tanh-corrected.
        """
        mean, log_std = self(features, state)  # [B, chunk_size, noise_dim]
        if deterministic:
            pre_tanh = mean
            log_prob = torch.zeros(mean.shape[0], device=mean.device, dtype=mean.dtype)
        else:
            std = log_std.exp()
            if noise_scale is not None:
                std = torch.sqrt(std.square() + float(noise_scale) ** 2)
            normal = torch.distributions.Normal(mean, std)
            pre_tanh = normal.rsample()
            log_prob = normal.log_prob(pre_tanh).sum(dim=(-1, -2))
            # tanh change-of-variables correction, exact softplus form (posttrain
            # parity): log(1 - tanh(u)^2) = 2*(log2 - u - softplus(-2u)).
            correction = 2.0 * (_LOG_2 - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh))
            log_prob = log_prob - correction.sum(dim=(-1, -2))
            if self.noise_bound != 1.0:
                log_prob = log_prob - self.chunk_size * self.noise_dim * math.log(self.noise_bound)
        noise = torch.tanh(pre_tanh) * self.noise_bound  # [B, chunk_size, noise_dim]

        if not self.noise_per_step:
            # Broadcast the single shared latent over the whole action horizon.
            noise = noise.expand(noise.shape[0], self.noise_horizon, self.noise_dim).contiguous()
        return noise, log_prob


__all__ = ["DSRLTransformerNoiseActor"]

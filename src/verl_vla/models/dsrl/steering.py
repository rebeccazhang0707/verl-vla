# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""DSRL latent-noise steering module for SAC-enabled VLA models."""

from __future__ import annotations

import torch
from torch import nn

from .actor import DSRLCNNActor, DSRLNoiseActor, DSRLTransformerNoiseActor
from .config import DSRLSteeringConfig

_DSRL_STATE_PREFIX = "dsrl.noise_actor."

# Noise-actor architectures selectable via ``adapter.dsrl.actor_type``.
NOISE_ACTORS = {
    "cnn": DSRLCNNActor,
    "mlp": DSRLNoiseActor,
    "transformer": DSRLTransformerNoiseActor,
}


class DSRLSteering(nn.Module):
    """Own the trainable DSRL noise actor and shared SAC transformations."""

    def __init__(
        self,
        config: DSRLSteeringConfig,
        *,
        feature_dim: int,
        state_dim: int,
        noise_dim: int,
        noise_horizon: int,
    ) -> None:
        super().__init__()
        self.config = config
        actor_type = str(getattr(config, "actor_type", "mlp")).lower()
        if actor_type not in NOISE_ACTORS:
            raise ValueError(f"dsrl actor_type must be one of {sorted(NOISE_ACTORS)}, got {actor_type!r}")
        self.actor_type = actor_type
        self.noise_actor = NOISE_ACTORS[actor_type](
            feature_dim=int(config.feature_dim or feature_dim),
            state_dim=int(config.state_dim or state_dim),
            noise_dim=int(noise_dim),
            noise_horizon=int(noise_horizon),
            config=config,
        )

    def sample(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
        noise_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        noise, log_probs = self.noise_actor.sample(
            features,
            state,
            deterministic=deterministic,
            noise_scale=noise_scale,
        )
        return noise, log_probs, {}

    def named_actor_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [
            (f"{_DSRL_STATE_PREFIX}{name}", parameter)
            for name, parameter in self.noise_actor.named_parameters()
            if parameter.requires_grad
        ]

    def select_critic_noise(self, actions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Select the steering noise stored in the SAC action contract."""
        noise = actions.get("steering_noise")
        if noise is None:
            noise = actions["action"]
        if noise.dim() == 3 and not self.noise_actor.noise_per_step:
            noise = noise[:, :1, :]
        return {"action": noise}


__all__ = ["DSRLSteering"]

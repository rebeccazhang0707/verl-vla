# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Pixel critic backend for PI0 DSRL training."""

from __future__ import annotations

import copy
from typing import Literal

import torch
from torch import nn

from ...dsrl import DSRLCNNEncoder
from .base import CriticBackend


class _CNNCriticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            linear = nn.Linear(current_dim, hidden_dim)
            nn.init.orthogonal_(linear.weight, gain=2.0**0.5)
            nn.init.zeros_(linear.bias)
            layers.extend([linear, nn.LayerNorm(hidden_dim), nn.ReLU()])
            current_dim = hidden_dim
        output = nn.Linear(current_dim, 1)
        nn.init.orthogonal_(output.weight, gain=2.0**0.5)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class PI0CNNCritic(nn.Module):
    """Shared CNN encoder with independently parameterized Q-functions."""

    def __init__(
        self,
        *,
        head_num: int,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        image_size: int,
        cnn_features: list[int],
        cnn_strides: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.encoder = DSRLCNNEncoder(
            image_size=image_size,
            features=cnn_features,
            strides=cnn_strides,
            latent_dim=latent_dim,
        )
        input_dim = int(latent_dim) + self.state_dim + int(action_dim)
        self.critic_heads = nn.ModuleList(
            [_CNNCriticHead(input_dim=input_dim, hidden_dims=hidden_dims) for _ in range(head_num)]
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_network_heads = copy.deepcopy(self.critic_heads)

    def forward(
        self,
        *,
        pixels: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        encoder = self.target_encoder if use_target_network else self.encoder
        heads = self.target_network_heads if use_target_network else self.critic_heads
        encoder.requires_grad_(requires_grad)
        heads.requires_grad_(requires_grad)

        encoded_pixels = encoder(pixels)
        states = states.reshape(states.shape[0], -1)[..., : self.state_dim].float()
        actions = actions.reshape(states.shape[0], -1).float()
        critic_input = torch.cat([encoded_pixels, states, actions], dim=-1)
        q_values = torch.cat([head(critic_input) for head in heads], dim=-1)
        if method == "cat":
            return q_values
        if method == "min":
            return q_values.min(dim=-1).values
        raise ValueError(f"Unknown PI0 CNN critic reduction: {method}")

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.critic_heads.parameters())

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        for target, source in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.mul_(1.0 - tau).add_(source, alpha=tau)
        for target, source in zip(self.target_network_heads.parameters(), self.critic_heads.parameters(), strict=True):
            target.mul_(1.0 - tau).add_(source, alpha=tau)


class PI0CNNCriticBackend(CriticBackend):
    uses_task_ids = False

    def init(self, model) -> None:
        if not model.config.dsrl.enabled:
            raise ValueError("The cnn critic requires dsrl.enabled=true.")
        cnn_config = model.config.critic.cnn
        action_dim = int(model.policy.max_action_dim)
        if bool(model.config.dsrl.noise_per_step):
            action_dim *= int(model.policy.n_action_steps)
        model.critic = PI0CNNCritic(
            head_num=int(cnn_config.head_num),
            state_dim=int(model.config.dsrl.state_dim or len(model.state_norm_stats["mean"])),
            action_dim=action_dim,
            hidden_dims=[int(dim) for dim in cnn_config.hidden_dims],
            image_size=int(cnn_config.image_size),
            cnn_features=[int(value) for value in cnn_config.features],
            cnn_strides=[int(value) for value in cnn_config.strides],
            latent_dim=int(cnn_config.latent_dim),
        )

    def forward(
        self,
        model,
        a: dict[str, torch.Tensor],
        state_features,
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        del task_ids
        _, _, (pixels, states) = state_features
        return model.critic(
            pixels=pixels,
            states=states,
            actions=a["action"],
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    def get_critic_parameters(self, model) -> list[nn.Parameter]:
        return model.critic.trainable_parameters()

    @torch.no_grad()
    def update_target_network(self, model, tau: float) -> None:
        model.critic.update_target_network(tau)


__all__ = ["PI0CNNCritic", "PI0CNNCriticBackend"]

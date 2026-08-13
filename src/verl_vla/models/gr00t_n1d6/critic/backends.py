# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from typing import Literal

import torch

from .base import CriticBackend
from .group import Gr00tCriticGroup
from .transformer import Gr00tTransformerCriticGroup


def _build_critic_group(model, *, pooling: str) -> Gr00tCriticGroup:
    config = model.adapter_config.critic
    return Gr00tCriticGroup(
        head_num=int(config.head_num),
        input_dim=int(model.critic_input_dim),
        backbone_feature_dim=int(model.backbone_feature_dim),
        pooling=pooling,
        prefix_attn_heads=int(config.prefix_attn_heads),
        layernorm=bool(config.layernorm),
        pool_proj_dim=int(config.pool_proj_dim),
        use_encoded_state=bool(config.use_encoded_state),
        state_real_dim=config.state_real_dim,
        action_horizon=int(model.critic_action_horizon),
        action_dim=int(model.critic_action_dim),
        mask_frozen_action=bool(config.mask_frozen_action),
        privileged_obs=bool(config.privileged_obs),
        privileged_obs_dim=int(config.privileged_obs_dim or 0),
        sac_action_train_mask=getattr(model, "sac_action_train_mask", None),
        sac_action_train_all=bool(getattr(model, "sac_action_train_all", True)),
    )


class CrossAttentionCriticBackend(CriticBackend):
    uses_task_ids = False

    def init(self, model) -> None:
        model.critic = _build_critic_group(model, pooling="attn")

    def forward(
        self,
        model,
        a: dict[str, torch.Tensor],
        state_features: dict[str, torch.Tensor],
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        del task_ids
        return model.critic(
            a=a,
            state_features=state_features,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    def get_critic_parameters(self, model) -> list[torch.nn.Parameter]:
        return model.critic.get_critic_parameters()

    @torch.no_grad()
    def update_target_network(self, model, tau: float) -> None:
        model.critic.update_target_network(tau)


class TransformerCriticBackend(CriticBackend):
    """posttrain-style Transformer sequence critic over [obs_token, actions].

    Used by DSRL: the observation token is GR00T's frozen ``pooled`` VL feature
    plus the raw state, and the action tokens are the steering noise. Keeps the
    CriticBackend contract intact.
    """

    uses_task_ids = False

    def init(self, model) -> None:
        config = model.adapter_config.critic
        obs_dim = int(model.backbone_feature_dim) + int(model.state_horizon) * int(model.max_state_dim)
        model.critic = Gr00tTransformerCriticGroup(
            head_num=int(config.head_num),
            obs_dim=obs_dim,
            action_horizon=int(model.critic_action_horizon),
            action_dim=int(model.critic_action_dim),
            d_model=int(config.transformer_d_model),
            nhead=int(config.transformer_nhead),
            num_encoder_layers=int(config.transformer_num_layers),
            dropout=float(config.transformer_dropout),
            activation=str(config.transformer_activation),
            action_embedding_dim=int(config.transformer_action_embedding_dim),
            pooling_strategy=str(config.transformer_pooling),
        )

    def forward(
        self,
        model,
        a: dict[str, torch.Tensor],
        state_features: dict[str, torch.Tensor],
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        del task_ids
        return model.critic(
            a=a,
            state_features=state_features,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    def get_critic_parameters(self, model) -> list[torch.nn.Parameter]:
        return model.critic.get_critic_parameters()

    @torch.no_grad()
    def update_target_network(self, model, tau: float) -> None:
        model.critic.update_target_network(tau)


class MeanPoolCriticBackend(CriticBackend):
    uses_task_ids = False

    def init(self, model) -> None:
        model.critic = _build_critic_group(model, pooling="mean")

    def forward(
        self,
        model,
        a: dict[str, torch.Tensor],
        state_features: dict[str, torch.Tensor],
        task_ids: torch.Tensor | None = None,
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        del task_ids
        return model.critic(
            a=a,
            state_features=state_features,
            use_target_network=use_target_network,
            method=method,
            requires_grad=requires_grad,
        )

    def get_critic_parameters(self, model) -> list[torch.nn.Parameter]:
        return model.critic.get_critic_parameters()

    @torch.no_grad()
    def update_target_network(self, model, tau: float) -> None:
        model.critic.update_target_network(tau)

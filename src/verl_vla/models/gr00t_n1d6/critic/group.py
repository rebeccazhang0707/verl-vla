# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared GR00T critic module group (heads + optional cross-attn / pool proj)."""

from __future__ import annotations

import copy
from typing import Literal

import torch
from torch import nn

from .mlp import CriticMLP


class Gr00tCriticGroup(nn.Module):
    """Ensemble critic heads over pooled VL features + state + normalised action."""

    def __init__(
        self,
        *,
        head_num: int,
        input_dim: int,
        backbone_feature_dim: int,
        pooling: str = "attn",
        prefix_attn_heads: int = 8,
        layernorm: bool = False,
        pool_proj_dim: int = 0,
        use_encoded_state: bool = False,
        state_real_dim: int | None = None,
        action_horizon: int,
        action_dim: int,
        mask_frozen_action: bool = False,
        privileged_obs: bool = False,
        privileged_obs_dim: int = 0,
        sac_action_train_mask: torch.Tensor | None = None,
        sac_action_train_all: bool = True,
    ) -> None:
        super().__init__()
        self.pooling = str(pooling).lower()
        self.use_encoded_state = bool(use_encoded_state)
        self.state_real_dim = int(state_real_dim) if state_real_dim is not None else None
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.mask_frozen_action = bool(mask_frozen_action)
        self.privileged_obs = bool(privileged_obs)
        self.privileged_obs_dim = int(privileged_obs_dim) if self.privileged_obs else 0
        self.sac_action_train_all = bool(sac_action_train_all)
        if sac_action_train_mask is not None:
            self.register_buffer("sac_action_train_mask", sac_action_train_mask.clone(), persistent=False)
        else:
            self.sac_action_train_mask = None

        self.critic_heads = nn.ModuleList(
            [CriticMLP(input_dim, use_layernorm=layernorm) for _ in range(head_num)]
        )
        self.target_critic_heads = copy.deepcopy(self.critic_heads)
        for p in self.target_critic_heads.parameters():
            p.requires_grad_(False)

        if pool_proj_dim > 0:
            self.critic_pool_proj = nn.Linear(backbone_feature_dim, pool_proj_dim)
            self.target_pool_proj = copy.deepcopy(self.critic_pool_proj)
            for p in self.target_pool_proj.parameters():
                p.requires_grad_(False)
        else:
            self.critic_pool_proj = None
            self.target_pool_proj = None

        if self.pooling == "attn":
            d = backbone_feature_dim
            # Learnable cross-attn query token held as nn.Embedding (NOT a bare Parameter)
            # so from_pretrained's _fast_init reaches + initialises it (avoids NaN query).
            self.critic_state_token = nn.Embedding(1, d)
            self.target_state_token = nn.Embedding(1, d)
            nn.init.normal_(self.critic_state_token.weight, mean=0.0, std=0.02)
            self.target_state_token.load_state_dict(self.critic_state_token.state_dict())
            for p in self.target_state_token.parameters():
                p.requires_grad_(False)
            self.critic_prefix_cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=prefix_attn_heads, batch_first=True
            )
            self.target_prefix_cross_attn = nn.MultiheadAttention(
                embed_dim=d, num_heads=prefix_attn_heads, batch_first=True
            )
            self.target_prefix_cross_attn.load_state_dict(self.critic_prefix_cross_attn.state_dict())
            for p in self.target_prefix_cross_attn.parameters():
                p.requires_grad_(False)
        else:
            self.critic_state_token = None
            self.target_state_token = None
            self.critic_prefix_cross_attn = None
            self.target_prefix_cross_attn = None

        if self.privileged_obs and self.privileged_obs_dim > 0:
            self.register_buffer(
                "priv_obs_running_mean", torch.zeros(self.privileged_obs_dim, dtype=torch.float32)
            )
            self.register_buffer(
                "priv_obs_running_var", torch.ones(self.privileged_obs_dim, dtype=torch.float32)
            )
            self.register_buffer("priv_obs_running_count", torch.zeros((), dtype=torch.float64))

    def _cross_attention_pool(
        self, vl_embeds: torch.Tensor, attn_mask: torch.Tensor, use_target_network: bool
    ) -> torch.Tensor:
        cross_attn = self.target_prefix_cross_attn if use_target_network else self.critic_prefix_cross_attn
        state_token = (self.target_state_token if use_target_network else self.critic_state_token).weight
        batch_size = vl_embeds.shape[0]
        mask_b = attn_mask.unsqueeze(-1).to(torch.bool)
        vl_safe = torch.where(mask_b, vl_embeds, torch.zeros_like(vl_embeds))
        query = state_token.view(1, 1, -1).expand(batch_size, -1, -1)
        key_padding_mask = ~attn_mask.to(dtype=torch.bool)
        pooled, _ = cross_attn(
            query=query,
            key=vl_safe,
            value=vl_safe,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return pooled.squeeze(1)

    def _normalize_priv_obs(self, priv: torch.Tensor, update: bool) -> torch.Tensor:
        eps = 1e-5
        if update and torch.is_grad_enabled():
            with torch.no_grad():
                finite = torch.isfinite(priv).all(dim=-1)
                x = priv[finite].to(torch.float64)
                n = x.shape[0]
                if n > 0:
                    batch_mean = x.mean(dim=0)
                    batch_var = x.var(dim=0, unbiased=False)
                    count = self.priv_obs_running_count
                    new_count = count + n
                    delta = batch_mean - self.priv_obs_running_mean.to(torch.float64)
                    new_mean = self.priv_obs_running_mean.to(torch.float64) + delta * (n / new_count)
                    m_a = self.priv_obs_running_var.to(torch.float64) * count
                    m_b = batch_var * n
                    m2 = m_a + m_b + (delta**2) * count * n / new_count
                    new_var = m2 / new_count
                    self.priv_obs_running_mean.copy_(new_mean.to(self.priv_obs_running_mean.dtype))
                    self.priv_obs_running_var.copy_(new_var.to(self.priv_obs_running_var.dtype))
                    self.priv_obs_running_count.copy_(new_count)
        mean = self.priv_obs_running_mean.to(priv.dtype)
        std = (self.priv_obs_running_var.to(priv.dtype) + eps).sqrt()
        return (priv - mean) / std

    @staticmethod
    def _action_from_dict(a: dict[str, torch.Tensor]) -> torch.Tensor:
        return a.get("full_action", a["action"])

    def _critic_input(
        self, a: dict[str, torch.Tensor], sf: dict[str, torch.Tensor], use_target_network: bool = False
    ) -> torch.Tensor:
        if self.pooling == "attn":
            pooled = self._cross_attention_pool(
                sf["backbone_features"], sf["backbone_attention_mask"], use_target_network
            )
        else:
            pooled = sf["pooled"]
        if self.critic_pool_proj is not None:
            proj = self.target_pool_proj if use_target_network else self.critic_pool_proj
            pooled = proj(pooled)
        batch_size = pooled.shape[0]
        state_src = sf["state_features"] if self.use_encoded_state else sf["state"]
        if (not self.use_encoded_state) and self.state_real_dim is not None:
            state_src = state_src[..., : self.state_real_dim]
        state_flat = state_src.reshape(batch_size, -1)
        full_action = self._action_from_dict(a).to(pooled.dtype)
        act = full_action[:, : self.action_horizon, : self.action_dim]
        if self.mask_frozen_action and not self.sac_action_train_all and self.sac_action_train_mask is not None:
            m = self.sac_action_train_mask[: self.action_dim].view(1, 1, -1)
            act = torch.where(m, act, torch.zeros_like(act))
        act = act.reshape(batch_size, -1)
        parts = [pooled, state_flat, act]
        if self.privileged_obs:
            priv = sf.get("priv_obs")
            if priv is not None:
                priv_flat = priv.reshape(batch_size, -1).to(pooled.dtype)
                priv_flat = self._normalize_priv_obs(priv_flat, update=not use_target_network)
            else:
                priv_flat = pooled.new_zeros(batch_size, self.privileged_obs_dim)
            parts.insert(2, priv_flat)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        a: dict[str, torch.Tensor],
        state_features: dict[str, torch.Tensor],
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        heads = self.target_critic_heads if use_target_network else self.critic_heads
        for p in heads.parameters():
            p.requires_grad_(requires_grad)
        if self.critic_pool_proj is not None:
            proj = self.target_pool_proj if use_target_network else self.critic_pool_proj
            for p in proj.parameters():
                p.requires_grad_(requires_grad)
        if self.pooling == "attn":
            attn = self.target_prefix_cross_attn if use_target_network else self.critic_prefix_cross_attn
            for p in attn.parameters():
                p.requires_grad_(requires_grad)
            tok = self.target_state_token if use_target_network else self.critic_state_token
            tok.requires_grad_(requires_grad)

        critic_input = self._critic_input(a, state_features, use_target_network)
        q_vals = torch.cat([h(critic_input) for h in heads], dim=-1)
        if method == "min":
            return q_vals.min(dim=-1).values
        return q_vals

    def get_critic_parameters(self) -> list[torch.nn.Parameter]:
        params = list(self.critic_heads.parameters())
        if self.critic_pool_proj is not None:
            params += list(self.critic_pool_proj.parameters())
        if self.pooling == "attn":
            params += list(self.critic_prefix_cross_attn.parameters())
            params += list(self.critic_state_token.parameters())
        return params

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        for p_online, p_target in zip(
            self.critic_heads.parameters(), self.target_critic_heads.parameters(), strict=True
        ):
            p_target.data.lerp_(p_online.data, tau)
        if self.critic_pool_proj is not None:
            for p_online, p_target in zip(
                self.critic_pool_proj.parameters(), self.target_pool_proj.parameters(), strict=True
            ):
                p_target.data.lerp_(p_online.data, tau)
        if self.pooling == "attn":
            for p_online, p_target in zip(
                self.critic_prefix_cross_attn.parameters(),
                self.target_prefix_cross_attn.parameters(),
                strict=True,
            ):
                p_target.data.lerp_(p_online.data, tau)
            for p_online, p_target in zip(
                self.critic_state_token.parameters(), self.target_state_token.parameters(), strict=True
            ):
                p_target.data.lerp_(p_online.data, tau)

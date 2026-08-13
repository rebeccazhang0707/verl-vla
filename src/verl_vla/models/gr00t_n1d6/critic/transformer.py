# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""GR00T SAC Transformer sequence critic (posttrain reference parity).

Mirrors the posttrain DSRL critic (``modules/transformer/critic.py``): each
ensemble member is a Transformer over the token sequence
``[obs_token, action_1, ..., action_K]`` — the observation embedding is the
first token and every steered flow step contributes one action token — pooled
(attention pooling by default) into a scalar Q value.

The observation token comes from GR00T's own frozen VL features (the same
``pooled`` mean-pooled prefix + raw ``state`` the DSRL noise actor consumes), so
actor and critic share one feature source rather than a separate DINOv2 tower.
The action tokens are the steering noise — the SAC action space under DSRL —
which :meth:`DSRLSteering.select_critic_noise` has already pulled out of the
``steering_noise`` slot and handed over under ``action``. The backend keeps the
exact interface of :class:`Gr00tCriticGroup` (``forward`` cat/min, target
network, polyak update, parameter enumeration) so the SAC training worker is
unchanged.
"""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (posttrain parity)."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class _TransformerCriticMember(nn.Module):
    """One Transformer Q-function over ``[obs_token, action tokens]``."""

    def __init__(
        self,
        *,
        obs_dim: int,
        action_horizon: int,
        action_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dropout: float,
        activation: str,
        action_embedding_dim: int,
        pooling_strategy: str,
    ) -> None:
        super().__init__()
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.pooling_strategy = str(pooling_strategy).lower()

        self.obs_projection = nn.Linear(obs_dim, d_model)
        self.action_embedding = nn.Linear(self.action_dim, action_embedding_dim)
        self.action_projection = nn.Linear(action_embedding_dim, d_model)

        self.input_norm = nn.LayerNorm(d_model)
        self.position_embedding = _PositionalEncoding(d_model, max_len=self.action_horizon + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, activation=activation, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model)
        )
        if self.pooling_strategy == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False)
            )
        else:
            self.attention_pool = None
        self.value_head = nn.Linear(d_model, 1)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "first":
            return tokens[:, 0, :]
        if self.pooling_strategy == "attention":
            weights = F.softmax(self.attention_pool(tokens), dim=1)
            return (tokens * weights).sum(dim=1)
        if self.pooling_strategy == "weighted_mean":
            seq_len = tokens.size(1)
            pos = torch.arange(seq_len, device=tokens.device, dtype=tokens.dtype)
            w = torch.exp(pos * 0.1)
            w[0] = w.max()
            w = (w / w.sum()).view(1, -1, 1)
            return (tokens * w).sum(dim=1)
        return tokens.mean(dim=1)

    def forward(self, pooled: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dtype = self.value_head.weight.dtype
        batch_size = pooled.shape[0]
        pooled = pooled.reshape(batch_size, -1).to(dtype)
        state = state.reshape(batch_size, -1).to(dtype)
        action = action[:, : self.action_horizon, : self.action_dim].to(dtype)

        obs_token = self.obs_projection(torch.cat([pooled, state], dim=-1)).unsqueeze(1)  # [B, 1, d]
        action_tokens = self.action_projection(self.action_embedding(action))  # [B, K, d]
        tokens = torch.cat([obs_token, action_tokens], dim=1)
        tokens = self.input_norm(tokens)
        tokens = self.position_embedding(tokens)
        tokens = self.transformer_encoder(tokens)
        return self.value_head(self._pool(tokens))  # [B, 1]


class Gr00tTransformerCriticGroup(nn.Module):
    """Ensemble of Transformer sequence critics over GR00T VL features + noise."""

    def __init__(
        self,
        *,
        head_num: int,
        obs_dim: int,
        action_horizon: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "gelu",
        action_embedding_dim: int = 128,
        pooling_strategy: str = "attention",
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"transformer critic d_model ({d_model}) must be divisible by nhead ({nhead})")
        if activation not in ("relu", "gelu"):
            raise ValueError(f"transformer critic activation must be 'relu' or 'gelu', got {activation}")

        def _member() -> _TransformerCriticMember:
            return _TransformerCriticMember(
                obs_dim=obs_dim,
                action_horizon=action_horizon,
                action_dim=action_dim,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dropout=dropout,
                activation=activation,
                action_embedding_dim=action_embedding_dim,
                pooling_strategy=pooling_strategy,
            )

        self.critic_members = nn.ModuleList([_member() for _ in range(int(head_num))])
        self.target_members = copy.deepcopy(self.critic_members)
        for p in self.target_members.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _action_from_dict(a: dict[str, torch.Tensor]) -> torch.Tensor:
        return a["full_action"] if "full_action" in a else a["action"]

    def forward(
        self,
        a: dict[str, torch.Tensor],
        state_features: dict[str, torch.Tensor],
        *,
        use_target_network: bool = False,
        method: Literal["cat", "min"] = "cat",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        members = self.target_members if use_target_network else self.critic_members
        for p in members.parameters():
            p.requires_grad_(requires_grad)

        pooled = state_features["pooled"]
        state = state_features["state"]
        action = self._action_from_dict(a)
        q_vals = torch.cat([m(pooled, state, action) for m in members], dim=-1)  # [B, head_num]
        if method == "min":
            return q_vals.min(dim=-1).values
        return q_vals

    def get_critic_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.critic_members.parameters())

    @torch.no_grad()
    def update_target_network(self, tau: float) -> None:
        for p_online, p_target in zip(self.critic_members.parameters(), self.target_members.parameters(), strict=True):
            p_target.data.lerp_(p_online.data, tau)


__all__ = ["Gr00tTransformerCriticGroup"]

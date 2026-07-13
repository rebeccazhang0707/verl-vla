# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""GR00T SAC critic MLP (SiLU + optional LayerNorm)."""

from __future__ import annotations

from torch import nn


class CriticMLP(nn.Module):
    """Two-layer MLP: input_dim -> 512 -> 256 -> 1.

    When ``use_layernorm`` is set, a ``LayerNorm`` is inserted after each hidden
    ``Linear`` and before the activation (DroQ/REDQ-style "LayerNorm critic").
    Disabled by default so existing checkpoints stay bit-identical.
    """

    def __init__(self, input_dim: int, use_layernorm: bool = False):
        super().__init__()
        if use_layernorm:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            )

    def forward(self, x):
        return self.net(x)

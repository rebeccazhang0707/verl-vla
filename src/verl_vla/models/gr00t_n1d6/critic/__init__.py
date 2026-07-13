# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""GR00T SAC critic backends.

Supported types: ``cross_attn`` and ``mean_pool`` (both ``uses_task_ids=False``).

Unlike pi05 LIBERO SAC (``multi_cross_attn`` + per-sample ``task_ids``), Arena
GR00T launchers train one task at a time (GR1 fridge, or a single LIBERO
suite/id). A shared single critic is therefore enough; do not add multitask
backends unless a true multi-task Arena recipe lands.
"""

from .base import CriticBackend
from .critic_cross_attn import CrossAttentionCriticBackend, MeanPoolCriticBackend
from .group import Gr00tCriticGroup
from .mlp import CriticMLP

__all__ = [
    "CriticBackend",
    "CriticMLP",
    "CrossAttentionCriticBackend",
    "Gr00tCriticGroup",
    "MeanPoolCriticBackend",
]

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""GR00T SAC critic backends.

Supported types: ``cross_attn``, ``mean_pool`` and ``transformer`` (all
``uses_task_ids=False``). ``transformer`` is the posttrain-parity DSRL critic.

Unlike pi05 LIBERO SAC (``multi_cross_attn`` + per-sample ``task_ids``), Arena
GR00T launchers train one task at a time (GR1 fridge, or a single LIBERO
suite/id). A shared single critic is therefore enough; do not add multitask
backends unless a true multi-task Arena recipe lands.
"""

from .backends import CrossAttentionCriticBackend, MeanPoolCriticBackend, TransformerCriticBackend
from .base import CriticBackend
from .group import Gr00tCriticGroup
from .mlp import CriticMLP
from .transformer import Gr00tTransformerCriticGroup

__all__ = [
    "CriticBackend",
    "CriticMLP",
    "CrossAttentionCriticBackend",
    "Gr00tCriticGroup",
    "Gr00tTransformerCriticGroup",
    "MeanPoolCriticBackend",
    "TransformerCriticBackend",
]

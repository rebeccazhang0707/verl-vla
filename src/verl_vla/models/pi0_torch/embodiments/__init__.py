# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""PI0 mappings for simulator and robot embodiments."""

from .arena import ArenaPi0Input, ArenaPi0Output
from .base import Pi0Input, Pi0Output
from .lerobot import LerobotPi0Input, LerobotPi0Output
from .libero import LiberoPi0Input, LiberoPi0Output

_PI0_EMBODIMENT_REGISTRY = {
    "arena": (ArenaPi0Input, ArenaPi0Output),
    "libero": (LiberoPi0Input, LiberoPi0Output),
    "lerobot": (LerobotPi0Input, LerobotPi0Output),
}


def get_pi0_embodiment_classes(embodiment: str) -> tuple[type[Pi0Input], type[Pi0Output]]:
    """Resolve the I/O mapping used for a simulator or robot embodiment.

    Older checkpoints using ``policy_type`` are normalized by ``PI0AdapterConfig``.
    """

    try:
        return _PI0_EMBODIMENT_REGISTRY[embodiment]
    except KeyError as exc:
        supported = ", ".join(sorted(_PI0_EMBODIMENT_REGISTRY))
        raise ValueError(f"Unknown pi0 embodiment: {embodiment}. Supported values: {supported}") from exc


__all__ = [
    "ArenaPi0Input",
    "ArenaPi0Output",
    "LerobotPi0Input",
    "LerobotPi0Output",
    "LiberoPi0Input",
    "LiberoPi0Output",
    "Pi0Input",
    "Pi0Output",
    "get_pi0_embodiment_classes",
]

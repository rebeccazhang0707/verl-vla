# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Environment adapters for the external GR00T N1.6 policy."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .arena_policy import ArenaGr00tInput, ArenaGr00tOutput
from .base import Gr00tInput, Gr00tOutput
from .libero_policy import LiberoGr00tInput, LiberoGr00tOutput, load_libero_statistics

_GR00T_POLICY_REGISTRY = {
    "arena": (ArenaGr00tInput, ArenaGr00tOutput),
    "libero": (LiberoGr00tInput, LiberoGr00tOutput),
}

# Embodiment-tag → optional statistics loader (flat LeRobot → nested processor stats).
_STATISTICS_LOADERS: dict[str, Callable[[str | Path], dict[str, Any]]] = {
    "libero_panda": load_libero_statistics,
}


def get_gr00t_policy_classes(policy_type: str) -> tuple[type[Gr00tInput], type[Gr00tOutput]]:
    try:
        return _GR00T_POLICY_REGISTRY[policy_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_GR00T_POLICY_REGISTRY))
        raise ValueError(f"Unknown gr00t policy_type: {policy_type}. Supported values: {supported}") from exc


def get_statistics_loader(
    embodiment_tag: str | None,
) -> Callable[[str | Path], dict[str, Any]] | None:
    """Return an embodiment-specific norm-stats loader, if one is registered."""
    if not embodiment_tag:
        return None
    return _STATISTICS_LOADERS.get(embodiment_tag)


__all__ = [
    "LiberoGr00tInput",
    "LiberoGr00tOutput",
    "Gr00tInput",
    "Gr00tOutput",
    "ArenaGr00tInput",
    "ArenaGr00tOutput",
    "get_gr00t_policy_classes",
    "get_statistics_loader",
]

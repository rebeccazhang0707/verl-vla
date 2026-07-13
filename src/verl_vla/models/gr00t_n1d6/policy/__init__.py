# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Environment adapters for the external GR00T N1.6 policy."""

from .arena_policy import ArenaGr00tInput, ArenaGr00tOutput
from .base import Gr00tPolicyInput, Gr00tPolicyOutput
from .libero_policy import LiberoGr00tInput, LiberoGr00tOutput
from .sac_io import Gr00tInput, Gr00tOutput

_GR00T_POLICY_REGISTRY = {
    "arena": (ArenaGr00tInput, ArenaGr00tOutput),
}


def get_gr00t_policy_classes(policy_type: str) -> tuple[type[Gr00tInput], type[Gr00tOutput]]:
    try:
        return _GR00T_POLICY_REGISTRY[policy_type]
    except KeyError as exc:
        supported = ", ".join(sorted(_GR00T_POLICY_REGISTRY))
        raise ValueError(f"Unknown gr00t policy_type: {policy_type}. Supported values: {supported}") from exc


__all__ = [
    "Gr00tPolicyInput",
    "Gr00tPolicyOutput",
    "LiberoGr00tInput",
    "LiberoGr00tOutput",
    "Gr00tInput",
    "Gr00tOutput",
    "ArenaGr00tInput",
    "ArenaGr00tOutput",
    "get_gr00t_policy_classes",
]

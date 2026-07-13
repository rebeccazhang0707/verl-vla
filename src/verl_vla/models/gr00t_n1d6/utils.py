# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GR00T N1.6 fallback constants + small helpers (gr00t-package-free).

Authoritative dims / embodiment_id come from the loaded checkpoint; these are
defaults for config fallbacks and the one Arena joint-space mapping we ship (GR1).
"""

import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "GR00TDim",
    "EmbodimentSpec",
    "GR1",
    "EMBODIMENTS",
    "load_embodiment_id",
    "split_flat_state_to_groups",
]


class GR00TDim(IntEnum):
    """Fallback model-level Gr00tN1d6 padded dims (embodiment-agnostic)."""

    ACTION_HORIZON = 16
    MAX_STATE_DIM = 128
    MAX_ACTION_DIM = 128
    STATE_HORIZON = 1


@dataclass(frozen=True)
class EmbodimentSpec:
    """Per-embodiment fallback metadata (projector id + joint group layout)."""

    name: str
    embodiment_id: int
    state_group_dims: "OrderedDict[str, int]"

    @property
    def action_dim(self) -> int:
        return sum(self.state_group_dims.values())


GR1 = EmbodimentSpec(
    name="gr1",
    embodiment_id=20,
    state_group_dims=OrderedDict(left_arm=7, right_arm=7, left_hand=6, right_hand=6),
)
assert GR1.action_dim == 26

EMBODIMENTS: "dict[str, EmbodimentSpec]" = {GR1.name: GR1}

_EMBODIMENT_ID_FILENAME = "embodiment_id.json"


def load_embodiment_id(tag: str, model_path: Optional[str] = None) -> int:
    """Projector index for ``tag``: checkpoint ``embodiment_id.json``, else :data:`EMBODIMENTS`."""
    if model_path:
        path = os.path.join(model_path, _EMBODIMENT_ID_FILENAME)
        try:
            with open(path) as f:
                mapping = json.load(f)
            if tag in mapping:
                return int(mapping[tag])
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s (%s); using EMBODIMENTS fallback", path, exc)
    try:
        return EMBODIMENTS[tag].embodiment_id
    except KeyError as exc:
        raise KeyError(f"Unknown embodiment tag {tag!r}; known: {list(EMBODIMENTS)}") from exc


def split_flat_state_to_groups(
    state_flat: np.ndarray,
    group_dims: "OrderedDict[str, int]",
) -> "OrderedDict[str, np.ndarray]":
    """Split ``(B, D)`` flat policy-order state into ``{group: (B, d)}``."""
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    start = 0
    for key, dim in group_dims.items():
        out[key] = state_flat[..., start : start + dim]
        start += dim
    return out

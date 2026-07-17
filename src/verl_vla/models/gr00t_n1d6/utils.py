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

"""GR00T N1.6 shared helpers (gr00t-package-free).

This module holds:

* Fallback model / embodiment constants used when checkpoint metadata is absent.
* Small geometry helpers (flat state → joint groups).
* Adapter checkpoint state-dict remapping / critic extraction (formerly
  ``checkpoint_utils``).

Authoritative dims / ``embodiment_id`` still come from the loaded checkpoint when
available; the constants below are defaults for config fallbacks and the Arena
joint-space mappings we ship (e.g. GR1).
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "GR00TDim",
    "EmbodimentSpec",
    "GR1",
    "LIBERO_PANDA",
    "EMBODIMENTS",
    "load_embodiment_id",
    "split_flat_state_to_groups",
    "normalize_adapter_state_dict",
    "extract_critic_state_dict",
]


class GR00TDim(IntEnum):
    """Fallback model-level Gr00tN1d6 padded dims (embodiment-agnostic).

    Used when adapter / config does not supply horizon or max state/action dims.
    Values match the stock N1.6 layout (padded to 128).
    """

    ACTION_HORIZON = 16
    MAX_STATE_DIM = 128
    MAX_ACTION_DIM = 128
    STATE_HORIZON = 1


@dataclass(frozen=True)
class EmbodimentSpec:
    """Per-embodiment fallback metadata (projector id + joint group layout).

    Attributes:
        name: Short tag used in configs / ``EMBODIMENTS`` lookup (e.g. ``\"gr1\"``).
        embodiment_id: Projector / embodiment index for the policy head.
        state_group_dims: Ordered map of joint-group name → dimension; order
            defines the flat state/action packing used by Arena-style envs.
    """

    name: str
    embodiment_id: int
    state_group_dims: OrderedDict[str, int]

    @property
    def action_dim(self) -> int:
        """Total action/state width as the sum of ``state_group_dims`` values."""
        return sum(self.state_group_dims.values())


################################################################################
# GR1 humanoid: 26-DoF joint layout used by the Arena GR1 mapping.
GR1 = EmbodimentSpec(
    name="gr1",
    embodiment_id=20,
    state_group_dims=OrderedDict(left_arm=7, right_arm=7, left_hand=6, right_hand=6),
)
assert GR1.action_dim == 26

# LIBERO Panda: state is 8-d (gripper width 2), action is 7-d (gripper command 1).
# ``EmbodimentSpec.action_dim`` sums state groups; prefer Adapter-derived action dims.
LIBERO_PANDA = EmbodimentSpec(
    name="libero_panda",
    embodiment_id=2,
    state_group_dims=OrderedDict(x=1, y=1, z=1, roll=1, pitch=1, yaw=1, gripper=2),
)

# Registry of known embodiment tags → fallback specs.
EMBODIMENTS: dict[str, EmbodimentSpec] = {
    GR1.name: GR1,
    LIBERO_PANDA.name: LIBERO_PANDA,
}

# Filename written next to a checkpoint with ``{tag: embodiment_id}`` mapping.
_EMBODIMENT_ID_FILENAME = "embodiment_id.json"
################################################################################


def load_embodiment_id(tag: str, model_path: Optional[str] = None) -> int:
    """Resolve projector ``embodiment_id`` for ``tag``.

    Prefers ``{model_path}/embodiment_id.json`` when present; otherwise falls
    back to :data:`EMBODIMENTS`.

    Args:
        tag: Embodiment name (e.g. ``\"gr1\"``, ``\"libero_panda\"``).
        model_path: Optional checkpoint / model directory containing
            ``embodiment_id.json``.

    Returns:
        Integer projector index for ``tag``.

    Raises:
        KeyError: If ``tag`` is missing from both the JSON file and
            :data:`EMBODIMENTS`.
    """
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
    group_dims: OrderedDict[str, int],
) -> OrderedDict[str, np.ndarray]:
    """Split a flat policy-order state into named joint groups.

    Args:
        state_flat: Array of shape ``(..., D)`` in the packed group order.
        group_dims: Ordered ``{group_name: dim}``; iteration order defines slices.

    Returns:
        OrderedDict mapping each group name to a view of shape ``(..., d)``.
    """
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    start = 0
    for key, dim in group_dims.items():
        out[key] = state_flat[..., start : start + dim]
        start += dim
    return out


def normalize_adapter_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap legacy critic key prefixes to the current ``critic.*`` layout.

    Aligns with pi05 ``load_state_dict`` conventions so older adapter checkpoints
    load cleanly:

    * ``critic_backend.*`` → ``critic.*``
    * ``auxiliary_modules.critic.*`` → ``critic.*``

    Keys already under ``critic.`` (or unrelated modules) are left unchanged.

    Args:
        state_dict: Raw adapter (or full) state dict to normalize.

    Returns:
        New dict with remapped keys; tensor values are shared references.
    """
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("critic_backend."):
            key = f"critic.{key.removeprefix('critic_backend.')}"
        elif key.startswith("auxiliary_modules.critic."):
            key = f"critic.{key.removeprefix('auxiliary_modules.critic.')}"
        normalized[key] = value
    return normalized


def extract_critic_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Extract critic (+ target) weights from a full adapter state dict.

    Selects keys starting with ``critic.`` and strips that prefix so the result
    can be passed to a standalone critic module's ``load_state_dict``.

    Args:
        state_dict: Preferably a normalized adapter state dict (see
            :func:`normalize_adapter_state_dict`).

    Returns:
        Dict of critic submodule keys → tensors (prefix removed).
    """
    prefix = "critic."
    return {name.removeprefix(prefix): value for name, value in state_dict.items() if name.startswith(prefix)}

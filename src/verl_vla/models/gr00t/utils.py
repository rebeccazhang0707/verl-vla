# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""GR00T **N1.6** embodiment specs + state helpers (gr00t-package-free).

These constants and pure-numpy utilities are split out of ``gr00t_policy.py`` so
they can be imported (and unit-tested) without pulling in the gr00t package or a
checkpoint. They are *fallback defaults / layout metadata* only — authoritative
dims come from the loaded checkpoint config / processor.
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
    "JointSpaceYamls",
    "EmbodimentSpec",
    "GR1",
    "EMBODIMENTS",
    "EMBODIMENT_ID_FALLBACK",
    "GR1_STATE_GROUP_DIMS",
    "get_embodiment_spec",
    "load_embodiment_id",
    "split_flat_state_to_groups",
]


class GR00TDim(IntEnum):
    """Fallback **model-level** ``Gr00tN1d6`` dims (embodiment-agnostic).

    Authoritative values live in the checkpoint config (this export:
    action_horizon=50, max_state_dim=128, max_action_dim=128,
    backbone_embedding_dim=2048).

    NOTE: the *real* (unpadded) action width is embodiment-specific and lives on
    :class:`EmbodimentSpec` (e.g. ``GR1.action_dim``), not here.
    """
    # Env-side default chunk-count fallback only — NOT the checkpoint action horizon.
    # The real checkpoint action_horizon is 50 (read from config.json / passed via
    # model.override_config.action_horizon). This 16 is used solely as a last-resort
    # default for ``num_action_chunks`` / the guard upper bound when neither the model
    # config nor override_config supplies a value; run scripts MUST set
    # ``+actor_rollout_ref.model.override_config.action_horizon=50`` so the guard's
    # upper bound matches the model and does not falsely reject chunks in (16, 50].
    ACTION_HORIZON = 16
    MAX_STATE_DIM = 128
    MAX_ACTION_DIM = 128
    STATE_HORIZON = 1


@dataclass(frozen=True)
class JointSpaceYamls:
    """Joint-space YAML filenames defining an embodiment's DOF layouts.

    These are the **single source of truth** for the policy ⇄ sim index tables
    (see ``verl_vla.envs.arena_env``); they ship under
    ``isaaclab_arena_gr00t/embodiments/<tag>``:
        policy: GR00T policy joint space (group -> [joint_name, ...]).
        action: full sim action joint space (joint_name -> column index).
        state:  full sim state joint space (joint_name -> column index).
    """

    policy: str
    action: str
    state: str


@dataclass(frozen=True)
class EmbodimentSpec:
    """Embodiment-specific GR00T constants, grouped in one place.

    Fallback defaults only — authoritative values come from the checkpoint
    (``embodiment_id.json`` for ``embodiment_id``, the ``*_joint_space.yaml`` /
    modality config for ``state_group_dims`` and the action width).

    Fields:
        name:              embodiment tag (matches ``gr00t.data.EmbodimentTag``).
        embodiment_id:     projector index into the action head.
        state_group_dims:  per-modality split of the flat policy-order state/action
                           vector, in joint-space yaml order.
        joint_space_yamls: filenames of this embodiment's joint-space YAMLs (policy /
                           action / state DOF layouts); ``None`` if not applicable.
    """

    name: str
    embodiment_id: int
    state_group_dims: "OrderedDict[str, int]"
    joint_space_yamls: Optional[JointSpaceYamls] = None

    @property
    def action_dim(self) -> int:
        """Real (unpadded) action width = sum of the per-group joint dims."""
        return sum(self.state_group_dims.values())

    @property
    def state_dim(self) -> int:
        return sum(self.state_group_dims.values())


# ---------------------------------------------------------------------------
# embodiment_id (projector index) resolution
# ---------------------------------------------------------------------------

# Backup copy of a GR00T checkpoint export's ``embodiment_id.json`` (tag -> projector
# index). embodiment_id is NOT in the joint-space YAMLs, so it cannot be derived; the
# authoritative source is the checkpoint's own file (see ``load_embodiment_id``) and
# this table is only the fallback when that file is unavailable.
EMBODIMENT_ID_FALLBACK: "dict[str, int]" = {
    "robocasa_panda_omron": 13,
    "gr1": 20,
    "behavior_r1_pro": 24,
    "unitree_g1": 8,
    "oxe_google": 0,
    "oxe_widowx": 1,
    "libero_panda": 2,
    "oxe_droid": 16,
    "new_embodiment": 10,
}
_EMBODIMENT_ID_FILENAME = "embodiment_id.json"


def load_embodiment_id(tag: str, model_path: Optional[str] = None) -> int:
    """Resolve an embodiment's projector index (embodiment_id).

    Authoritative source = the checkpoint's ``<model_path>/embodiment_id.json``.
    Falls back to the copied :data:`EMBODIMENT_ID_FALLBACK` table when no
    ``model_path`` is given or that file is missing / unreadable / lacks the tag.
    """
    if model_path:
        path = os.path.join(model_path, _EMBODIMENT_ID_FILENAME)
        try:
            with open(path) as f:
                mapping = json.load(f)
            if tag in mapping:
                return int(mapping[tag])
            logger.warning("%s has no tag %r; using fallback embodiment_id table", path, tag)
        except FileNotFoundError:
            logger.warning("%s not found; using fallback embodiment_id table", path)
        except Exception as exc:  # noqa: BLE001 - never let a bad json break loading
            logger.warning("Failed to read %s (%s); using fallback table", path, exc)
    try:
        return int(EMBODIMENT_ID_FALLBACK[tag])
    except KeyError as exc:
        raise KeyError(
            f"Unknown embodiment tag {tag!r}; known tags: {list(EMBODIMENT_ID_FALLBACK)}"
        ) from exc


# GR1 arms-only: joint groups in the order of gr00t_26dof_joint_space.yaml, plus
# the projector index from the embodiment_id table (``"gr1": 20``) and the Arena
# joint-space YAML filenames (26-DOF policy / 36-DOF action / 54-DOF state).
GR1 = EmbodimentSpec(
    name="gr1",
    embodiment_id=EMBODIMENT_ID_FALLBACK["gr1"],
    state_group_dims=OrderedDict(left_arm=7, right_arm=7, left_hand=6, right_hand=6),
    joint_space_yamls=JointSpaceYamls(
        policy="gr00t_26dof_joint_space.yaml",
        action="36dof_joint_space.yaml",
        state="54dof_joint_space.yaml",
    ),
)
assert GR1.action_dim == 26


# Registry + lookup so callers can resolve a spec from an embodiment tag.
EMBODIMENTS: "dict[str, EmbodimentSpec]" = {GR1.name: GR1}


def get_embodiment_spec(tag: str) -> EmbodimentSpec:
    try:
        return EMBODIMENTS[tag]
    except KeyError as exc:
        raise KeyError(
            f"Unknown embodiment tag {tag!r}; known tags: {list(EMBODIMENTS)}"
        ) from exc


# Backward-compatible alias (derive from the spec; do not duplicate values).
GR1_STATE_GROUP_DIMS: "OrderedDict[str, int]" = GR1.state_group_dims


def split_flat_state_to_groups(
    state_flat: np.ndarray,
    group_dims: "OrderedDict[str, int]" = GR1_STATE_GROUP_DIMS,
) -> "OrderedDict[str, np.ndarray]":
    """Split (B, D) flat policy-order joints into {group: (B, d)} per modality key."""
    assert state_flat.shape[-1] == sum(group_dims.values()), (
        f"flat state width {state_flat.shape[-1]} != sum(group_dims)={sum(group_dims.values())}"
    )
    out: OrderedDict[str, np.ndarray] = OrderedDict()
    start = 0
    for key, dim in group_dims.items():
        out[key] = state_flat[..., start : start + dim]
        start += dim
    return out

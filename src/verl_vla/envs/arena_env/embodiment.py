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

"""GR1 embodiment ⇄ Isaac Lab Arena sim joint-space layout.

The index tables are **derived from the Arena embodiment YAMLs** rather than
hand-copied, so they stay in sync with Arena. We replicate the exact name-based
lookup of ``isaaclab_arena_gr00t.utils.joints_conversion``:

    policy (26-DOF, gr00t_26dof_joint_space.yaml): group -> [joint_name, ...]
    state  (54-DOF, 54dof_joint_space.yaml):       joint_name -> column index
    action (36-DOF, 36dof_joint_space.yaml):       joint_name -> column index

Flattening the policy groups (left_arm, right_arm, left_hand, right_hand) gives
the 26 GR00T joint names in order; looking each one up in the state/action
``name -> index`` dicts yields the two index tables bundled on ``GR1_ARENA``
(an :class:`ArenaJointMapping`):

    GR1_ARENA.state_full_to_policy[i] = state_cfg[name_i]   # gather 54 -> 26
    GR1_ARENA.policy_to_action[i]     = action_cfg[name_i]  # scatter 26 -> 36

These mappings are embodiment- AND simulator-specific and are kept out of the env
wrapper (which only calls ``GR1_ARENA`` methods) so the wrapper stays generic.

The joint-space YAMLs are the **single source of truth** (no hardcoded fallback
index tables — everything is keyed by joint name). Discovery order:
  1. ``$ARENA_GR1_JOINT_SPACE_DIR``
  2. the installed ``isaaclab_arena_gr00t`` package (embodiments/gr1)
If none is found, a clear error is raised.
"""

import importlib.util
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from verl_vla.models.gr00t.utils import GR1, EmbodimentSpec

logger = logging.getLogger(__name__)

__all__ = [
    "ArenaJointMapping",
    "GR1_ARENA",
    "build_index_maps_from_yaml",
]


@dataclass(frozen=True)
class ArenaJointMapping:
    """GR00T policy-order joints ⇄ Isaac Lab Arena sim joint-space, bound to a spec.

    One cohesive object per (embodiment, simulator): it pairs the model-side
    :class:`~verl_vla.models.gr00t.utils.EmbodimentSpec` (which joint
    groups / dims the GR00T checkpoint speaks) with the simulator-specific
    gather/scatter index tables, and exposes the conversions as methods so callers
    never touch raw index lists.

    Fields (all derived from the Arena joint-space YAMLs; see module docstring):
        spec:                 model-side embodiment spec (e.g. ``GR1``).
        state_full_to_policy: gather indices, full sim state -> policy order
                              (len == ``spec.action_dim``).
        policy_to_action:     scatter indices, policy order -> full sim action
                              (len == ``spec.action_dim``).
        sim_action_dim:       width of the sim action vector.
        state_full_dim:       width of the full sim state vector.
    """

    spec: EmbodimentSpec
    state_full_to_policy: list[int]
    policy_to_action: list[int]
    sim_action_dim: int
    state_full_dim: int

    def __post_init__(self):
        assert len(self.state_full_to_policy) == self.spec.action_dim, (
            f"state map len {len(self.state_full_to_policy)} != spec.action_dim {self.spec.action_dim}"
        )
        assert len(self.policy_to_action) == self.spec.action_dim, (
            f"action map len {len(self.policy_to_action)} != spec.action_dim {self.spec.action_dim}"
        )

    @property
    def policy_dim(self) -> int:
        """Real (unpadded) policy action/state width (== ``spec.action_dim``)."""
        return self.spec.action_dim

    def gather_state(self, full_state):
        """Full sim state -> policy order. ``(B, state_full_dim) -> (B, policy_dim)``.

        Accepts a numpy array or torch tensor (name-based column gather).
        """
        return full_state[:, self.state_full_to_policy]

    def scatter_action(self, policy_action):
        """Policy action -> full sim action. ``(B, policy_dim) -> (B, sim_action_dim)``.

        Joints not controlled by the policy stay at zero; dtype/device preserved.
        Expects a torch tensor (uses ``new_zeros`` + advanced-index assignment).
        """
        sim_action = policy_action.new_zeros(policy_action.shape[0], self.sim_action_dim)
        sim_action[:, self.policy_to_action] = policy_action
        return sim_action

    def extract_action(self, sim_action):
        """Inverse of :meth:`scatter_action`. ``(B, sim_action_dim) -> (B, policy_dim)``."""
        return sim_action[:, self.policy_to_action]

# YAML file names within the gr1 embodiment dir.
_POLICY_YAML = "gr00t_26dof_joint_space.yaml"
_ACTION_YAML = "36dof_joint_space.yaml"
_STATE_YAML = "54dof_joint_space.yaml"


def _find_gr1_joint_space_dir() -> Optional[Path]:
    """Locate the Arena gr1 embodiment YAML dir (without importing Arena)."""
    env = os.environ.get("ARENA_GR1_JOINT_SPACE_DIR")
    if env and Path(env).is_dir():
        return Path(env)

    # Installed package location (use find_spec to avoid import side effects).
    try:
        spec = importlib.util.find_spec("isaaclab_arena_gr00t")
        if spec is not None and spec.submodule_search_locations:
            cand = Path(list(spec.submodule_search_locations)[0]) / "embodiments" / "gr1"
            if cand.is_dir():
                return cand
    except (ImportError, ValueError):
        pass

    return None


def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def build_index_maps_from_yaml(
    joint_dir: "str | Path",
) -> "tuple[list[int], list[int], int, int]":
    """Derive (state_54_to_26, action_26_to_36, sim_action_dim, state_full_dim).

    Replicates the name-based lookup Arena uses in ``joints_conversion``.
    """
    joint_dir = Path(joint_dir)
    policy_groups = _load_yaml(joint_dir / _POLICY_YAML)["joints"]   # group -> [name]
    action_yaml = _load_yaml(joint_dir / _ACTION_YAML)
    state_yaml = _load_yaml(joint_dir / _STATE_YAML)
    action_cfg = action_yaml["joints"]                              # name -> idx
    state_cfg = state_yaml["joints"]                                # name -> idx

    # Flatten policy groups (in YAML order) into the 26 GR00T joint names.
    flat_names = [name for names in policy_groups.values() for name in names]

    # Sanity: YAML policy grouping must match the EmbodimentSpec.
    yaml_group_dims = OrderedDict((g, len(names)) for g, names in policy_groups.items())
    if list(yaml_group_dims.items()) != list(GR1.state_group_dims.items()):
        logger.warning(
            "Arena policy YAML groups %s differ from GR1.state_group_dims %s",
            dict(yaml_group_dims), dict(GR1.state_group_dims),
        )

    state_indices = [state_cfg[name] for name in flat_names]
    action_map = [action_cfg[name] for name in flat_names]
    sim_action_dim = int(action_yaml.get("total_joints", len(action_cfg)))
    state_full_dim = int(state_yaml.get("total_joints", len(state_cfg)))
    return state_indices, action_map, sim_action_dim, state_full_dim


def _resolve_maps() -> "tuple[list[int], list[int], int, int]":
    """Derive the GR1 ⇄ Arena-sim index tables from the joint-space YAMLs.

    The YAMLs are the single source of truth (no hardcoded fallback): the
    embodiment ships them under ``isaaclab_arena_gr00t/embodiments/gr1`` so discovery
    should always succeed. A clear error is raised otherwise.
    """
    joint_dir = _find_gr1_joint_space_dir()
    if joint_dir is None:
        raise RuntimeError(
            "Arena GR1 joint-space YAMLs not found; cannot derive joint maps. "
            "Set ARENA_GR1_JOINT_SPACE_DIR to the embodiments/gr1 dir "
            f"(expected files: {_POLICY_YAML}, {_STATE_YAML}, {_ACTION_YAML})."
        )
    derived = build_index_maps_from_yaml(joint_dir)
    logger.info("Loaded GR1 joint-space maps from %s", joint_dir)
    return derived


# Resolve the GR1 ⇄ Arena-sim index tables (YAML-derived) and bundle them with the
# model-side GR1 spec into a single cohesive mapping object.
# ArenaJointMapping.__post_init__ cross-checks the table widths against GR1.action_dim.
_state_idx, _action_map, _sim_action_dim, _state_full_dim = _resolve_maps()

GR1_ARENA = ArenaJointMapping(
    spec=GR1,
    state_full_to_policy=_state_idx,
    policy_to_action=_action_map,
    sim_action_dim=_sim_action_dim,
    state_full_dim=_state_full_dim,
)

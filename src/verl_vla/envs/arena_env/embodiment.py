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

"""Embodiment ⇄ Isaac Lab Arena sim joint-space layout.

:class:`ArenaJointMapping` pairs a model-side
:class:`~verl_vla.models.gr00t.utils.EmbodimentSpec` with the simulator-specific
gather/scatter index tables, exposing the conversions as methods so callers never
touch raw index lists. The tables are **derived from the Arena embodiment YAMLs**
(named by ``spec.joint_space_yamls``) rather than hand-copied, so they stay in sync
with Arena; this replicates the name-based lookup of
``isaaclab_arena_gr00t.utils.joints_conversion``:

    policy: group -> [joint_name, ...]          (GR00T policy joint order)
    action: joint_name -> column index          (full sim action vector)
    state:  joint_name -> column index          (full sim state vector)

Flattening the policy groups (in YAML order) gives the policy-order joint names;
looking each one up in the state/action ``name -> index`` dicts yields the two
index tables bundled on the mapping:

    mapping.state_full_to_policy[i] = state_cfg[name_i]   # gather full state -> policy
    mapping.policy_to_action[i]     = action_cfg[name_i]  # scatter policy -> full action

These mappings are embodiment- AND simulator-specific and are kept out of the env
wrapper (which only calls the mapping's methods) so the wrapper stays generic.

The joint-space YAMLs are the **single source of truth** (no hardcoded fallback
index tables — everything is keyed by joint name). YAML discovery + parsing lives in
:mod:`verl_vla.envs.arena_env.utils` (``resolve_joint_maps``); this module only
bundles the resulting tables with the model-side spec. ``GR1_ARENA`` is the concrete
binding for the GR1 embodiment; add more by calling ``ArenaJointMapping.from_spec``.
"""

from dataclasses import dataclass
from pathlib import Path

from verl_vla.envs.arena_env.utils import build_index_maps_from_yaml, resolve_joint_maps
from verl_vla.models.gr00t.utils import GR1, EmbodimentSpec

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

    @classmethod
    def from_spec(
        cls,
        spec: EmbodimentSpec,
        joint_dir: "str | Path | None" = None,
    ) -> "ArenaJointMapping":
        """Build a mapping by deriving the index tables from the joint-space YAMLs.

        The YAML filenames come from ``spec.joint_space_yamls`` and ``spec.name`` drives
        directory discovery; parsing is delegated to
        :func:`verl_vla.envs.arena_env.utils.resolve_joint_maps`. The YAML grouping is
        cross-checked against ``spec.state_group_dims`` and ``__post_init__`` cross-checks
        the table widths against ``spec.action_dim``.
        """
        if spec.joint_space_yamls is None:
            raise ValueError(
                f"EmbodimentSpec {spec.name!r} has no joint_space_yamls; "
                "cannot derive Arena-sim joint maps."
            )
        state_idx, action_map, sim_action_dim, state_full_dim = resolve_joint_maps(
            spec.name, spec.joint_space_yamls, joint_dir, expected_group_dims=spec.state_group_dims
        )
        return cls(
            spec=spec,
            state_full_to_policy=state_idx,
            policy_to_action=action_map,
            sim_action_dim=sim_action_dim,
            state_full_dim=state_full_dim,
        )


# Derive the GR1 ⇄ Arena-sim index tables (YAML-derived) and bundle them with the
# model-side GR1 spec into a single cohesive mapping object.
GR1_ARENA = ArenaJointMapping.from_spec(GR1)

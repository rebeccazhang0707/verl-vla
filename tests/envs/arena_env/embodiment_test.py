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

"""Unit tests for the GR1 ⇄ Arena-sim joint-space mapping (``ArenaJointMapping``).

The mapping methods are simulator-agnostic (pure index gather/scatter), so they
are tested directly on ``GR1_ARENA`` without launching Isaac Sim. Importing the
module still pulls in the full ``verl`` package (torch, ...), so the whole module
is skipped when those dependencies are unavailable.
"""

import importlib

import pytest

torch = pytest.importorskip("torch")

# ``embodiment.py`` runs ``_resolve_maps()`` at import time, which raises
# ``RuntimeError`` (not ``ImportError``) when the GR1 joint-space YAMLs cannot be
# found (no ``isaaclab_arena_gr00t`` package and no ``ARENA_GR1_JOINT_SPACE_DIR``;
# see ``conftest.py``). ``pytest.importorskip`` only catches ``ImportError``, so we
# catch both explicitly to skip cleanly instead of erroring at collection time.
try:
    embodiment = importlib.import_module("verl_vla.envs.arena_env.embodiment")
except (ImportError, RuntimeError) as e:
    pytest.skip(f"arena embodiment unavailable: {e}", allow_module_level=True)

GR1_ARENA = embodiment.GR1_ARENA


# ---------------------------------------------------------------------------
# Index tables
# ---------------------------------------------------------------------------


def test_state_gather_indices_valid():
    idx = GR1_ARENA.state_full_to_policy
    assert len(idx) == GR1_ARENA.policy_dim == 26
    assert len(set(idx)) == 26, "state indices must be unique"
    assert all(0 <= i < GR1_ARENA.state_full_dim for i in idx)


def test_action_scatter_indices_valid():
    idx = GR1_ARENA.policy_to_action
    assert len(idx) == GR1_ARENA.policy_dim == 26
    assert len(set(idx)) == 26, "action target indices must be unique"
    assert all(0 <= i < GR1_ARENA.sim_action_dim for i in idx)


def test_policy_dim_matches_spec():
    assert GR1_ARENA.policy_dim == int(GR1_ARENA.spec.action_dim)
    assert len(GR1_ARENA.policy_to_action) == int(GR1_ARENA.spec.action_dim)


# ---------------------------------------------------------------------------
# Action conversion (policy 26-DOF <-> sim 36-DOF)
# ---------------------------------------------------------------------------


def test_scatter_and_extract_round_trip():
    batch = 3
    policy = torch.randn(batch, GR1_ARENA.policy_dim)

    sim = GR1_ARENA.scatter_action(policy)
    assert sim.shape == (batch, GR1_ARENA.sim_action_dim)

    recovered = GR1_ARENA.extract_action(sim)
    assert torch.allclose(recovered, policy)


def test_scatter_zero_fills_uncontrolled_joints():
    policy = torch.ones(1, GR1_ARENA.policy_dim)
    sim = GR1_ARENA.scatter_action(policy)

    controlled = set(GR1_ARENA.policy_to_action)
    for joint in range(GR1_ARENA.sim_action_dim):
        if joint not in controlled:
            assert sim[0, joint].item() == 0.0


def test_scatter_preserves_dtype_and_device():
    policy = torch.zeros(2, GR1_ARENA.policy_dim, dtype=torch.float64)
    sim = GR1_ARENA.scatter_action(policy)
    assert sim.dtype == torch.float64
    assert sim.device == policy.device


# ---------------------------------------------------------------------------
# State gather (full sim state -> policy order)
# ---------------------------------------------------------------------------


def test_gather_state_shape_and_columns():
    full = torch.arange(GR1_ARENA.state_full_dim, dtype=torch.float32).unsqueeze(0).repeat(4, 1)
    policy = GR1_ARENA.gather_state(full)
    assert policy.shape == (4, GR1_ARENA.policy_dim)
    # gathered columns must equal the configured index order
    assert torch.allclose(policy[0], full[0, GR1_ARENA.state_full_to_policy])

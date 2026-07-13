# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Pure-python tests for the Arena embodiment adapters.

These deliberately avoid importing ``isaaclab*`` / ``gr00t`` / ``lightwheel_sdk``
so they run in the plain verl-vla environment. They cover:

* the gather/scatter round trip of :class:`ArenaJointMapping`;
* the YAML-derived index map builder;
* G1 WBC byte-for-byte parity with the legacy inline wrapper logic
  (``policy_to_sim_action`` identity, ``robot_joint_pos`` state, camera dict);
* the Franka LIBERO task-space passthrough (action/state identity).
"""

from __future__ import annotations

import argparse
import types

import numpy as np
import pytest
import torch

from verl_vla.envs.arena.embodiment import (
    DEFAULT_ARENA_STATE_MODE,
    ArenaJointMapping,
    JointSpaceEmbodiment,
    TaskSpaceEmbodiment,
    make_arena_embodiment,
)


def _fake_spec(action_dim: int):
    return types.SimpleNamespace(action_dim=action_dim)


# ---------------------------------------------------------------------------
# ArenaJointMapping gather/scatter
# ---------------------------------------------------------------------------


def test_arena_joint_mapping_gather_scatter_roundtrip():
    # 3 policy joints; state has 5 columns, sim action has 6 columns.
    mapping = ArenaJointMapping(
        spec=_fake_spec(3),
        state_full_to_policy=[4, 0, 2],
        policy_to_action=[1, 5, 3],
        sim_action_dim=6,
        state_full_dim=5,
    )
    assert mapping.policy_dim == 3

    full_state = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5)
    gathered = mapping.gather_state(full_state)
    assert gathered.shape == (2, 3)
    torch.testing.assert_close(gathered, full_state[:, [4, 0, 2]])

    policy_action = torch.tensor([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0]])
    sim_action = mapping.scatter_action(policy_action)
    assert sim_action.shape == (2, 6)
    # Scattered columns carry the policy values; the rest stay zero.
    torch.testing.assert_close(sim_action[:, [1, 5, 3]], policy_action)
    zero_cols = [c for c in range(6) if c not in (1, 5, 3)]
    assert torch.count_nonzero(sim_action[:, zero_cols]) == 0

    # gather_action is the inverse of scatter_action.
    torch.testing.assert_close(mapping.gather_action(sim_action), policy_action)


def test_arena_joint_mapping_rejects_bad_widths():
    with pytest.raises(AssertionError):
        ArenaJointMapping(
            spec=_fake_spec(3),
            state_full_to_policy=[0, 1],  # wrong length
            policy_to_action=[0, 1, 2],
            sim_action_dim=4,
            state_full_dim=3,
        )


# ---------------------------------------------------------------------------
# YAML-derived index maps
# ---------------------------------------------------------------------------


def _write_gr1_like_yamls(tmp_path):
    (tmp_path / "gr00t_26dof_joint_space.yaml").write_text(
        "joints:\n  left_arm:\n    - a\n    - b\n  right_arm:\n    - c\n"
    )
    (tmp_path / "36dof_joint_space.yaml").write_text("joints:\n  a: 0\n  b: 1\n  c: 2\ntotal_joints: 5\n")
    (tmp_path / "54dof_joint_space.yaml").write_text("joints:\n  a: 10\n  b: 11\n  c: 12\ntotal_joints: 20\n")


def test_build_index_maps_from_yaml(tmp_path):
    _write_gr1_like_yamls(tmp_path)
    state_idx, action_map, sim_action_dim, state_full_dim = ArenaJointMapping.build_index_maps_from_yaml(tmp_path)
    # Flatten policy groups in order -> [a, b, c]; look each up in state/action dicts.
    assert state_idx == [10, 11, 12]
    assert action_map == [0, 1, 2]
    assert sim_action_dim == 5
    assert state_full_dim == 20


# ---------------------------------------------------------------------------
# G1 WBC byte-for-byte parity with the legacy inline wrapper logic
# ---------------------------------------------------------------------------


# G1 WBC is now the config-driven JointSpaceEmbodiment (no per-robot class); the values
# that used to be G1 class defaults live in the config (arena.yaml), so the test cfg
# supplies them here to mirror that single source of truth.
def _g1_cfg(**overrides):
    cfg = {
        "arena_state_mode": "g1_wbc_joint",
        "camera_names": ("robot_head_cam_rgb",),
        "enable_cameras": True,
        "action_dim": 50,
        "state_dim": None,
        "use_policy_action": False,
        "stable_hold_joint_slice": 43,
        "base_height_index": 46,
        "base_height_command": 0.75,
    }
    cfg.update(overrides)
    return cfg


# GR1 is the *mapped* JointSpaceEmbodiment; the joint-space spec + cameras come from the
# config (arena_gr1.yaml), mirrored here.
def _gr1_cfg(**overrides):
    cfg = {
        "arena_state_mode": "gr1_joint",
        "camera_names": ("robot_pov_cam_rgb",),
        "arena_joint_space_spec": "gr1",
        "action_dim": 26,
    }
    cfg.update(overrides)
    return cfg


def test_g1_stable_hold_constants():
    emb = make_arena_embodiment(_g1_cfg(), num_envs=2)
    assert isinstance(emb, JointSpaceEmbodiment)
    assert emb.state_mode == "g1_wbc_joint"
    assert emb.stable_hold_joint_slice == 43
    assert emb.base_height_index == 46
    assert emb.base_height_command == 0.75


def test_make_arena_embodiment_defaults_to_g1():
    # A config that omits arena_state_mode must fall back to g1_wbc_joint.
    assert DEFAULT_ARENA_STATE_MODE == "g1_wbc_joint"
    emb = make_arena_embodiment({"camera_names": ("robot_head_cam_rgb",), "enable_cameras": True}, num_envs=1)
    assert isinstance(emb, JointSpaceEmbodiment)
    assert emb.state_mode == "g1_wbc_joint"


def test_make_arena_embodiment_unknown_mode_raises():
    with pytest.raises(ValueError):
        make_arena_embodiment({"arena_state_mode": "nope"}, num_envs=1)


def test_g1_policy_to_sim_action_is_identity():
    emb = make_arena_embodiment(_g1_cfg(), num_envs=2)
    action = np.random.default_rng(0).standard_normal((2, 50)).astype(np.float32)

    out = emb.policy_to_sim_action(action, device="cpu")
    # Legacy behaviour: torch.as_tensor(action, dtype=float32, device=...).
    expected = torch.as_tensor(action, dtype=torch.float32, device="cpu")
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected)
    assert np.array_equal(out.cpu().numpy(), action)


def test_g1_extract_state_matches_robot_joint_pos():
    emb = make_arena_embodiment(_g1_cfg(), num_envs=2)
    joint_pos = torch.arange(2 * 50, dtype=torch.float32).reshape(2, 50)
    raw_obs = {"policy": {"robot_joint_pos": joint_pos}}

    state = emb.extract_state(raw_obs, scene=None)
    # Legacy _extract_state: robot_joint_pos -> float32 numpy, untouched values.
    assert state.dtype == np.float32
    assert np.array_equal(state, joint_pos.numpy().astype(np.float32))


def test_g1_extract_state_empty_policy_fallback():
    emb = make_arena_embodiment(_g1_cfg(action_dim=50), num_envs=3)
    state = emb.extract_state({"policy": {}}, scene=None)
    assert state.shape == (3, 50)
    assert not state.any()


def test_g1_extract_images_keys_and_uint8():
    emb = make_arena_embodiment(_g1_cfg(), num_envs=2)
    rgb = (np.random.default_rng(1).random((2, 8, 8, 3))).astype(np.float32)  # 0..1 floats
    raw_obs = {"camera_obs": {"robot_head_cam_rgb": torch.from_numpy(rgb)}}

    images = emb.extract_images(raw_obs)
    assert set(images) == {"observation.images.robot_head_cam_rgb"}
    out = images["observation.images.robot_head_cam_rgb"]
    assert out.dtype == np.uint8
    # Legacy _to_uint8_rgb: floats in [0,1] scaled by 255 then clipped.
    expected = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    assert np.array_equal(out, expected)


def test_g1_extract_images_missing_camera_raises():
    emb = make_arena_embodiment(_g1_cfg(camera_names=("does_not_exist",)), num_envs=1)
    rgb = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(KeyError):
        emb.extract_images({"camera_obs": {"robot_head_cam_rgb": rgb}})


def test_g1_extract_images_no_cameras_raises_when_enabled():
    emb = make_arena_embodiment(_g1_cfg(enable_cameras=True), num_envs=1)
    with pytest.raises(KeyError):
        emb.extract_images({"camera_obs": {}})


# ---------------------------------------------------------------------------
# Task-space (LIBERO / eef_pose, rotvec policy + quat_xyzw sim)
# ---------------------------------------------------------------------------


def _task_space_cfg(**overrides):
    cfg = {
        "arena_state_mode": "eef_pose",
        "action_dim": 7,
        "state_dim": 7,
        "camera_names": ("agentview_cam_rgb", "eye_in_hand_cam_rgb"),
    }
    cfg.update(overrides)
    return cfg


def test_task_space_policy_to_sim_action_converts_rotvec_to_quat_xyzw():
    emb = make_arena_embodiment(_task_space_cfg(), num_envs=1)
    assert isinstance(emb, TaskSpaceEmbodiment)
    # Identity rotation in axis-angle -> quat_xyzw (0,0,0,1).
    action = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.5]])
    out = emb.policy_to_sim_action(action, device="cpu")
    assert out.shape == (1, 8)
    torch.testing.assert_close(out[0, :3], action[0, :3])
    torch.testing.assert_close(out[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(out[0, 7:], action[0, 6:7])


def test_task_space_extract_state_from_concatenated_policy():
    emb = make_arena_embodiment(_task_space_cfg(), num_envs=1)
    # Sim policy obs: pos(3) + quat_xyzw(4) + gripper fingers(2).
    policy = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.04, 0.03]])
    state = emb.extract_state({"policy": policy}, scene=None)
    assert state.shape == (1, 7)
    np.testing.assert_allclose(state[0, :3], [0.1, 0.2, 0.3], rtol=0, atol=1e-6)
    np.testing.assert_allclose(state[0, 3:6], [0.0, 0.0, 0.0], rtol=0, atol=1e-5)
    np.testing.assert_allclose(state[0, 6:], [0.04], rtol=0, atol=1e-6)


def test_task_space_extract_state_from_dict_policy():
    emb = make_arena_embodiment(_task_space_cfg(), num_envs=1)
    policy = {
        "eef_pose": torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]]),
        "gripper_pos": torch.tensor([[0.04, 0.03]]),
    }
    state = emb.extract_state({"policy": policy}, scene=None)
    assert state.shape == (1, 7)
    np.testing.assert_allclose(state[0, 6:], [0.04], rtol=0, atol=1e-6)


def test_task_space_add_cli_args_reads_libero_task_keys():
    cfg = {
        "arena_state_mode": "eef_pose",
        "libero_task_suite": "libero_spatial",
        "libero_task_id": 3,
        "libero_randomize_object_pose": True,
        "libero_robot_init_noise_std": 0.05,
    }
    emb = make_arena_embodiment(cfg, num_envs=1)
    args = argparse.Namespace()
    emb.add_cli_args(args, cfg)
    assert args.task_suite == "libero_spatial"
    assert args.task_id == 3
    assert args.randomize_object_pose is True
    assert args.robot_init_noise_std == 0.05


def test_gr1_joint_map_is_lazy_and_import_safe():
    # Building a GR1 adapter must not require isaac/gr00t; the joint map is only
    # resolved on first use (and would raise a clear error if the YAMLs are absent).
    emb = make_arena_embodiment(_gr1_cfg(), num_envs=1)
    assert isinstance(emb, JointSpaceEmbodiment)
    assert emb.state_mode == "gr1_joint"
    assert emb.camera_names == ["robot_pov_cam_rgb"]
    # Lazy: constructing with a spec configured must NOT have built the mapping yet.
    assert emb._joint_map is None


# ---------------------------------------------------------------------------
# use_policy_action: config-driven (G1 zero-regression teleop path)
# ---------------------------------------------------------------------------


def test_use_policy_action_is_config_driven():
    # G1 WBC keeps the legacy stable-hold / teleop path. With an explicit
    # use_policy_action=False in cfg (as in arena.yaml) OR when the knob is omitted
    # on an identity joint-space (no arena_joint_space_spec), the adapter defaults
    # to False. Policy-driven GR1 joint + Franka LIBERO keep True.
    g1 = make_arena_embodiment(_g1_cfg(), num_envs=1)
    g1_default = make_arena_embodiment(
        {k: v for k, v in _g1_cfg().items() if k != "use_policy_action"}, num_envs=1
    )
    gr1 = make_arena_embodiment(_gr1_cfg(), num_envs=1)
    franka = make_arena_embodiment(_task_space_cfg(), num_envs=1)
    assert g1.use_policy_action is False
    assert g1_default.use_policy_action is False
    assert gr1.use_policy_action is True
    assert franka.use_policy_action is True


# ---------------------------------------------------------------------------
# policy_action_dim (Phase 2 review item #5): recorder uses the POLICY action width
# ---------------------------------------------------------------------------


def test_g1_policy_action_dim_is_none_uses_sim_dim():
    # Identity embodiment: None => the wrapper falls back to the sim action_dim
    # (unchanged G1 recorder behaviour).
    emb = make_arena_embodiment(_g1_cfg(), num_envs=1)
    assert emb.policy_action_dim is None


def test_gr1_policy_action_dim_is_policy_width():
    # GR1 records the 26-DOF GR00T policy action, not the scattered 36-DOF sim action.
    # Inject a fake joint map so we do not need the Arena YAMLs / gr00t here.
    emb = make_arena_embodiment(_gr1_cfg(), num_envs=1)
    emb._joint_map = ArenaJointMapping(
        spec=_fake_spec(26),
        state_full_to_policy=list(range(26)),
        policy_to_action=list(range(26)),
        sim_action_dim=36,
        state_full_dim=54,
    )
    assert emb.policy_action_dim == 26


def test_task_space_policy_action_dim_from_cfg():
    emb = make_arena_embodiment(_task_space_cfg(action_dim=7), num_envs=1)
    assert emb.policy_action_dim == 7

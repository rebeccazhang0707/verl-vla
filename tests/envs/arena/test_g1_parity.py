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

"""G1 parity tests for state/action extraction (no isaac / gr00t deps).

Camera extraction is strict: configured ``camera_names`` must exist in
``camera_obs`` when ``enable_cameras=True`` (no fallback / zero-image placeholder).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from verl_vla.envs.arena.embodiment import make_arena_embodiment

# ---------------------------------------------------------------------------
# Reference helpers for G1 identity joint-space parity (uint8 RGB + state extract).
# Camera path is strict (configured names must exist); state/action stay identity.
# ---------------------------------------------------------------------------


def _legacy_to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _legacy_extract_camera_images(raw_obs, *, camera_names) -> dict[str, np.ndarray]:
    camera_obs = raw_obs.get("camera_obs", {}) if isinstance(raw_obs, dict) else {}
    return {f"observation.images.{name}": _legacy_to_uint8_rgb(camera_obs[name]) for name in camera_names}


def _legacy_extract_state(raw_obs, *, num_envs, state_dim) -> np.ndarray:
    policy_obs = raw_obs.get("policy", {}) if isinstance(raw_obs, dict) else {}
    if "robot_joint_pos" in policy_obs:
        state = policy_obs["robot_joint_pos"]
    else:
        parts = list(policy_obs.values())
        if not parts:
            return np.zeros((num_envs, state_dim), dtype=np.float32)
        tensors = [part if isinstance(part, torch.Tensor) else torch.as_tensor(part) for part in parts]
        state = torch.cat([tensor.reshape(tensor.shape[0], -1) for tensor in tensors], dim=-1)

    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    return np.asarray(state, dtype=np.float32)


def _g1(**overrides):
    cfg = {
        "arena_state_mode": "g1_wbc_joint",
        "camera_names": ("robot_head_cam_rgb",),
        "enable_cameras": True,
        "action_dim": 50,
        "state_dim": None,
    }
    cfg.update(overrides)
    return make_arena_embodiment(cfg, num_envs=overrides.get("num_envs", 2))


def _assert_image_dicts_equal(new: dict, ref: dict) -> None:
    assert set(new) == set(ref), f"keys differ: {set(new)} vs {set(ref)}"
    for key in ref:
        assert new[key].dtype == ref[key].dtype == np.uint8
        assert np.array_equal(new[key], ref[key]), f"pixels differ for {key}"


# ---------------------------------------------------------------------------
# G1 identity joint-space: camera / state / action vs reference helpers
# ---------------------------------------------------------------------------


def test_g1_extract_images_parity_float_uint8_rgba():
    rng = np.random.default_rng(7)
    num_envs = 2
    cam = "robot_head_cam_rgb"

    cases = {
        "float01": torch.from_numpy(rng.random((num_envs, 6, 6, 3)).astype(np.float32)),  # [0,1] -> *255
        "uint8": torch.from_numpy(rng.integers(0, 256, (num_envs, 6, 6, 3), dtype=np.uint8)),  # untouched
        "rgba_gt1": torch.from_numpy((rng.random((num_envs, 6, 6, 4)) * 3.0).astype(np.float32)),  # >1, drop alpha+clip
    }
    for label, frame in cases.items():
        emb = _g1(num_envs=num_envs)
        raw_obs = {"camera_obs": {cam: frame}}
        new = emb.extract_images(raw_obs)
        ref = _legacy_extract_camera_images(raw_obs, camera_names=[cam])
        _assert_image_dicts_equal(new, ref)
        assert set(new) == {f"observation.images.{cam}"}, label


def test_g1_extract_images_missing_camera_raises():
    num_envs = 1
    emb = _g1(camera_names=("does_not_exist",), num_envs=num_envs)
    raw_obs = {"camera_obs": {"robot_head_cam_rgb": np.zeros((num_envs, 4, 4, 3), dtype=np.uint8)}}
    with pytest.raises(KeyError):
        emb.extract_images(raw_obs)


def test_g1_extract_images_empty_camera_names_returns_empty():
    emb = _g1(camera_names=(), enable_cameras=False, num_envs=3)
    assert emb.extract_images({"camera_obs": {}}) == {}


def test_g1_extract_images_disabled_camera_missing_raises():
    emb = _g1(enable_cameras=False, num_envs=3)
    with pytest.raises(KeyError):
        emb.extract_images({"camera_obs": {}})


def test_g1_extract_state_parity_all_branches():
    num_envs = 2
    # (a) robot_joint_pos present.
    jp = torch.arange(num_envs * 50, dtype=torch.float32).reshape(num_envs, 50)
    obs_a = {"policy": {"robot_joint_pos": jp}}
    # (b) no robot_joint_pos -> concat of all policy values (reshaped to (B, -1)).
    obs_b = {
        "policy": {
            "a": torch.arange(num_envs * 3, dtype=torch.float32).reshape(num_envs, 3),
            "b": torch.arange(num_envs * 2, dtype=torch.float32).reshape(num_envs, 2),
        }
    }
    # (c) empty policy -> zeros((num_envs, state_dim)).
    obs_c = {"policy": {}}

    for obs in (obs_a, obs_b, obs_c):
        emb = _g1(num_envs=num_envs)
        new = emb.extract_state(obs, scene=None)
        ref = _legacy_extract_state(obs, num_envs=num_envs, state_dim=emb.state_dim)
        assert new.dtype == np.float32 == ref.dtype
        assert new.shape == ref.shape
        assert np.array_equal(new, ref)


def test_g1_policy_to_sim_action_parity_identity():
    # Legacy env_step did exactly torch.as_tensor(action, float32, device).
    emb = _g1(num_envs=4)
    action = np.random.default_rng(3).standard_normal((4, 50)).astype(np.float64)  # non-float32 in
    new = emb.policy_to_sim_action(action, device="cpu")
    ref = torch.as_tensor(action, dtype=torch.float32, device="cpu")
    assert new.dtype == torch.float32
    assert torch.equal(new, ref)

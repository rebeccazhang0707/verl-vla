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

"""Unit tests for the gr00t-free Arena policy IO adapters (no gr00t package needed).

Covers the pure tensor transforms in ``ArenaGr00tInput.from_env_obs`` and
``ArenaGr00tOutput`` -- the parts that must stay correct independent of the gr00t
checkpoint / processor.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("verl")
from verl import DataProto  # noqa: E402

from verl_vla.models.gr00t_n1d6.policy import (  # noqa: E402
    ArenaGr00tInput,
    ArenaGr00tOutput,
    get_gr00t_policy_classes,
)
from verl_vla.models.gr00t_n1d6.policy.arena_policy import (  # noqa: E402
    _image_batch_to_bhwc_uint8,
)

B, H, W = 2, 8, 6

# Sample camera obs keys (any ``observation.images.*`` names work; the adapter maps
# cameras onto the checkpoint video_keys by order, not by these names).
ARENA_HEAD_CAMERA_KEY = "observation.images.robot_head_cam_rgb"
ARENA_WRIST_CAMERA_KEY = "observation.images.robot_wrist_cam_rgb"
ARENA_STATE_KEY = "observation.state"


def _make_obs(images: torch.Tensor, state: torch.Tensor, wrist: torch.Tensor | None = None) -> DataProto:
    tensors = {ARENA_HEAD_CAMERA_KEY: images, ARENA_STATE_KEY: state}
    if wrist is not None:
        tensors[ARENA_WRIST_CAMERA_KEY] = wrist
    non_tensors = {"task": np.asarray(["pick up the cube"] * images.shape[0], dtype=object)}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)


def test_registry_resolves_arena():
    input_cls, output_cls = get_gr00t_policy_classes("arena")
    assert input_cls is ArenaGr00tInput
    assert output_cls is ArenaGr00tOutput


def test_registry_unknown_raises():
    with pytest.raises(ValueError):
        get_gr00t_policy_classes("does_not_exist")


def test_image_bhwc_uint8_passthrough():
    imgs = torch.randint(0, 256, (B, H, W, 3), dtype=torch.uint8)
    out = _image_batch_to_bhwc_uint8(imgs)
    assert out.shape == (B, H, W, 3)
    assert out.dtype == torch.uint8
    assert torch.equal(out, imgs)


def test_image_bchw_to_bhwc():
    imgs = torch.randint(0, 256, (B, 3, H, W), dtype=torch.uint8)
    out = _image_batch_to_bhwc_uint8(imgs)
    assert out.shape == (B, H, W, 3)
    assert torch.equal(out, imgs.permute(0, 2, 3, 1))


def test_image_float01_scaled_to_uint8():
    imgs = torch.ones((B, H, W, 3), dtype=torch.float32) * 0.5
    out = _image_batch_to_bhwc_uint8(imgs)
    assert out.dtype == torch.uint8
    assert torch.all(out == 128)  # round(0.5 * 255) == 128


def test_image_drops_alpha_channel():
    imgs = torch.randint(0, 256, (B, H, W, 4), dtype=torch.uint8)
    out = _image_batch_to_bhwc_uint8(imgs)
    assert out.shape == (B, H, W, 3)
    assert torch.equal(out, imgs[..., :3])


# ---------------------------------------------------------------------------
# from_env_obs passes every observation.images.* through as a camera dict
# (no head/wrist resolution; the adapter maps cameras onto video_keys by order).
# ---------------------------------------------------------------------------


def test_from_env_obs_passes_all_cameras_through():
    imgs = torch.randint(0, 256, (B, H, W, 3), dtype=torch.uint8)
    wrist = torch.randint(0, 256, (B, H, W, 3), dtype=torch.uint8)
    state = torch.randn(B, 26, dtype=torch.float32)
    obs = DataProto.from_dict(
        tensors={
            "observation.images.robot_pov_cam_rgb": imgs,
            "observation.images.right_wrist_cam_rgb": wrist,
            ARENA_STATE_KEY: state,
        },
        non_tensors={"task": np.asarray(["open the fridge"] * B, dtype=object)},
    )
    model_input = ArenaGr00tInput.from_env_obs(obs)
    # Dict insertion order must match obs camera order (adapter maps by position).
    assert list(model_input.images) == [
        "observation.images.robot_pov_cam_rgb",
        "observation.images.right_wrist_cam_rgb",
    ]
    assert torch.equal(model_input.images["observation.images.robot_pov_cam_rgb"], imgs)
    assert torch.equal(model_input.images["observation.images.right_wrist_cam_rgb"], wrist)


def test_from_env_obs_head_only():
    imgs = torch.randint(0, 256, (B, H, W, 3), dtype=torch.uint8)
    state = torch.randn(B, 26, dtype=torch.float32)
    obs = _make_obs(imgs, state)

    model_input = ArenaGr00tInput.from_env_obs(obs)
    head = model_input.images[ARENA_HEAD_CAMERA_KEY]
    assert head.shape == (B, H, W, 3)
    assert head.dtype == torch.uint8
    assert len(model_input.images) == 1
    assert model_input.state.dtype == torch.float32
    assert torch.allclose(model_input.state, state)
    assert model_input.task == ["pick up the cube"] * B


def test_from_env_obs_with_wrist():
    imgs = torch.randint(0, 256, (B, H, W, 3), dtype=torch.uint8)
    wrist = torch.randint(0, 256, (B, H, W, 3), dtype=torch.uint8)
    state = torch.randn(B, 26, dtype=torch.float32)
    obs = _make_obs(imgs, state, wrist=wrist)

    model_input = ArenaGr00tInput.from_env_obs(obs)
    assert ARENA_WRIST_CAMERA_KEY in model_input.images
    assert model_input.images[ARENA_WRIST_CAMERA_KEY].shape == (B, H, W, 3)
    assert torch.equal(model_input.images[ARENA_WRIST_CAMERA_KEY], wrist)


def test_output_from_model_output_chunks_and_carries_full_action():
    horizon, max_action_dim, action_dim = 16, 128, 26
    full_action = torch.randn(B, horizon, max_action_dim)
    decoded = torch.randn(B, horizon, action_dim)
    log_probs = torch.randn(B)

    output = ArenaGr00tOutput.from_model_output(
        {
            "full_action": full_action,
            "decoded_action": decoded,
            "log_probs": log_probs,
            "num_action_chunks": 8,
        }
    )
    assert output.action.shape == (B, 8, action_dim)
    assert torch.equal(output.action, decoded[:, :8])
    assert torch.equal(output.full_action, full_action)
    assert torch.equal(output.log_prob, log_probs)


def test_output_to_data_proto_keys():
    full_action = torch.randn(B, 16, 128)
    decoded = torch.randn(B, 16, 26)
    output = ArenaGr00tOutput.from_model_output(
        {"full_action": full_action, "decoded_action": decoded, "log_probs": torch.randn(B), "num_action_chunks": 16}
    )
    proto = output.to_data_proto()
    assert "action" in proto.batch.keys()
    assert "full_action" in proto.batch.keys()
    assert "log_prob" in proto.batch.keys()
    assert proto.batch["action"].shape == (B, 16, 26)
    assert proto.batch["full_action"].shape == (B, 16, 128)


def test_output_without_log_prob_omits_key():
    full_action = torch.randn(B, 16, 128)
    decoded = torch.randn(B, 16, 26)
    output = ArenaGr00tOutput.from_model_output(
        {"full_action": full_action, "decoded_action": decoded, "num_action_chunks": 16}
    )
    proto = output.to_data_proto()
    assert output.log_prob is None
    assert "log_prob" not in proto.batch.keys()


def test_output_missing_decoded_falls_back_to_full_action():
    # Actor-side / no-decode path: without ``decoded_action`` the env-facing ``action``
    # must fall back to the normalised ``full_action`` so callers still get a chunk.
    # (Guards the action double-track: the fallback keeps the two aligned in shape.)
    full_action = torch.randn(B, 16, 128)
    output = ArenaGr00tOutput.from_model_output({"full_action": full_action, "num_action_chunks": 4})
    assert output.action.shape == (B, 4, 128)
    assert torch.equal(output.action, full_action[:, :4])
    assert torch.equal(output.full_action, full_action)


def test_output_chunk_clamped_to_available_horizon():
    # num_action_chunks larger than the decoded horizon must clamp, never over-index.
    full_action = torch.randn(B, 16, 128)
    decoded = torch.randn(B, 12, 26)
    output = ArenaGr00tOutput.from_model_output(
        {"full_action": full_action, "decoded_action": decoded, "num_action_chunks": 999}
    )
    assert output.action.shape == (B, 12, 26)
    assert torch.equal(output.action, decoded)


def test_output_default_chunk_uses_full_decoded_horizon():
    # When num_action_chunks is absent it defaults to the full decoded horizon.
    full_action = torch.randn(B, 16, 128)
    decoded = torch.randn(B, 10, 26)
    output = ArenaGr00tOutput.from_model_output({"full_action": full_action, "decoded_action": decoded})
    assert output.action.shape == (B, 10, 26)

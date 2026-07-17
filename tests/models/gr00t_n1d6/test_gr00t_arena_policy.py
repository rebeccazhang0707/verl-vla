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
checkpoint / processor -- plus the SAC action double-track invariant (the critic
scores the NORMALISED ``full_action``, never the decoded env ``action``).
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
from verl_vla.utils.data import (  # noqa: E402
    add_transition_prefixes,
    flatten_trajectories,
    get_dataproto_from_prefix,
    stack_dataproto_with_padding,
)

B, H, W = 2, 8, 6

# Sample camera obs keys (any ``observation.images.*`` names work; the adapter maps
# cameras onto the checkpoint video_keys by order, not by these names).
ARENA_HEAD_CAMERA_KEY = "observation.images.robot_head_cam_rgb"
ARENA_STATE_KEY = "observation.state"


def test_registry_resolves_arena():
    input_cls, output_cls = get_gr00t_policy_classes("arena")
    assert input_cls is ArenaGr00tInput
    assert output_cls is ArenaGr00tOutput


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


def test_from_env_obs_state_and_task():
    imgs = torch.randint(0, 256, (B, H, W, 3), dtype=torch.uint8)
    state = torch.randn(B, 26, dtype=torch.float32)
    obs = DataProto.from_dict(
        tensors={ARENA_HEAD_CAMERA_KEY: imgs, ARENA_STATE_KEY: state},
        non_tensors={"task": np.asarray(["pick up the cube"] * B, dtype=object)},
    )

    model_input = ArenaGr00tInput.from_env_obs(obs)
    head = model_input.images[ARENA_HEAD_CAMERA_KEY]
    assert head.shape == (B, H, W, 3)
    assert head.dtype == torch.uint8
    assert model_input.state.dtype == torch.float32
    assert torch.allclose(model_input.state, state)
    assert model_input.task == ["pick up the cube"] * B


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


# ---------------------------------------------------------------------------
# SAC action double-track: the critic scores the NORMALISED action
# ---------------------------------------------------------------------------


def test_full_action_survives_replay_plumbing():
    """End-to-end double-track invariant (rollout output -> replay transition).

    The normalised ``full_action`` from ``ArenaGr00tOutput.to_data_proto`` must
    survive the env-loop replay plumbing (``stack_dataproto_with_padding`` ->
    ``add_transition_prefixes`` -> ``flatten_trajectories``) so the ``t0.action.*``
    dict the SAC critic reads still carries the NORMALISED action, distinct from the
    DECODED env action. A rename/drop anywhere in that chain would flip the critic
    onto the wrong (decoded) action space; this guards it.
    """
    batch, decoded_chunk, decoded_dim, full_horizon, max_action_dim = 2, 16, 26, 50, 128
    full_action = torch.randn(batch, full_horizon, max_action_dim)
    decoded = torch.randn(batch, decoded_chunk, decoded_dim)
    rollout = ArenaGr00tOutput.from_model_output(
        {"full_action": full_action, "decoded_action": decoded, "num_action_chunks": decoded_chunk}
    ).to_data_proto()

    # Env-loop namespacing: one rollout step -> keys "action.action" / "action.full_action".
    stacked = stack_dataproto_with_padding([rollout], "action")
    assert set(stacked) == {"action.action", "action.full_action"}
    data = DataProto.from_dict(tensors=stacked)

    # Rollout slot -> t0/t1 transition fields, then flatten (B, steps, ...) -> (B*steps, ...).
    data = flatten_trajectories(add_transition_prefixes(data))

    a0 = get_dataproto_from_prefix(data, "t0.action.").batch
    assert "full_action" in a0.keys() and "action" in a0.keys()
    # The critic-space selection (a.get("full_action", a["action"])) must land on the
    # NORMALISED full_action, not the decoded env action.
    assert a0["full_action"].shape[-1] == max_action_dim
    assert not torch.equal(a0["full_action"], a0["action"])

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from verl import DataProto

pytest.importorskip("gr00t", reason="GR00T N1.6 is an optional dependency")

from verl_vla.models.gr00t_n1d6.policy.libero_policy import (
    LIBERO_IMAGE_KEYS,
    LIBERO_KEYS,
    LiberoGr00tInput,
    LiberoGr00tOutput,
    image_to_uint8_hwc,
    libero_gripper_to_gr00t,
    load_libero_statistics,
    prepare_libero_gripper_action,
)

PACKAGE_DIR = Path(__file__).parents[2] / "src/verl_vla/models/gr00t_n1d6"


def test_source_commit_is_pinned():
    package_init = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "e29d8fc50b0e4745120ae3fb72447986fe638aa6" in package_init


def test_load_flat_libero_statistics(tmp_path):
    stats = {
        modality: {
            name: [float(index + offset) for index in range(8 if modality == "state" else 7)]
            for offset, name in enumerate(("min", "max", "mean", "std", "q01", "q99"))
        }
        for modality in ("state", "action")
    }
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")

    nested = load_libero_statistics(path)["libero_panda"]

    assert tuple(nested["state"]) == LIBERO_KEYS
    assert nested["state"]["x"]["min"] == [0.0]
    assert nested["state"]["gripper"]["min"] == [6.0, 7.0]
    assert nested["action"]["gripper"] == {
        "min": [-3.0],
        "max": [-2.5],
        "mean": [-3.5],
        "std": [4.5],
        "q01": [-5.0],
        "q99": [-4.5],
    }


def test_libero_gripper_to_gr00t_matches_official_training_semantics():
    action = np.zeros((3, 7), dtype=np.float32)
    action[:, -1] = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    converted = libero_gripper_to_gr00t(action)

    np.testing.assert_array_equal(converted[:, -1], np.array([1.0, 0.5, 0.0]))
    np.testing.assert_array_equal(action[:, -1], np.array([-1.0, 0.0, 1.0]))


def test_image_to_uint8_hwc_scales_chw_float_image():
    image = np.ones((3, 4, 5), dtype=np.float32) * 0.5

    converted = image_to_uint8_hwc(image)

    assert converted.shape == (4, 5, 3)
    assert converted.dtype == np.uint8
    assert np.all(converted == 127)


def test_image_to_uint8_hwc_accepts_bfloat16_tensor():
    image = torch.full((3, 4, 5), 0.5, dtype=torch.bfloat16)

    converted = image_to_uint8_hwc(image)

    assert converted.shape == (4, 5, 3)
    assert converted.dtype == np.uint8
    assert np.all(converted == 127)


def test_flat_statistics_require_min_and_max(tmp_path):
    stats = {
        modality: {name: [0.0] * (8 if modality == "state" else 7) for name in ("mean", "std", "q01", "q99")}
        for modality in ("state", "action")
    }
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 8 state statistics"):
        load_libero_statistics(path)


def test_prepare_libero_gripper_action_matches_official_semantics():
    action = np.zeros((1, 3, 7), dtype=np.float32)
    action[..., -1] = np.array([0.1, 0.5, 0.9], dtype=np.float32)

    prepared = prepare_libero_gripper_action(action)

    np.testing.assert_array_equal(prepared[0, :, -1], np.array([1.0, 0.0, -1.0]))
    np.testing.assert_array_equal(
        action[0, :, -1],
        np.array([0.1, 0.5, 0.9], dtype=np.float32),
    )


def _raw_libero_batch() -> tuple[DataProto, torch.Tensor, torch.Tensor]:
    actions = torch.zeros((1, 16, 7), dtype=torch.float32)
    action_valid_mask = torch.ones((1, 16), dtype=torch.float32)
    action_valid_mask[:, -1] = 0
    obs = DataProto.from_dict(
        tensors={
            "observation.images.image": torch.zeros((1, 3, 8, 8)),
            "observation.images.wrist_image": torch.zeros((1, 3, 8, 8)),
            "observation.state": torch.arange(8, dtype=torch.float32).reshape(1, 8),
            "action": actions,
            "action_is_pad": ~action_valid_mask.bool(),
        },
        non_tensors={"task": np.asarray(["pick up the bowl"], dtype=object)},
    )
    return obs, actions, action_valid_mask


def test_libero_input_from_env_obs_exposes_adapter_tensors():
    obs, _, _ = _raw_libero_batch()

    policy_input = LiberoGr00tInput.from_env_obs(obs)

    assert list(policy_input.images) == list(LIBERO_IMAGE_KEYS)
    assert policy_input.images[LIBERO_IMAGE_KEYS[0]].dtype == torch.uint8
    assert policy_input.images[LIBERO_IMAGE_KEYS[0]].shape == (1, 8, 8, 3)
    assert policy_input.state.shape == (1, 8)
    assert policy_input.task == ["pick up the bowl"]
    assert policy_input.actions is None


def test_libero_input_from_data_proto_converts_gripper():
    obs, actions, _ = _raw_libero_batch()
    actions = actions.clone()
    actions[..., -1] = -1.0

    policy_input = LiberoGr00tInput.from_data_proto(obs, actions=actions)

    assert policy_input.actions is not None
    assert policy_input.actions.shape == (1, 16, 7)
    torch.testing.assert_close(policy_input.actions[..., -1], torch.ones(1, 16))


def test_libero_input_accepts_bfloat16_dataproto():
    obs, actions, _ = _raw_libero_batch()
    for key, value in obs.batch.items():
        if torch.is_floating_point(value):
            obs.batch[key] = value.to(torch.bfloat16)

    policy_input = LiberoGr00tInput.from_data_proto(obs, actions=actions.to(torch.bfloat16))

    assert policy_input.state.dtype == torch.float32
    assert policy_input.actions is not None
    assert policy_input.actions.dtype == torch.float32


def test_libero_output_applies_gripper_and_chunks():
    decoded = torch.zeros((1, 4, 7), dtype=torch.float32)
    decoded[0, :, -1] = torch.tensor([0.2, 0.8, 0.1, 0.9])
    full_action = torch.zeros((1, 4, 128), dtype=torch.float32)

    output = LiberoGr00tOutput.from_model_output(
        {
            "full_action": full_action,
            "decoded_action": decoded,
            "num_action_chunks": 2,
        }
    )

    assert output.action.shape == (1, 2, 7)
    torch.testing.assert_close(output.action[0, :, -1], torch.tensor([1.0, -1.0]))
    assert torch.equal(output.full_action, full_action)

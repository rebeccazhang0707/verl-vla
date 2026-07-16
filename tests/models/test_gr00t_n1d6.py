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

"""GR00T N1.6 unit tests: adapter/critic config, state-dict helpers, LIBERO data.

The config and state-dict sections are gr00t-package-free; the LIBERO data
section needs the optional ``gr00t`` package and is skipped without it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from verl_vla.models.gr00t_n1d6.adapter_config import Gr00tAdapterConfig, Gr00tCriticConfig
from verl_vla.models.gr00t_n1d6.utils import (
    extract_critic_state_dict,
    normalize_adapter_state_dict,
)

HAS_GROOT = importlib.util.find_spec("gr00t") is not None
requires_gr00t = pytest.mark.skipif(not HAS_GROOT, reason="GR00T N1.6 is an optional dependency")

if HAS_GROOT:
    from verl import DataProto

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


# ---------------------------------------------------------------------------
# Adapter / critic config (gr00t-package-free)
# ---------------------------------------------------------------------------


def test_critic_defaults_disabled():
    critic = Gr00tCriticConfig()
    assert critic.enabled is False
    assert critic.type == "cross_attn"
    assert critic.pooling == "attn"
    assert critic.head_num == 10


def test_sac_enable_legacy_alias_enables_critic():
    cfg = Gr00tAdapterConfig(sac_enable=True, critic_type="mean_pool", critic_head_num=4)
    assert cfg.critic.enabled is True
    assert cfg.sac_enable is True
    assert cfg.critic.type == "mean_pool"
    assert cfg.critic.head_num == 4
    assert cfg.critic.pooling == "mean"


def test_nested_critic_can_enable_without_legacy_alias():
    cfg = Gr00tAdapterConfig(
        critic={"enabled": True, "type": "cross_attn", "head_num": 8},
    )
    assert cfg.critic.enabled is True
    assert cfg.critic.head_num == 8


def test_legacy_sac_enable_overrides_nested_enabled():
    cfg = Gr00tAdapterConfig(
        sac_enable=False,
        critic={"enabled": True, "type": "cross_attn", "head_num": 8},
    )
    # Flat legacy fields are applied after nested critic (same as pi0).
    assert cfg.critic.enabled is False
    assert cfg.critic.head_num == 8


def test_adapter_to_dict_includes_critic():
    cfg = Gr00tAdapterConfig(policy_type="arena", action_dim=26)
    payload = cfg.to_dict()
    assert payload["policy_type"] == "arena"
    assert payload["action_dim"] == 26
    assert "critic" in payload
    assert payload["critic"]["enabled"] is False


def test_libero_sft_defaults_via_overrides():
    cfg = Gr00tAdapterConfig(
        policy_type="libero",
        embodiment_tag="libero_panda",
        action_dim=7,
        embodiment_id=2,
        num_action_chunks=8,
        override_modality_configs=True,
        use_relative_action=True,
        critic={"enabled": False},
    )
    assert cfg.policy_type == "libero"
    assert cfg.embodiment_tag == "libero_panda"
    assert cfg.action_dim == 7
    assert cfg.override_modality_configs is True
    assert cfg.use_relative_action is True
    assert cfg.critic.enabled is False
    assert "norm_stats_path" in cfg.to_dict()


# ---------------------------------------------------------------------------
# Adapter state-dict helpers (no gr00t runtime required)
# ---------------------------------------------------------------------------


def test_normalize_adapter_state_dict_remaps_legacy_critic_prefixes():
    state = {
        "policy.backbone.weight": torch.ones(1),
        "critic_backend.critic_heads.0.weight": torch.ones(2),
        "auxiliary_modules.critic.target_critic_heads.0.weight": torch.ones(3),
        "critic.already_ok": torch.ones(4),
    }
    normalized = normalize_adapter_state_dict(state)
    assert "policy.backbone.weight" in normalized
    assert "critic.critic_heads.0.weight" in normalized
    assert "critic.target_critic_heads.0.weight" in normalized
    assert "critic.already_ok" in normalized
    assert "critic_backend.critic_heads.0.weight" not in normalized
    assert "auxiliary_modules.critic.target_critic_heads.0.weight" not in normalized


def test_extract_critic_state_dict_strips_prefix():
    state = {
        "policy.x": torch.ones(1),
        "critic.heads.0.weight": torch.ones(2),
        "critic.target_heads.0.weight": torch.ones(3),
    }
    critic = extract_critic_state_dict(state)
    assert set(critic) == {"heads.0.weight", "target_heads.0.weight"}
    assert "policy.x" not in critic


# ---------------------------------------------------------------------------
# LIBERO data transforms (requires the optional gr00t package)
# ---------------------------------------------------------------------------


@requires_gr00t
def test_source_commit_is_pinned():
    package_init = (PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "e29d8fc50b0e4745120ae3fb72447986fe638aa6" in package_init


@requires_gr00t
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


@requires_gr00t
def test_libero_gripper_to_gr00t_matches_official_training_semantics():
    action = np.zeros((3, 7), dtype=np.float32)
    action[:, -1] = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    converted = libero_gripper_to_gr00t(action)

    np.testing.assert_array_equal(converted[:, -1], np.array([1.0, 0.5, 0.0]))
    np.testing.assert_array_equal(action[:, -1], np.array([-1.0, 0.0, 1.0]))


@requires_gr00t
def test_image_to_uint8_hwc_scales_chw_float_image():
    image = np.ones((3, 4, 5), dtype=np.float32) * 0.5

    converted = image_to_uint8_hwc(image)

    assert converted.shape == (4, 5, 3)
    assert converted.dtype == np.uint8
    assert np.all(converted == 127)


@requires_gr00t
def test_image_to_uint8_hwc_accepts_bfloat16_tensor():
    image = torch.full((3, 4, 5), 0.5, dtype=torch.bfloat16)

    converted = image_to_uint8_hwc(image)

    assert converted.shape == (4, 5, 3)
    assert converted.dtype == np.uint8
    assert np.all(converted == 127)


@requires_gr00t
def test_flat_statistics_require_min_and_max(tmp_path):
    stats = {
        modality: {name: [0.0] * (8 if modality == "state" else 7) for name in ("mean", "std", "q01", "q99")}
        for modality in ("state", "action")
    }
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(stats), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 8 state statistics"):
        load_libero_statistics(path)


@requires_gr00t
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


@requires_gr00t
def test_libero_input_from_env_obs_exposes_adapter_tensors():
    obs, _, _ = _raw_libero_batch()

    policy_input = LiberoGr00tInput.from_env_obs(obs)

    assert list(policy_input.images) == list(LIBERO_IMAGE_KEYS)
    assert policy_input.images[LIBERO_IMAGE_KEYS[0]].dtype == torch.uint8
    assert policy_input.images[LIBERO_IMAGE_KEYS[0]].shape == (1, 8, 8, 3)
    assert policy_input.state.shape == (1, 8)
    assert policy_input.task == ["pick up the bowl"]
    assert policy_input.actions is None


@requires_gr00t
def test_libero_input_from_data_proto_converts_gripper():
    obs, actions, _ = _raw_libero_batch()
    actions = actions.clone()
    actions[..., -1] = -1.0

    policy_input = LiberoGr00tInput.from_data_proto(obs, actions=actions)

    assert policy_input.actions is not None
    assert policy_input.actions.shape == (1, 16, 7)
    torch.testing.assert_close(policy_input.actions[..., -1], torch.ones(1, 16))


@requires_gr00t
def test_libero_input_accepts_bfloat16_dataproto():
    obs, actions, _ = _raw_libero_batch()
    for key, value in obs.batch.items():
        if torch.is_floating_point(value):
            obs.batch[key] = value.to(torch.bfloat16)

    policy_input = LiberoGr00tInput.from_data_proto(obs, actions=actions.to(torch.bfloat16))

    assert policy_input.state.dtype == torch.float32
    assert policy_input.actions is not None
    assert policy_input.actions.dtype == torch.float32


@requires_gr00t
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
